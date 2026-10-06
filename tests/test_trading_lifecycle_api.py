"""订单生命周期结算闭环的 API 端到端测试。"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.trading.base import Order, OrderSide, OrderType, Decimal


@pytest.fixture
def client():
    with TestClient(app) as c:
        resp = c.post("/api/trading/connect", json={
            "adapter_type": "simulation",
            "config": {"initial_cash": 100000, "default_quote_price": 10.0},
        })
        assert resp.status_code == 200
        yield c


def _place_pending_buy(client, oid, qty=1000, price=9.0):
    """绕过自动撮合直接登记一笔在途限价买单。"""
    from app.controllers.trading_controller import trading_service

    order = Order(
        order_id=oid,
        stock_code="000001",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=qty,
        price=Decimal(str(price)),
    )
    return trading_service.adapter.place_order(order)


class TestPartialFillCancelLifecycle:
    def test_partial_fill_then_cancel(self, client):
        _place_pending_buy(client, "API1", 1000, 9.0)

        t = datetime(2026, 10, 6, 10, 0, 0).isoformat()
        # 第一笔分片
        r = client.post("/api/trading/orders/API1/trades", json={
            "trade_id": "API1-T1", "quantity": 300, "price": 9.0, "traded_at": t,
        })
        assert r.status_code == 200
        assert r.json()["is_new"] is True

        order = client.get("/api/trading/orders/API1").json()
        assert order["status"] == "partial"
        assert order["filled_quantity"] == 300
        assert order["remaining_quantity"] == 700

        # 重复回报幂等
        r2 = client.post("/api/trading/orders/API1/trades", json={
            "trade_id": "API1-T1", "quantity": 300, "price": 9.0, "traded_at": t,
        })
        assert r2.json()["is_new"] is False
        assert client.get("/api/trading/orders/API1").json()["filled_quantity"] == 300

        # 部分成交后撤单：只释放剩余
        r = client.delete("/api/trading/orders/API1")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "cancelled"
        assert data["filled_quantity"] == 300
        assert data["cancelled_quantity"] == 700
        assert data["remaining_quantity"] == 0
        assert len(data["trades"]) == 1
        assert data["trades"][0]["price"] == 9.0

        # 撤单后不能再撤
        assert client.delete("/api/trading/orders/API1").status_code == 400

        # 成交明细保留原始价格时间
        trades = client.get("/api/trading/orders/API1/trades").json()["trades"]
        assert trades[0]["trade_id"] == "API1-T1"
        assert trades[0]["traded_at"] == t

    def test_cancel_releases_frozen_cash(self, client):
        _place_pending_buy(client, "API2", 1000, 9.0)
        client.post("/api/trading/orders/API2/trades", json={
            "trade_id": "API2-T1", "quantity": 400, "price": 9.0,
        })
        client.delete("/api/trading/orders/API2")
        account = client.get("/api/trading/account").json()
        assert account["frozen_cash"] == 0
        assert account["available_cash"] == pytest.approx(100000 - 400 * 9.0 - 5)

    def test_oversized_report_rejected(self, client):
        _place_pending_buy(client, "API3", 1000, 9.0)
        r = client.post("/api/trading/orders/API3/trades", json={
            "trade_id": "API3-T1", "quantity": 1001, "price": 9.0,
        })
        assert r.status_code == 400
        order = client.get("/api/trading/orders/API3").json()
        assert order["status"] == "submitted"
        assert order["filled_quantity"] == 0

    def test_unknown_order_report_400(self, client):
        r = client.post("/api/trading/orders/GHOST/trades", json={
            "trade_id": "X", "quantity": 100, "price": 9.0,
        })
        assert r.status_code == 400


class TestRolloverAndReconcile:
    def test_rollover_and_reconcile(self, client):
        _place_pending_buy(client, "API4", 1000, 9.0)
        client.post("/api/trading/orders/API4/trades", json={
            "trade_id": "API4-T1", "quantity": 400, "price": 9.0,
        })
        r = client.post("/api/trading/rollover")
        assert r.status_code == 200
        assert "API4" in r.json()["order_ids"]

        order = client.get("/api/trading/orders/API4").json()
        assert order["status"] == "cancelled"
        assert order["filled_quantity"] == 400
        assert order["cancelled_quantity"] == 600

        rec = client.get("/api/trading/reconcile").json()
        assert rec["balanced"] is True
        assert rec["problems"] == []


class TestPagination:
    def test_orders_pagination(self, client):
        for i in range(5):
            _place_pending_buy(client, f"PG{i}", 100, 9.0)

        seen = set()
        cursor = None
        pages = 0
        while True:
            params = {"limit": 2}
            if cursor:
                params["cursor"] = cursor
            data = client.get("/api/trading/orders", params=params).json()
            pages += 1
            for o in data["orders"]:
                assert o["order_id"] not in seen
                seen.add(o["order_id"])
            if not data["has_more"]:
                break
            cursor = data["next_cursor"]
        assert seen == {f"PG{i}" for i in range(5)}
        assert pages == 3

    def test_trades_pagination_and_filter(self, client):
        _place_pending_buy(client, "PGT", 1000, 9.0)
        for i in range(3):
            client.post("/api/trading/orders/PGT/trades", json={
                "trade_id": f"PGT-T{i}", "quantity": 100, "price": 9.0,
            })
        data = client.get("/api/trading/trades", params={"order_id": "PGT", "limit": 2}).json()
        assert len(data["items"]) == 2
        assert data["has_more"] is True
        data2 = client.get("/api/trading/trades", params={
            "order_id": "PGT", "limit": 2, "cursor": data["next_cursor"],
        }).json()
        all_ids = {t["trade_id"] for t in data["items"]} | {t["trade_id"] for t in data2["items"]}
        assert all_ids == {"PGT-T0", "PGT-T1", "PGT-T2"}

    def test_legacy_orders_list_still_works(self, client):
        _place_pending_buy(client, "LEG1", 100, 9.0)
        data = client.get("/api/trading/orders").json()
        assert "orders" in data
        assert any(o["order_id"] == "LEG1" for o in data["orders"])
