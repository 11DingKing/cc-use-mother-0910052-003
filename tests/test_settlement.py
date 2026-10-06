"""订单生命周期与成交分片结算闭环测试。

覆盖：部分成交后撤单只释放剩余、分片原始价格/时间/费用保留、
重复/迟到/超量回报、跨日收盘作废、失败重试幂等、分页稳定性、
重启重放与任意恢复点的账户-明细对账。
"""

import os
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from app.trading.base import Order, OrderSide, OrderStatus, OrderType
from app.trading.simulation_adapter import SimulationAdapter
from app.trading.settlement import SettlementError, SettlementLedger
from app.services.trading_service import TradingService, TradingException


INITIAL_CASH = Decimal("1000000")


@pytest.fixture
def adapter():
    adapter = SimulationAdapter({"initial_cash": str(INITIAL_CASH)})
    adapter.connect()
    adapter.set_quote("000001", 10.0)
    return adapter


def _resting_buy(adapter, quantity=1000, price="9.0", code="000001"):
    """下一笔不会立即成交的限价买单（当前价 10 > 限价 9）。"""
    order = Order(
        order_id=adapter._generate_order_id(),
        stock_code=code,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=Decimal(price),
    )
    return adapter.place_order(order)


class TestPartialFillCancel:
    """部分成交后撤单：只释放剩余数量。"""

    def test_partial_then_cancel_releases_only_remaining(self, adapter):
        order = _resting_buy(adapter, quantity=1000, price="9.0")
        assert order.status == OrderStatus.SUBMITTED

        account = adapter.get_account()
        assert account.frozen_cash == Decimal("9000.00")
        assert account.available_cash == INITIAL_CASH - Decimal("9000")

        t1 = adapter.process_trade_report(
            order.order_id, "RPT-1", 300, Decimal("9.0"),
            trade_time=datetime(2026, 10, 6, 10, 0),
        )
        t2 = adapter.process_trade_report(
            order.order_id, "RPT-2", 300, Decimal("8.9"),
            trade_time=datetime(2026, 10, 6, 10, 30),
        )

        order = adapter.get_order(order.order_id)
        assert order.status == OrderStatus.PARTIAL_FILLED
        assert order.filled_quantity == 600
        assert order.remaining_quantity == 400

        # 撤单成功（部分成交单可撤）
        assert adapter.cancel_order(order.order_id) is True
        order = adapter.get_order(order.order_id)
        assert order.status == OrderStatus.CANCELLED
        assert order.cancelled_quantity == 400
        assert order.filled_quantity == 600
        assert order.remaining_quantity == 0

        # 账户：冻结全部释放，只扣两片实际成交金额与费用
        account = adapter.get_account()
        assert account.frozen_cash == Decimal("0")
        fee1 = t1.commission
        fee2 = t2.commission
        expected_cash = (
            INITIAL_CASH
            - Decimal("300") * Decimal("9.0")
            - Decimal("300") * Decimal("8.9")
            - fee1 - fee2
        )
        assert account.available_cash == expected_cash

        # 持仓 600 股，加权成本 = (2700 + 2670)/600
        pos = adapter.get_position("000001")
        assert pos.quantity == 600
        assert pos.available_quantity == 600
        assert pos.avg_cost == Decimal("5370") / Decimal("600")

        # 每个分片保留原始价格与时间
        assert len(order.trades) == 2
        assert order.trades[0].price == Decimal("9.0")
        assert order.trades[1].price == Decimal("8.9")
        assert order.trades[0].trade_time == datetime(2026, 10, 6, 10, 0)
        assert order.trades[1].trade_time == datetime(2026, 10, 6, 10, 30)
        # 订单聚合价为加权均价，不覆盖分片原价
        assert order.filled_price == Decimal("5370") / Decimal("600")

        assert adapter.reconcile() == []

    def test_each_fragment_has_own_commission(self, adapter):
        order = _resting_buy(adapter, quantity=1000, price="9.0")
        t1 = adapter.process_trade_report(order.order_id, "A", 100, Decimal("9.0"))
        t2 = adapter.process_trade_report(order.order_id, "B", 100, Decimal("9.0"))

        # 每片金额 900，按 0.0003 不足 5 元，各收最低佣金 5 元
        assert t1.commission == Decimal("5")
        assert t2.commission == Decimal("5")
        refreshed = adapter.get_order(order.order_id)
        assert refreshed.commission == Decimal("10")
        assert refreshed.commission == t1.commission + t2.commission

    def test_sell_partial_cancel_restores_remaining_shares(self, adapter):
        # 先建仓 1000 股
        filled = adapter.buy("000001", 1000, 10.0, OrderType.LIMIT)
        assert filled.status == OrderStatus.FILLED

        order = Order(
            order_id=adapter._generate_order_id(),
            stock_code="000001",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=800,
            price=Decimal("11.0"),
        )
        adapter.place_order(order)
        pos = adapter.get_position("000001")
        assert pos.quantity == 1000
        assert pos.available_quantity == 200
        assert pos.frozen_quantity == 800

        adapter.process_trade_report(order.order_id, "S1", 300, Decimal("11.0"))
        assert adapter.cancel_order(order.order_id) is True

        pos = adapter.get_position("000001")
        assert pos.quantity == 700          # 1000 - 300 成交
        assert pos.frozen_quantity == 0
        assert pos.available_quantity == 700  # 剩余 500 全部解冻

        refreshed = adapter.get_order(order.order_id)
        assert refreshed.filled_quantity == 300
        assert refreshed.cancelled_quantity == 500
        assert adapter.reconcile() == []


class TestIdempotentReports:
    """重复、迟到、超量回报。"""

    def test_duplicate_report_settles_once(self, adapter):
        order = _resting_buy(adapter, quantity=1000)
        t1 = adapter.process_trade_report(order.order_id, "DUP", 200, Decimal("9.0"))

        before = adapter.get_account().available_cash
        again = adapter.process_trade_report(order.order_id, "DUP", 200, Decimal("9.0"))

        assert again.trade_id == t1.trade_id
        # 现金、持仓、分片数均不变
        assert adapter.get_account().available_cash == before
        assert adapter.get_position("000001").quantity == 200
        assert len(adapter.get_order(order.order_id).trades) == 1

    def test_late_report_after_cancel_rejected(self, adapter):
        order = _resting_buy(adapter, quantity=1000)
        adapter.process_trade_report(order.order_id, "L1", 200, Decimal("9.0"))
        assert adapter.cancel_order(order.order_id) is True

        with pytest.raises(SettlementError, match="迟到"):
            adapter.process_trade_report(order.order_id, "L2", 100, Decimal("9.0"))

        # 被拒回报不产生任何分片
        assert len(adapter.get_order(order.order_id).trades) == 1
        assert adapter.reconcile() == []

    def test_report_after_full_fill_rejected(self, adapter):
        order = _resting_buy(adapter, quantity=100)
        adapter.process_trade_report(order.order_id, "F1", 100, Decimal("9.0"))
        with pytest.raises(SettlementError):
            adapter.process_trade_report(order.order_id, "F2", 1, Decimal("9.0"))

    def test_over_quantity_report_rejected(self, adapter):
        order = _resting_buy(adapter, quantity=100)
        with pytest.raises(SettlementError, match="超过剩余"):
            adapter.process_trade_report(order.order_id, "O1", 101, Decimal("9.0"))
        assert adapter.get_order(order.order_id).status == OrderStatus.SUBMITTED
        assert adapter.get_account().frozen_cash == Decimal("900.00")

    def test_unknown_order_report_rejected(self, adapter):
        with pytest.raises(SettlementError, match="订单不存在"):
            adapter.process_trade_report("NOPE", "X", 100, Decimal("9"))

    def test_trade_id_belonging_to_other_order_rejected(self, adapter):
        o1 = _resting_buy(adapter, 100, "9.0")
        o2 = _resting_buy(adapter, 100, "9.0")
        adapter.process_trade_report(o1.order_id, "SHARED", 100, Decimal("9.0"))
        with pytest.raises(SettlementError, match="属于其他订单"):
            adapter.process_trade_report(o2.order_id, "SHARED", 100, Decimal("9.0"))


class TestDayEnd:
    """跨日收盘。"""

    def test_expire_active_orders_releases_freeze(self, adapter):
        order = _resting_buy(adapter, quantity=1000, price="9.0")
        adapter.process_trade_report(order.order_id, "D1", 200, Decimal("9.0"))

        expired = adapter.expire_day_orders()
        assert len(expired) == 1

        refreshed = adapter.get_order(order.order_id)
        assert refreshed.status == OrderStatus.EXPIRED
        assert refreshed.cancel_reason == "收盘作废"
        assert refreshed.cancelled_quantity == 800
        assert adapter.get_account().frozen_cash == 0

        # 收盘后的回报一律拒绝
        with pytest.raises(SettlementError):
            adapter.process_trade_report(order.order_id, "D2", 100, Decimal("9.0"))
        # 重复收盘幂等
        assert adapter.expire_day_orders() == []
        assert adapter.reconcile() == []

    def test_expired_order_not_matched_by_later_quote(self, adapter):
        order = _resting_buy(adapter, quantity=1000, price="9.0")
        adapter.expire_day_orders()
        # 次日行情跌入限价区间，也不应再成交
        adapter.set_quote("000001", 8.0)
        assert adapter.get_order(order.order_id).status == OrderStatus.EXPIRED
        assert adapter.get_order(order.order_id).filled_quantity == 0


class TestIdempotentPlace:
    """失败重试：同一 client_order_id 不重复下单。"""

    def test_retry_with_same_client_id_returns_same_order(self, adapter):
        order = Order(
            order_id=adapter._generate_order_id(),
            client_order_id="REQ-77",
            stock_code="000001",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=Decimal("9.0"),
        )
        first = adapter.place_order(order)

        retry = Order(
            order_id=adapter._generate_order_id(),  # 重试生成了新 ID
            client_order_id="REQ-77",
            stock_code="000001",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=Decimal("9.0"),
        )
        second = adapter.place_order(retry)

        assert second.order_id == first.order_id
        assert adapter.count_orders() == 1
        # 冻结只发生一次
        assert adapter.get_account().frozen_cash == Decimal("9000.00")

    def test_rejected_order_retry_keeps_stable_result(self, adapter):
        order = Order(
            order_id=adapter._generate_order_id(),
            client_order_id="REQ-BAD",
            stock_code="000001",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10_000_000,
            price=Decimal("9.0"),
        )
        result = adapter.place_order(order)
        assert result.status == OrderStatus.REJECTED

        retry = Order(
            order_id=adapter._generate_order_id(),
            client_order_id="REQ-BAD",
            stock_code="000001",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10_000_000,
            price=Decimal("9.0"),
        )
        again = adapter.place_order(retry)
        assert again.order_id == result.order_id
        assert again.status == OrderStatus.REJECTED


class TestPagination:
    """查询分页稳定性。"""

    def test_orders_pagination_stable_and_complete(self, adapter):
        ids = []
        for i in range(5):
            order = _resting_buy(adapter, quantity=100, price="9.0")
            ids.append(order.order_id)

        seen = []
        for offset in range(0, 5, 2):
            page = adapter.get_orders(limit=2, offset=offset)
            seen.extend(o.order_id for o in page)
        assert sorted(seen) == sorted(ids)
        assert len(set(seen)) == 5
        assert adapter.count_orders() == 5

    def test_trades_pagination_stable_and_complete(self, adapter):
        order = _resting_buy(adapter, quantity=1000)
        for i in range(5):
            adapter.process_trade_report(
                order.order_id, f"T{i}", 100, Decimal("9.0"),
                trade_time=datetime(2026, 10, 6, 10, i),
            )
        seen = []
        for offset in range(0, 5, 2):
            page = adapter.get_trades(order_id=order.order_id, limit=2, offset=offset)
            seen.extend(t.trade_id for t in page)
        assert seen == [f"T{i}" for i in range(5)]

    def test_trades_filter_by_date(self, adapter):
        order = _resting_buy(adapter, quantity=1000)
        adapter.process_trade_report(
            order.order_id, "D-OLD", 100, Decimal("9.0"),
            trade_time=datetime(2026, 10, 6, 10, 0),
        )
        adapter.process_trade_report(
            order.order_id, "D-NEW", 100, Decimal("9.0"),
            trade_time=datetime(2026, 10, 7, 10, 0),
        )
        day1 = adapter.get_trades(trade_date="2026-10-06")
        day2 = adapter.get_trades(trade_date="2026-10-07")
        assert [t.trade_id for t in day1] == ["D-OLD"]
        assert [t.trade_id for t in day2] == ["D-NEW"]


class TestReplayRecovery:
    """重启后重放：任意恢复点账户汇总与订单明细一致。"""

    def test_replay_after_partial_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_url = f"sqlite:///{os.path.join(tmp, 'trade.db')}"
            cfg = {"initial_cash": "1000000", "db_url": db_url}

            first = SimulationAdapter(cfg)
            first.connect()
            first.set_quote("000001", 10.0)
            order = _resting_buy(first, quantity=1000, price="9.0")
            first.process_trade_report(order.order_id, "R1", 300, Decimal("9.0"))
            first.process_trade_report(order.order_id, "R2", 300, Decimal("8.9"))
            first.cancel_order(order.order_id)
            expected_cash = first.get_account().available_cash
            expected_positions = {
                p.stock_code: (p.quantity, p.available_quantity, p.avg_cost)
                for p in first.get_positions()
            }
            assert first.reconcile() == []

            # 重新启动：行情不在持久化范围内，账本状态应完整恢复
            second = SimulationAdapter(cfg)
            second.connect()
            assert second.get_account().available_cash == expected_cash
            assert second.get_account().frozen_cash == Decimal("0")

            restored = second.get_order(order.order_id)
            assert restored.status == OrderStatus.CANCELLED
            assert restored.filled_quantity == 600
            assert restored.cancelled_quantity == 400
            assert [t.trade_id for t in restored.trades] == ["R1", "R2"]
            assert restored.trades[1].price == Decimal("8.9")

            actual_positions = {
                p.stock_code: (p.quantity, p.available_quantity, p.avg_cost)
                for p in second.get_positions()
            }
            assert actual_positions == expected_positions
            assert second.reconcile() == []

    def test_replay_with_active_order_rebuilds_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_url = f"sqlite:///{os.path.join(tmp, 'trade2.db')}"
            cfg = {"initial_cash": "1000000", "db_url": db_url}

            first = SimulationAdapter(cfg)
            first.connect()
            first.set_quote("000001", 10.0)
            order = _resting_buy(first, quantity=1000, price="9.0")
            first.process_trade_report(order.order_id, "R1", 400, Decimal("9.0"))
            # 未撤单、未收完，处于部分成交状态时重启

            second = SimulationAdapter(cfg)
            second.connect()
            restored = second.get_order(order.order_id)
            assert restored.status == OrderStatus.PARTIAL_FILLED
            assert restored.remaining_quantity == 600

            account = second.get_account()
            assert account.frozen_cash == Decimal("5400.00")
            assert second.get_position("000001").quantity == 400
            assert second.reconcile() == []

            # 重启后仍可继续成交与撤单
            second.process_trade_report(order.order_id, "R2", 200, Decimal("9.0"))
            assert second.cancel_order(order.order_id) is True
            assert second.get_account().frozen_cash == 0
            assert second.reconcile() == []

    def test_replay_sell_freeze_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_url = f"sqlite:///{os.path.join(tmp, 'sell.db')}"
            cfg = {"initial_cash": "1000000", "db_url": db_url}

            first = SimulationAdapter(cfg)
            first.connect()
            first.set_quote("000001", 10.0)
            first.buy("000001", 1000, 10.0, OrderType.LIMIT)
            sell = Order(
                order_id=first._generate_order_id(),
                stock_code="000001",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=800,
                price=Decimal("11.0"),
            )
            first.place_order(sell)

            second = SimulationAdapter(cfg)
            second.connect()
            pos = second.get_position("000001")
            assert pos.quantity == 1000
            assert pos.available_quantity == 200
            assert pos.frozen_quantity == 800
            assert second.reconcile() == []

    def test_replay_duplicate_report_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_url = f"sqlite:///{os.path.join(tmp, 'dup.db')}"
            cfg = {"initial_cash": "1000000", "db_url": db_url}

            first = SimulationAdapter(cfg)
            first.connect()
            first.set_quote("000001", 10.0)
            order = _resting_buy(first, quantity=1000, price="9.0")
            first.process_trade_report(order.order_id, "ONCE", 200, Decimal("9.0"))
            cash_before = first.get_account().available_cash

            second = SimulationAdapter(cfg)
            second.connect()
            # 重启后券商网关重推同一回报，不能二次结算
            again = second.process_trade_report(
                order.order_id, "ONCE", 200, Decimal("9.0")
            )
            assert again.quantity == 200
            assert second.get_account().available_cash == cash_before
            assert second.reconcile() == []


class TestServiceLayer:
    """服务层与接口语义。"""

    @pytest.fixture
    def service(self):
        svc = TradingService()
        svc.connect("simulation", {"initial_cash": "1000000"})
        return svc

    def test_service_cancel_partial_order(self, service):
        service.adapter.set_quote("000001", 10.0)
        result = service.buy("000001", 1000, 9.0)
        order_id = result["order_id"]

        service.report_trade(order_id, "W1", 300, 9.0)
        cancelled = service.cancel_order(order_id)
        assert cancelled["status"] == "cancelled"
        assert cancelled["filled_quantity"] == 300
        assert cancelled["cancelled_quantity"] == 700
        # 已成交后再撤被拒
        with pytest.raises(TradingException):
            service.cancel_order(order_id)

    def test_service_late_report_raises(self, service):
        service.adapter.set_quote("000001", 10.0)
        result = service.buy("000001", 1000, 9.0)
        service.cancel_order(result["order_id"])
        with pytest.raises(TradingException, match="迟到"):
            service.report_trade(result["order_id"], "X1", 100, 9.0)

    def test_service_reconcile_balanced(self, service):
        service.adapter.set_quote("000001", 10.0)
        result = service.buy("000001", 500, 10.0)
        report = service.reconcile()
        assert report["balanced"] is True
        assert report["problems"] == []
        assert result["trades"][0]["price"] == 10.0

    def test_service_orders_page_shape(self, service):
        service.adapter.set_quote("000001", 10.0)
        service.buy("000001", 100, 10.0)
        page = service.get_orders(limit=1)
        assert page["total"] == 1
        assert page["limit"] == 1
        assert len(page["items"]) == 1

    def test_service_idempotent_buy_retry(self, service):
        service.adapter.set_quote("000001", 10.0)
        first = service.buy("000001", 1000, 9.0, client_order_id="C-1")
        second = service.buy("000001", 1000, 9.0, client_order_id="C-1")
        assert first["order_id"] == second["order_id"]

    def test_service_day_end(self, service):
        service.adapter.set_quote("000001", 10.0)
        service.buy("000001", 1000, 9.0)
        result = service.settle_day_end()
        assert result["expired_count"] == 1
        assert result["orders"][0]["status"] == "expired"


class TestMarketOrderAndMultiFragment:
    """市价单滑点结算与多分片场景。"""

    def test_market_buy_freezes_and_settles_with_slippage(self, adapter):
        cash0 = adapter.get_account().available_cash
        order = Order(
            order_id=adapter._generate_order_id(),
            stock_code="000001",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,
        )
        result = adapter.place_order(order)
        assert result.status == OrderStatus.FILLED
        trade = result.trades[0]
        # 成交价 = 10 * (1 + 0.001)
        assert trade.price == Decimal("10.00") * Decimal("1.001")
        account = adapter.get_account()
        assert account.frozen_cash == 0
        assert account.available_cash == cash0 - trade.price * 1000 - trade.commission
        assert adapter.reconcile() == []

    def test_three_fragments_then_full_fill(self, adapter):
        order = _resting_buy(adapter, quantity=1000, price="9.0")
        adapter.process_trade_report(order.order_id, "P1", 200, Decimal("9.0"))
        adapter.process_trade_report(order.order_id, "P2", 300, Decimal("9.1"))
        adapter.process_trade_report(order.order_id, "P3", 500, Decimal("8.8"))

        refreshed = adapter.get_order(order.order_id)
        assert refreshed.status == OrderStatus.FILLED
        assert refreshed.filled_quantity == 1000
        assert refreshed.remaining_quantity == 0
        assert len(refreshed.trades) == 3
        # 终态订单撤单被拒
        assert adapter.cancel_order(order.order_id) is False
        assert adapter.get_account().frozen_cash == 0
        assert adapter.reconcile() == []

    def test_sell_full_lifecycle_with_reconcile(self, adapter):
        adapter.buy("000001", 1000, 10.0, OrderType.LIMIT)
        order = Order(
            order_id=adapter._generate_order_id(),
            stock_code="000001",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=Decimal("11.0"),
        )
        adapter.place_order(order)
        adapter.process_trade_report(order.order_id, "Q1", 400, Decimal("11.0"))
        adapter.process_trade_report(order.order_id, "Q2", 600, Decimal("11.2"))

        refreshed = adapter.get_order(order.order_id)
        assert refreshed.status == OrderStatus.FILLED
        assert adapter.get_position("000001") is None
        assert adapter.get_account().frozen_cash == 0
        # 两笔卖出分片分别保留原始价格
        assert [t.price for t in refreshed.trades] == [
            Decimal("11.0"), Decimal("11.2")
        ]
        assert adapter.reconcile() == []


class TestSettlementLedgerUnit:
    """账本费用与直接结算单元测试。"""

    def test_buy_commission_includes_min(self):
        ledger = SettlementLedger(Decimal("100000"))
        amount = Decimal("900")
        fee = ledger.calculate_commission(OrderSide.BUY, amount)
        assert fee == Decimal("5")

    def test_sell_commission_includes_stamp_tax(self):
        ledger = SettlementLedger(
            Decimal("100000"),
            commission_rate=Decimal("0.0003"),
            min_commission=Decimal("5"),
            stamp_tax_rate=Decimal("0.001"),
        )
        amount = Decimal("100000")
        fee = ledger.calculate_commission(OrderSide.SELL, amount)
        assert fee == Decimal("100000") * Decimal("0.0003") + Decimal("100000") * Decimal("0.001")

    def test_reject_buy_beyond_available(self):
        ledger = SettlementLedger(Decimal("100"))
        order = Order(
            order_id="X", stock_code="000001", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, price=Decimal("2"),
        )
        with pytest.raises(SettlementError, match="可用资金不足"):
            ledger.freeze_for_order(order, Decimal("2"))
        # 拒绝后无任何冻结副作用
        assert ledger.get_account().frozen_cash == 0
        assert ledger.get_account().available_cash == Decimal("100")
