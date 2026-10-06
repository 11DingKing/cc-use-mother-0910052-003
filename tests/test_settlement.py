"""订单生命周期与成交分片结算闭环测试。

覆盖：部分成交后撤单、分片原始价格/时间、重复/迟到回报、跨日收盘、
失败重试、键集分页、对账、SQLite 重启重放与落账原子性。
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from app.trading.base import Order, OrderSide, OrderStatus, OrderType
from app.trading.settlement import (
    SettlementLedger,
    SettlementError,
    TradeReport,
)
from app.trading.simulation_adapter import SimulationAdapter
from app.trading.repository import (
    InMemoryTradeRepository,
    SqliteTradeRepository,
    DuplicateTradeError,
)


def make_buy(oid="O1", qty=1000, price="9.00"):
    return Order(
        order_id=oid,
        stock_code="000001",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=qty,
        price=Decimal(price),
    )


@pytest.fixture
def ledger():
    return SettlementLedger(
        initial_cash=Decimal("100000"),
        commission_rate=Decimal("0.0003"),
        min_commission=Decimal("5"),
        stamp_tax_rate=Decimal("0.001"),
    )


class TestPartialFillAndCancel:
    """部分成交后撤单：只释放剩余数量，保留每笔分片原始价格与时间。"""

    def test_partial_fill_then_cancel_releases_only_remaining(self, ledger):
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        ledger.set_clock(lambda: t0)
        order = ledger.submit_order(make_buy("O1", 1000, "9.00"))

        t1 = t0 + timedelta(minutes=1)
        ledger.apply_trade_report(TradeReport("O1", "T1", 300, Decimal("9.10"), t1))

        assert order.status == OrderStatus.PARTIAL_FILLED
        assert order.filled_quantity == 300
        assert order.remaining_quantity == 700
        # 在途买单冻结剩余 700 股的资金
        account = ledger.get_account()
        assert account.frozen_cash == Decimal("9.00") * 700
        assert account.available_cash == Decimal("100000") - Decimal("9.10") * 300 - Decimal("5") - Decimal("9.00") * 700

        t2 = t0 + timedelta(minutes=2)
        ledger.cancel_order("O1", cancelled_at=t2)

        order = ledger.get_order("O1")
        assert order.status == OrderStatus.CANCELLED
        assert order.filled_quantity == 300          # 已成交部分保留
        assert order.cancelled_quantity == 700       # 只释放剩余
        assert order.remaining_quantity == 0
        assert order.cancelled_at == t2

        account = ledger.get_account()
        assert account.frozen_cash == 0              # 剩余冻结全部释放
        # 现金只扣已成交 300 股的价款与费用
        assert account.available_cash == Decimal("100000") - Decimal("9.10") * 300 - Decimal("5")

        # 持仓只含已成交部分
        pos = ledger.get_position("000001")
        assert pos.quantity == 300
        assert pos.available_quantity == 300
        assert pos.avg_cost == Decimal("9.10")

    def test_each_trade_keeps_original_price_and_time(self, ledger):
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        ledger.set_clock(lambda: t0)
        order = ledger.submit_order(make_buy("O1", 1000, "9.50"))

        ledger.apply_trade_report(TradeReport("O1", "T1", 200, Decimal("9.10"), t0 + timedelta(seconds=5)))
        ledger.apply_trade_report(TradeReport("O1", "T2", 300, Decimal("9.30"), t0 + timedelta(seconds=9)))
        ledger.cancel_order("O1")

        trades = order.trades
        assert [(t.trade_id, t.quantity, t.price) for t in trades] == [
            ("T1", 200, Decimal("9.10")),
            ("T2", 300, Decimal("9.30")),
        ]
        assert trades[0].traded_at < trades[1].traded_at
        # 订单加权均价来自分片，且不覆盖分片原始价
        assert order.filled_price == (Decimal("9.10") * 200 + Decimal("9.30") * 300) / 500
        assert trades[0].price == Decimal("9.10")
        # 费用按片累计，每片单独收最低佣金
        assert order.commission == Decimal("10")

    def test_sell_partial_cancel_releases_position(self, ledger):
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        ledger.set_clock(lambda: t0)
        # 先持仓 1000 股
        buy = Order(order_id="B1", stock_code="000001", side=OrderSide.BUY,
                    order_type=OrderType.LIMIT, quantity=1000, price=Decimal("10"))
        ledger.submit_order(buy)
        ledger.apply_trade_report(TradeReport("B1", "BT", 1000, Decimal("10"), t0))

        sell = Order(order_id="S1", stock_code="000001", side=OrderSide.SELL,
                     order_type=OrderType.LIMIT, quantity=1000, price=Decimal("11"))
        ledger.submit_order(sell)
        ledger.apply_trade_report(TradeReport("S1", "ST1", 400, Decimal("11"), t0 + timedelta(seconds=1)))
        ledger.cancel_order("S1")

        pos = ledger.get_position("000001")
        assert pos.quantity == 600
        assert pos.available_quantity == 600       # 撤单释放剩余持仓
        assert sell.filled_quantity == 400
        assert sell.cancelled_quantity == 600

    def test_cancel_then_fill_rest_keeps_filled_quantity(self, ledger):
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        ledger.set_clock(lambda: t0)
        order = ledger.submit_order(make_buy("O1", 1000, "9"))
        ledger.apply_trade_report(TradeReport("O1", "T1", 1000, Decimal("9"), t0))
        # 已全部成交，撤单被拒
        with pytest.raises(SettlementError):
            ledger.cancel_order("O1")
        assert order.status == OrderStatus.FILLED


class TestIdempotentReports:
    """重复回报与迟到回报必须稳定。"""

    def test_duplicate_report_not_settled_twice(self, ledger):
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        ledger.set_clock(lambda: t0)
        ledger.submit_order(make_buy("O1", 1000, "9"))
        report = TradeReport("O1", "T1", 300, Decimal("9"), t0)

        trade1, new1 = ledger.apply_trade_report(report)
        trade2, new2 = ledger.apply_trade_report(report)  # 重复回报

        assert new1 is True and new2 is False
        assert trade1 is trade2
        order = ledger.get_order("O1")
        assert len(order.trades) == 1
        assert order.filled_quantity == 300
        # 现金只扣一次
        account = ledger.get_account()
        # 现金只扣一次（在途剩余 700 股仍冻结，可用+冻结口径核对）
        assert account.available_cash + account.frozen_cash == (
            Decimal("100000") - Decimal("9") * 300 - Decimal("5")
        )
        assert ledger.reconcile()["balanced"] is True

    def test_late_report_after_cancel_absorbed_within_cancelled(self, ledger):
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        ledger.set_clock(lambda: t0)
        order = ledger.submit_order(make_buy("O1", 1000, "9"))
        ledger.apply_trade_report(TradeReport("O1", "T1", 400, Decimal("9"), t0))
        ledger.cancel_order("O1", cancelled_at=t0 + timedelta(seconds=10))
        assert order.cancelled_quantity == 600

        cash_before = ledger.get_account().available_cash

        # 迟到回报 300 股：吃掉撤销余量，仍属已成交
        trade, is_new = ledger.apply_trade_report(
            TradeReport("O1", "T2", 300, Decimal("9"), t0 + timedelta(seconds=20))
        )
        assert is_new is True
        assert order.filled_quantity == 700
        assert order.cancelled_quantity == 300
        assert order.status == OrderStatus.CANCELLED  # 仍有 300 未成交
        assert ledger.get_position("000001").quantity == 700
        # 迟到分片正常扣款
        assert ledger.get_account().available_cash == cash_before - Decimal("9") * 300 - Decimal("5")

        # 超出可承接数量的迟到回报被拒绝，不产生任何副作用
        cash_before2 = ledger.get_account().available_cash
        with pytest.raises(SettlementError):
            ledger.apply_trade_report(
                TradeReport("O1", "T3", 400, Decimal("9"), t0 + timedelta(seconds=30))
            )
        assert order.filled_quantity == 700
        assert ledger.get_account().available_cash == cash_before2
        assert ledger.reconcile()["balanced"] is True

    def test_report_exceeding_quantity_rejected(self, ledger):
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        ledger.set_clock(lambda: t0)
        order = ledger.submit_order(make_buy("O1", 1000, "9"))
        with pytest.raises(SettlementError):
            ledger.apply_trade_report(TradeReport("O1", "T1", 1001, Decimal("9"), t0))
        # 被拒回报不落任何账
        assert order.filled_quantity == 0
        assert order.trades == []
        assert order.status == OrderStatus.SUBMITTED

    def test_report_unknown_order_rejected(self, ledger):
        with pytest.raises(SettlementError):
            ledger.apply_trade_report(
                TradeReport("NOPE", "T1", 100, Decimal("9"), datetime(2026, 10, 6))
            )


class TestRollover:
    """跨日收盘：在途单撤销，分片归属各自成交日。"""

    def test_rollover_cancels_active_and_keeps_trades(self, ledger):
        day1 = datetime(2026, 10, 6, 14, 59, 0)
        ledger.set_clock(lambda: day1)
        o1 = ledger.submit_order(make_buy("O1", 1000, "9"))
        ledger.apply_trade_report(TradeReport("O1", "T1", 400, Decimal("9"), day1))
        o2 = ledger.submit_order(
            Order(order_id="O2", stock_code="000002", side=OrderSide.BUY,
                  order_type=OrderType.LIMIT, quantity=200, price=Decimal("5"))
        )

        close = datetime(2026, 10, 6, 15, 0, 0)
        closed = ledger.rollover_trading_day(close_time=close)

        assert {o.order_id for o in closed} == {"O1", "O2"}
        assert o1.status == OrderStatus.CANCELLED and o1.filled_quantity == 400
        assert o1.cancelled_quantity == 600
        assert o2.status == OrderStatus.CANCELLED and o2.cancelled_quantity == 200
        # 在途冻结全部释放，已成交分片保留
        assert ledger.get_account().frozen_cash == 0
        assert ledger.get_position("000001").quantity == 400

        # 收盘后再无在途单可撤
        assert ledger.rollover_trading_day(close_time=close) == []

    def test_trades_keep_own_trade_date_and_daily_summary(self, ledger):
        d1 = datetime(2026, 10, 6, 10, 0, 0)
        d2 = datetime(2026, 10, 7, 10, 0, 0)
        ledger.set_clock(lambda: d1)
        o = ledger.submit_order(make_buy("O1", 1000, "9"))
        ledger.apply_trade_report(TradeReport("O1", "T1", 400, Decimal("9"), d1))
        ledger.cancel_order("O1", cancelled_at=datetime(2026, 10, 6, 15))

        # 次日另一笔订单
        ledger.set_clock(lambda: d2)
        o2 = ledger.submit_order(
            Order(order_id="O2", stock_code="000001", side=OrderSide.BUY,
                  order_type=OrderType.LIMIT, quantity=200, price=Decimal("9"))
        )
        ledger.apply_trade_report(TradeReport("O2", "T2", 200, Decimal("9"), d2))

        page_d1 = ledger.get_trades(trade_date="2026-10-06")
        assert [t.trade_id for t in page_d1] == ["T1"]
        page_d2 = ledger.get_trades(trade_date="2026-10-07")
        assert [t.trade_id for t in page_d2] == ["T2"]

        summary = ledger.get_daily_summary("2026-10-06")
        assert summary["buy_quantity"] == 400
        assert summary["buy_amount"] == 3600.0
        assert summary["sell_quantity"] == 0


class TestRetryAndRejection:
    """失败重试：重复提交不重复冻结，拒绝单不产生现金/持仓影响。"""

    def test_resubmit_same_order_id_is_idempotent(self, ledger):
        order = make_buy("O1", 1000, "9")
        first = ledger.submit_order(order)
        second = ledger.submit_order(make_buy("O1", 1000, "9"))
        assert first is second
        assert len(ledger.get_orders(limit=100).items) == 1
        # 冻结只有一份
        assert ledger.get_account().frozen_cash == Decimal("9") * 1000

    def test_insufficient_funds_rejected_no_side_effect(self, ledger):
        with pytest.raises(SettlementError):
            ledger.submit_order(make_buy("BIG", 100000, "9"))
        account = ledger.get_account()
        assert account.available_cash == Decimal("100000")
        assert account.frozen_cash == 0
        assert ledger.get_positions() == []

    def test_persistence_failure_rolls_back_ledger_state(self):
        repo = InMemoryTradeRepository()

        class FlakyRepository(InMemoryTradeRepository):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def save_fill(self, trade, order):
                self.calls += 1
                raise RuntimeError("模拟数据库宕机")

        flaky = FlakyRepository()
        led = SettlementLedger(initial_cash=Decimal("100000"), repository=flaky)
        led.submit_order(make_buy("O1", 1000, "9"))
        with pytest.raises(RuntimeError):
            led.apply_trade_report(
                TradeReport("O1", "T1", 100, Decimal("9"), datetime(2026, 10, 6, 10))
            )
        # 落账失败：内存状态整体回滚，无“半片成交”
        order = led.get_order("O1")
        assert order.filled_quantity == 0
        assert order.trades == []
        assert order.status == OrderStatus.SUBMITTED

        # 重试（同一 trade_id 语义上重新送达）可以成功
        ok_repo = InMemoryTradeRepository()
        # 直接在同一账本换仓储后重试
        led._repository = ok_repo
        trade, is_new = led.apply_trade_report(
            TradeReport("O1", "T1", 100, Decimal("9"), datetime(2026, 10, 6, 10))
        )
        assert is_new is True
        assert led.get_order("O1").filled_quantity == 100


class TestPagination:
    """查询分页：稳定全序，翻页不重不漏。"""

    def test_orders_keyset_pagination_stable(self, ledger):
        base = datetime(2026, 10, 6, 9, 30, 0)
        for i in range(5):
            ledger.set_clock(lambda i=i: base + timedelta(seconds=i))
            ledger.submit_order(make_buy(f"O{i}", 100, "9"))

        seen = []
        cursor = None
        for _ in range(10):
            page = ledger.get_orders(limit=2, cursor=cursor)
            seen.extend(page.items)
            if not page.has_more:
                assert page.next_cursor is None
                break
            cursor = page.next_cursor
        assert len(seen) == 5
        assert len({o.order_id for o in seen}) == 5
        # 倒序稳定
        assert [o.order_id for o in seen] == [f"O{i}" for i in range(4, -1, -1)]

    def test_trades_keyset_pagination(self, ledger):
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        ledger.set_clock(lambda: t0)
        order = ledger.submit_order(make_buy("O1", 1000, "9"))
        for i in range(5):
            ledger.apply_trade_report(
                TradeReport("O1", f"T{i}", 100, Decimal("9"), t0 + timedelta(seconds=i))
            )

        seen, cursor = [], None
        while True:
            page = ledger.get_trades(order_id="O1", limit=2, cursor=cursor)
            seen.extend(page.items)
            if not page.has_more:
                break
            cursor = page.next_cursor
        assert [t.trade_id for t in seen] == [f"T{i}" for i in range(4, -1, -1)]

    def test_invalid_cursor_raises(self, ledger):
        with pytest.raises(SettlementError):
            ledger.get_orders(cursor="not-a-cursor")

    def test_filters(self, ledger):
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        ledger.set_clock(lambda: t0)
        ledger.submit_order(make_buy("O1", 1000, "9"))
        ledger.apply_trade_report(TradeReport("O1", "T1", 1000, Decimal("9"), t0))
        other = Order(order_id="O2", stock_code="000002", side=OrderSide.BUY,
                      order_type=OrderType.LIMIT, quantity=100, price=Decimal("5"))
        ledger.submit_order(other)

        assert {o.order_id for o in ledger.get_orders(stock_code="000001", limit=100)} == {"O1"}
        assert {o.order_id for o in ledger.get_orders(status=OrderStatus.FILLED, limit=100)} == {"O1"}
        assert {o.order_id for o in ledger.get_orders(status=OrderStatus.SUBMITTED, limit=100)} == {"O2"}
        assert {o.order_id for o in ledger.get_orders(trade_date="2026-10-06", limit=100)} == {"O1", "O2"}


class TestReconcile:
    """任意恢复点账户汇总与订单明细可核对。"""

    def test_reconcile_balanced_after_mixed_flow(self, ledger):
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        ledger.set_clock(lambda: t0)
        # 买单部分成交后撤单
        o1 = ledger.submit_order(make_buy("O1", 1000, "9"))
        ledger.apply_trade_report(TradeReport("O1", "T1", 400, Decimal("9"), t0))
        ledger.cancel_order("O1")
        # 卖单（针对持仓）部分成交
        s1 = Order(order_id="S1", stock_code="000001", side=OrderSide.SELL,
                   order_type=OrderType.LIMIT, quantity=400, price=Decimal("10"))
        ledger.submit_order(s1)
        ledger.apply_trade_report(TradeReport("S1", "ST", 100, Decimal("10"), t0 + timedelta(seconds=1)))

        result = ledger.reconcile()
        assert result["balanced"] is True, result["problems"]
        # 现金 = 初始 - 买入价款费用 + 卖出价款费用
        expect_cash = (
            Decimal("100000")
            - Decimal("9") * 400 - Decimal("5")
            + Decimal("10") * 100
            - (max(Decimal("10") * 100 * Decimal("0.0003"), Decimal("5")) + Decimal("10") * 100 * Decimal("0.001"))
        )
        account = ledger.get_account()
        assert account.available_cash + account.frozen_cash == expect_cash
        # 在途卖单冻结 300 股
        pos = ledger.get_position("000001")
        assert pos.quantity == 300
        assert pos.available_quantity == 0


class TestSqliteRecovery:
    """SQLite 持久化：重启重放分片，幂等与对账在恢复后继续成立。"""

    def _url(self, tmp_path):
        return f"sqlite:///{tmp_path / 'trading.db'}"

    def test_restart_replays_trades_and_account(self, tmp_path):
        url = self._url(tmp_path)
        t0 = datetime(2026, 10, 6, 10, 0, 0)

        led1 = SettlementLedger(
            initial_cash=Decimal("100000"),
            repository=SqliteTradeRepository(url),
        )
        led1.set_clock(lambda: t0)
        order = led1.submit_order(make_buy("O1", 1000, "9"))
        led1.apply_trade_report(TradeReport("O1", "T1", 400, Decimal("9"), t0))
        led1.cancel_order("O1", cancelled_at=t0 + timedelta(seconds=5))
        account_before = led1.get_account()
        pos_before = led1.get_position("000001")

        # 重启：新账本从同一数据库重放
        led2 = SettlementLedger(
            initial_cash=Decimal("100000"),
            repository=SqliteTradeRepository(url),
        )
        restored = led2.get_order("O1")
        assert restored.status == OrderStatus.CANCELLED
        assert restored.filled_quantity == 400
        assert restored.cancelled_quantity == 600
        assert [(t.trade_id, t.quantity, str(t.price), t.traded_at) for t in restored.trades] == [
            ("T1", 400, "9.000000", t0)
        ]
        assert led2.get_account().available_cash == account_before.available_cash
        pos2 = led2.get_position("000001")
        assert pos2.quantity == pos_before.quantity
        assert led2.reconcile()["balanced"] is True

        # 恢复后重复回报仍幂等，不会二次结算
        cash = led2.get_account().available_cash
        _, is_new = led2.apply_trade_report(TradeReport("O1", "T1", 400, Decimal("9"), t0))
        assert is_new is False
        assert led2.get_account().available_cash == cash
        assert led2.reconcile()["balanced"] is True

        # 初始资金等配置也从库中恢复
        assert led2.initial_cash == Decimal("100000")

    def test_trade_and_order_commit_atomically(self, tmp_path):
        url = self._url(tmp_path)
        repo = SqliteTradeRepository(url)
        led = SettlementLedger(initial_cash=Decimal("100000"), repository=repo)
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        led.set_clock(lambda: t0)
        led.submit_order(make_buy("O1", 1000, "9"))
        led.apply_trade_report(TradeReport("O1", "T1", 400, Decimal("9"), t0))

        # 直接向仓储重复写同一 trade_id：唯一约束拒绝，订单累计不被改动
        order = led.get_order("O1")
        with pytest.raises(DuplicateTradeError):
            repo.save_fill(
                order.trades[0],
                order,
            )
        reloaded = SettlementLedger(
            initial_cash=Decimal("100000"),
            repository=SqliteTradeRepository(url),
        )
        assert len(reloaded.get_order("O1").trades) == 1

    def test_rejected_order_persisted(self, tmp_path):
        url = self._url(tmp_path)
        led = SettlementLedger(
            initial_cash=Decimal("100"),
            repository=SqliteTradeRepository(url),
        )
        order = make_buy("O1", 1000, "9")
        with pytest.raises(SettlementError):
            led.submit_order(order)
        led.register_terminal_new(order, OrderStatus.REJECTED, "资金不足")

        led2 = SettlementLedger(
            initial_cash=Decimal("100"),
            repository=SqliteTradeRepository(url),
        )
        assert led2.get_order("O1").status == OrderStatus.REJECTED


class TestSnapshotRestore:
    """内存快照恢复：流水不可变，账户派生，无二次结算。"""

    def test_snapshot_restore_roundtrip(self, ledger):
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        ledger.set_clock(lambda: t0)
        ledger.submit_order(make_buy("O1", 1000, "9"))
        ledger.apply_trade_report(TradeReport("O1", "T1", 400, Decimal("9"), t0))
        ledger.cancel_order("O1")
        snap = ledger.snapshot()
        cash = ledger.get_account().available_cash

        led2 = SettlementLedger(initial_cash=Decimal("100000"))
        led2.restore(snap)
        assert led2.get_order("O1").status == OrderStatus.CANCELLED
        assert led2.get_account().available_cash == cash
        assert led2.reconcile()["balanced"] is True


class TestAdapterIntegration:
    """适配器层：部分成交推送、撤单、收盘、查询与 API 友好结构。"""

    @pytest.fixture
    def adapter(self):
        ad = SimulationAdapter({"initial_cash": 100000})
        ad.connect()
        ad.set_quote("000001", 10.0)
        return ad

    def _pending_buy(self, adapter, oid, qty=1000, price=9.0):
        order = Order(
            order_id=oid,
            stock_code="000001",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=qty,
            price=Decimal(str(price)),
        )
        return adapter.place_order(order)

    def test_partial_fill_cancel_flow(self, adapter):
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        order = self._pending_buy(adapter, "O1", 1000, 9.0)
        assert order.status == OrderStatus.SUBMITTED

        adapter.simulate_partial_fill("O1", 300, 9.0, traded_at=t0)
        order = adapter.get_order("O1")
        assert order.status == OrderStatus.PARTIAL_FILLED

        assert adapter.cancel_order("O1") is True
        order = adapter.get_order("O1")
        assert order.status == OrderStatus.CANCELLED
        assert order.remaining_quantity == 0
        data = order.to_dict()
        assert data["filled_quantity"] == 300
        assert data["cancelled_quantity"] == 700
        assert len(data["trades"]) == 1
        assert data["trades"][0]["price"] == 9.0
        assert data["trades"][0]["traded_at"] == t0.isoformat()
        assert adapter.reconcile()["balanced"] is True

    def test_duplicate_report_callback_fires_once(self, adapter):
        t0 = datetime(2026, 10, 6, 10, 0, 0)
        self._pending_buy(adapter, "O1", 1000, 9.0)
        events = []
        adapter.register_callback("on_trade", lambda t: events.append(t.trade_id))

        report = TradeReport("O1", "T1", 100, Decimal("9"), t0)
        adapter.report_trade(report)
        adapter.report_trade(report)
        assert events == ["T1"]

    def test_rollover_and_paginated_orders(self, adapter):
        for i in range(3):
            self._pending_buy(adapter, f"O{i}", 100, 9.0)
        closed = adapter.rollover_trading_day(datetime(2026, 10, 6, 15))
        assert len(closed) == 3

        page = adapter.get_orders(limit=2)
        assert len(page.items) == 2 and page.has_more is True
        page2 = adapter.get_orders(limit=2, cursor=page.next_cursor)
        ids = {o.order_id for o in page.items} | {o.order_id for o in page2.items}
        assert ids == {"O0", "O1", "O2"}

    def test_legacy_get_orders_returns_page_object(self, adapter):
        self._pending_buy(adapter, "O1", 100, 9.0)
        page = adapter.get_orders()
        assert len(page) == 1
