"""订单生命周期与成交分片的结算账本。

设计要点
--------
1. 账本的权威状态只有两类不可丢失的数据：
   - 订单（Order）：生命周期状态机，filled/cancelled/remaining 三者闭合；
   - 成交流水（Trade）：每一笔回报一条不可变分片，trade_id 为全局幂等键。
2. 账户现金、持仓、冻结全部从「成交 + 在途订单」派生，落账（写流水）与
   过账（账户/持仓变动）是同一个原子单元：持久化时同事务提交，内存提交时
   先校验后变更。任何恢复点重放流水都得到同一账户汇总，因此账户与明细
   天然可对账，重复/迟到回报不会产生第二次结算。
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.trading.base import (
    Account,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Trade,
)

logger = logging.getLogger(__name__)

# 可撤销 / 在途的状态
ACTIVE_STATUSES = (
    OrderStatus.PENDING,
    OrderStatus.SUBMITTED,
    OrderStatus.PARTIAL_FILLED,
)
# 终态：不再接受成交或撤单
TERMINAL_STATUSES = (
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
    OrderStatus.FAILED,
)


class SettlementError(Exception):
    """结算被拒绝（数量越界、状态非法等），调用方应将订单置为失败/拒绝。"""


@dataclass
class TradeReport:
    """外部（交易所/撮合器）推送的一笔成交回报。"""

    order_id: str
    trade_id: str
    quantity: int
    price: Decimal
    traded_at: Optional[datetime] = None
    commission: Optional[Decimal] = None  # 缺省由账本费率计算

    def __post_init__(self):
        self.price = Decimal(str(self.price))
        if self.commission is not None:
            self.commission = Decimal(str(self.commission))


@dataclass
class Page:
    """稳定键集分页结果。

    排序键为 (时间, 主键) 全序，cursor 为不透明令牌；
    同一数据状态下，翻页结果不重复、不遗漏，插入新数据不影响向后翻页。
    """

    items: List[Any]
    next_cursor: Optional[str] = None
    has_more: bool = False

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [_to_dict(i) for i in self.items],
            "next_cursor": self.next_cursor,
            "has_more": self.has_more,
        }


def _to_dict(item: Any) -> Any:
    return item.to_dict() if hasattr(item, "to_dict") else item


def encode_cursor(ts: datetime, key: str) -> str:
    raw = f"{ts.isoformat()}|{key}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_cursor(cursor: str) -> Tuple[datetime, str]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
        ts_iso, key = raw.split("|", 1)
        return datetime.fromisoformat(ts_iso), key
    except Exception as exc:  # 非法游标按“无游标”处理，结果仍稳定
        raise SettlementError(f"非法分页游标: {cursor}") from exc


class SettlementLedger:
    """内存结算账本，可挂载持久化仓储实现重启恢复。"""

    def __init__(
        self,
        initial_cash: Decimal = Decimal("1000000"),
        commission_rate: Decimal = Decimal("0.0003"),
        min_commission: Decimal = Decimal("5"),
        stamp_tax_rate: Decimal = Decimal("0.001"),
        slippage_rate: float = 0.001,
        account_id: Optional[str] = None,
        broker: str = "模拟交易",
        time_func: Callable[[], datetime] = datetime.now,
        repository: Optional["TradeRepository"] = None,
    ):
        self.initial_cash = Decimal(str(initial_cash))
        self.commission_rate = Decimal(str(commission_rate))
        self.min_commission = Decimal(str(min_commission))
        self.stamp_tax_rate = Decimal(str(stamp_tax_rate))
        self.slippage_rate = slippage_rate
        self.account_id = account_id or ("SIM_" + datetime.now().strftime("%Y%m%d%H%M%S"))
        self.broker = broker
        self._now = time_func
        self._repository = repository

        self._orders: Dict[str, Order] = {}
        self._quotes: Dict[str, Decimal] = {}  # 最近价，仅用于市值/盈亏展示

        if self._repository is not None:
            self._restore_from_repository()
            self._persist_meta()

    # ------------------------------------------------------------------
    # 时钟
    # ------------------------------------------------------------------
    def set_clock(self, time_func: Callable[[], datetime]) -> None:
        """注入时钟（跨日收盘测试用）。"""
        self._now = time_func

    def now(self) -> datetime:
        return self._now()

    # ------------------------------------------------------------------
    # 下单 / 状态登记
    # ------------------------------------------------------------------
    def submit_order(
        self,
        order: Order,
        reference_price: Optional[Decimal] = None,
    ) -> Order:
        """登记一笔新订单并做资金/持仓可承接校验。

        同一 order_id 重复提交（失败重试）直接返回既有订单，不重复冻结。
        reference_price 用于市价单的资金校验/冻结参考。
        """
        existing = self._orders.get(order.order_id)
        if existing is not None:
            return existing

        order.status = OrderStatus.SUBMITTED
        order.updated_at = self._now()
        if not order.created_at:
            order.created_at = order.updated_at

        account, positions = self._build_account_positions()

        if order.side == OrderSide.BUY:
            limit = order.price or (
                Decimal(str(reference_price)) if reference_price is not None else None
            )
            if limit is None:
                raise SettlementError("下单缺少可用价格")
            required = limit * order.quantity
            if required > account.available_cash:
                raise SettlementError(
                    f"可用资金不足，需要 {required:.2f}，可用 {account.available_cash:.2f}"
                )
        else:
            pos = positions.get(order.stock_code)
            available = pos.available_quantity if pos else 0
            if available < order.quantity:
                raise SettlementError(
                    f"可用持仓不足，需要 {order.quantity}，可用 {available}"
                )

        self._orders[order.order_id] = order
        self._persist_order(order)
        return order

    def register_terminal_new(
        self,
        order: Order,
        status: OrderStatus,
        reason: str,
    ) -> Order:
        """未通过校验/连接失败的订单：登记为拒绝或失败终态，不产生冻结与成交。"""
        existing = self._orders.get(order.order_id)
        if existing is not None:
            return existing
        order.status = status
        order.error_message = reason
        order.updated_at = self._now()
        if not order.created_at:
            order.created_at = order.updated_at
        self._orders[order.order_id] = order
        self._persist_order(order)
        return order

    def reject_order(self, order_id: str, reason: str) -> Optional[Order]:
        """下单被拒/失败：登记终态，不产生任何成交或冻结。"""
        order = self._orders.get(order_id)
        if order is None or order.status in TERMINAL_STATUSES:
            return order
        order.status = OrderStatus.REJECTED
        order.error_message = reason
        order.updated_at = self._now()
        self._persist_order(order)
        return order

    def fail_order(self, order_id: str, reason: str) -> Optional[Order]:
        order = self._orders.get(order_id)
        if order is None or order.status in TERMINAL_STATUSES:
            return order
        order.status = OrderStatus.FAILED
        order.error_message = reason
        order.updated_at = self._now()
        self._persist_order(order)
        return order

    # ------------------------------------------------------------------
    # 成交回报（幂等核心）
    # ------------------------------------------------------------------
    def apply_trade_report(self, report: TradeReport) -> Tuple[Trade, bool]:
        """处理一笔成交回报，返回 (流水, 是否本次新入账)。

        - 重复回报（trade_id 已存在）：原样返回既有流水，不二次结算；
        - 迟到回报（订单已撤单）：在已撤销数量内承接，超出则拒绝；
        - 数量越界（超过在途+已撤可承接量）：拒绝且不落任何账；
        - 成功入账：流水与订单状态在同一单元内提交（持久化同事务）。
        """
        order = self._orders.get(report.order_id)
        if order is None:
            raise SettlementError(f"成交回报对应订单不存在: {report.order_id}")

        if report.quantity <= 0:
            raise SettlementError("成交数量必须为正数")
        if report.price is None or report.price <= 0:
            raise SettlementError("成交价格必须为正数")

        # 幂等：重复回报
        existing = order.get_trade(report.trade_id)
        if existing is not None:
            return existing, False

        traded_at = report.traded_at or self._now()

        # 可承接数量 = 在途剩余 + 已撤销数量（撤单与成交在交易所侧存在竞争）
        absorbable = order.remaining_quantity + order.cancelled_quantity
        already = order.filled_quantity
        if already + report.quantity > order.quantity or report.quantity > absorbable:
            raise SettlementError(
                f"成交数量 {report.quantity} 超过订单 {order.order_id} 可承接数量 {absorbable}"
            )

        commission = report.commission
        if commission is None:
            commission = self._calc_commission(order.side, report.price, report.quantity)

        trade = Trade(
            trade_id=report.trade_id,
            order_id=order.order_id,
            stock_code=order.stock_code,
            side=order.side,
            quantity=report.quantity,
            price=report.price,
            commission=commission,
            traded_at=traded_at,
            settled=True,  # 随落账一起过账，不存在“已落账未过账”的中间态
        )

        # ==== 原子单元：先在内存变更，持久化同事务；失败整体回滚 ====
        snapshot = self._snapshot_order(order)
        try:
            absorbed_from_cancelled = 0
            if order.status == OrderStatus.CANCELLED:
                # 迟到回报只能吃掉此前“撤销释放”的那部分
                absorbed_from_cancelled = min(report.quantity, order.cancelled_quantity)

            order.filled_quantity += report.quantity
            order.cancelled_quantity -= absorbed_from_cancelled
            order.commission += commission
            order.trades.append(trade)
            order.trades.sort(key=lambda t: (t.traded_at, t.trade_id))
            order.filled_price = self._weighted_avg_price(order)
            order.updated_at = self._now()

            if order.filled_quantity + order.cancelled_quantity >= order.quantity:
                order.status = (
                    OrderStatus.FILLED
                    if order.filled_quantity == order.quantity
                    else OrderStatus.CANCELLED
                )
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIAL_FILLED

            self._persist_fill(trade, order)
        except Exception:
            self._restore_order(order, snapshot)
            raise

        return trade, True

    def _calc_commission(self, side: OrderSide, price: Decimal, quantity: int) -> Decimal:
        amount = price * quantity
        commission = max(amount * self.commission_rate, self.min_commission)
        if side == OrderSide.SELL:
            commission += amount * self.stamp_tax_rate
        return commission.quantize(Decimal("0.000001"))

    @staticmethod
    def _weighted_avg_price(order: Order) -> Decimal:
        total_qty = sum(t.quantity for t in order.trades)
        total_amount = sum(t.amount for t in order.trades)
        return total_amount / total_qty if total_qty else Decimal("0")

    # ------------------------------------------------------------------
    # 撤单：只释放剩余数量，已成交分片原样保留
    # ------------------------------------------------------------------
    def cancel_order(self, order_id: str, cancelled_at: Optional[datetime] = None) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            raise SettlementError(f"订单不存在: {order_id}")
        if order.status in TERMINAL_STATUSES:
            raise SettlementError(f"订单状态为 {order.status.value}，无法撤销")
        if order.remaining_quantity <= 0:
            raise SettlementError("订单已无剩余数量可撤销")

        order.cancelled_quantity += order.remaining_quantity
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = cancelled_at or self._now()
        order.updated_at = order.cancelled_at
        self._persist_order(order)
        return order

    # ------------------------------------------------------------------
    # 跨日收盘：撤掉所有在途单（交易所日单失效），分片归属各自成交日
    # ------------------------------------------------------------------
    def rollover_trading_day(self, close_time: Optional[datetime] = None) -> List[Order]:
        closed_at = close_time or self._now()
        cancelled = []
        for order in self._orders.values():
            if order.status in ACTIVE_STATUSES and order.remaining_quantity > 0:
                order.cancelled_quantity += order.remaining_quantity
                order.status = OrderStatus.CANCELLED
                order.cancelled_at = closed_at
                order.updated_at = closed_at
                cancelled.append(order)
                self._persist_order(order)
        if cancelled:
            logger.info("收盘撤单 %d 笔，成交分片按原成交日保留", len(cancelled))
        return cancelled

    # ------------------------------------------------------------------
    # 账户 / 持仓：全部从流水与在途订单派生
    # ------------------------------------------------------------------
    def set_quote(self, stock_code: str, price: Decimal) -> None:
        self._quotes[stock_code] = Decimal(str(price))

    def _build_account_positions(self) -> Tuple[Account, Dict[str, Position]]:
        cash = self.initial_cash

        # 1) 从已结算流水重建持仓与现金（每笔分片只被累加一次）
        raw: Dict[str, Dict[str, Decimal]] = {}
        all_trades = [t for o in self._orders.values() for t in o.trades]
        for t in all_trades:
            slot = raw.setdefault(
                t.stock_code,
                {"qty": Decimal("0"), "buy_amount": Decimal("0"), "fee": Decimal("0")},
            )
            if t.side == OrderSide.BUY:
                cash -= t.amount + t.commission
                slot["buy_amount"] += t.amount
                slot["qty"] += t.quantity
            else:
                cash += t.amount - t.commission
                slot["qty"] -= t.quantity
            slot["fee"] += t.commission

        # 2) 在途卖单冻结持仓、在途买单冻结资金（撤单自动释放）
        frozen_qty: Dict[str, int] = {}
        frozen_cash = Decimal("0")
        for order in self._orders.values():
            if order.status not in ACTIVE_STATUSES:
                continue
            remaining = order.remaining_quantity
            if remaining <= 0:
                continue
            if order.side == OrderSide.SELL:
                frozen_qty[order.stock_code] = frozen_qty.get(order.stock_code, 0) + remaining
            elif order.price is not None:
                frozen_cash += order.price * remaining

        positions: Dict[str, Position] = {}
        market_value = Decimal("0")
        total_pnl = Decimal("0")
        for stock_code, slot in raw.items():
            qty = int(slot["qty"])
            if qty <= 0:
                continue
            avg_cost = slot["buy_amount"] / qty if slot["buy_amount"] else Decimal("0")
            current_price = self._quotes.get(stock_code) or avg_cost
            mv = current_price * qty
            pnl = (current_price - avg_cost) * qty
            frozen = frozen_qty.get(stock_code, 0)
            positions[stock_code] = Position(
                stock_code=stock_code,
                stock_name=stock_code,
                quantity=qty,
                available_quantity=qty - frozen,
                avg_cost=avg_cost,
                current_price=current_price,
                market_value=mv,
                profit_loss=pnl,
                profit_loss_ratio=float(pnl / avg_cost) if avg_cost > 0 else 0.0,
                updated_at=self._now(),
            )
            market_value += mv
            total_pnl += pnl

        total_assets = cash + market_value
        denominator = total_assets - total_pnl
        account = Account(
            account_id=self.account_id,
            broker=self.broker,
            total_assets=total_assets,
            available_cash=cash - frozen_cash,
            frozen_cash=frozen_cash,
            market_value=market_value,
            profit_loss=total_pnl,
            profit_loss_ratio=float(total_pnl / denominator) if denominator > 0 else 0.0,
            updated_at=self._now(),
        )
        return account, positions

    def get_account(self) -> Account:
        account, _ = self._build_account_positions()
        return account

    def get_positions(self) -> List[Position]:
        _, positions = self._build_account_positions()
        return sorted(positions.values(), key=lambda p: p.stock_code)

    def get_position(self, stock_code: str) -> Optional[Position]:
        _, positions = self._build_account_positions()
        return positions.get(stock_code)

    # ------------------------------------------------------------------
    # 查询：稳定排序 + 键集分页
    # ------------------------------------------------------------------
    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_trade(self, trade_id: str) -> Optional[Trade]:
        for order in self._orders.values():
            found = order.get_trade(trade_id)
            if found:
                return found
        return None

    def get_orders(
        self,
        stock_code: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        trade_date: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Page:
        """订单按 (created_at DESC, order_id DESC) 全序键集分页。"""
        limit = self._normalize_limit(limit)
        orders = list(self._orders.values())
        if stock_code:
            orders = [o for o in orders if o.stock_code == stock_code]
        if status:
            orders = [o for o in orders if o.status == status]
        if trade_date:
            orders = [o for o in orders if o.trade_date.isoformat() == trade_date]

        orders.sort(key=lambda o: (o.created_at, o.order_id), reverse=True)
        return self._paginate(orders, cursor, limit, lambda o: (o.created_at, o.order_id))

    def get_trades(
        self,
        order_id: Optional[str] = None,
        stock_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Page:
        """成交流水按 (traded_at DESC, trade_id DESC) 全序键集分页。"""
        limit = self._normalize_limit(limit)
        trades = [t for o in self._orders.values() for t in o.trades]
        if order_id:
            trades = [t for t in trades if t.order_id == order_id]
        if stock_code:
            trades = [t for t in trades if t.stock_code == stock_code]
        if trade_date:
            trades = [t for t in trades if t.trade_date.isoformat() == trade_date]

        trades.sort(key=lambda t: (t.traded_at, t.trade_id), reverse=True)
        return self._paginate(trades, cursor, limit, lambda t: (t.traded_at, t.trade_id))

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        if limit is None or limit <= 0:
            return 50
        return min(limit, 500)

    def _paginate(self, items: List[Any], cursor: Optional[str], limit: int, key_fn) -> Page:
        if cursor:
            cur_ts, cur_key = decode_cursor(cursor)
            # items 已按时间倒序；保留严格“早于游标”的行
            filtered = []
            for item in items:
                ts, key = key_fn(item)
                if (ts, key) < (cur_ts, cur_key):
                    filtered.append(item)
            items = filtered

        has_more = len(items) > limit
        page_items = items[:limit]
        next_cursor = None
        if has_more and page_items:
            ts, key = key_fn(page_items[-1])
            next_cursor = encode_cursor(ts, key)
        return Page(items=page_items, next_cursor=next_cursor, has_more=has_more)

    def get_daily_summary(self, trade_date: str) -> Dict[str, Any]:
        """某日成交汇总，直接从该日分片派生（跨日收盘后仍可查）。"""
        buys = {"amount": Decimal("0"), "qty": 0, "commission": Decimal("0"), "count": 0}
        sells = {"amount": Decimal("0"), "qty": 0, "commission": Decimal("0"), "count": 0}
        stamp_tax = Decimal("0")
        for order in self._orders.values():
            for t in order.trades:
                if t.trade_date.isoformat() != trade_date:
                    continue
                bucket = buys if t.side == OrderSide.BUY else sells
                bucket["amount"] += t.amount
                bucket["qty"] += t.quantity
                bucket["commission"] += t.commission
                bucket["count"] += 1
                if t.side == OrderSide.SELL:
                    stamp_tax += t.amount * self.stamp_tax_rate

        return {
            "trade_date": trade_date,
            "buy_count": buys["count"],
            "buy_quantity": buys["qty"],
            "buy_amount": float(buys["amount"]),
            "sell_count": sells["count"],
            "sell_quantity": sells["qty"],
            "sell_amount": float(sells["amount"]),
            "commission": float(buys["commission"] + sells["commission"]),
            "stamp_tax": float(stamp_tax),
            "net_cash_flow": float(sells["amount"] - buys["amount"]),
        }

    # ------------------------------------------------------------------
    # 对账：账户汇总必须与订单/成交明细闭合
    # ------------------------------------------------------------------
    def reconcile(self) -> Dict[str, Any]:
        problems: List[str] = []
        for order in self._orders.values():
            trade_qty = sum(t.quantity for t in order.trades)
            if trade_qty != order.filled_quantity:
                problems.append(
                    f"订单 {order.order_id} 分片数量和 {trade_qty} != filled_quantity {order.filled_quantity}"
                )
            closed = order.filled_quantity + order.cancelled_quantity
            if order.status in TERMINAL_STATUSES and closed != order.quantity:
                problems.append(
                    f"订单 {order.order_id} 终态未闭合: {closed} != {order.quantity}"
                )
            if order.status in ACTIVE_STATUSES and closed > order.quantity:
                problems.append(f"订单 {order.order_id} 在途数量越界")
            fee = sum(t.commission for t in order.trades)
            if fee != order.commission:
                problems.append(f"订单 {order.order_id} 费用累计与分片不一致")
            if trade_qty > 0:
                avg = self._weighted_avg_price(order)
                if order.filled_price is None or abs(order.filled_price - avg) > Decimal("0.000001"):
                    problems.append(f"订单 {order.order_id} 加权均价与分片不一致")
            if len({t.trade_id for t in order.trades}) != len(order.trades):
                problems.append(f"订单 {order.order_id} 存在重复 trade_id")

        # 账户现金与分片独立重算
        account, positions = self._build_account_positions()
        expect_cash = self.initial_cash
        stock_qty: Dict[str, int] = {}
        for order in self._orders.values():
            for t in order.trades:
                if t.side == OrderSide.BUY:
                    expect_cash -= t.amount + t.commission
                    stock_qty[t.stock_code] = stock_qty.get(t.stock_code, 0) + t.quantity
                else:
                    expect_cash += t.amount - t.commission
                    stock_qty[t.stock_code] = stock_qty.get(t.stock_code, 0) - t.quantity
        frozen_cash = sum(
            o.price * o.remaining_quantity
            for o in self._orders.values()
            if o.status in ACTIVE_STATUSES and o.side == OrderSide.BUY and o.price
        )
        if account.available_cash + account.frozen_cash != expect_cash:
            problems.append("账户现金与分片现金流不一致")
        if account.frozen_cash != frozen_cash:
            problems.append("冻结资金与在途买单不一致")
        for code, qty in stock_qty.items():
            if qty <= 0:
                if code in positions:
                    problems.append(f"持仓 {code} 应已清仓但仍存在")
                continue
            pos = positions.get(code)
            if pos is None or pos.quantity != qty:
                problems.append(f"持仓 {code} 数量与分片不一致")

        return {
            "balanced": not problems,
            "problems": problems,
            "order_count": len(self._orders),
            "trade_count": sum(len(o.trades) for o in self._orders.values()),
            "cash": float(expect_cash),
            "available_cash": float(account.available_cash),
            "frozen_cash": float(account.frozen_cash),
            "positions": {code: pos.quantity for code, pos in positions.items()},
        }

    # ------------------------------------------------------------------
    # 快照 / 恢复（重启重放）
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "config": {
                "initial_cash": str(self.initial_cash),
                "commission_rate": str(self.commission_rate),
                "min_commission": str(self.min_commission),
                "stamp_tax_rate": str(self.stamp_tax_rate),
                "account_id": self.account_id,
                "broker": self.broker,
            },
            "orders": [o.to_dict(include_trades=True) for o in self._orders.values()],
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """从快照重建：流水不可变，账户/持仓派生，不存在“再次结算”。"""
        self._orders.clear()
        for od in snapshot["orders"]:
            order = Order(
                order_id=od["order_id"],
                stock_code=od["stock_code"],
                side=OrderSide(od["side"]),
                order_type=OrderType(od["order_type"]),
                quantity=od["quantity"],
                price=Decimal(str(od["price"])) if od["price"] is not None else None,
                stop_price=Decimal(str(od["stop_price"])) if od.get("stop_price") else None,
                status=OrderStatus(od["status"]),
                filled_quantity=od["filled_quantity"],
                filled_price=Decimal(str(od["filled_price"])) if od["filled_price"] is not None else None,
                commission=Decimal(str(od["commission"])),
                created_at=datetime.fromisoformat(od["created_at"]),
                updated_at=datetime.fromisoformat(od["updated_at"]),
                cancelled_quantity=od.get("cancelled_quantity", 0),
                cancelled_at=datetime.fromisoformat(od["cancelled_at"]) if od.get("cancelled_at") else None,
                strategy_name=od.get("strategy_name"),
                signal_type=od.get("signal_type"),
                signal_strength=od.get("signal_strength", 0.0) or 0.0,
            )
            for td in od.get("trades", []):
                order.trades.append(
                    Trade(
                        trade_id=td["trade_id"],
                        order_id=td["order_id"],
                        stock_code=td["stock_code"],
                        side=OrderSide(td["side"]),
                        quantity=td["quantity"],
                        price=Decimal(str(td["price"])),
                        commission=Decimal(str(td["commission"])),
                        traded_at=datetime.fromisoformat(td["traded_at"]),
                        settled=td.get("settled", True),
                        created_at=datetime.fromisoformat(td["created_at"]) if td.get("created_at") else None,
                    )
                )
            self._orders[order.order_id] = order

    def _restore_from_repository(self) -> None:
        data = self._repository.load_all()  # type: ignore[union-attr]
        meta = data.get("meta", {})
        if meta:
            self.initial_cash = Decimal(meta.get("initial_cash", str(self.initial_cash)))
            self.commission_rate = Decimal(
                meta.get("commission_rate", str(self.commission_rate))
            )
            self.min_commission = Decimal(
                meta.get("min_commission", str(self.min_commission))
            )
            self.stamp_tax_rate = Decimal(
                meta.get("stamp_tax_rate", str(self.stamp_tax_rate))
            )
            self.account_id = meta.get("account_id", self.account_id)
            self.broker = meta.get("broker", self.broker)
        for order, trades in data["orders"]:
            self._orders[order.order_id] = order
            order.trades = trades
        logger.info(
            "账本恢复: %d 笔订单, %d 笔成交",
            len(self._orders),
            sum(len(o.trades) for o in self._orders.values()),
        )

    # ------------------------------------------------------------------
    # 持久化钩子
    # ------------------------------------------------------------------
    def _persist_meta(self) -> None:
        """账本配置首次落库；库中已有配置时以库为准，不覆盖。"""
        if self._repository is None:
            return
        for key, value in (
            ("initial_cash", str(self.initial_cash)),
            ("commission_rate", str(self.commission_rate)),
            ("min_commission", str(self.min_commission)),
            ("stamp_tax_rate", str(self.stamp_tax_rate)),
            ("account_id", self.account_id),
            ("broker", self.broker),
        ):
            self._repository.save_meta_if_absent(key, value)

    def _persist_order(self, order: Order) -> None:
        if self._repository is not None:
            self._repository.save_order(order)

    def _persist_fill(self, trade: Trade, order: Order) -> None:
        if self._repository is not None:
            self._repository.save_fill(trade, order)
        # 内存账本此处即落账即过账：账户/持仓在下一次查询时由流水派生

    @staticmethod
    def _snapshot_order(order: Order) -> Dict[str, Any]:
        return {
            "status": order.status,
            "filled_quantity": order.filled_quantity,
            "cancelled_quantity": order.cancelled_quantity,
            "commission": order.commission,
            "filled_price": order.filled_price,
            "updated_at": order.updated_at,
            "trades": list(order.trades),
        }

    @staticmethod
    def _restore_order(order: Order, snap: Dict[str, Any]) -> None:
        order.status = snap["status"]
        order.filled_quantity = snap["filled_quantity"]
        order.cancelled_quantity = snap["cancelled_quantity"]
        order.commission = snap["commission"]
        order.filled_price = snap["filled_price"]
        order.updated_at = snap["updated_at"]
        order.trades = snap["trades"]
