"""模拟交易适配器：订单生命周期 + 成交分片结算闭环。

关键语义：

* 下单即冻结（买单冻结限价资金，卖单冻结股数），限价单挂单等待撮合；
* 每次撮合产生一个带唯一 ``trade_id`` 的成交分片，分片独立结算、
  保留原始价格/时间/费用，订单状态随累计成交量推进；
* 撤单只释放 *剩余数量* 的冻结，已成交分片原样保留；
* ``process_trade_report`` 以 trade_id 幂等：重复回报返回首次结果，
  迟到（订单已撤/已成/已作废）与超量回报被拒绝；
* 所有订单与分片写入持久层，重启后重放分片即可重建账户与持仓；
* 查询使用稳定排序 + limit/offset 分页。
"""

import logging
import threading
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from app.trading.base import (
    Account,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Trade,
    TradingAdapter,
)
from app.trading.persistence import DuplicateTrade, TradingStore
from app.trading.settlement import SettlementError, SettlementLedger

logger = logging.getLogger(__name__)


class SimulationAdapter(TradingAdapter):
    """业务模块说明。"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})

        initial_cash = (
            Decimal(str(self.config.get("initial_cash", 1000000)))
            if self.config
            else Decimal("1000000")
        )
        self.commission_rate = (
            Decimal(str(self.config.get("commission_rate", 0.0003)))
            if self.config
            else Decimal("0.0003")
        )
        self.min_commission = (
            Decimal(str(self.config.get("min_commission", 5)))
            if self.config
            else Decimal("5")
        )
        self.stamp_tax_rate = (
            Decimal(str(self.config.get("stamp_tax_rate", 0.001)))
            if self.config
            else Decimal("0.001")
        )
        self.slippage_rate = (
            self.config.get("slippage_rate", 0.001) if self.config else 0.001
        )
        self.default_quote_price = (
            Decimal(str(self.config.get("default_quote_price", 10)))
            if self.config
            else Decimal("10")
        )

        db_url = (
            self.config.get("db_url", "sqlite:///:memory:")
            if self.config
            else "sqlite:///:memory:"
        )
        self._store = TradingStore(db_url)

        account_id = self._store.get_meta("account_id")
        self.ledger = SettlementLedger(
            initial_cash=initial_cash,
            commission_rate=self.commission_rate,
            min_commission=self.min_commission,
            stamp_tax_rate=self.stamp_tax_rate,
            slippage_rate=Decimal(str(self.slippage_rate)),
            account_id=account_id,
        )
        if account_id is None:
            self._store.set_meta("account_id", self.ledger.account_id)
        self._store.set_meta("initial_cash", str(initial_cash))

        self._orders: Dict[str, Order] = {}
        self._lock = threading.RLock()
        self._replaying = False

        # 模拟行情（内存）
        self._quotes: Dict[str, Dict] = {}

        # 从持久层重放，恢复任意恢复点的完整状态
        self._replay()

    # -------------------------------------------------------------- 重放

    def _replay(self) -> None:
        """按账本顺序重放全部订单与成交分片，重建内存状态。

        订单只恢复元数据（冻结由成交/剩余量推导），成交分片重新结算；
        活跃订单未成交部分的冻结在重放末尾按订单剩余量补冻结。
        """
        self._replaying = True
        try:
            order_records = self._store.load_all_orders()
            for rec in order_records:
                order = self._order_from_record(rec)
                self._orders[order.order_id] = order

            trade_records = self._store.load_all_trades()
            for rec in trade_records:
                order = self._orders[rec.order_id]
                trade = self._trade_from_record(rec)
                self.ledger.restore_trade(order, trade)
                order.trades.append(trade)
                order.filled_price = order.average_fill_price

            # 补冻结：活跃订单按剩余量重建下单冻结
            for order in self._orders.values():
                if order.status.is_active and order.remaining_quantity > 0:
                    self.ledger.restore_active_freeze(
                        order, order.remaining_quantity
                    )
        finally:
            self._replaying = False

    @staticmethod
    def _order_from_record(rec) -> Order:
        return Order(
            order_id=rec.order_id,
            client_order_id=rec.client_order_id,
            stock_code=rec.stock_code,
            side=OrderSide(rec.side),
            order_type=OrderType(rec.order_type),
            quantity=rec.quantity,
            price=Decimal(rec.price) if rec.price else None,
            stop_price=Decimal(rec.stop_price) if rec.stop_price else None,
            freeze_price=Decimal(rec.freeze_price) if rec.freeze_price else None,
            status=OrderStatus(rec.status),
            filled_quantity=rec.filled_quantity,
            cancelled_quantity=rec.cancelled_quantity,
            cancel_reason=rec.cancel_reason,
            commission=Decimal(rec.commission),
            error_message=rec.error_message,
            strategy_name=rec.strategy_name,
            signal_type=rec.signal_type,
            signal_strength=float(rec.signal_strength or 0.0),
            created_at=rec.created_at,
            updated_at=rec.updated_at,
        )

    @staticmethod
    def _trade_from_record(rec) -> Trade:
        return Trade(
            trade_id=rec.trade_id,
            order_id=rec.order_id,
            stock_code=rec.stock_code,
            side=OrderSide(rec.side),
            quantity=rec.quantity,
            price=Decimal(rec.price),
            commission=Decimal(rec.commission),
            trade_time=rec.trade_time,
            sequence=rec.sequence,
        )

    # -------------------------------------------------------------- 连接

    def connect(self) -> bool:
        """业务模块说明。"""
        self._connected = True
        logger.info("Simulation adapter connected")
        return True

    def disconnect(self) -> None:
        """业务模块说明。"""
        self._connected = False
        logger.info("Simulation adapter disconnected")

    def get_account(self) -> Optional[Account]:
        """业务模块说明。"""
        with self._lock:
            return self.ledger.get_account()

    def get_positions(self) -> List[Position]:
        """业务模块说明。"""
        with self._lock:
            return self.ledger.get_positions()

    def get_position(self, stock_code: str) -> Optional[Position]:
        """业务模块说明。"""
        with self._lock:
            return self.ledger.get_position(stock_code)

    # -------------------------------------------------------------- 下单

    def place_order(self, order: Order) -> Order:
        """业务模块说明。"""
        with self._lock:
            # 客户端幂等：失败重试复用同一 client_order_id，绝不重复下单
            if order.client_order_id:
                existing = self._store.find_order_by_client_id(order.client_order_id)
                if existing is not None:
                    logger.info(
                        "Idempotent place_order hit: %s -> %s",
                        order.client_order_id,
                        existing.order_id,
                    )
                    return self._orders[existing.order_id]

            if not self._connected:
                order.status = OrderStatus.FAILED
                order.error_message = "交易连接已断开"
                return order

            quote = self.get_quote(order.stock_code)
            if not quote:
                order.status = OrderStatus.REJECTED
                order.error_message = "无法获取行情数据"
                self._orders[order.order_id] = order
                self._persist_order(order)
                return order

            current_price = Decimal(str(quote["last_price"]))

            # 冻结资金/股份；不足则拒绝（不产生任何冻结副作用）
            try:
                self.ledger.freeze_for_order(order, current_price)
            except SettlementError as exc:
                order.status = OrderStatus.REJECTED
                order.error_message = str(exc)
                self._orders[order.order_id] = order
                self._persist_order(order)
                return order

            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now()
            self._orders[order.order_id] = order
            self._persist_order(order)

            # 尝试撮合（市价单立即全部成交；限价单满足价格时成交）
            self._try_fill_order(order, current_price)

            self._emit("on_order", order)
            return order

    # -------------------------------------------------------------- 撮合

    def _try_fill_order(self, order: Order, current_price: Decimal) -> None:
        """业务模块说明。"""
        fill_price = None

        if order.order_type == OrderType.MARKET:
            slippage = current_price * Decimal(str(self.slippage_rate))
            if order.side == OrderSide.BUY:
                fill_price = current_price + slippage
            else:
                fill_price = current_price - slippage
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                if current_price <= order.price:
                    fill_price = order.price
            else:
                if current_price >= order.price:
                    fill_price = order.price

        if fill_price is not None:
            self._apply_fill(order, order.remaining_quantity, fill_price)

    def _apply_fill(
        self, order: Order, quantity: int, fill_price: Decimal,
        fill_time: Optional[datetime] = None, trade_id: Optional[str] = None,
    ) -> Trade:
        """生成并结算一个成交分片，推进订单状态。内部调用，已持锁。"""
        if quantity <= 0:
            raise SettlementError("成交数量必须为正数")
        if quantity > order.remaining_quantity:
            raise SettlementError(
                f"成交数量 {quantity} 超过订单剩余 {order.remaining_quantity}"
            )
        if not order.status.is_active:
            raise SettlementError(f"订单 {order.order_id} 已终态，不能成交")

        now = fill_time or datetime.now()
        sequence = len(order.trades) + 1
        trade = Trade(
            trade_id=trade_id
            or f"{order.order_id}_F{sequence:04d}_{now.strftime('%H%M%S%f')}",
            order_id=order.order_id,
            stock_code=order.stock_code,
            side=order.side,
            quantity=quantity,
            price=fill_price,
            commission=Decimal("0"),
            trade_time=now,
            sequence=sequence,
        )

        # 先落账本（trade_id 唯一约束兜底），再结算；任一步失败订单状态不变
        try:
            self._store.append_trade(trade)
        except DuplicateTrade:
            raise
        commission = self.ledger.settle_trade(order, trade)

        order.trades.append(trade)
        order.filled_quantity += quantity
        order.commission += commission
        order.filled_price = order.average_fill_price
        order.status = (
            OrderStatus.FILLED
            if order.filled_quantity >= order.quantity
            else OrderStatus.PARTIAL_FILLED
        )
        order.updated_at = now
        self._persist_order(order)

        self._emit("on_trade", trade)
        self._emit("on_order", order)
        logger.info(
            "Trade %s: order=%s %s %s %s@%s seq=%d",
            trade.trade_id, order.order_id, order.side.value,
            order.stock_code, quantity, fill_price, sequence,
        )
        return trade

    # ------------------------------------------------------- 外部成交回报

    def process_trade_report(
        self,
        order_id: str,
        trade_id: str,
        quantity: int,
        price: Decimal,
        trade_time: Optional[datetime] = None,
    ) -> Trade:
        """处理外部成交回报，幂等且对乱序安全。

        * 重复回报（trade_id 已存在）：返回原分片，不再次结算；
        * 迟到回报（订单已撤/已成/收盘作废）：拒绝并说明；
        * 超量回报（累计超过订单数量）：拒绝，不写入账本。
        """
        with self._lock:
            price = Decimal(str(price))

            existing = self._store.get_trade(trade_id)
            if existing is not None:
                if existing.order_id != order_id:
                    raise SettlementError(
                        f"成交号 {trade_id} 属于其他订单 {existing.order_id}"
                    )
                order = self._orders[order_id]
                return next(t for t in order.trades if t.trade_id == trade_id)

            order = self._orders.get(order_id)
            if order is None:
                raise SettlementError(f"订单不存在: {order_id}")
            if not order.status.is_active:
                raise SettlementError(
                    f"订单状态为 {order.status.value}，迟到成交回报被拒绝"
                )
            if quantity > order.remaining_quantity:
                raise SettlementError(
                    f"成交数量 {quantity} 超过剩余 {order.remaining_quantity}"
                )
            if quantity <= 0:
                raise SettlementError("成交数量必须为正数")

            return self._apply_fill(
                order, quantity, price, fill_time=trade_time, trade_id=trade_id
            )

    # -------------------------------------------------------------- 撤单

    def cancel_order(self, order_id: str) -> bool:
        """部分成交后撤单：只释放剩余数量的冻结，成交分片原样保留。"""
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                return False
            if not order.status.is_active:
                return False

            remaining = order.remaining_quantity
            try:
                self.ledger.release_frozen(order, remaining)
            except SettlementError as exc:
                logger.error("Cancel release failed for %s: %s", order_id, exc)
                return False

            order.cancelled_quantity += remaining
            order.status = OrderStatus.CANCELLED
            order.cancel_reason = order.cancel_reason or "用户撤单"
            order.updated_at = datetime.now()
            self._persist_order(order)

            self._emit("on_order", order)
            logger.info(
                "Order %s cancelled: filled=%d released=%d",
                order_id, order.filled_quantity, remaining,
            )
            return True

    def expire_day_orders(self, reason: str = "收盘作废") -> List[Order]:
        """跨日收盘：作废所有仍活跃的订单并释放剩余冻结。"""
        with self._lock:
            expired: List[Order] = []
            for order in list(self._orders.values()):
                if not order.status.is_active:
                    continue
                remaining = order.remaining_quantity
                try:
                    self.ledger.release_frozen(order, remaining)
                except SettlementError as exc:
                    logger.error(
                        "Expire release failed for %s: %s", order.order_id, exc
                    )
                    continue
                order.cancelled_quantity += remaining
                order.status = OrderStatus.EXPIRED
                order.cancel_reason = reason
                order.updated_at = datetime.now()
                self._persist_order(order)
                expired.append(order)
                self._emit("on_order", order)
            if expired:
                logger.info("Expired %d day orders", len(expired))
            return expired

    # -------------------------------------------------------------- 查询

    def get_order(self, order_id: str) -> Optional[Order]:
        """业务模块说明。"""
        with self._lock:
            return self._orders.get(order_id)

    def get_orders(
        self,
        stock_code: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Order]:
        """稳定排序（创建时间倒序、order_id 兜底）分页查询。"""
        with self._lock:
            orders = list(self._orders.values())
            if stock_code:
                orders = [o for o in orders if o.stock_code == stock_code]
            if status:
                orders = [o for o in orders if o.status == status]
            orders.sort(key=lambda o: (o.created_at, o.order_id), reverse=True)
            return orders[offset : offset + limit]

    def count_orders(
        self,
        stock_code: Optional[str] = None,
        status: Optional[OrderStatus] = None,
    ) -> int:
        """业务模块说明。"""
        with self._lock:
            orders = self._orders.values()
            if stock_code:
                orders = [o for o in orders if o.stock_code == stock_code]
            if status:
                orders = [o for o in orders if o.status == status]
            return len(orders)

    def get_trades(
        self,
        order_id: Optional[str] = None,
        stock_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Trade]:
        """按订单/股票/交易日查询成交分片，账本顺序稳定分页。"""
        with self._lock:
            trades: List[Trade] = []
            if order_id:
                order = self._orders.get(order_id)
                trades = list(order.trades) if order else []
                if stock_code:
                    trades = [t for t in trades if t.stock_code == stock_code]
                if trade_date:
                    trades = [
                        t for t in trades
                        if t.trade_date.isoformat() == trade_date
                    ]
            else:
                for order in self._orders.values():
                    if stock_code and order.stock_code != stock_code:
                        continue
                    for t in order.trades:
                        if trade_date and t.trade_date.isoformat() != trade_date:
                            continue
                        trades.append(t)
            trades.sort(key=lambda t: (t.trade_time, t.sequence, t.trade_id))
            return trades[offset : offset + limit]

    def count_trades(
        self,
        order_id: Optional[str] = None,
        stock_code: Optional[str] = None,
        trade_date: Optional[str] = None,
    ) -> int:
        """业务模块说明。"""
        with self._lock:
            return len(
                self.get_trades(order_id, stock_code, trade_date, limit=10**9)
            )

    # -------------------------------------------------------------- 对账

    def reconcile(self) -> List[str]:
        """账户汇总与订单/成交明细交叉核对，空列表表示完全平衡。"""
        with self._lock:
            all_trades = [t for o in self._orders.values() for t in o.trades]
            problems = self.ledger.reconcile(
                list(self._orders.values()), all_trades
            )
            # 额外校验：内存分片数与持久账本一致
            stored = self._store.count_trades()
            if stored != len(all_trades):
                problems.append(
                    f"成交分片数不一致：内存 {len(all_trades)} != 账本 {stored}"
                )
            return problems

    # -------------------------------------------------------------- 行情

    def get_quote(self, stock_code: str) -> Optional[Dict]:
        """业务模块说明。"""
        if stock_code not in self._quotes:
            self.set_quote(stock_code, float(self.default_quote_price))

        quote = self._quotes[stock_code]
        quote["bid_price_1"] = quote["last_price"] * 0.999
        quote["ask_price_1"] = quote["last_price"] * 1.001
        quote["datetime"] = datetime.now().isoformat()

        with self._lock:
            self.ledger.mark_price(
                stock_code, Decimal(str(quote["last_price"]))
            )
        return quote

    def set_quote(self, stock_code: str, price: float) -> None:
        """设置行情并对所有触及该价的活跃限价单做一次撮合。"""
        self._quotes[stock_code] = {
            "stock_code": stock_code,
            "last_price": price,
            "open": price,
            "high": price * 1.02,
            "low": price * 0.98,
            "close": price,
            "volume": 1000000,
            "bid_price_1": price * 0.999,
            "ask_price_1": price * 1.001,
            "bid_volume_1": 1000,
            "ask_volume_1": 1000,
            "datetime": datetime.now().isoformat(),
        }
        with self._lock:
            self.ledger.mark_price(stock_code, Decimal(str(price)))
            # 行情驱动撮合：价格满足条件的活跃限价单逐笔成交
            for order in list(self._orders.values()):
                if not order.status.is_active or order.stock_code != stock_code:
                    continue
                self._try_fill_order(order, Decimal(str(price)))

    # -------------------------------------------------------------- 内部

    def _persist_order(self, order: Order) -> None:
        if self._replaying:
            return
        self._store.save_order(order)
