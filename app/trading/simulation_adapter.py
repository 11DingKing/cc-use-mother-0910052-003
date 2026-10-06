"""模拟交易适配器。

撮合逻辑只负责“产生成交回报”，订单生命周期、分片入账、费用、现金与
持仓结算全部由 SettlementLedger 完成；因此部分成交、撤单释放、重复/迟到
回报、重启恢复与分页对账的语义在适配器层自动成立。
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from app.trading.base import (
    TradingAdapter,
    Order,
    OrderStatus,
    OrderType,
    OrderSide,
    Position,
    Account,
)
from app.trading.settlement import (
    SettlementLedger,
    SettlementError,
    TradeReport,
    Page,
)

logger = logging.getLogger(__name__)


class SimulationAdapter(TradingAdapter):
    """业务模块说明。"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})

        cfg = config or {}
        # 初始资金
        initial_cash = Decimal(str(cfg.get("initial_cash", 1000000)))

        # 交易成本配置
        self.commission_rate = Decimal(str(cfg.get("commission_rate", 0.0003)))
        self.min_commission = Decimal(str(cfg.get("min_commission", 5)))
        self.stamp_tax_rate = Decimal(str(cfg.get("stamp_tax_rate", 0.001)))
        self.slippage_rate = cfg.get("slippage_rate", 0.001)
        self.default_quote_price = Decimal(str(cfg.get("default_quote_price", 10)))

        # 持久化仓储：默认内存账本；传入 repository/url 时可跨重启恢复
        repository = cfg.get("repository")
        self.ledger = SettlementLedger(
            initial_cash=initial_cash,
            commission_rate=self.commission_rate,
            min_commission=self.min_commission,
            stamp_tax_rate=self.stamp_tax_rate,
            slippage_rate=self.slippage_rate,
            account_id=cfg.get("account_id"),
            repository=repository,
        )

        # 模拟行情（买卖盘口）
        self._quotes: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """业务模块说明。"""
        self._connected = True
        logger.info("Simulation adapter connected")
        return True

    def disconnect(self) -> None:
        """业务模块说明。"""
        self._connected = False
        logger.info("Simulation adapter disconnected")

    # ------------------------------------------------------------------
    # 账户 / 持仓（全部由账本从成交分片派生）
    # ------------------------------------------------------------------
    def get_account(self) -> Optional[Account]:
        """业务模块说明。"""
        return self.ledger.get_account()

    def get_positions(self) -> List[Position]:
        """业务模块说明。"""
        return self.ledger.get_positions()

    def get_position(self, stock_code: str) -> Optional[Position]:
        """业务模块说明。"""
        return self.ledger.get_position(stock_code)

    # ------------------------------------------------------------------
    # 下单与撮合
    # ------------------------------------------------------------------
    def place_order(self, order: Order) -> Order:
        """业务模块说明。"""
        if not self._connected:
            return self.ledger.register_terminal_new(order, OrderStatus.FAILED, "交易连接已断开")

        # 获取行情
        quote = self.get_quote(order.stock_code)
        if not quote:
            return self.ledger.register_terminal_new(order, OrderStatus.REJECTED, "无法获取行情数据")

        current_price = Decimal(str(quote["last_price"]))

        # 登记订单并校验资金/持仓；失败只留终态订单，不产生冻结与成交
        try:
            self.ledger.submit_order(order, reference_price=current_price)
        except SettlementError as exc:
            status = (
                OrderStatus.REJECTED
                if "持仓" in str(exc) or "资金" in str(exc) or "价格" in str(exc)
                else OrderStatus.FAILED
            )
            return self.ledger.register_terminal_new(order, status, str(exc))

        self._emit("on_order", order)

        # 尝试即时撮合（限价单不满足条件则保持在途，等待后续回报）
        fill_price = self._match_price(order, current_price)
        if fill_price is not None:
            trade, _ = self.report_trade(
                TradeReport(
                    order_id=order.order_id,
                    trade_id=self._next_trade_id(order),
                    quantity=order.quantity,
                    price=fill_price,
                    traded_at=self.ledger.now(),
                )
            )
            logger.info(
                f"Order filled: {order.order_id} {order.side.value} "
                f"{order.stock_code} {trade.quantity}@{fill_price}"
            )

        return order

    def _match_price(self, order: Order, current_price: Decimal) -> Optional[Decimal]:
        """返回本次可成交价；不可成交返回 None。"""
        if order.order_type == OrderType.MARKET:
            slippage = current_price * Decimal(str(self.slippage_rate))
            if order.side == OrderSide.BUY:
                return current_price + slippage
            return current_price - slippage

        if order.order_type == OrderType.LIMIT and order.price is not None:
            if order.side == OrderSide.BUY and current_price <= order.price:
                return order.price
            if order.side == OrderSide.SELL and current_price >= order.price:
                return order.price
        return None

    def report_trade(self, report: TradeReport):
        """提交一笔外部成交回报（重复/迟到回报由账本幂等处理）。"""
        before = self.ledger.get_order(report.order_id)
        before_status = before.status if before else None
        trade, is_new = self.ledger.apply_trade_report(report)
        if is_new:
            order = self.ledger.get_order(report.order_id)
            self._emit("on_trade", trade)
            if order.status != before_status:
                self._emit("on_order", order)
        return trade, is_new

    def simulate_partial_fill(
        self,
        order_id: str,
        quantity: int,
        price: float,
        traded_at: Optional[datetime] = None,
        trade_id: Optional[str] = None,
    ):
        """测试/模拟用：对在途单推送一笔部分成交。"""
        order = self.ledger.get_order(order_id)
        if order is None:
            raise SettlementError(f"订单不存在: {order_id}")
        tid = trade_id or self._next_trade_id(order)
        return self.report_trade(
            TradeReport(
                order_id=order_id,
                trade_id=tid,
                quantity=quantity,
                price=Decimal(str(price)),
                traded_at=traded_at,
            )
        )

    @staticmethod
    def _next_trade_id(order: Order) -> str:
        return f"{order.order_id}:T{len(order.trades) + 1}"

    # ------------------------------------------------------------------
    # 撤单：部分成交后只释放剩余数量
    # ------------------------------------------------------------------
    def cancel_order(self, order_id: str) -> bool:
        """业务模块说明。"""
        order = self.ledger.get_order(order_id)
        if order is None:
            return False
        try:
            self.ledger.cancel_order(order_id)
        except SettlementError:
            return False
        self._emit("on_order", order)
        return True

    def rollover_trading_day(self, close_time: Optional[datetime] = None) -> List[Order]:
        """跨日收盘：撤销全部在途单，成交分片按原成交日保留。"""
        closed = self.ledger.rollover_trading_day(close_time)
        for order in closed:
            self._emit("on_order", order)
        return closed

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_order(self, order_id: str) -> Optional[Order]:
        """业务模块说明。"""
        return self.ledger.get_order(order_id)

    def get_orders(
        self,
        stock_code: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        trade_date: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Page:
        """业务模块说明。订单键集分页，排序 (created_at, order_id) 稳定。"""
        return self.ledger.get_orders(
            stock_code=stock_code,
            status=status,
            trade_date=trade_date,
            limit=limit,
            cursor=cursor,
        )

    def get_trades(
        self,
        order_id: Optional[str] = None,
        stock_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Page:
        """成交流水分页查询。"""
        return self.ledger.get_trades(
            order_id=order_id,
            stock_code=stock_code,
            trade_date=trade_date,
            limit=limit,
            cursor=cursor,
        )

    def reconcile(self) -> Dict[str, Any]:
        """账户汇总与订单/成交明细对账。"""
        return self.ledger.reconcile()

    def get_quote(self, stock_code: str) -> Optional[Dict]:
        """业务模块说明。"""
        if stock_code not in self._quotes:
            self.set_quote(stock_code, float(self.default_quote_price))

        quote = self._quotes[stock_code]
        quote["bid_price_1"] = quote["last_price"] * 0.999
        quote["ask_price_1"] = quote["last_price"] * 1.001
        quote["datetime"] = datetime.now().isoformat()

        # 行情用于账本派生持仓市值/盈亏
        self.ledger.set_quote(stock_code, Decimal(str(quote["last_price"])))

        return quote

    def set_quote(self, stock_code: str, price: float) -> None:
        """业务模块说明。"""
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
        self.ledger.set_quote(stock_code, Decimal(str(price)))
