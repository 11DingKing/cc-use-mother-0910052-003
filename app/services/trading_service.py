"""业务模块说明。"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any

from app.trading.base import (
    TradingAdapter,
    Order,
    OrderStatus,
    OrderType,
    OrderSide,
    Position,
    Account,
    RiskManager,
)
from app.trading.simulation_adapter import SimulationAdapter
from app.trading.settlement import SettlementError, TradeReport
from app.trading.vnpy_adapter import VnpyAdapter
from app.services.analysis_service import AnalysisService
from app.middleware.exception_handler import AppException

logger = logging.getLogger(__name__)


class TradingException(AppException):
    """业务模块说明。"""
    
    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        stock_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="TRADING_ERROR",
            status_code=400,
            details={
                "order_id": order_id,
                "stock_code": stock_code,
                **(details or {}),
            },
        )


class TradingService:
    """业务模块说明。"""
    
    def __init__(self, adapter: Optional[TradingAdapter] = None):
        self.adapter = adapter or SimulationAdapter()
        self.risk_manager = RiskManager()
        self.analysis_service = AnalysisService()
        self._auto_trade_enabled = False
    
    def connect(self, adapter_type: str = "simulation", config: Optional[Dict] = None) -> bool:
        """业务模块说明。"""
        config = config or {}
        if adapter_type == "vnpy":
            self.adapter = VnpyAdapter(config)
        else:
            adapter_config = dict(config)
            # 传入 db_url 时启用 SQLite 持久化，重启后重放成交分片恢复账本
            db_url = adapter_config.pop("db_url", None)
            if db_url:
                from app.trading.repository import SqliteTradeRepository

                adapter_config["repository"] = SqliteTradeRepository(db_url)
            self.adapter = SimulationAdapter(adapter_config)

        return self.adapter.connect()
    
    def disconnect(self) -> None:
        """业务模块说明。"""
        if self.adapter:
            self.adapter.disconnect()
    
    def get_account(self) -> Dict[str, Any]:
        """业务模块说明。"""
        account = self.adapter.get_account()
        if not account:
            raise TradingException("无法获取账户信息，请检查交易连接")
        return account.to_dict()
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """业务模块说明。"""
        positions = self.adapter.get_positions()
        return [p.to_dict() for p in positions]
    
    def get_position(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """业务模块说明。"""
        position = self.adapter.get_position(stock_code)
        return position.to_dict() if position else None
    
    def buy(
        self,
        stock_code: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: str = "limit",
        signal_type: Optional[str] = None,
        signal_strength: float = 0.0,
    ) -> Dict[str, Any]:
        """业务模块说明。"""
        # 数量校验
        if quantity <= 0 or quantity % 100 != 0:
            raise TradingException(
                "买入数量必须是100的整数倍",
                stock_code=stock_code,
            )
        
        # 构建订单
        ot = OrderType.LIMIT if order_type == "limit" else OrderType.MARKET
        
        if ot == OrderType.LIMIT and price is None:
            raise TradingException("限价单必须指定价格", stock_code=stock_code)
        
        order = Order(
            order_id=self.adapter._generate_order_id(),
            stock_code=stock_code,
            side=OrderSide.BUY,
            order_type=ot,
            quantity=quantity,
            price=Decimal(str(price)) if price else None,
            signal_type=signal_type,
            signal_strength=signal_strength,
        )
        
        # 风控检查
        account = self.adapter.get_account()
        positions = self.adapter.get_positions()
        
        passed, reason = self.risk_manager.check_order(order, account, positions)
        if not passed:
            raise TradingException(
                f"风控检查未通过: {reason}",
                stock_code=stock_code,
            )
        
        # 执行下单
        result = self.adapter.place_order(order)
        
        if result.status in (OrderStatus.REJECTED, OrderStatus.FAILED):
            raise TradingException(
                f"下单失败: {result.error_message}",
                order_id=result.order_id,
                stock_code=stock_code,
            )
        
        # 记录已成交部分的交易金额（支持部分成交）
        if result.filled_quantity > 0 and result.filled_price is not None:
            self.risk_manager.record_trade(result.filled_price * result.filled_quantity)

        return result.to_dict()
    
    def sell(
        self,
        stock_code: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: str = "limit",
        signal_type: Optional[str] = None,
        signal_strength: float = 0.0,
    ) -> Dict[str, Any]:
        """业务模块说明。"""
        # 持仓检查
        position = self.adapter.get_position(stock_code)
        if not position or position.available_quantity < quantity:
            available = position.available_quantity if position else 0
            raise TradingException(
                f"可用持仓不足，需要 {quantity}，可用 {available}",
                stock_code=stock_code,
            )
        
        # 构建订单
        ot = OrderType.LIMIT if order_type == "limit" else OrderType.MARKET
        
        if ot == OrderType.LIMIT and price is None:
            raise TradingException("限价单必须指定价格", stock_code=stock_code)
        
        order = Order(
            order_id=self.adapter._generate_order_id(),
            stock_code=stock_code,
            side=OrderSide.SELL,
            order_type=ot,
            quantity=quantity,
            price=Decimal(str(price)) if price else None,
            signal_type=signal_type,
            signal_strength=signal_strength,
        )
        
        # 执行下单
        result = self.adapter.place_order(order)
        
        if result.status in (OrderStatus.REJECTED, OrderStatus.FAILED):
            raise TradingException(
                f"下单失败: {result.error_message}",
                order_id=result.order_id,
                stock_code=stock_code,
            )
        
        return result.to_dict()
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """业务模块说明。部分成交后撤单只释放剩余数量，成交分片原样保留。"""
        order = self.adapter.get_order(order_id)
        if not order:
            raise TradingException("订单不存在", order_id=order_id)

        if order.status not in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL_FILLED,
        ):
            raise TradingException(
                f"订单状态为 {order.status.value}，无法撤销",
                order_id=order_id,
            )

        success = self.adapter.cancel_order(order_id)
        if not success:
            raise TradingException("撤单失败", order_id=order_id)

        order = self.adapter.get_order(order_id)
        return order.to_dict()

    def report_trade(
        self,
        order_id: str,
        trade_id: str,
        quantity: int,
        price: float,
        traded_at: Optional[datetime] = None,
        commission: Optional[float] = None,
    ) -> Dict[str, Any]:
        """推送成交回报；重复 trade_id 幂等，迟到回报在撤单余量内承接。"""
        if self.adapter.get_order(order_id) is None:
            raise TradingException("订单不存在", order_id=order_id)

        report = TradeReport(
            order_id=order_id,
            trade_id=trade_id,
            quantity=quantity,
            price=Decimal(str(price)),
            traded_at=traded_at,
            commission=Decimal(str(commission)) if commission is not None else None,
        )
        try:
            if hasattr(self.adapter, "report_trade"):
                trade, is_new = self.adapter.report_trade(report)
            else:
                raise TradingException("当前交易通道不支持成交回报", order_id=order_id)
        except SettlementError as exc:
            raise TradingException(str(exc), order_id=order_id)

        return {"trade": trade.to_dict(), "is_new": is_new}

    def get_order_trades(self, order_id: str) -> List[Dict[str, Any]]:
        """订单的每一笔成交分片（原始价格与时间）。"""
        order = self.adapter.get_order(order_id)
        if not order:
            raise TradingException("订单不存在", order_id=order_id)
        return [t.to_dict() for t in order.trades]

    def rollover_trading_day(self) -> Dict[str, Any]:
        """跨日收盘：撤销全部在途单，保留当日及历史成交分片。"""
        if not hasattr(self.adapter, "rollover_trading_day"):
            raise TradingException("当前交易通道不支持收盘处理")
        closed = self.adapter.rollover_trading_day()
        return {"cancelled_count": len(closed), "order_ids": [o.order_id for o in closed]}

    def reconcile(self) -> Dict[str, Any]:
        """账户汇总与订单/成交明细对账。"""
        if not hasattr(self.adapter, "reconcile"):
            raise TradingException("当前交易通道不支持对账")
        return self.adapter.reconcile()
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """业务模块说明。"""
        order = self.adapter.get_order(order_id)
        if not order:
            raise TradingException("订单不存在", order_id=order_id)
        return order.to_dict()
    
    def get_orders(
        self,
        stock_code: Optional[str] = None,
        status: Optional[str] = None,
        trade_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """业务模块说明。"""
        order_status = OrderStatus(status) if status else None
        page = self.adapter.get_orders(
            stock_code=stock_code,
            status=order_status,
            trade_date=trade_date,
            limit=500,
        )
        return [o.to_dict() for o in page.items]

    def query_orders(
        self,
        stock_code: Optional[str] = None,
        status: Optional[str] = None,
        trade_date: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """订单键集分页查询（排序稳定，翻页不重不漏）。"""
        order_status = OrderStatus(status) if status else None
        page = self.adapter.get_orders(
            stock_code=stock_code,
            status=order_status,
            trade_date=trade_date,
            limit=limit,
            cursor=cursor,
        )
        return page.to_dict()

    def query_trades(
        self,
        order_id: Optional[str] = None,
        stock_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """成交流水键集分页查询。"""
        if not hasattr(self.adapter, "get_trades"):
            raise TradingException("当前交易通道不支持成交查询")
        page = self.adapter.get_trades(
            order_id=order_id,
            stock_code=stock_code,
            trade_date=trade_date,
            limit=limit,
            cursor=cursor,
        )
        return page.to_dict()
    
    def get_quote(self, stock_code: str) -> Dict[str, Any]:
        """业务模块说明。"""
        quote = self.adapter.get_quote(stock_code)
        if not quote:
            raise TradingException("无法获取行情数据", stock_code=stock_code)
        return quote
    
    def execute_signal(
        self,
        stock_code: str,
        signal_type: str,
        signal_strength: float,
        price: float,
        position_ratio: float = 0.1,
    ) -> Optional[Dict[str, Any]]:
        """业务模块说明。"""
        if not self._auto_trade_enabled:
            logger.info(f"Auto trade disabled, signal ignored: {signal_type}")
            return None
        
        account = self.adapter.get_account()
        if not account:
            return None
        
        is_buy = signal_type.startswith("BUY")
        
        if is_buy:
            # 计算买入数量
            available = float(account.available_cash)
            buy_amount = available * position_ratio * signal_strength
            quantity = int(buy_amount / price / 100) * 100  # 100股整数倍
            
            if quantity >= 100:
                return self.buy(
                    stock_code=stock_code,
                    quantity=quantity,
                    price=price,
                    order_type="limit",
                    signal_type=signal_type,
                    signal_strength=signal_strength,
                )
        else:
            # 卖出
            position = self.adapter.get_position(stock_code)
            if position and position.available_quantity > 0:
                # 根据信号强度决定卖出比例
                sell_quantity = int(position.available_quantity * signal_strength / 100) * 100
                if sell_quantity >= 100:
                    return self.sell(
                        stock_code=stock_code,
                        quantity=sell_quantity,
                        price=price,
                        order_type="limit",
                        signal_type=signal_type,
                        signal_strength=signal_strength,
                    )
        
        return None
    
    def enable_auto_trade(self, enabled: bool = True) -> None:
        """业务模块说明。"""
        self._auto_trade_enabled = enabled
        logger.info(f"Auto trade {'enabled' if enabled else 'disabled'}")
    
    def check_stop_loss_take_profit(self) -> List[Dict[str, Any]]:
        """业务模块说明。"""
        results = []
        positions = self.adapter.get_positions()
        
        for pos in positions:
            if self.risk_manager.check_stop_loss(pos):
                # 触发止损
                quote = self.adapter.get_quote(pos.stock_code)
                if quote:
                    try:
                        result = self.sell(
                            stock_code=pos.stock_code,
                            quantity=pos.available_quantity,
                            price=quote["bid_price_1"],
                            order_type="limit",
                            signal_type="STOP_LOSS",
                        )
                        result["trigger"] = "stop_loss"
                        results.append(result)
                    except TradingException as e:
                        logger.error(f"Stop loss failed: {e}")
            
            elif self.risk_manager.check_take_profit(pos):
                # 触发止盈
                quote = self.adapter.get_quote(pos.stock_code)
                if quote:
                    try:
                        result = self.sell(
                            stock_code=pos.stock_code,
                            quantity=pos.available_quantity,
                            price=quote["bid_price_1"],
                            order_type="limit",
                            signal_type="TAKE_PROFIT",
                        )
                        result["trigger"] = "take_profit"
                        results.append(result)
                    except TradingException as e:
                        logger.error(f"Take profit failed: {e}")
        
        return results
