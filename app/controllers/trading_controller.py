"""业务模块说明。"""

from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.services.trading_service import TradingService

router = APIRouter(prefix="/api/trading", tags=["trading"])
trading_service = TradingService()


class ConnectRequest(BaseModel):
    """业务模块说明。"""
    adapter_type: str = "simulation"  # simulation, vnpy
    config: Optional[dict] = None


class BuyRequest(BaseModel):
    """业务模块说明。"""
    stock_code: str
    quantity: int
    price: Optional[float] = None
    order_type: str = "limit"  # limit, market
    signal_type: Optional[str] = None
    signal_strength: float = 0.0
    client_order_id: Optional[str] = None  # 失败重试幂等键


class SellRequest(BaseModel):
    """业务模块说明。"""
    stock_code: str
    quantity: int
    price: Optional[float] = None
    order_type: str = "limit"
    signal_type: Optional[str] = None
    signal_strength: float = 0.0
    client_order_id: Optional[str] = None


class SignalTradeRequest(BaseModel):
    """业务模块说明。"""
    stock_code: str
    signal_type: str
    signal_strength: float
    price: float
    position_ratio: float = 0.1


class TradeReportRequest(BaseModel):
    """外部成交回报。"""
    order_id: str
    trade_id: str
    quantity: int
    price: float
    trade_time: Optional[str] = None


@router.post("/connect")
async def connect(request: ConnectRequest):
    """业务模块说明。"""
    success = trading_service.connect(request.adapter_type, request.config)
    return {
        "success": success,
        "adapter_type": request.adapter_type,
        "message": "连接成功" if success else "连接失败",
    }


@router.post("/disconnect")
async def disconnect():
    """业务模块说明。"""
    trading_service.disconnect()
    return {"success": True, "message": "已断开连接"}


@router.get("/account")
async def get_account():
    """业务模块说明。"""
    return trading_service.get_account()


@router.get("/positions")
async def get_positions():
    """业务模块说明。"""
    return {"positions": trading_service.get_positions()}


@router.get("/positions/{stock_code}")
async def get_position(stock_code: str):
    """业务模块说明。"""
    position = trading_service.get_position(stock_code)
    if not position:
        return {"error": "未持有该股票"}
    return position


@router.post("/buy")
async def buy(request: BuyRequest):
    """业务模块说明。"""
    return trading_service.buy(
        stock_code=request.stock_code,
        quantity=request.quantity,
        price=request.price,
        order_type=request.order_type,
        signal_type=request.signal_type,
        signal_strength=request.signal_strength,
        client_order_id=request.client_order_id,
    )


@router.post("/sell")
async def sell(request: SellRequest):
    """业务模块说明。"""
    return trading_service.sell(
        stock_code=request.stock_code,
        quantity=request.quantity,
        price=request.price,
        order_type=request.order_type,
        signal_type=request.signal_type,
        signal_strength=request.signal_strength,
        client_order_id=request.client_order_id,
    )


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """业务模块说明。"""
    return trading_service.cancel_order(order_id)


@router.get("/orders/{order_id}")
async def get_order(order_id: str):
    """业务模块说明。"""
    return trading_service.get_order(order_id)


@router.get("/orders")
async def get_orders(
    stock_code: Optional[str] = Query(default=None, description="股票代码"),
    status: Optional[str] = Query(default=None, description="订单状态"),
    limit: int = Query(default=50, ge=1, le=500, description="每页数量"),
    offset: int = Query(default=0, ge=0, description="偏移量"),
):
    """分页查询订单（稳定排序）。"""
    page = trading_service.get_orders(stock_code, status, limit, offset)
    # 保留 orders 字段兼容旧客户端
    return {
        "orders": page["items"],
        "items": page["items"],
        "total": page["total"],
        "limit": page["limit"],
        "offset": page["offset"],
    }


@router.get("/trades")
async def get_trades(
    order_id: Optional[str] = Query(default=None, description="订单编号"),
    stock_code: Optional[str] = Query(default=None, description="股票代码"),
    trade_date: Optional[str] = Query(default=None, description="交易日 YYYY-MM-DD"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """分页查询成交分片。"""
    return trading_service.get_trades(
        order_id, stock_code, trade_date, limit, offset
    )


@router.post("/trades/report")
async def report_trade(request: TradeReportRequest):
    """提交外部成交回报（幂等：重复 trade_id 返回首次结果）。"""
    return trading_service.report_trade(
        order_id=request.order_id,
        trade_id=request.trade_id,
        quantity=request.quantity,
        price=request.price,
        trade_time=request.trade_time,
    )


@router.post("/day-end")
async def settle_day_end():
    """跨日收盘：作废活跃订单并释放剩余冻结。"""
    return trading_service.settle_day_end()


@router.get("/reconcile")
async def reconcile():
    """账户汇总与订单/成交明细交叉核对。"""
    return trading_service.reconcile()


@router.get("/quote/{stock_code}")
async def get_quote(stock_code: str):
    """业务模块说明。"""
    return trading_service.get_quote(stock_code)


@router.post("/signal-trade")
async def execute_signal_trade(request: SignalTradeRequest):
    """业务模块说明。"""
    result = trading_service.execute_signal(
        stock_code=request.stock_code,
        signal_type=request.signal_type,
        signal_strength=request.signal_strength,
        price=request.price,
        position_ratio=request.position_ratio,
    )
    
    if result:
        return result
    return {"message": "自动交易未启用或条件不满足"}


@router.post("/auto-trade/enable")
async def enable_auto_trade():
    """业务模块说明。"""
    trading_service.enable_auto_trade(True)
    return {"success": True, "message": "自动交易已启用"}


@router.post("/auto-trade/disable")
async def disable_auto_trade():
    """业务模块说明。"""
    trading_service.enable_auto_trade(False)
    return {"success": True, "message": "自动交易已禁用"}


@router.post("/check-stop-loss")
async def check_stop_loss():
    """业务模块说明。"""
    results = trading_service.check_stop_loss_take_profit()
    return {
        "triggered_count": len(results),
        "orders": results,
    }
