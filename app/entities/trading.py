"""交易订单与成交流水的持久化实体。

订单表保存生命周期最终/当前状态，成交表保存每一笔不可变分片，
trade_id 为全局幂等键；二者在同一事务内写入，是重启后重放结算的依据。
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Numeric,
    Index,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class OrderRecord(Base):
    """订单生命周期记录（一行一单，随状态推进更新）。"""

    __tablename__ = "trading_orders"

    order_id = Column(String(64), primary_key=True)
    stock_code = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)          # buy / sell
    order_type = Column(String(16), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric(20, 6), nullable=True)
    stop_price = Column(Numeric(20, 6), nullable=True)
    status = Column(String(16), nullable=False)

    filled_quantity = Column(Integer, nullable=False, default=0)
    filled_price = Column(Numeric(20, 6), nullable=True)  # 成交加权均价
    commission = Column(Numeric(20, 6), nullable=False, default=0)
    cancelled_quantity = Column(Integer, nullable=False, default=0)
    cancelled_at = Column(DateTime, nullable=True)

    error_message = Column(String(500), nullable=True)
    strategy_name = Column(String(100), nullable=True)
    signal_type = Column(String(32), nullable=True)
    signal_strength = Column(Float, nullable=True)

    trade_date = Column(String(10), nullable=False)   # YYYY-MM-DD
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_orders_stock_created", "stock_code", "created_at"),
        Index("ix_orders_status_created", "status", "created_at"),
        Index("ix_orders_trade_date", "trade_date"),
    )

    def __repr__(self):
        return f"<OrderRecord({self.order_id}, {self.status}, {self.filled_quantity}/{self.quantity})>"


class TradeRecord(Base):
    """成交流水（一行一笔成交分片，只追加，不更新、不删除）。"""

    __tablename__ = "trading_trades"

    trade_id = Column(String(80), primary_key=True)
    order_id = Column(String(64), nullable=False)
    stock_code = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric(20, 6), nullable=False)
    commission = Column(Numeric(20, 6), nullable=False, default=0)

    traded_at = Column(DateTime, nullable=False)      # 交易所回报原始时间
    trade_date = Column(String(10), nullable=False)
    settled = Column(Integer, nullable=False, default=0)  # 是否已完成账户过账
    created_at = Column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint("trade_id", name="uix_trade_id"),
        Index("ix_trades_order", "order_id", "traded_at"),
        Index("ix_trades_stock_date", "stock_code", "trade_date"),
        Index("ix_trades_trade_date", "trade_date"),
    )

    def __repr__(self):
        return f"<TradeRecord({self.trade_id}, {self.quantity}@{self.price})>"


class TradingMetaRecord(Base):
    """交易账本元数据（单例行，保存初始资金等重建账户所需参数）。"""

    __tablename__ = "trading_meta"

    key = Column(String(64), primary_key=True)
    value = Column(String(255), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
