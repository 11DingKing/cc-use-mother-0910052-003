"""交易结算持久化层。

两张表构成结算闭环的存储基础：

- ``trading_orders``：订单当前快照（含状态、已成交、已释放数量与客户端幂等键）；
- ``trading_trades``：成交分片账本，``trade_id`` 全局唯一，是每一笔成交
  原始价格、时间与费用的不可变凭证，也是重启重放与去重的依据。

所有金额以字符串存储 Decimal，避免二进制浮点与 SQLite NUMERIC 亲和度带来的误差。
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

Base = declarative_base()


class OrderRecord(Base):
    """订单快照记录。"""

    __tablename__ = "trading_orders"

    order_id = Column(String(64), primary_key=True)
    client_order_id = Column(String(64), nullable=True, unique=True)
    stock_code = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    order_type = Column(String(16), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(String(32), nullable=True)
    stop_price = Column(String(32), nullable=True)
    freeze_price = Column(String(32), nullable=True)
    status = Column(String(16), nullable=False, index=True)
    filled_quantity = Column(Integer, nullable=False, default=0)
    cancelled_quantity = Column(Integer, nullable=False, default=0)
    cancel_reason = Column(String(100), nullable=True)
    commission = Column(String(32), nullable=False, default="0")
    error_message = Column(String(300), nullable=True)
    strategy_name = Column(String(64), nullable=True)
    signal_type = Column(String(20), nullable=True)
    signal_strength = Column(String(16), nullable=True, default="0.0")
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    def __repr__(self) -> str:
        return f"<OrderRecord({self.order_id} {self.status} filled={self.filled_quantity})>"


class TradeRecord(Base):
    """成交分片记录（不可变账本）。"""

    __tablename__ = "trading_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(96), nullable=False, unique=True)
    order_id = Column(String(64), nullable=False, index=True)
    stock_code = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    sequence = Column(Integer, nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(String(32), nullable=False)
    commission = Column(String(32), nullable=False)
    trade_time = Column(DateTime, nullable=False)
    trade_date = Column(String(10), nullable=False, index=True)

    __table_args__ = (
        # 同一订单内分片序号严格递增，重放顺序确定
        UniqueConstraint("order_id", "sequence", name="uix_trade_order_sequence"),
        Index("ix_trade_stock_date", "stock_code", "trade_date"),
    )

    def __repr__(self) -> str:
        return f"<TradeRecord({self.trade_id} {self.quantity}@{self.price})>"


class MetaRecord(Base):
    """适配器元信息（初始资金、账户编号等），用于重启重建账本。"""

    __tablename__ = "trading_meta"

    key = Column(String(64), primary_key=True)
    value = Column(String(256), nullable=False)


class DuplicateTrade(Exception):
    """成交分片 trade_id 已存在（重复/重试回报）。"""

    def __init__(self, trade_id: str):
        self.trade_id = trade_id
        super().__init__(f"重复成交回报: {trade_id}")


class TradingStore:
    """订单与成交分片的数据访问对象，线程内使用。"""

    def __init__(self, url: str = "sqlite:///:memory:"):
        engine_kwargs = {"echo": False}
        if url == "sqlite:///:memory:":
            # 内存库需要单连接共享，否则不同 session 看到不同的库
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            engine_kwargs["poolclass"] = StaticPool
        elif url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        self._engine = create_engine(url, **engine_kwargs)
        self._session_factory = sessionmaker(
            bind=self._engine, autocommit=False, autoflush=False
        )
        Base.metadata.create_all(self._engine)

    # -------------------------------------------------------------- 元信息

    def set_meta(self, key: str, value: str) -> None:
        """写入/更新键值元信息（如初始资金、账户编号）。"""
        session = self._session_factory()
        try:
            existing = session.get(MetaRecord, key)
            if existing is None:
                session.add(MetaRecord(key=key, value=value))
            else:
                existing.value = value
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_meta(self, key: str) -> Optional[str]:
        """业务模块说明。"""
        session = self._session_factory()
        try:
            record = session.get(MetaRecord, key)
            return record.value if record else None
        finally:
            session.close()

    def save_order(self, order) -> None:
        """插入或更新订单快照（以 order_id 为准）。"""
        session = self._session_factory()
        try:
            existing = session.get(OrderRecord, order.order_id)
            values = self._order_values(order)
            if existing is None:
                session.add(OrderRecord(**values))
            else:
                for key, value in values.items():
                    setattr(existing, key, value)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _order_values(order) -> dict:
        return {
            "order_id": order.order_id,
            "client_order_id": order.client_order_id,
            "stock_code": order.stock_code,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "price": str(order.price) if order.price is not None else None,
            "stop_price": str(order.stop_price) if order.stop_price is not None else None,
            "freeze_price": (
                str(order.freeze_price) if order.freeze_price is not None else None
            ),
            "status": order.status.value,
            "filled_quantity": order.filled_quantity,
            "cancelled_quantity": order.cancelled_quantity,
            "cancel_reason": order.cancel_reason,
            "commission": str(order.commission),
            "error_message": order.error_message,
            "strategy_name": order.strategy_name,
            "signal_type": order.signal_type,
            "signal_strength": str(order.signal_strength),
            "created_at": order.created_at,
            "updated_at": order.updated_at,
        }

    def append_trade(self, trade) -> None:
        """追加成交分片；trade_id 冲突时抛出 :class:`DuplicateTrade`。"""
        session = self._session_factory()
        try:
            session.add(
                TradeRecord(
                    trade_id=trade.trade_id,
                    order_id=trade.order_id,
                    stock_code=trade.stock_code,
                    side=trade.side.value,
                    sequence=trade.sequence,
                    quantity=trade.quantity,
                    price=str(trade.price),
                    commission=str(trade.commission),
                    trade_time=trade.trade_time,
                    trade_date=trade.trade_date.isoformat(),
                )
            )
            session.commit()
        except IntegrityError:
            session.rollback()
            raise DuplicateTrade(trade.trade_id)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """业务模块说明。"""
        session = self._session_factory()
        try:
            return session.query(TradeRecord).filter_by(trade_id=trade_id).one_or_none()
        finally:
            session.close()

    def find_order_by_client_id(self, client_order_id: str) -> Optional[OrderRecord]:
        """业务模块说明。"""
        session = self._session_factory()
        try:
            return (
                session.query(OrderRecord)
                .filter_by(client_order_id=client_order_id)
                .one_or_none()
            )
        finally:
            session.close()

    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        """业务模块说明。"""
        session = self._session_factory()
        try:
            return session.get(OrderRecord, order_id)
        finally:
            session.close()

    def list_orders(
        self,
        stock_code: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[OrderRecord]:
        """稳定排序分页：created_at DESC, order_id DESC 作为并列次序。"""
        session = self._session_factory()
        try:
            query = session.query(OrderRecord)
            if stock_code:
                query = query.filter_by(stock_code=stock_code)
            if status:
                query = query.filter_by(status=status)
            query = query.order_by(
                OrderRecord.created_at.desc(), OrderRecord.order_id.desc()
            )
            return query.limit(limit).offset(offset).all()
        finally:
            session.close()

    def count_orders(
        self,
        stock_code: Optional[str] = None,
        status: Optional[str] = None,
    ) -> int:
        """业务模块说明。"""
        session = self._session_factory()
        try:
            query = session.query(OrderRecord)
            if stock_code:
                query = query.filter_by(stock_code=stock_code)
            if status:
                query = query.filter_by(status=status)
            return query.count()
        finally:
            session.close()

    def list_trades(
        self,
        order_id: Optional[str] = None,
        stock_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TradeRecord]:
        """稳定排序分页：账本自增 id ASC，与成交发生顺序一致。"""
        session = self._session_factory()
        try:
            query = session.query(TradeRecord)
            if order_id:
                query = query.filter_by(order_id=order_id)
            if stock_code:
                query = query.filter_by(stock_code=stock_code)
            if trade_date:
                query = query.filter_by(trade_date=trade_date)
            query = query.order_by(TradeRecord.id.asc())
            return query.limit(limit).offset(offset).all()
        finally:
            session.close()

    def count_trades(
        self,
        order_id: Optional[str] = None,
        stock_code: Optional[str] = None,
        trade_date: Optional[str] = None,
    ) -> int:
        """业务模块说明。"""
        session = self._session_factory()
        try:
            query = session.query(TradeRecord)
            if order_id:
                query = query.filter_by(order_id=order_id)
            if stock_code:
                query = query.filter_by(stock_code=stock_code)
            if trade_date:
                query = query.filter_by(trade_date=trade_date)
            return query.count()
        finally:
            session.close()

    def load_all_orders(self) -> List[OrderRecord]:
        """重放用：按创建时间加载全部订单。"""
        session = self._session_factory()
        try:
            return (
                session.query(OrderRecord)
                .order_by(OrderRecord.created_at.asc(), OrderRecord.order_id.asc())
                .all()
            )
        finally:
            session.close()

    def load_all_trades(self) -> List[TradeRecord]:
        """重放用：按账本顺序加载全部成交分片。"""
        session = self._session_factory()
        try:
            return session.query(TradeRecord).order_by(TradeRecord.id.asc()).all()
        finally:
            session.close()
