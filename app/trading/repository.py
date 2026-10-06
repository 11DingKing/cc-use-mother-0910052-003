"""结算账本的持久化仓储。

落账（追加不可变成交行）与过账（更新订单累计状态）在同一个数据库事务内
提交；重启时按 trade_id 去重重放，账户/持仓全部由流水派生，因此崩溃恢复
后不会对同一成交分片二次结算。
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.entities.trading import Base, OrderRecord, TradeRecord, TradingMetaRecord
from app.trading.base import Order, OrderSide, OrderStatus, OrderType, Trade

logger = logging.getLogger(__name__)


class TradeRepository:
    """订单/成交仓储接口（内存实现见 InMemoryTradeRepository）。"""

    def save_order(self, order: Order) -> None:
        raise NotImplementedError

    def save_fill(self, trade: Trade, order: Order) -> None:
        raise NotImplementedError

    def save_meta(self, key: str, value: str) -> None:
        raise NotImplementedError

    def save_meta_if_absent(self, key: str, value: str) -> None:
        raise NotImplementedError

    def load_all(self) -> Dict[str, Any]:
        raise NotImplementedError


class InMemoryTradeRepository(TradeRepository):
    """进程内持久化替身：同样保证 save_fill 的原子性与 trade_id 去重。"""

    def __init__(self, meta: Optional[Dict[str, str]] = None):
        self._orders: Dict[str, OrderRecord] = {}
        self._trades: Dict[str, TradeRecord] = {}
        self._meta: Dict[str, str] = dict(meta or {})

    def save_order(self, order: Order) -> None:
        self._orders[order.order_id] = _order_to_record(order)

    def save_fill(self, trade: Trade, order: Order) -> None:
        # 原子单元：任一步失败都不留中间态
        if trade.trade_id in self._trades:
            raise DuplicateTradeError(trade.trade_id)
        order_record = _order_to_record(order)
        trade_record = _trade_to_record(trade)
        self._trades[trade.trade_id] = trade_record
        self._orders[order.order_id] = order_record

    def save_meta(self, key: str, value: str) -> None:
        self._meta[key] = value

    def save_meta_if_absent(self, key: str, value: str) -> None:
        self._meta.setdefault(key, value)

    def load_all(self) -> Dict[str, Any]:
        orders: List[Tuple[Order, List[Trade]]] = []
        trades_by_order: Dict[str, List[TradeRecord]] = {}
        for tr in self._trades.values():
            trades_by_order.setdefault(tr.order_id, []).append(tr)
        for rec in self._orders.values():
            order = _record_to_order(rec)
            trades = [_record_to_trade(t) for t in trades_by_order.get(rec.order_id, [])]
            trades.sort(key=lambda t: (t.traded_at, t.trade_id))
            orders.append((order, trades))
        return {"orders": orders, "meta": dict(self._meta)}


class DuplicateTradeError(Exception):
    """trade_id 唯一约束冲突（重复回报穿透到持久层）。"""


class SqliteTradeRepository(TradeRepository):
    """基于 SQLAlchemy 的 SQLite 仓储。"""

    def __init__(self, url: str = "sqlite:///:memory:", engine=None):
        if engine is not None:
            self._engine = engine
        else:
            self._engine = create_engine(
                url,
                connect_args={"check_same_thread": False} if "sqlite" in url else {},
            )
        Base.metadata.create_all(self._engine)
        self._SessionLocal = sessionmaker(
            bind=self._engine, autocommit=False, autoflush=False
        )

    @classmethod
    def from_session_factory(cls, session_factory) -> "SqliteTradeRepository":
        repo = cls(engine=session_factory.kw["bind"])
        repo._SessionLocal = session_factory
        return repo

    def save_order(self, order: Order) -> None:
        session: Session = self._SessionLocal()
        try:
            self._upsert_order(session, order)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_fill(self, trade: Trade, order: Order) -> None:
        session: Session = self._SessionLocal()
        try:
            existing = (
                session.query(TradeRecord)
                .filter(TradeRecord.trade_id == trade.trade_id)
                .first()
            )
            if existing is not None:
                # 重复回报：整事务回滚，不产生任何副作用
                session.rollback()
                raise DuplicateTradeError(trade.trade_id)
            session.add(_trade_to_record(trade))
            self._upsert_order(session, order)
            session.commit()
        except DuplicateTradeError:
            raise
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_meta(self, key: str, value: str) -> None:
        session = self._SessionLocal()
        try:
            row = session.get(TradingMetaRecord, key)
            if row is None:
                session.add(TradingMetaRecord(key=key, value=value))
            else:
                row.value = value
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_meta_if_absent(self, key: str, value: str) -> None:
        session = self._SessionLocal()
        try:
            row = session.get(TradingMetaRecord, key)
            if row is None:
                session.add(TradingMetaRecord(key=key, value=value))
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _upsert_order(self, session: Session, order: Order) -> None:
        rec = session.get(OrderRecord, order.order_id)
        payload = _order_to_record(order)
        if rec is None:
            session.add(payload)
        else:
            for col in (
                "stock_code", "side", "order_type", "quantity", "price", "stop_price",
                "status", "filled_quantity", "filled_price", "commission",
                "cancelled_quantity", "cancelled_at", "error_message",
                "strategy_name", "signal_type", "signal_strength",
                "trade_date", "created_at", "updated_at",
            ):
                setattr(rec, col, getattr(payload, col))

    def load_all(self) -> Dict[str, Any]:
        session = self._SessionLocal()
        try:
            order_recs = session.query(OrderRecord).all()
            trade_recs = (
                session.query(TradeRecord)
                .order_by(TradeRecord.traded_at.asc(), TradeRecord.trade_id.asc())
                .all()
            )
            meta = {
                row.key: row.value
                for row in session.query(TradingMetaRecord).all()
            }
        finally:
            session.close()

        trades_by_order: Dict[str, List[Trade]] = {}
        seen: set = set()
        for tr in trade_recs:
            if tr.trade_id in seen:  # 双保险：唯一约束之外再去重
                continue
            seen.add(tr.trade_id)
            trades_by_order.setdefault(tr.order_id, []).append(_record_to_trade(tr))

        orders: List[Tuple[Order, List[Trade]]] = []
        for rec in order_recs:
            order = _record_to_order(rec)
            orders.append((order, trades_by_order.get(rec.order_id, [])))
        return {"orders": orders, "meta": meta}


# ----------------------------------------------------------------------
# Record <-> domain 转换
# ----------------------------------------------------------------------
def _order_to_record(order: Order) -> OrderRecord:
    return OrderRecord(
        order_id=order.order_id,
        stock_code=order.stock_code,
        side=order.side.value,
        order_type=order.order_type.value,
        quantity=order.quantity,
        price=order.price,
        stop_price=order.stop_price,
        status=order.status.value,
        filled_quantity=order.filled_quantity,
        filled_price=order.filled_price,
        commission=order.commission,
        cancelled_quantity=order.cancelled_quantity,
        cancelled_at=order.cancelled_at,
        error_message=order.error_message,
        strategy_name=order.strategy_name,
        signal_type=order.signal_type,
        signal_strength=order.signal_strength,
        trade_date=order.trade_date.isoformat(),
        created_at=order.created_at,
        updated_at=order.updated_at,
    )


def _trade_to_record(trade: Trade) -> TradeRecord:
    return TradeRecord(
        trade_id=trade.trade_id,
        order_id=trade.order_id,
        stock_code=trade.stock_code,
        side=trade.side.value,
        quantity=trade.quantity,
        price=trade.price,
        commission=trade.commission,
        traded_at=trade.traded_at,
        trade_date=trade.trade_date.isoformat(),
        settled=1 if trade.settled else 0,
        created_at=trade.created_at or datetime.now(),
    )


def _record_to_order(rec: OrderRecord) -> Order:
    return Order(
        order_id=rec.order_id,
        stock_code=rec.stock_code,
        side=OrderSide(rec.side),
        order_type=OrderType(rec.order_type),
        quantity=rec.quantity,
        price=Decimal(str(rec.price)) if rec.price is not None else None,
        stop_price=Decimal(str(rec.stop_price)) if rec.stop_price is not None else None,
        status=OrderStatus(rec.status),
        filled_quantity=rec.filled_quantity or 0,
        filled_price=Decimal(str(rec.filled_price)) if rec.filled_price is not None else None,
        commission=Decimal(str(rec.commission or 0)),
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        cancelled_quantity=rec.cancelled_quantity or 0,
        cancelled_at=rec.cancelled_at,
        error_message=rec.error_message,
        strategy_name=rec.strategy_name,
        signal_type=rec.signal_type,
        signal_strength=rec.signal_strength or 0.0,
    )


def _record_to_trade(rec: TradeRecord) -> Trade:
    return Trade(
        trade_id=rec.trade_id,
        order_id=rec.order_id,
        stock_code=rec.stock_code,
        side=OrderSide(rec.side),
        quantity=rec.quantity,
        price=Decimal(str(rec.price)),
        commission=Decimal(str(rec.commission or 0)),
        traded_at=rec.traded_at,
        settled=bool(rec.settled),
        created_at=rec.created_at,
    )
