"""结算账本：订单生命周期与成交分片之间的结算闭环。

结算不变量（任意恢复点账户汇总均可由成交分片重放得到）：

* 买入下单：``frozen_cash += 限价 * 数量``，``available_cash`` 同额减少；
* 买入成交分片：释放该分片冻结，按分片实际成交价扣款并扣除分片费用；
* 卖出下单：冻结对应股数（``available_quantity`` 减少，``quantity`` 不变）；
* 卖出成交分片：扣减持仓数量，按分片价格回款并扣除分片费用；
* 撤单/收盘作废：只释放 *剩余数量* 对应的冻结，已成交部分原样保留。

每个分片独立结算且仅结算一次，因此重复/迟到回报与重启重放都不会重复扣账。
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from app.trading.base import Account, Order, OrderSide, OrderStatus, Position, Trade

logger = logging.getLogger(__name__)


class SettlementError(Exception):
    """结算被拒绝（资金/持仓不足、超量成交、终态订单等）。"""


class SettlementLedger:
    """线程内结算账本，金额全部使用 Decimal。"""

    def __init__(
        self,
        initial_cash: Decimal,
        commission_rate: Decimal = Decimal("0.0003"),
        min_commission: Decimal = Decimal("5"),
        stamp_tax_rate: Decimal = Decimal("0.001"),
        slippage_rate: Decimal = Decimal("0.001"),
        account_id: Optional[str] = None,
        broker: str = "模拟交易",
    ):
        self.initial_cash = Decimal(str(initial_cash))
        self.commission_rate = Decimal(str(commission_rate))
        self.min_commission = Decimal(str(min_commission))
        self.stamp_tax_rate = Decimal(str(stamp_tax_rate))
        self.slippage_rate = Decimal(str(slippage_rate))

        self._cash = self.initial_cash
        self._frozen_cash = Decimal("0")
        self._positions: Dict[str, Position] = {}
        self.account_id = account_id or "SIM_" + datetime.now().strftime("%Y%m%d%H%M%S")
        self.broker = broker

    # ------------------------------------------------------------------ 费用

    def calculate_commission(self, side: OrderSide, amount: Decimal) -> Decimal:
        """单笔成交分片的费用：佣金（含最低佣金）+ 卖出印花税。"""
        commission = max(amount * self.commission_rate, self.min_commission)
        if side == OrderSide.SELL:
            commission += amount * self.stamp_tax_rate
        return commission

    # ------------------------------------------------------------ 生命周期

    def freeze_for_order(self, order: Order, reference_price: Decimal) -> None:
        """下单接受时冻结资金/股份。失败抛出 SettlementError，不产生任何副作用。"""
        if order.side == OrderSide.BUY:
            freeze_price = order.price or reference_price
            # 市价单按含滑点的预期成交价冻结，避免成交时冻结不足
            if order.price is None:
                freeze_price = reference_price * (Decimal("1") + self.slippage_rate)
            required = freeze_price * order.quantity
            if required > self._cash:
                raise SettlementError(
                    f"可用资金不足，需要 {required}，可用 {self._cash}"
                )
            self._cash -= required
            self._frozen_cash += required
            order.freeze_price = freeze_price
        else:
            pos = self._positions.get(order.stock_code)
            if not pos or pos.available_quantity < order.quantity:
                available = pos.available_quantity if pos else 0
                raise SettlementError(
                    f"可用持仓不足，需要 {order.quantity}，可用 {available}"
                )
            pos.available_quantity -= order.quantity
            pos.frozen_quantity += order.quantity
            pos.updated_at = datetime.now()

    def settle_trade(self, order: Order, trade: Trade) -> Decimal:
        """结算单个成交分片，返回该分片费用。

        买入分片先按 *下单价* 释放该数量的冻结资金，再按 *分片实际成交价*
        扣款（二者价差自然回到可用现金）；卖出分片把冻结股数转为实际卖出。
        调用方负责保证同一分片不会被结算两次。
        """
        amount = trade.price * trade.quantity
        commission = self.calculate_commission(trade.side, amount)
        trade.commission = commission

        pos = self._positions.get(trade.stock_code)

        if trade.side == OrderSide.BUY:
            # 释放该分片下单时冻结的资金（按冻结基准价）
            freeze_price = order.freeze_price or order.price or trade.price
            released = freeze_price * trade.quantity
            if released > self._frozen_cash:
                raise SettlementError("冻结资金余额不足以结算成交分片")
            self._frozen_cash -= released
            self._cash += released
            # 按实际成交价扣款
            self._cash -= amount + commission

            if pos is not None:
                total_cost = pos.avg_cost * pos.quantity + trade.price * trade.quantity
                new_qty = pos.quantity + trade.quantity
                pos.avg_cost = total_cost / new_qty
                pos.quantity = new_qty
                pos.available_quantity += trade.quantity
                pos.current_price = trade.price
            else:
                self._positions[trade.stock_code] = Position(
                    stock_code=trade.stock_code,
                    stock_name=trade.stock_code,
                    quantity=trade.quantity,
                    available_quantity=trade.quantity,
                    frozen_quantity=0,
                    avg_cost=trade.price,
                    current_price=trade.price,
                    market_value=trade.price * trade.quantity,
                    profit_loss=Decimal("0"),
                    profit_loss_ratio=0.0,
                )
        else:
            if pos is None or pos.quantity < trade.quantity:
                held = pos.quantity if pos else 0
                raise SettlementError(
                    f"成交数量超过持仓，需要 {trade.quantity}，持仓 {held}"
                )
            # 冻结股数转为实际卖出，回款扣费
            pos.frozen_quantity -= trade.quantity
            pos.quantity -= trade.quantity
            pos.current_price = trade.price
            self._cash += amount - commission

        self._refresh_position(trade.stock_code)
        if self._positions.get(trade.stock_code) is not None and (
            self._positions[trade.stock_code].quantity <= 0
        ):
            del self._positions[trade.stock_code]
        logger.debug(
            "Settled trade %s %s %s %s@%s fee=%s",
            trade.trade_id,
            trade.side.value,
            trade.stock_code,
            trade.quantity,
            trade.price,
            commission,
        )
        return commission

    def restore_trade(self, order: Order, trade: Trade) -> Decimal:
        """重启重放专用：按分片直接入账，不假设存在下单冻结。

        买入直接扣 ``成交额+费用``、卖出直接增 ``成交额-费用`` 并减持仓；
        活跃订单的剩余冻结在全部分片重放后统一重建。
        """
        amount = trade.price * trade.quantity
        commission = self.calculate_commission(trade.side, amount)
        trade.commission = commission

        pos = self._positions.get(trade.stock_code)
        if trade.side == OrderSide.BUY:
            self._cash -= amount + commission
            if pos is not None:
                total_cost = pos.avg_cost * pos.quantity + trade.price * trade.quantity
                new_qty = pos.quantity + trade.quantity
                pos.avg_cost = total_cost / new_qty
                pos.quantity = new_qty
                pos.available_quantity += trade.quantity
                pos.current_price = trade.price
            else:
                self._positions[trade.stock_code] = Position(
                    stock_code=trade.stock_code,
                    stock_name=trade.stock_code,
                    quantity=trade.quantity,
                    available_quantity=trade.quantity,
                    frozen_quantity=0,
                    avg_cost=trade.price,
                    current_price=trade.price,
                    market_value=trade.price * trade.quantity,
                    profit_loss=Decimal("0"),
                    profit_loss_ratio=0.0,
                )
        else:
            if pos is None or pos.quantity < trade.quantity:
                held = pos.quantity if pos else 0
                raise SettlementError(
                    f"重放卖出分片超过持仓，需要 {trade.quantity}，持仓 {held}"
                )
            pos.quantity -= trade.quantity
            pos.available_quantity -= trade.quantity
            pos.current_price = trade.price
            self._cash += amount - commission

        self._refresh_position(trade.stock_code)
        if self._positions.get(trade.stock_code) is not None and (
            self._positions[trade.stock_code].quantity <= 0
        ):
            del self._positions[trade.stock_code]
        return commission

    def restore_active_freeze(self, order: Order, quantity: int) -> None:
        """重放末尾为活跃订单的剩余数量补建冻结。"""
        if quantity <= 0:
            return
        if order.side == OrderSide.BUY:
            freeze_price = order.freeze_price or order.price
            if freeze_price is None:
                return
            frozen = freeze_price * quantity
            self._frozen_cash += frozen
            self._cash -= frozen
        else:
            pos = self._positions.get(order.stock_code)
            if pos is not None:
                pos.frozen_quantity += quantity
                pos.available_quantity -= quantity

    def release_frozen(self, order: Order, quantity: int) -> None:
        """撤单/收盘时释放订单剩余数量对应的冻结（只释放，不碰已成交部分）。"""
        if quantity <= 0:
            return

        if order.side == OrderSide.BUY:
            freeze_price = order.freeze_price or order.price
            if freeze_price is None:
                # 市价单下单即成交，正常不会有剩余走到这里
                raise SettlementError("无法释放缺少价格的市价单冻结")
            released = freeze_price * quantity
            if released > self._frozen_cash:
                raise SettlementError("冻结资金余额不足以释放")
            self._frozen_cash -= released
            self._cash += released
        else:
            pos = self._positions.get(order.stock_code)
            if not pos or pos.frozen_quantity < quantity:
                frozen = pos.frozen_quantity if pos else 0
                raise SettlementError(
                    f"冻结持仓不足，需要释放 {quantity}，冻结 {frozen}"
                )
            pos.frozen_quantity -= quantity
            pos.available_quantity += quantity
            pos.updated_at = datetime.now()

    # --------------------------------------------------------------- 行情

    def mark_price(self, stock_code: str, price: Decimal) -> None:
        """行情更新时刷新持仓市值与盈亏。"""
        pos = self._positions.get(stock_code)
        if pos:
            pos.current_price = Decimal(str(price))
            self._refresh_position(stock_code)

    def _refresh_position(self, stock_code: str) -> None:
        pos = self._positions.get(stock_code)
        if not pos:
            return
        pos.market_value = pos.current_price * pos.quantity
        if pos.avg_cost > 0 and pos.quantity > 0:
            pos.profit_loss = (pos.current_price - pos.avg_cost) * pos.quantity
            pos.profit_loss_ratio = float(
                (pos.current_price - pos.avg_cost) / pos.avg_cost
            )
        else:
            pos.profit_loss = Decimal("0")
            pos.profit_loss_ratio = 0.0
        pos.updated_at = datetime.now()

    # --------------------------------------------------------------- 视图

    def get_positions(self) -> List[Position]:
        """业务模块说明。"""
        return [p for p in self._positions.values() if p.quantity > 0]

    def get_position(self, stock_code: str) -> Optional[Position]:
        """业务模块说明。"""
        pos = self._positions.get(stock_code)
        return pos if pos and pos.quantity > 0 else None

    def get_account(self) -> Account:
        """业务模块说明。"""
        market_value = sum(p.market_value for p in self.get_positions())
        total_assets = self._cash + self._frozen_cash + market_value
        profit_loss = sum(p.profit_loss for p in self.get_positions())
        cost_basis = total_assets - profit_loss
        ratio = float(profit_loss / cost_basis) if cost_basis > 0 else 0.0
        return Account(
            account_id=self.account_id,
            broker=self.broker,
            total_assets=total_assets,
            available_cash=self._cash,
            frozen_cash=self._frozen_cash,
            market_value=market_value,
            profit_loss=profit_loss,
            profit_loss_ratio=ratio,
            updated_at=datetime.now(),
        )

    # --------------------------------------------------------------- 对账

    def reconcile(self, orders: List[Order], trades: List[Trade]) -> List[str]:
        """账户汇总与订单/成交明细交叉核对，返回不一致说明列表（空列表表示平衡）。"""
        problems: List[str] = []

        # 1) 成交数量合计 == 订单 filled_quantity
        filled_by_order: Dict[str, int] = {}
        amount_by_order: Dict[str, Decimal] = {}
        for t in trades:
            filled_by_order[t.order_id] = filled_by_order.get(t.order_id, 0) + t.quantity
            amount_by_order[t.order_id] = (
                amount_by_order.get(t.order_id, Decimal("0")) + t.price * t.quantity
            )

        closed_statuses = (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        )
        for order in orders:
            frag_qty = filled_by_order.get(order.order_id, 0)
            if frag_qty != order.filled_quantity:
                problems.append(
                    f"订单 {order.order_id} 成交分片数量合计 {frag_qty} "
                    f"与订单已成交 {order.filled_quantity} 不一致"
                )
            if order.status in closed_statuses and (
                order.quantity != order.filled_quantity + order.cancelled_quantity
            ):
                problems.append(
                    f"订单 {order.order_id} 终态数量不闭合: "
                    f"{order.quantity} != {order.filled_quantity}+"
                    f"{order.cancelled_quantity}"
                )
            if order.status == OrderStatus.PARTIAL_FILLED and order.filled_quantity <= 0:
                problems.append(f"订单 {order.order_id} 标记部分成交但无成交数量")
            if order.filled_quantity > 0 and not order.trades:
                problems.append(f"订单 {order.order_id} 有成交量但缺少成交分片")

        # 2) 重建现金：初始资金 - 买入成交额 - 总费用 + 卖出成交额
        cash_built = self.initial_cash
        total_fee = Decimal("0")
        for t in trades:
            amount = t.price * t.quantity
            cash_built += -amount if t.side == OrderSide.BUY else amount
            total_fee += t.commission
        cash_built -= total_fee
        # 活跃买单的冻结仍在 frozen_cash 中，需要加回才等于可用现金
        active_buy_frozen = Decimal("0")
        for order in orders:
            if order.side == OrderSide.BUY and order.status.is_active:
                basis = order.freeze_price or order.price
                if basis:
                    active_buy_frozen += basis * order.remaining_quantity
        cash_built -= active_buy_frozen

        if cash_built != self._cash:
            problems.append(
                f"现金不平：分片重放 {cash_built} != 账本 {self._cash}"
            )

        # 3) 持仓数量：买入分片合计 - 卖出分片合计 == 持仓（含冻结）
        net_qty: Dict[str, int] = {}
        for t in trades:
            delta = t.quantity if t.side == OrderSide.BUY else -t.quantity
            net_qty[t.stock_code] = net_qty.get(t.stock_code, 0) + delta
        for code, qty in net_qty.items():
            pos = self._positions.get(code)
            held = pos.quantity if pos else 0
            if qty != held:
                problems.append(
                    f"持仓 {code} 不平：分片净量 {qty} != 账本持仓 {held}"
                )

        return problems
