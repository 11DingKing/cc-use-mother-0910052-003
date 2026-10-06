# 部分成交后的撤单结算业务服务

这是一个使用 Python、FastAPI 与 SQLite 实现的纯后端业务服务，包含领域模型、数据访问、业务编排、接口和异常路径测试。项目可在单个 Linux 应用容器内离线运行，使用本地 SQLite 或内存替身，不依赖外部运行服务。

## 安装

```bash
python3 -m pip install -r requirements.txt
```

## 测试

```bash
python3 -m pytest -q
```

## 构建检查

```bash
python3 -m compileall -q app
```

## API 导入冒烟

```bash
python3 -c "from app.main import app; print(len(app.routes))"
```

## 启动

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 订单生命周期与成交分片结算闭环

模拟适配器（`SimulationAdapter`，连接配置支持 `db_url`，默认内存 SQLite，
可传文件路径如 `sqlite:////data/trading.db` 实现跨重启持久化）保证以下语义：

- **下单即冻结**：买单按限价（市价单含滑点缓冲）冻结资金，卖单冻结股数。
- **成交分片（Trade）**：每次部分成交生成带唯一 `trade_id` 的不可变分片，
  独立结算费用、保留原始价格与成交时间；订单聚合字段只做加权汇总，
  不覆盖任何分片原值。
- **撤单只释放剩余**：部分成交后撤单仅释放 `remaining_quantity`
  对应冻结，状态变为 `cancelled`，已成交分片与持仓原样保留。
- **回报幂等**：`POST /api/trading/trades/report` 以 `trade_id` 去重，
  重复回报返回首次结果；迟到（撤单/全成/收盘后）与超量回报被拒绝且不落账。
- **失败重试**：下单携带 `client_order_id`，同一键的重试返回同一订单，不重复冻结。
- **跨日收盘**：`POST /api/trading/day-end` 将仍活跃订单作废（`expired`）
  并释放剩余冻结，作废后行情不再撮合、回报不再接受。
- **稳定分页**：订单按 `created_at, order_id` 倒序、成交按账本顺序，
  `GET /api/trading/orders` 与 `GET /api/trading/trades` 支持 `limit/offset`
  并返回 `total`。
- **重启重放与对账**：重启后按成交分片重放重建现金、持仓与活跃单冻结；
  `GET /api/trading/reconcile` 从分片独立重算现金/持仓并与订单数量闭合关系
  交叉核对，任意恢复点均应返回 `{"balanced": true}`。

