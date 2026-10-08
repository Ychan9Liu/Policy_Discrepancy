# 03 第一阶段：logging-only 工程验收

日期：2026-10-08。源码基准：`e935ff7bb01085fec8a1672e31ddfe0536c0f9c0`；本阶段算法与测试代码验收提交：`a8ca727dbd3f3698bb217c00d15997681ee644ef`。本文件记录工程 fixture，不是 M2 效果实验或正式校准协议。

## 实现与配置

- `agent.rep_probe.mode`：`off`（默认）或 `logging`。其他模式明确报错；Dt、constant、shuffle 留待第二阶段。
- `agent.rep_probe.alpha`：默认 `-1.0`，仅为“未提供”的技术占位；`logging` 必须显式给出有限、非负的浮点数。测试中的 `0` 和 `0.7` 均仅为 fixture，不是正式科研取值。`off` 不计算 probe，也不验证动作空间，保留原 baseline 支持范围。
- `logging` 只接受标准 `rssm`、全连续 action heads 和 `bounded_normal`。在正常 actor/value 前向及 repval loss 之后、总 loss 聚合之前，用 RSSM 已经计算的 prior logits 和 raw rep KL 做只读探针；`report()` 路径不重复做探针。probe 使用同一个 `feat2tensor`、同一个 actor、unimix 后逐因子 mode 的 one-hot，显式 stop-gradient；原采样的 posterior 和 imagination 路径保持原样。
- 对每个 head 的 Gaussian 均值与标准差以 float32 计算解析对称 KL：每 event 维只聚合一次，按所有 head 的总标量维数归一化为 `[B,T]` 的 Dt。使用非负的平方差形式，避免对称 KL 中对数项的浮点抵消；不做任意负值裁剪。非有限或非正标准差、非有限 Dt 产生 `NaN`，并在 `dt/invalid_D_frac` 显示，不转成有效候选权重。`w=1/(1+alpha*D)` 完全 stop-gradient。
- logging-only 的实际 rep 权重恒为 1，原 `losses['rep']` 没有乘候选 w。诊断输出均在 `train/dt/`：Dt 与源 w 的均值及 p10/p50/p90、raw rep KL 的均值及分位数、raw KL 与 Dt 的协方差、free-nats 后/实际加权后的 rep 均值、活跃比例、活跃区候选权重均值、实际权重均值、实际与候选的加权超额 KL、逐因子/整向量 mode 差异率、actor 均值/标准差差异、活跃但 Dt 接近零的总体比例，以及非法 Dt 比例。空活跃集时 `active_weight_available=0`、`active_weight_mean=NaN`；不伪造零均值。`raw_active_D_near_zero_frac` 的工程阈值为 `1e-6`，不是科研效果判据。

## 验收环境与容差

sv1，`/data/Policy_Discrepancy/envs/dreamer`，Python 3.11.16、JAX 0.4.33；依赖锁文件 SHA-256 为 `cd50c7936690f9adbc5c35f99b3df2f7a977cef3222582f988979b11f65b6f22`。GPU fixture 使用 A100-SXM4-80GB、驱动 570.158.01、CUDA/bfloat16；另用 CPU/float32。固定 seed 7、连续二维 action、B=2、world-model T=3、replay_context=1、imag_last=2、`free_nats=0.1`。另用 `replay_context=0`、`free_nats=1.0` 检查空活跃集合。所有 fixture 均用极小 debug 模型；没有运行四个真实 DMC 任务的正式训练。

一致性比较将未改动 `e935ff7` 源码单独导出到 `/data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/baseline-src`，与 off/logging 使用相同输入、carry、seed 和配置，分别在全新进程运行。对参数键、初值、原始全参数梯度、独立 loss 分支、采样 rep 特征、Ninjax 训练状态及一次 optimizer 更新后的全部状态要求**逐位相同**（容差 0）；CPU/float32 与 GPU/bfloat16 各自比较，不跨后端要求相同。bfloat16 快照以 float32 无损保存其数值。解析 KL fixture 的 float32 相对容差为 `1e-6`；零梯度与同分布 Dt=0 fixture 要求精确零。没有通过放宽一致性容差来掩盖差异。

## 命令与结果

工作目录均为服务器 `/data/Policy_Discrepancy/repo`，Python 为 `/data/Policy_Discrepancy/envs/dreamer/bin/python`；未改动基准进程的工作目录为上述 `baseline-src`，`PYTHONPATH=$PWD` 指向其源码。测试快照只存于 `/data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/`。

```sh
python -m unittest tests/test_rep_probe.py -v
python tests/snapshot_agent_identity.py --mode off --output /data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/off-cpu-latest.npz --replay_context 1 --free_nats 0.1 --imag_last 2 --alpha 0.7 --bench_updates 100
python tests/snapshot_agent_identity.py --mode logging --output /data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/logging-cpu-latest.npz --replay_context 1 --free_nats 0.1 --imag_last 2 --alpha 0.7 --bench_updates 100
python tests/compare_identity.py /data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/baseline-final.npz /data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/off-cpu-latest.npz /data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/logging-cpu-latest.npz
CUDA_VISIBLE_DEVICES=0 python tests/snapshot_agent_identity.py --mode logging --output /data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/logging-gpu-latest.npz --replay_context 1 --free_nats 0.1 --imag_last 2 --alpha 0.7 --platform cuda --dtype bfloat16
python tests/compare_identity.py /data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/baseline-gpu2.npz /data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/off-gpu2.npz /data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/logging-gpu-latest.npz
```

基准与 GPU off 快照使用同一 `snapshot_agent_identity.py` 命令，分别设 `--mode baseline` 或 `--mode off`，输出到上列对应文件；基准运行在 `baseline-src`，但脚本从当前仓库的 `tests/` 读取。五项 unittest 均通过；CPU 和 GPU 各有 618 个共享数组对基准逐位相同。两后端的一次 loss 分别为 `8.606865882873535` 和 `8.605982780456543`，各自三模式一致。GPU logging 的 `invalid_D_frac=0`、`rep_active_frac=1`、`active_weight_mean=0.99999398`、实际权重均值=1；CPU 结果同样有效。空活跃 fixture 的 `active_weight_mean=NaN`，按约定显式不可用。logging 诊断对 actor 参数、posterior/prior logits 和 history 的梯度 fixture 均为精确零；独立 rep KL 对 posterior logits 的梯度非零。

在此极小 CPU fixture 上，预热后 100 次更新 off 为 0.3931 秒、logging 为 0.4223 秒，约增加 7.4%；进程峰值 RSS 分别为 613720 和 625180 KiB，约增加 11.2 MiB。这只是工程成本样本，不能外推正式模型或真实 DMC 吞吐；GPU fixture 的峰值 RSS 约 1.34/1.35 GiB，也包含编译和并行运行影响。

## 剩余风险与第二阶段边界

- 尚未用已选四个真实 DMC 任务和最终模型规模测量 logging 开销、有效诊断分布或复现完整环境训练。没有正式 baseline 或 Dt 效果结论。
- 本阶段只验证了 off/logging；Dt 权重对活跃 rep KL 梯度的逐点缩放、constant、全局 shuffle、独立 RNG 及跨 shard 置换尚未实现和验收。这些属于第二阶段，不用第一阶段的 identity 结果替代。
- 第二阶段实现可以在第一阶段验收后继续，alpha/c 仍作为显式配置与 fixture 值；正式运行须先冻结 `docs/METHOD_SPEC.md` 第 6 节的校准、模型规模、指标、seed 与预算等协议。
