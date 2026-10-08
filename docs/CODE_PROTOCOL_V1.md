# 03 · M2 v1 协议链路工程验收

日期：2026-10-09。依据：`docs/EXPERIMENTS.md` 已确认 v1；交接提交 `e4d9b67940f12d1ae67f7b92c500c316b423f2f2`。本报告记录缩小预算工程 fixture，**没有运行四任务的正式百万步比较，也没有 M2 效果结论**。第一、二阶段算法与基准证据继续见 `CODE_PHASE1.md`、`CODE_PHASE2.md`。

## 提交与改动

- 主 runner 代码提交 `95fe492`，后续代码修正至 `5df5df5`；最终工程测试提交 `b20ffeda07fe741deb1b9ee285174a3e34bcaaf8`。本报告完成后的文档提交另记为最终 HEAD，不改变已验收代码或测试。
- `embodied/run/protocol_v1.py`：精确动作计数、动作步评价网格、每次成功更新收据、匹配累计、独立报告/评价随机流、检查点及运行清单。
- `embodied/core/driver.py`：`step_selected()` 只推进所选训练环境，以多环境配置精确停在评价/终点动作边界。
- `embodied/jax/agent.py`：评价 policy 使用独立 seed/counter；显式同步当前参数快照；训练 batch 在消费时同步分配训练 key，报告复用已消费 batch 的只读切片。
- `dreamerv3/main.py`、`dreamerv3/configs.yaml`：`protocol_v1` 入口和 `m2_v1` 完整预设。预设使用 size50m、纯视觉 DMC、环境 seed、训练 seed0、alpha20、窗口 `[100000,300000)`、预算 1000000、间隔 50000、每点 10 episode；固定 `run.train_ratio=256` 作为四组共用的工程配置。
- `dreamerv3/rep_probe.py`：在同一次成功优化更新的有效 `[B,T]` 位置全局求 `sum(1[raw KL>free_nats] × w)`、active count 和非法值 count。logging 的实际训练权重仍为 1，候选 w 单独统计。两卡测试检查跨 shard 求和。
- `dreamerv3/freeze_c.py`：从完整 logging baseline 的原始 update 收据重算 c，核对最终状态、评价网格及匹配结果，写只读来源信息和 SHA-256。constant 正式入口校验任务、seed、alpha、窗口、阈值、c 和产物校验值；工程 fixture 可绕开正式产物要求，不能作为正式 c。
- `embodied/envs/dummy_cont.py`、`tests/test_protocol_v1.py`、`tests/compare_protocol_runs.py`、`tests/run_protocol_fixtures.py`：缩小预算的连续动作完整 agent 验收。未改动基准与 overlay 测试文件保持不变。

## 实现口径

- `train_action_steps` 是所有训练环境实际执行的非 reset 动作总数；replay 预填充也计入。reset 单列，评价动作单列。runner 在每个动作边界选择最多 `min(envs, remaining)` 个环境，避免并行环境超额。训练更新由原 `train_ratio / (batch_size × batch_length)` 触发。
- 评价在初始化的 0 步及每个动作网格点执行。先同步优化后的 policy 参数，再保存该点快照；此时训练暂停，用独立评价环境顺序采样恰好指定数量的完整 episode。评价结果记录点位、episode seed、return、长度及快照对应的 update ID。
- 每次优化开始记录动作步 `s_k` 和单调 update ID，**成功完成并验证 loss/梯度有限后**记录 `S_k,N_k`。只累加 `100000<=s_k<300000`，不是按 replay transition 时间或日志写出时间。原始收据写 `update_receipts.jsonl`，累计量和 ID ledger 写 checkpoint；同 ID 同值只计一次，冲突报错，不去重正常 replay 重采样。
- 训练 policy/更新沿用原 JAX seed 序列；replay 使用稳定 SHA-256 派生的独立 seed。训练环境 seed 按 `(root_seed,task,train_env,index)` 派生。报告用 `(root_seed,task,report,report_id)` 派生独立 NumPy key，并复用当前更新已消费的 batch。评价环境按 `(root_seed,task,eval_env,grid_index,episode_id)` 派生；评价 policy 的 PCG64 seed 含独立 `EVAL` 命名空间和该 episode seed。shuffle 保留第二阶段独立 `fold_in(0x4454, optimizer_step)` key。实际映射、配置、提交、主机、设备类型、Python/JAX/dm-control/MuJoCo 版本与依赖锁哈希均写 `protocol_manifest.json`。
- 无活跃位置、非法 D/w/raw KL、非法 c、非有限 loss/梯度均停止报告。`train/dt/` 保留 Dt、源/实际权重、raw KL、活跃数与比例、活跃位置权重和实际超额 KL 减少量，可区分权重没作用与作用后没有任务收益。v1 的单 seed 不能用于跨 seed 稳定性结论。

## 缩小预算验收

sv1 (`lyg2153`)，`/data/Policy_Discrepancy/envs/dreamer/bin/python`，Python 3.11.16、JAX 0.4.33、dm-control 1.0.48、MuJoCo 3.15.0，依赖锁 SHA-256 `cd50c7936690f9adbc5c35f99b3df2f7a977cef3222582f988979b11f65b6f22`；CPU/float32 与 A100 CUDA/bfloat16。所有运行产物在 `/data/Policy_Discrepancy/runs/verify-protocol-v1-95fe492/`。`dummycont_test`、12 训练动作、间隔 3、窗口 `[3,9)`、2 环境、2 评价 episode、`free_nats=0.1` 均只用于**工程 fixture**，不改变正式协议。报告开关、评价 episode 数、环境并行方式分别改变；完整四模式链路均从头运行。生产参数从不使用 `debug` 覆盖规模。

在服务器仓库执行的主要命令：

```sh
CUDA_VISIBLE_DEVICES=0,1 /data/Policy_Discrepancy/envs/dreamer/bin/python -m unittest tests/test_rep_probe.py tests/test_rep_overlay.py tests/test_protocol_v1.py -v
CUDA_VISIBLE_DEVICES=0 /data/Policy_Discrepancy/envs/dreamer/bin/python -m tests.run_protocol_fixtures --output_dir /data/Policy_Discrepancy/runs/verify-protocol-v1-95fe492/matrix-b20ffed
/data/Policy_Discrepancy/envs/dreamer/bin/python tests/run_phase2_snapshots.py --output_dir /data/Policy_Discrepancy/runs/verify-protocol-v1-95fe492/agent-b20ffed --baseline_src /data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/baseline-src --platform cpu --dtype float32
CUDA_VISIBLE_DEVICES=0 /data/Policy_Discrepancy/envs/dreamer/bin/python tests/run_phase2_snapshots.py --output_dir /data/Policy_Discrepancy/runs/verify-protocol-v1-95fe492/agent-b20ffed --baseline_src /data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/baseline-src --platform cuda --dtype bfloat16
```

17 项 unittest 通过：窗口起点纳入/终点排除、按更新开始步归属、重复 replay、同 ID 去重与冲突、不同 active count 的 S/N、N=0/非法值停止、多环境局部步进、并行 worker、可重放种子、正式视觉预设、冻结产物校验，以及 phase1/2 的梯度隔离和跨 shard shuffle。单位 fixture 的边界是精确值；完整 agent 训练状态比较要求逐位相同（容差 0）。CPU 有效 overlay 的 loss/梯度分解用 `atol=2e-6, rtol=2e-5`；CUDA/bfloat16 用 `atol=0.002, rtol=0.04`，未改动分支仍逐位相同。这些后端容差沿用第二阶段。

最终提交的完整 agent 工程矩阵：baseline 及报告开启、1 episode/点评价、并行环境四种运行的最终参数 SHA-256 均为 `4317c6095b04e61698f26beb6e7c077c6c5d49e1baf135a716e3b3849c7d74b2`；每个都是 12 个训练动作、5 次 reset、9 次更新、9 条相同收据，匹配 `S=12.999326229095459,N=13`。评价网格精确为 `[0,3,6,9,12]`；2 episode/点的评价动作是 30，1 episode/点为 15；共同 episode 的 seed/return/长度一致。报告与评价次数变化没有改变后续训练参数、训练 policy/batch key 计数或匹配收据。Dt、constant、shuffle 三组也完成同样网格和精确预算；工程 c 只由该小 baseline 算出，不用于真实任务。

关闭、logging、Dt alpha=0、constant c=1 在 CPU/float32 与 CUDA/bfloat16 均与未改动 `e935ff7` 基准的 789 个共享数组逐位相同；有效 Dt/constant/shuffle 各有 490 个未改动数组逐位相同，活跃 rep loss 与梯度如预期变化。CPU 的有效活跃权重约为 Dt 0.700708、constant 0.600000、shuffle 0.700708；CUDA 对应约 0.700287、0.600000、0.700287；shuffle 两后端均为 6 位置中 5 个实际换位。跨 shard 原始匹配 S/N 与全局置换由两卡单元测试覆盖。对比命令为 `tests/compare_overlay.py`，对各后端八个快照使用上列容差；两次比较均通过。CUDA 的最大总梯度变化分别为 0.01503、0.01799、0.01547，非平凡作用不是零活跃伪通过。

故意在 6 动作 checkpoint 后中断，再用相同命令恢复，会明确拒绝训练续跑；已完成运行重复启动不追加评价或收据，实测两个 JSONL 的重开前后 SHA-256 完全相同。另用 `free_nats=1000` 的全不活跃工程 fixture 运行到终点后报 `No active rep loss positions in matching window`，没有生成 c。replay 和环境内部状态不能完整无缝恢复，因此**不声称可从中途检查点无缝续跑**。检查点保存动作/reset/评价计数、更新 ID、匹配 S/N 与 ledger、agent 参数和已有随机计数；完整运行可只读核对。需要中途故障处理时应保留失败产物，从头在新的目录运行并说明原因，不将其伪装为同一条连续训练。正式失败运行的选择/排除须与 04 的执行记录一同公开。

## 04 的真实任务工程试跑入口

先在本机确定提交并通过 Git 同步，服务器核对 `git rev-parse HEAD` 和干净工作区；每个 task/mode 用新的独立目录。以下是命令形状，**不授权本轮启动正式百万步比较**：

```sh
cd /data/Policy_Discrepancy/repo
PY=/data/Policy_Discrepancy/envs/dreamer/bin/python
$PY -m dreamerv3.main --configs m2_v1 --task dmc_hopper_hop --logdir /data/Policy_Discrepancy/runs/UNIQUE_BASELINE --agent.rep_probe.mode logging
$PY -m dreamerv3.freeze_c --baseline_dir /data/Policy_Discrepancy/runs/UNIQUE_BASELINE --output /data/Policy_Discrepancy/runs/UNIQUE_BASELINE/c_frozen.json
$PY -m dreamerv3.main --configs m2_v1 --task dmc_hopper_hop --logdir /data/Policy_Discrepancy/runs/UNIQUE_CONSTANT --agent.rep_probe.mode constant --agent.rep_probe.c C_FROM_ARTIFACT --run.frozen_c_file /data/Policy_Discrepancy/runs/UNIQUE_BASELINE/c_frozen.json
```

Dt、shuffle 分别改 `--agent.rep_probe.mode dt`、`shuffle`，alpha20 已在预设中；四任务分别从头训练。运行前按 `config.yaml` 核对 size50m、纯视觉、repeat1、batch/replay/环境数、dtype/设备及四组唯一差异；记录 Git SHA、服务器/GPU、环境版本。真实 size50m 的显存、吞吐、四个 DMC 的完整长链路、异常失败与跨服务器运行尚未验收。先做独立工程集成试跑并检查诊断、磁盘及评价快照，再由 04 按已冻结 v1 执行正式比较。正式 c 实测数值尚未产生，不能用本文工程 fixture c。

评价逐点 mean 已存 `evaluations.jsonl`。正式主指标按 21 点均值作梯形积分后除以 1000000；末段指标是 800000、850000、900000、950000、1000000 五点 mean 的算术平均。四组差值和四任务等权汇总按 `EXPERIMENTS.md`，此处不产生效果判断。
