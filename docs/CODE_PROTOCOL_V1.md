# 03 · M2 v1 协议链路工程验收

日期：2026-10-09。依据：`docs/EXPERIMENTS.md` 已确认 v1；交接提交 `e4d9b67940f12d1ae67f7b92c500c316b423f2f2`。本报告记录缩小预算工程 fixture，**没有运行四任务的正式百万步比较，也没有 M2 效果结论**。第一、二阶段算法与基准证据继续见 `CODE_PHASE1.md`、`CODE_PHASE2.md`。

同步：代码及测试均先在本机提交，以 Git bundle 快进同步到 sv1 的同一确定提交后才运行；本轮结束时本机 `main` 与 GitHub `origin/main` 已快进一致。最终文档提交再次通过 bundle 同步到 sv1；本机、GitHub、sv1 的最终 SHA 与 bundle SHA-256 以完成报告为准。服务器未直接修改项目源码。

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

## 原始 return 与分层结果表落实（2026-10-09，当前）

接手版本 `22fe4b3bf712bb43019f6e88001e4df6353cf324`，工作区干净；核对 `20ab3486bfaa067cc7950d3f5c8408fe2ed92ec3` 至接手版本仅文档变化。依据 `EXPERIMENTS.md` C/C.1，原始 return 及冻结指标已经确认，不再等待02选择 score。科学设置、gate、匹配、训练随机流和正式预算均未修改；本轮没有真实任务新训练或正式百万步比较。

### 代码及兼容语义

- 代码提交 `6eef3696dbdfff74a6f9ede61c28186ecd0d93f7`；首轮测试/完整 agent fixture 提交 `c4de170928e9314559be793634209320d97fe0b4`；最后统计代码/测试实测提交 `1f5b5a385346da01e1108c90114fed721c8bbaf2`。后者相比完整 agent 版本只增强离线行号/快照审计、测试和使用说明，runner与累计算法无变化；完成后的文档提交另记最终HEAD。
- `dreamerv3/return_stats.py` 的 `EpisodeReturn` 按环境接口 reward 顺序累加，排除reset、包含终止动作非零奖励，不折扣、不除长度、不乘repeat。缺少完整episode、非有限reward或中途reset拼接直接报错。DMC ActionRepeat先累计primitive reward，统计端只加一次接口返回。
- `protocol_v1.evaluate` 使用该累加器并校验完整episode；保留原 `scores/lengths/seeds/mean/update_id`，schema v2增加 `returns_raw/mean_return_raw`、目标/实际快照动作步、快照tag和update ID、逐episode ID/完整性/reward来源/环境seed/动作随机流。每个已保存快照增加小文件 `evaluation_snapshot.json`，不修改模型、奖励接口或训练目标。
- `analysis/return_metrics.py` 为标准库离线工具。生产CLI固定B=1000000、21点评价/10完整episode、末段固定五点、seed0及四任务；按实际x积分，然后任务内seed均值、四任务等权。保留每任务负向差值，输出六张表：episode、评价点、seed指标、任务均值、总体、三项Dt差值；另有JSON表和输入/工具/字段/完整性审计。使用与run index格式见 `analysis/README.md`。
- AUC和tail分别校验必需点、episode、实际步与固定快照，缺失/冲突分别标不完整。多条相同评价重写只保留一次并记录计数；不同快照/内容冲突不任选。源JSONL行号包含空行/损坏行的真实偏移；损坏未知行不会被静默丢弃来完成指标。同task/group/seed多个运行不择优或平均。总体缺任何预定组成即留空，不重加权。额外评价记录保留，不替代指定末段五点。
- 旧字段语义通过**已审查源码提交**和快照证据映射，`scores := episode_return_raw`、`mean := mean_episode_return_raw`；不从字段名推断归一化。旧聚合日志缺少逐transition terminal/reward轨迹，不能重新直接复算reward总和；episode完整性依据为源代码每次Driver只跑一个完整episode及旧长度/seed/快照读审计，输出注明 `reviewed_legacy_source`。只读审查 `evaluations.jsonl`，不将缺少快照索引的训练 `scores.jsonl` 混入评价曲线。
- 点均值浮点核对容差事前固定 `atol=1e-9, rtol=1e-12`（float64求和顺序差），动作步、数量、ID与快照匹配为精确核对。缺失指标JSON为null、CSV留空，不是0；`--require-complete` 仍生成审计表，然后返回2。输出目录已存在则拒绝覆盖。

### 本轮测试与只读产物核查

本机 Python3.12.14，sv1 dreamer Python3.11.16。纯统计测试15项通过；sv1 CPU、两个逻辑host设备上的相关回归共39项通过，保留phase1/2的probe、梯度、free-nats、shuffle及protocol测试。首个服务器命令误用 `JAX_PLATFORM_NAME` 隔离CUDA，1项设备初始化失败，日志保留；改为明确 `JAX_PLATFORMS=cpu` 后通过，没有修改代码或放宽容差。最终测试命令：

```sh
python -m unittest tests.test_return_metrics -v
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 /data/Policy_Discrepancy/envs/dreamer/bin/python -m unittest tests.test_return_metrics tests.test_protocol_v1 tests.test_rep_probe tests.test_rep_overlay -v
python -m tests.run_return_metrics_fixtures --output analysis/outputs/NEW_HAND_FIXTURE
python -m analysis.return_metrics --index analysis/outputs/RUN_INDEX.json --output analysis/outputs/NEW_RETURN_AUDIT
```

手算fixture覆盖：不同长度/负奖励/非零终止奖励、接口repeat不重复加、算术均值；实际非等间距x的面积 `4+6+15=25` 再除固定100万；逐seed积分再任务均值、任务值 `[2,4,20,30]` 等权为14，Dt任务差值 `[2,-1,4,-2]` 总体0.75；乱序与额外记录；指定末段五点；缺失初始/终点/中间点、完整episode、快照/均值冲突；旧score映射和未知语义拒绝。另有纯合成21点×10episode曲线 `r=1+x/100000`，AUC/B=6、末段五点均值10。多seed只为统计fixture，不新增训练seed或科研结论。

`c4de170...` 上沿用完整连续动作CPU/float32 agent矩阵：12动作/9更新、网格0/3/6/9/12，报告开关、评价episode数及并行环境变体的参数摘要仍为 `4317c6095b04e61698f26beb6e7c077c6c5d49e1baf135a716e3b3849c7d74b2`，与先前协议fixture一致；训练收据逐位一致，四模式均完成，中断拒绝也保留。最后离线reader读其4个模式的schema v2，40条显式完整episode/20个源快照匹配通过。它是CPU/debug工程fixture，不能替代04真实size50m CUDA补验。前轮20ab的真实CUDA修复证据仍保留其原范围。

既有04主矩阵只读统计：本机三份小证据包哈希与 `SERVER_INTEGRATION_SIZE50M` 一致；实时只读三台服务器16目录的manifest/evaluations/config/execution/audit/final，共96文件哈希与本机镜像全同。原始运行SHA仍为 `1393548fb6e46d28f92a834204f7b09495a3eab1`，没有重跑或修改。48个0/2049/4098工程评价点、96条episode的旧return映射、算术mean、长度/seed、快照更新counter及文件来源核对通过。源点 `source_complete=true` 只表示各自2episode工程配置满足；正式 `protocol_complete=false`。16条正式AUC/tail及总体/差值全部null，明确短预算/不足10episode/缺少冻结点，不缩区间或除4098来冒充正式指标。`--require-complete` 实测返回2。

### 证据位置、同步和剩余条件

- 本机（均Git忽略）：`analysis/outputs/return-v1-hand-fixture-1f5b5a3/`、`return-v1-engineering-audit-1f5b5a3/` 六表/JSON/audit；`return-v1-source-20261009/` 的run index、小日志、三台实时哈希和 `live-source-parity.json`；`return-v1-validation-summary-1f5b5a3.json` 保存核查结论及产物SHA-256；测试stdout日志也在outputs。
- sv1 CPU agent与schema审计：`/data/Policy_Discrepancy/repo/analysis/outputs/return-v1-agent-fixtures-c4de170/`、`return-v1-agent-schema-audit-1f5b5a3/`。确定提交、测试日志和bundle记录：`/data/Policy_Discrepancy/runs/statistics-raw-return-v1-20261009-c4de170/`。代码从本机Git提交后通过校验bundle快进同步；没有服务器直接改项目代码，没有使用GPU或干扰现有GPU作业。GitHub及最终文档同步范围以完成回复核对为准。
- 统计代码、分层结果表和本轮允许的只读核查已完成。未产生正式指标或方法效果；旧episode缺少逐reward原始轨迹的证据限制保留。下一步由**现有04**独立核查源目录/快照/新schema及表格，完成此前真实size50m工程补验、完整配置与资源质量检查。原始return指标已冻结，归一化score讨论不阻塞；正式比较仍须工程闭环和用户另行授权。不得自动创建chat或发送消息。
