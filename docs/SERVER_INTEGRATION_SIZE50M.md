# 04 · 四任务 size50m 真实 DMC 工程集成试跑

日期：2026-10-09（Asia/Shanghai）。授权范围：服务器核查、确定版本同步、依赖检查和隔离工程短跑；不启动正式百万动作步比较。

**当前补充（2026-10-09）**：本文件保留修复前验收事实和原始16条四模式数据。03修复/统计实现后的04独立复核、四任务新schema CUDA报告补验、冻结输入及恢复边界见 `docs/SERVER_RETURN_CUDA_AUDIT.md`。下文“未闭环/指标重审”属于当时状态；原始return现已冻结，有限工程补验已完成，正式运行仍需用户另行授权。新核查发现CUDA卡选择未约束EGL，已在执行包装器显式绑定/记录渲染设备，并在sv2核查实际落卡；旧资源峰值不能读成独占总显存，其他卡/服务器映射启动前仍需检查。

**结果：16 个主运行的训练、评价、匹配、工程 c 冻结和重开检查通过；整体验收尚未闭环。** CUDA 开启报告有明确异常；真实 hopper 同配置重跑不逐位一致，使评价次数变化的端到端隔离比较无法作因果判断。首选下一任务是 03 修复报告路径并定位重跑差异，再由 04 补验。没有正式 c、正式 M2 结果或方法效果结论。

## 版本、环境与同步

- 接手本机 HEAD `c4ef89c1413bc440d1d3a2d7583ff2138c9a6084`，工作区干净。比 03 交接 `9aafc0ba24fb2aa27a4620fcdf43e81182a61282` 仅多 AGENTS/CHATS 的交接规则；03 最后测试 `b20ffeda07fe741deb1b9ee285174a3e34bcaaf8` 之后没有算法或测试文件变化。
- 主矩阵、完成重开、报告故障复现和中断检查的实际 SHA：`1393548fb6e46d28f92a834204f7b09495a3eab1`。新增外部 supervisor，不修改 agent、runner、DMC 或科学协议。
- hopper 的独立评价补验、同配置重跑实际 SHA：`63d36a111d247aa8cd05af173f74e43e4e35d2e1`。相对主矩阵仅增加资源分析脚本、GPU 占用检查和评价补验入口；算法/runner/测试不变。
- 最终报告及审计脚本提交在完成回复中另给完整 SHA；不能把文档提交写成重新完成了训练验收。最终审计脚本已从本机复制到产物目录执行，源文件 SHA-256 为 `aa0cf95b02bdba040620262acec418ee9701155117087cea77e7829bc9c0f61e`。
- 初查 sv1 已是 `c4ef89c...`；sv2/sv3 是 `e935ff7bb01085fec8a1672e31ddfe0536c0f9c0`，三者工作区干净。主矩阵启动前通过校验 Git bundle 将三台快进至 `1393548...`；仅承担补验的 sv1 随后快进至 `63d36a1...`。服务器未直接修改仓库代码。bundle 首次 fetch 使用不存在的 `main` ref 失败，改用 bundle 内的 `HEAD` 后成功，无实验受影响。
- execution bundle SHA-256：`fd28d55fdcd0a0239cf827ef878973a9cf6f4a9d85613e53ad2b227c10d0a2ab`；supplement bundle：`3bc9ce396980d9dfe27319b02dd3a3715ee0ed679a3a13bf532b222745ae4f18`。本机及接收服务器哈希一致、`git bundle verify` 通过。最终 GitHub/服务器同步状态以完成回复为准，运行 manifest 不随报告同步改变。

三台环境均为 `/data/Policy_Discrepancy/envs/dreamer`，Python 3.11.16、JAX/jaxlib 0.4.33、dm-control 1.0.48、MuJoCo 3.15.0；`pip check` 通过，实际 `pip freeze` 的 SHA-256 都是 `cd50c7936690f9adbc5c35f99b3df2f7a977cef3222582f988979b11f65b6f22`，与依赖锁一致。虚拟环境继承 `base-py311`，未删除或改动基础环境。GPU 全为 A100-SXM4-80GB，驱动 570.158.01。CUDA 编译日志提示驱动支持版本低于 PTX compiler，禁用并行编译；本轮未升级驱动，记录为编译耗时限制。

| 服务器 / hostname | 初查 GPU 作业 | 初查 `/data` 可用 | 初查根分区可用 | 主矩阵任务与物理 GPU |
| --- | --- | --- | --- | --- |
| sv1 / lyg2153 | 8 卡空闲，无计算进程 | 923 GiB | 101 GiB | hopper_hop: 0；quadruped_run: 1 |
| sv2 / lyg0360 | 8 卡空闲，无计算进程 | 约 2 TiB | 708 GiB | quadruped_walk: 0 |
| sv3 / lyg0326 | 8 卡空闲，无计算进程 | 约 1.2 TiB | 323 GiB | reacher_hard: 0 |

完整 GPU UUID、作业/磁盘/内存快照见各服务器产物根目录的 `preflight.txt`、`postflight.txt`；依赖列表见 `pip-freeze.txt`。2026-10-09 本次检查已否定旧 STATUS 中“sv3 根分区仍满”的当前状态表述，没有执行磁盘清理。

## 工程配置与正式协议区别

不使用 `debug` 模型预设。固定 clean、仅视觉 64×64 RGB、`image=True/proprio=False/use_seed=True`、repeat=1、默认相机（hopper/reacher=0，quadruped=2）、原 DMC horizon。size50m 的 RSSM 为 deter4096/hidden512/stoch32/classes32，encoder/decoder depth32、各 head units512；日志实际 optimizer 参数约 41.64–41.66M，属于该预设的实际参数数目。free-nats=1，dyn/rep scale=1/0.1；ac_grads=false、reward_grad=true、repval_grad=true、repval_loss=true；alpha=20，训练 seed=0。

| 字段 | 本次工程值 | 当前 v1 原配置/正式值 |
| --- | --- | --- |
| action_budget | 4098 | 1000000 |
| eval_every_actions / 网格 | 2049 / `[0,2049,4098]` | 50000 / 21 点 |
| eval_eps | 主矩阵 2；独立评价补验 1 | 10 |
| matching window | `[2200,3600)` | `[100000,300000)` |
| engineering_fixture | True | False |
| train_ratio / batch | 256 / 16×64，replay_context=1 | 同 m2_v1，不缩减 |
| 训练/评价环境数 | 16 / 1，parallel driver | 同 m2_v1 |
| replay | capacity=5e6、chunksize=1024、online=True、uniform=1 | 同 m2_v1 |
| dtype / 拓扑 | CUDA/bfloat16，单卡 policy+train，逻辑设备 `[0]` | 单卡预设；正式运行前须记录实际物理 UUID |
| jax.prealloc | False，便于观察实际分配 | 原预设 True，本轮不验证 True 的显存预留 |
| logger.outputs | `[jsonl]` | 原预设 `[jsonl,scope]` |
| report_every_actions | 主矩阵 0；故障补验 2049 | 原预设 0 |
| engineering_stop_after_actions | 主矩阵 0；中断检查 2049 | 正式禁止此工程开关 |
| constant 来源 | 本任务工程 baseline，工程窗口 c | 必须是完整正式 baseline 及正式窗口冻结产物 |

2049 不能被 16 整除；每个评价边界和终点实际停在精确动作步，没有并行 overshoot。保留默认训练规模和比例，预算足以在 replay 预填充后产生 517 次成功更新，并让 350 次更新进入缩小匹配窗口。该窗口只是工程验收窗口，不是正式 c 的近似校准。

每个目录保存原始完整 `config.yaml`、含完整 flat config 与配置哈希依据的 `protocol_manifest.json`、实际命令/环境/时间/退出码的 `execution-*.json`。本机的 `docs/size50m_integration_evidence.json` 提供完整配置及资源/审计索引。四组实际配置只改变模式、工程 c 和 logdir；模型、梯度开关、seed、replay、更新频率及评价保持一致。独立评价补验仅另改 eval_eps；同配置重跑仅另改 logdir。

所有子进程使用 MUJOCO_GL=egl、PYOPENGL_PLATFORM=egl、OMP/OPENBLAS/MKL_NUM_THREADS=1、运行内 TMPDIR 和指定 CUDA_VISIBLE_DEVICES；没有安装/升级依赖。GPU 种类、版本、16 个训练环境 seed、replay seed 及评价每条 episode seed 均可从 manifest/评价文件核查。

## 运行索引与工程 c

共同根目录：`/data/Policy_Discrepancy/runs/engineering-size50m-20261009-1393548/`，以下简称 EROOT。各任务在表中服务器有 `EROOT/<task>/{logging,dt,constant,shuffle}/` 四个独立目录，四组全部从头训练。完整检查点、replay、快照留在服务器，不提交到 Git。每个模式的 `stdout-initial.log` 有行级 epoch 时间，`resources-initial.jsonl` 有约 1 秒采样，执行命令见 `execution-initial.json`。

| task / action 维数 | 服务器 / 物理 GPU UUID | 四模式退出/主审计 | 工程 S | 工程 N | 工程 c |
| --- | --- | --- | --- | --- | --- |
| dmc_hopper_hop / 4 | sv1 GPU0 / GPU-e2e89de9-6fa9-23b1-806d-4cfb6fed08c8 | 全部 0 / 通过 | 343051.2319946289 | 343068 | 0.9999511233767909 |
| dmc_quadruped_run / 12 | sv1 GPU1 / GPU-78457590-ae80-0dbd-aec7-f3bb224824c9 | 全部 0 / 通过 | 358377.1010131836 | 358400 | 0.9999361077376775 |
| dmc_quadruped_walk / 12 | sv2 GPU0 / GPU-76344ce3-10a1-952d-68f5-7720a3cda667 | 全部 0 / 通过 | 358378.59173583984 | 358400 | 0.9999402671200889 |
| dmc_reacher_hard / 2 | sv3 GPU0 / GPU-aae57745-85b7-eab0-731a-1fb814457489 | 全部 0 / 通过 | 358366.5705566406 | 358392 | 0.9999290457282546 |

每任务 c 在完整工程 baseline 完成后由 `python -m dreamerv3.freeze_c --engineering_fixture` 生成 `logging/c_frozen_engineering.json` 及 `.json.sha256`，再启动 constant。artifact 明确 `engineering_fixture=true`、窗口、task、baseline SHA、完整配置哈希、receipt 哈希、纳入 update IDs、S/N/c/阈值/alpha/UTC 冻结时间。本轮每任务纳入 update IDs 42–391，共 350 个，开始动作步 2200–3596；3600 不纳入。用同一工程目录尝试正式 freeze 被明确拒绝，没有生成正式 c 文件。

工程 constant 使用 fixture 入口，正式入口的冻结文件校验不会在 fixture 模式强制执行；外部审计另外核对 artifact 哈希、任务、来源、收据及配置 c 精确一致。不能把这写成真实任务正式入口全量验收。正式入口的旧 fixture 证据见 CODE_PROTOCOL_V1。

主运行复现形状（具体命令保存于每个 execution JSON；以下不会启动正式预算）：

```sh
cd /data/Policy_Discrepancy/repo
/data/Policy_Discrepancy/envs/dreamer/bin/python -u scripts/run_size50m_integration.py \
  --task dmc_hopper_hop --gpu 0 \
  --root /data/Policy_Discrepancy/runs/NEW_ENGINEERING_ID/dmc_hopper_hop
```

supervisor 明确固定 4098 动作的工程配置，拒绝覆盖既有目录。最终版本在每次子进程启动前检查该 GPU 的显存占用，已占用则拒绝启动；这是即时检查，不能替代长期独占资源安排。c 不允许人工填入 fixture 值或从其他任务复制。

## 主链路验收证据

1. 所有主运行最终：4098 个非 reset 训练动作、16 次初始 reset、517 次成功更新/517 条连续唯一收据、259 次训练 policy 调用及 517 个训练 batch key。首次更新在动作 2032，预填充属于 4098 的预算；reset 与评价计数单列。16 个训练环境本次各只推进约 256 动作，尚未覆盖真实训练环境的完整 episode 结束/reset 循环。
2. 每个主运行评价点精确为 0/2049/4098，update ID 分别 0/5/517。每点 2 条完整 episode，非 reset 长度均 1000；评价动作共 6000，不混入训练预算。共有 48 个实际参数快照。读取 checkpoint 后核对各快照的 agent 更新计数；0 点 actions/updates 都为 0。终点快照、评价后的最终参数、最终 checkpoint 的参数摘要一致，终点评价未修改训练参数。
3. 每条收据验证成功 update ID、开始动作步、非法值计数；窗口从原始收据全局 S/N 重算，与 matching_result/final_state/冻结 artifact 一致。对真实收据列表再次提交相同 ID，更新数与 S/N 不变；同 ID 故意改统计会拒绝。没有对正常 replay 重采样去重，也没有把 active_weight_mean 的日志均值用作 c。
4. 所有四模式损失、梯度范数及 Dt 诊断有限，invalid_D_frac/match_invalid_count 为 0；未出现空活跃均值不可用。本次只验证成功路径，真实任务的 N=0/非法值失败注入未重复做，旧 fixture 证据继续适用其工程范围。
5. logging 实际权重为 1、超额 KL 实际减少为 0；源候选 w 单独用于匹配。Dt/constant/shuffle 在活跃 rep KL 位置有非零权重减少和超额 KL 减少。参数轨迹不同是观察事实，因真实重跑本身不同，不能用组间参数差异作方法因果证据。本轮没有固定真实 batch 的成对原始梯度快照，梯度隔离/逐点缩放的已有证据范围仍是03 fixture。shuffle 全局覆盖单卡完整有效 `[16,64]`，实际换位比例末段均约 0.999083；未跑真实 size50m 的多卡 shard，不能代替此前两卡 fixture 或声称真实多卡已验收。
6. 四个完整 logging 运行用完全相同命令重开均退出 0，提示已完成；evaluations、update_receipts、final_state、matching_result 的 SHA-256 均未变。新 supervisor 日志、资源日志是新的审计文件，不声称整个目录所有字节不变。

随机流的命名空间/派生方式与实际 seed 已记录：训练初始化/更新和 policy 原 JAX counters；replay 独立 SHA-256 派生；评价按 task/grid/episode 派生环境 seed，动作/latent 使用 EVAL 命名空间；report 与 shuffle 使用独立方案。源码与计数核对不替代下面真实端到端差异的调查。

末段第二训练区间的聚合诊断（不是最后一批，也不是全程机制结论）：

| task | Dt rep_active_frac | Dt 活跃位置平均权重减少 | Dt 加权超额 KL 平均减少 | shuffle 换位比例 |
| --- | --- | --- | --- | --- |
| hopper_hop | 0.969980 | 7.01365e-5 | 5.64254e-4 | 0.999083 |
| quadruped_run | 1.000000 | 7.66520e-5 | 6.47981e-4 | 0.999083 |
| quadruped_walk | 1.000000 | 7.41000e-5 | 6.09632e-4 | 0.999083 |
| reacher_hard | 0.999977 | 7.27558e-5 | 5.25628e-4 | 0.999083 |

Dt 平均约 3.5–3.8e-6，工程 c 接近 1，短跑中的实际压缩减弱很弱。保留 alpha20，不改变模型、free-nats、梯度开关或窗口以追求更强作用。非零工程作用不证明控制收益、选择性或 M2 有效；这些工程 c 也不能预示正式窗口的 c。

## 补验、故障与未闭环条件

### CUDA report 的已复现代码异常

sv1：`EROOT/dmc_hopper_hop/isolation/`，实际 SHA `1393548...`，eval_eps=1、report_every_actions=2049，其余训练配置同主矩阵。进程退出 1；只完成 0 点评价和 5 次训练更新，没有 final_state，没有 c。本次未在服务器修代码或关闭 transfer guard 绕过。

```text
embodied/run/protocol_v1.py:162, report_on_batch
batch = {k: v[:, :length] for k, v in batch.items() if k != 'seed'}
jaxlib.xla_extension.XlaRuntimeError: INVALID_ARGUMENT:
Disallowed host-to-device transfer: aval=ShapedArray(int32[]),
dst_sharding=NamedSharding(... PartitionSpec() ...)
```

完整命令、堆栈、配置和资源记录分别在 `execution-initial.json`、`stdout-initial.log`、`config.yaml`、`resources-initial.jsonl`；外层 supervisor 的 AssertionError 是子进程失败后的检查，不是根因。CODE_PROTOCOL_V1 的 CPU 小 fixture 报告测试不能代替这个 CUDA size50m 路径。主矩阵 report_every_actions=0，所以不能据其成功声称报告开启路径通过。03 应修复设备上的 batch 切片/传输语义，保留 report 只读、复用已消费 batch、独立 RNG 的协议。

报告失败是在非零训练进度、尚未保存该进度 checkpoint 时发生。只读检查确认最新 checkpoint 的 actions=updates=0、ledger 空、S=N=0，而目录已有 5 条成功更新收据。保留原目录，不在其中重开，不把从旧 checkpoint 重跑称为续跑；03 还应检查这个拒绝/去重边界，不能仅依赖 state.actions>0 的恢复分支。此额外恢复边界本轮没有主动续跑测试。

### 真实 hopper 重跑差异，不等于已定位 RNG 泄漏

sv1 的 `evaluation-isolation/`（eval_eps=1，report=0）完整运行退出 0，但 `tests/compare_protocol_runs.py` 对 baseline 的严格参数哈希比较失败。新增 `same-config-repeat/`（eval_eps=2，report=0）作同配置对照，也完整运行退出 0、却不逐位一致。两者实际 SHA `63d36a1...`，相对主矩阵算法/runner/测试没有变化，完整配置差异已检查，分别只有 logdir+eval_eps、只有 logdir。

| 对照 | 最终 params SHA-256 | matching S / N | 相对原 baseline 不同的收据 |
| --- | --- | --- | --- |
| 原 logging baseline | `2f27117744d43da5ac2929131c91f2d021f6c3112bb959e69ff2454c8c80e237` | 343051.2319946289 / 343068 | — |
| eval_eps=1 | `dcd9ab3c64f2f844083bdb6227e1a5b5d34f1d5a63a4323c4c0ed095e8a372b3` | 342896.1499328613 / 342913 | 494 / 517 |
| 同 eval_eps=2 重跑 | `5dcfd35dac251e089465686d1d7cdb1792b821d1aa63c8ad4152eaa253eaeb03` | 343050.236907959 / 343067 | 489 / 517 |

三者 actions=4098、resets=16、policy calls=259、batch keys=updates=517。初始化快照的全部 294 个数组相同；前两条成功收据相同；两个补验的首次收据分歧都在 update ID=2、开始动作 2040：原 S_k=1023.4505615234375，对照 S_k=1023.4481201171875，N_k 都为 1024。三点共同 episode 的 seed、长度、return 相同（hopper 此短跑对应的共同 return 均为 0，证据辨别力有限）。

排除 optimizer 状态后，同配置重跑在 2049 点的模型数组最大绝对差约 `7.45524e-7`，到终点约 `7.90520e-4`；eval_eps=1 对照分别约 `6.78003e-7`、`1.06410e-3`。原始检查点参数摘要包含 optimizer/normalizer 状态，不能将 optimizer 的较大差值混称为模型权重差。完整键级差异、计数、配置和首条分歧见 `EROOT/reproducibility-comparison.json`。

**证据边界**：同配置重跑自身已不同，不能把 1/2 episode 的参数差异直接认定为评价消耗训练 RNG；也不能用相同 counters 声称完整真实轨迹已经可重放。本轮没有采集逐步图像/环境状态/batch/carry/原始梯度的完整哈希，尚不能区分渲染、训练数值、异步时序或其他实现原因。03 应在固定输入对照和真实环境重跑之间定位首次分歧、给出技术结论及补验，而不是运行后放宽容差或更改科学协议来宣布通过。

### 中断处理与资源安排疏漏

sv3：`EROOT/dmc_reacher_hard/interrupted/`，同主配置、engineering_stop_after_actions=2049。完成两点评价和 5 次更新后，按工程开关故意退出 1；相同命令重开退出 1，明确 `Training continuation refused: environment and replay RNG state are not completely checkpointed; start a fresh run`。评价与收据的 SHA-256 重开前后不变，没有 final_state 或冻结 c。证据在 `interruption_audit.json`、两个 execution JSON、`stdout-initial.log`/`stdout-restore.log`。故意中断不是未知训练崩溃，也不能当作真正故障恢复成功。

该补验启动前 04 未重新核查 GPU 占用，当时 sv3 GPU0 已有别的项目新增作业，约 177 秒的中断/重开检查与其共享了 GPU。未停止或修改其他作业，可能的资源竞争影响未测量；这是执行安排疏漏，不能写成全程未干扰已有作业。其整卡采样峰值 26581 MiB 含外部进程，**不纳入本报告的 size50m 资源估算**。之后更新本机 supervisor、Git 同步到承担下一补验的 sv1，加入每次启动前的占用拒绝检查；其后的评价与同配置补验选择了即时空闲 GPU。最新资源快照仍显示 sv3 GPU0 有外部作业，后续安排须避开。

## 资源实测与正式运行估算

主矩阵四任务同时进行，sv1 同时两卡、sv2/sv3 各一张卡；每任务四模式依次运行。资源只取这些最初空闲 GPU 上的主运行，不取污染的中断补验。显存按 nvidia-smi 整卡约 1 秒采样，属于观察到的采样峰值，不能保证捕获更短瞬时峰值。不是 JAX allocator 的精确进程峰值。

吞吐取第二训练区间：排除评价暂停和初始编译，并再跳过 64 次更新预热，实际计时约 71–73 秒；用成功收据的动作步/update ID 差除以采样时间差。包含环境、replay、收据 IO 的训练墙钟，不是纯 GPU 更新速度。评价时间从该点 `Saving checkpoint: eval_snapshots/...` 到评价后开始保存训练 checkpoint，含参数快照 IO、环境创建和顺序 episode；不含其后的训练 checkpoint 保存完成时间。初始评价另有首调用开销。

| task | 四模式采样峰值 MiB / GiB | 预热训练动作/s 范围 | 更新/s 范围 | 2 episode/点评价秒（含初始） | 完整单模式墙钟秒 | 每模式磁盘 GB（十进制） |
| --- | --- | --- | --- | --- | --- | --- |
| hopper_hop | 9529 / 9.31 | 24.65–24.74 | 6.16–6.19 | 13.32–15.63 | 218–226 | 2.132 |
| quadruped_run | 6640 / 6.48 | 24.47–24.62 | 6.12–6.16 | 13.83–16.31 | 223–225 | 2.144 |
| quadruped_walk | 8078 / 7.89 | 24.72–24.88 | 6.18–6.22 | 12.95–15.17 | 207–219 | 2.144 |
| reacher_hard | 8078 / 7.89 | 24.65–24.83 | 6.16–6.21 | 12.47–15.22 | 202–205 | 2.129 |

每模式三个 eval 快照约 1.539 GB，最新训练 checkpoint 约 0.513 GB；工程 replay 约 hopper 0.03654 GB、quadruped 两任务 0.0482–0.0486 GB、reacher 0.03336 GB。工程根目录总占用约 sv1 22.394 GB、sv2 8.576 GB、sv3 10.071 GB，包含补验和失败产物，全部保留。最终完成核查时本项目没有遗留训练进程。

按当前 v1 100万动作、21点×10 episode 作资源估算（指标口径另待 02）：

| task | 训练小时 `1e6/实测平均吞吐` | 210 episode 评价分钟 | 合计小时（不含长程额外开销） | replay 压缩磁盘线性外推 GB | 含 21 快照/最新 ckpt 的磁盘 GB |
| --- | --- | --- | --- | --- | --- |
| hopper_hop | 11.25 | 23.78 | 11.65 | 8.92 | 20.20 |
| quadruped_run | 11.30 | 24.96 | 11.72 | 11.86 | 23.14 |
| quadruped_walk | 11.20 | 22.87 | 11.58 | 11.75 | 23.04 |
| reacher_hard | 11.23 | 22.44 | 11.60 | 8.14 | 19.42 |

评价按去除初始点后每 episode 的中位耗时，再取四模式平均，乘 210；不能把 3 点工程评价作为正式评价预算。磁盘采用 `工程 replay 字节/4098×1e6 + 21×单快照 + 最新 checkpoint`，仅是压缩比不变的规划估算，不是实际长程上界。原始 replay 每位置仅 image+deter+stoch 已有 32768 bytes（64×64×3 uint8，4096 float32，32×32 float32），百万位置约 32.8 GB 原始数组，再加索引/其他字段/预取/序列和快照临时开销；主机内存与落盘压缩大小不能混用。

本次只训练每个环境约 256 动作，未跨越真实训练 episode 末尾、长期 replay chunk 周转或完整百万步；后期策略、活跃度、同步 IO 和并发资源竞争都可能改变速度。显存实测基于 prealloc=False/jsonl；原预设 prealloc=True/scope 没有被这一峰值验收。原 `run.steps=1e10` 仍存在配置中，但 protocol_v1 实际停止依据是 action_budget，不能把它读成执行了100亿步。

**调度建议（尚未执行）**：正式配置冻结后，每条运行独占一张空闲 A100，四组使用相同单卡拓扑/dtype/batch/envs。先沿本轮四路并发执行四任务正式 logging baseline，全部完成后工具才能冻结正式 c；同任务固定服务器，再以四路并发完成 Dt/constant/shuffle 三波。暂估约 48–60 小时、16 条运行约 190 GPU 小时，给每条预留 60 GB `/data`（16 条共 960 GB），主机 RAM 按至少 40–60 GiB/运行规划并监测，实际需求须按后期增长修正。sv1 两任务共 8 条预留 480 GB，sv2/sv3 各 4 条预留 240 GB；最新可用量约 903 GiB/2 TiB/1.2 TiB 可支持该规划，但启动前再检查。

sv3 后续优先选空闲 GPU1 或其他空闲卡，避开现有 GPU0 作业；sv1/2 当前空闲不代表未来空闲。扩展到 12 路比较并行在卡数上可行，但本轮最多仅验证 sv1 两路并行，没有验证六路/十二路吞吐和资源竞争，不能承诺仍为约12小时完成。正式 c 工具要求 baseline 完整完成，不能按本报告自行改为300000步提前冻结或从 baseline checkpoint 分叉。

## 产物完整性、限制与下一步

各服务器 EROOT 有 `resource-summary.json`、`analysis-audit.log`，全部主运行的 `audit.json`、任务级 `task_status.json`；补验有比较/中断审计文件。完整 config、命令、依赖、seed/设备、收据、评价、SHA、状态可追溯。服务器另存小证据包 `/data/Policy_Discrepancy/runs/engineering-size50m-20261009-1393548-evidence.tar.gz`，仅包含 JSON/日志/配置/审计源代码等小文件，排除 checkpoint、replay、eval 快照和 tmp。本机镜像在忽略的 `runs/engineering-size50m-20261009/`，不提交大产物；证据包哈希在完成核验后记入下表。

| 服务器 | 最终小证据包 SHA-256 |
| --- | --- |
| sv1 | `43844e7ea4bc987f0ebb1fdee1b36e42d97456691c8562cc3a7c7cd0dd930af0` |
| sv2 | `b07f9051ec3795bce86b92d600f452f6110d567d3feba55cc9cb2e1b181d2be6` |
| sv3 | `0c925e5b89de465bc2da27559d5a4684eb7da33b84d8b21742cad7ec691a33f9` |

当前状态不满足“全部工程验收通过”。立即可开展的是 03 的 CUDA report 修复、真实重跑差异定位/隔离补验方案，以及 02 的 return/score 指标审议；二者不自动向其他 chat 派发。本轮没有改方法代码，没有正式训练，没有正式匹配 c，没有跨 seed、控制充分性或抗干扰结论。

正式启动条件：03 给出确定修复 SHA及必要回归，并在 CUDA size50m 真实任务上完成报告/评价隔离补验，明确真实重跑可复现边界；02 将最终指标及汇总口径写入项目文档并获确认；正式完整工程配置（含 prealloc、logger 和设备安排）记录一致；04 重新核查空闲资源、干净同版本、完整新目录和产物记录；随后由用户授权正式运行。没有满足条件前，不把本次工程 c 或 return 混入正式结果。

## 可复制的首选交接 prompt（交给现有 03 chat，用户手动发送）

```text
你是 03 · 代码与测试。项目根目录 D:\program\PD\dmr3。
从 04 完成回复给出的最终完整交付 SHA 接手；HEAD 如更新，先核查差异，不回退。
本报告实际主测试 SHA 1393548fb6e46d28f92a834204f7b09495a3eab1，
评价/同配置重跑 SHA 63d36a111d247aa8cd05af173f74e43e4e35d2e1；算法相同。
必读 AGENTS.md、docs/CHATS.md、STATUS.md、SERVER_INTEGRATION_SIZE50M.md、
CODE_PROTOCOL_V1.md、EXPERIMENTS.md、METHOD_SPEC.md、RESEARCH.md。
本轮16个四任务size50m工程主运行已完成；不是正式比较，工程c不能复用。
已确认四个clean仅视觉DMC、size50m、seed0、alpha20、free_nats1、
dyn/rep1/0.1、ac_grads=false、reward_grad/repval_grad=true和四组。
正式匹配窗口、预算、评价网格仍按现行协议；return/score由02重审，
你不得自行选择新指标、改alpha/窗口/c/seed/模型或增加科研因素。

首选任务：修复CUDA report_on_batch的切片transfer guard异常，
并定位真实hopper同配置重跑从update2开始的分歧，落实RNG隔离补验。
server EROOT=/data/Policy_Discrepancy/runs/engineering-size50m-20261009-1393548。
sv1的dmc_hopper_hop/isolation有完整report异常命令/堆栈/配置；
logging、evaluation-isolation、same-config-repeat以及根目录的
reproducibility-comparison.json提供严格比较证据；初始294数组相同，
训练计数相同，但同配置本身不逐位一致，不得直接认定评价RNG泄漏。
同时核查report失败时零动作checkpoint已有成功收据的恢复拒绝边界。

允许本机修改必要工程代码/测试，通过Git同步到实际承担测试的服务器；
服务器仓库/data/Policy_Discrepancy/repo，Python环境
/data/Policy_Discrepancy/envs/dreamer，不能删除base-py311。
不在服务器直接改代码，不动已有作业，每次启动前检查GPU占用，
新验证目录保存完整配置/版本/seed/命令/环境/状态；原失败产物保留。
工程参数4098动作、间隔2049、窗口[2200,3600)、每点2episode；
报告复现开启report_every_actions=2049（原故障eval_eps=1），
固定真实size50m/CUDA/bfloat16/16环境/batch16x64/train_ratio256。
不得用debug模型或CPU-only fixture替代该真实CUDA缺陷的修复验收。
不能仅全局关闭transfer guard或放宽协议/事后容差来宣布通过；
如采用受限显式传输方案，记录设备/shape/数据流及对应测试。
冻结输入的同参数/batch/carry/seed对照用于区分工程随机流和环境差异；
记录诊断依据，不能自行把未定位的差异写成已解决。

验收：report可执行且只读，不消费训练/replay随机流；训练counter/收据/
参数隔离有可复核证据；真实重跑差异原因或明确技术边界与补验策略记录；
原17项相关测试及受影响完整agent回归按需完成；恢复/重开无静默覆盖或重复。
产物在服务器新工程目录，文档在CODE_PROTOCOL_V1/STATUS及必要修复报告。
交付最终完整SHA、实际测试SHA、差异和同步范围、通过/未通过项、
条件与完整可复制prompt，首选交回04补验；不启动正式百万步训练。
不自动创建chat、不向其他chat发送消息。02确认指标且工程闭环后，
再由用户授权04进行正式logging baseline、逐任务冻结正式c和其余三组。
```
