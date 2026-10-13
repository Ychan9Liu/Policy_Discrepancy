# DtLatch 100k 探索实验：DL-pilot-r2

## 授权与科学范围（2026-10-13）

用户在“直接跑一组实验”和“PD_1 能否对照”的讨论后明确要求：“跑100k吧，sv2，gpu4~7”。
本次落实此前单任务建议：`dmc_quadruped_walk`、训练seed0、N（DtLatch），在sv2物理GPU4运行；
GPU5–7不增加任务。100k为实际训练环境动作，含预填充，不含reset或评价。
这是新授权的探索性性能实验，使用独立入口；不是原DL-protocol-r1预验证通过后的18条试跑，
也不是正式M2-v1的预算修改。原筛选失败、native RGB身份失败及机制证据不足保留。

**已确认决定**：本次启动、100k及sv2/GPU4–7资源范围来自用户。
**采用的初值**：助手此前推荐α=20、ρ=0.5、H=1，随本次试跑冻结执行；
不称这些值获效果证据或用户逐项确认。任务/seed/历史对照沿用此前单任务建议。
**工作假设**：双信号选择性减轻rep压缩可能改善早期学习。
**未证实主张**：第二信号能识别关键观测、保护后验的因果机制、跨seed性能提升。

## 固定设置

| 项目 | 本次设置 |
| --- | --- |
| 方法 | 已实现DL-code-r1；`dt_latch.mode=dtlatch`，`rep_probe.mode=off` |
| 主要参数 | α20、ρ0.5；不搜索或按中途回报调整 |
| 第二信号 | 真实下一步奖励的TwoHot支持差 `e=max(exp(-d_q)-exp(-d_p),0)` |
| 保护权重 | `v=sg[1-ρ e (1-1/(1+α D))]`，有效邻接mask；不可用位置权重1 |
| 结构 | H1，q/p categorical mode、同h、记录的下一步实际动作；raw KL不参与gate |
| 模型/观测 | size50m、64×64 RGB、proprio=False、camera=-1→quadruped实际2 |
| 更新 | batch16×64/context1、16训练环境、train_ratio256、consec_train1 |
| 损失/梯度 | free nats1、dyn1、rep0.1；reward_grad/repval_grad=True、ac_grads=False |
| 预算/评价 | 100000训练动作、repeat1；0/50000/100000各10完整episode |
| 评价策略 | 原生sampled actor，独立eval RNG，原始非reset环境reward无折扣和 |
| 匹配/置换 | 仅N，不执行c/κ匹配或shuffle；κ/bins未绑定且不使用 |
| 计算/日志 | 一张A100-SXM4-80GB、BF16、prealloc=False、逻辑CUDA0、JSONL |
| 上限/恢复 | 完整入口14400秒（含初始化/JIT/评价/审计）；不自动延长、不续跑或拼接 |

主要自由方法参数有两个。H1是结构选择，KL tolerance=1e-5是数值检查。
PD_1的constant c属于原Dt匹配，不能代替DL的κ；100k不覆盖原[100k,300k)匹配窗。
因此本次不启动匹配强度或条件shuffle臂，不声称排除了通用正则强度解释。

## 历史对照与可比性

原PD_1实际训练SHA为 `48d71f1541721208322442f5a1d5977124f3e888`。

- B：`/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-seed0-20261009/dmc_quadruped_walk/logging-attempt01`
- D：`/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-dt-seed0-20261009/dmc_quadruped_walk/dt-attempt01`

绑定各条 `protocol_manifest.json`、`evaluations.jsonl`、`final_state.json` SHA256；
将0/50k/100k完整原始前缀保存为小型只读证据。新配置逐字段对照实际B配置，
只允许预算/入口/日志/输出路径与明确方法开关差异，其他模型/环境/更新/评价配置须一致。
两条旧0步checkpoint文件SHA256相同：
`6ad6dfae1a8b6a160b877eac9e663f12dbe232a0d0e4daae01911bf1d9661f94`；
完整294叶params/optimizer/normalization初始化digest为
`18199ec5fb45b13f18ec29ae6dcc438d3018807888ce9f2c5e6ee64b621a9267`。
新N从头初始化并核对此digest，不加载旧已训练checkpoint。

沿用原 `train_env`、`train_replay`、`eval_env` 命名空间，不能使用旧DL入口的 `DL_*`。
seed32为NUL连接字符串SHA256前4字节big-endian；replay seed3974293035。
评价seed按root0/task/`eval_env`/grid_index/episode0–9派生；sampled policy为
PCG64(root0,EVAL namespace,env_seed,call)。成功更新数应为50k时11993、100k时24493。
评价前后完整params、训练计数、replay sampler RNG/metrics须不变。

已知旧B/D即使初始checkpoint同字节，0点评价回报仍略有差异。
本次采用通常在线训练/评价对象，不承诺逐像素配对，不放宽旧native诊断的pixel identity门；
联报渲染/轨迹随机性、历史控制和代码SHA不同的局限。相同seed不等于轨迹逐字节相同。
只比较相同100k前缀的梯形积分/100000、100k终点与各点raw return；
不能与PD_1的1M AUC或80–100万tail直接比较。
单训练seed/10评价episode仅支持描述性方向，不支持训练seed层面显著性或方法有效结论。

## 工程、设备与验收

新入口 `scripts/dl_pilot_train.py`；原 `scripts/dl_train.py` 和
`embodied/run/protocol_v1.py` 的准入/科学检查不修改。工程在本机修改、测试、提交，
只同步sv2隔离干净checkout；sv2 canonical旧PD_1 checkout不改。
Python `/data/Policy_Discrepancy/envs/dreamer/bin/python`；Python3.11.16、JAX/JAXLIB0.4.33、
NumPy1.26.4、dm-control1.0.48、MuJoCo3.15.0；lock SHA256
`cd50c7936690f9adbc5c35f99b3df2f7a977cef3222582f988979b11f65b6f22`。

GPU4 UUID `GPU-7bf636ec-fa03-aa3f-7314-dd803c1658be`；`CUDA_VISIBLE_DEVICES`设此UUID，
`CUDA_DEVICE_ORDER=PCI_BUS_ID`、逻辑CUDA0、`MUJOCO_EGL_DEVICE_ID=4`、
`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`。启动前完整XML核查4–7；
实际零动作CUDA/EGL probe证明同卡，再fresh核查GPU4。父进程监督训练/环境子进程的
独立process group，定期按PID查询实际compute/graphics UUID；不共享或设备逃逸。
超时/故障只终止本次owned group，保留失败；本机修复、测试、提交后新attempt，不干扰未知作业。

验收：clean Linux确定SHA及CPU契约通过、初始化digest一致、100k动作/24493更新、
3点×10完整1000步episode、全更新收据连续且loss/grad有限、评价不改训练状态、
实际CUDA/EGL同卡、worker exit0、父审计与GPU释放证据齐全。
CPU double只验工程流程，不算真实任务或方法验收。完整命令/index/config/probe/sourceSHA
及日志存服务器独立计划/run目录；实际SHA/运行状态/资源时间更新到STATUS与DL_VALIDATION_LOG。

## 未决与下一步

优先由当前chat完成已授权这条100k、处理普通工程故障和审计，之后交“05｜实验分析”
比较PD_1前缀与有效释放统计。只读CPU分析可立即做；增加任务/训练seed/预算、
匹配强度/条件shuffle、改gate或阈值须另行明确，不能因阳性自动追加。
机制诊断、合法nuisance/错误依赖、独立留出及训练seed重复仍需证据；
本轮阴性也不能排除长预算或稀有条件下收益。
