# 04 · 四任务正式 logging-only baseline 与 c/return 索引

日期：2026-10-09（Asia/Shanghai）。用户本轮明确授权使用 **sv2物理GPU0–3** 执行四任务正式baseline，生成逐任务正式c及所需原始return指标。本轮范围为四条logging-only，不包含Dt/constant/shuffle；不新增seed、校准训练或改变冻结协议。

## 当前状态：配置声明缺项，训练未启动

用户指定资源已明确；四任务科学参数已经确认，无需重复批准。`AGENTS.md`、`docs/CHATS.md`、STATUS当前规则及正式准备报告明确：04此前提出的prealloc=False/仅JSONL仍是未批准候选。当前尚无用户与03共同声明该项执行设置的记录。已集中提出两个完整方案：候选False/jsonl，或03预设默认True/jsonl+scope；等待声明，不以GPU分配或启动授权替代未确认配置。

两种方案均按同一03生产Flags解析为四份完整配置；唯一配置差异为 `jax.prealloc`、`logger.outputs`，科学参数、预算、窗口、评价和随机流不变。审阅位置：本机及sv2 repo `analysis/outputs/formal-baseline-choices-40fd48d/`，8份JSON/8份YAML及SHA-256清单；本机镜像16份文件哈希核对通过。该目录仅是声明备选，不是运行结果。

启动授权和GPU分配已经收到。设置明确后在现有04继续执行，**不再请求启动确认**；启动前重新核查指定卡占用、确定版本及完整有效环境。发生卡冲突按AGENTS等待用户调整，不换卡、增卡或共享。

## 资源分配与现场证据

初查UTC2026-10-09 13:38:12（北京时间21:38:12）：sv2/lyg0360，全部8卡0MiB，没有compute或pmon进程；因此指定0–3无本项目、其他项目或未知作业。/data可用2148586446848 bytes，RAM available1065507381248 bytes。其他服务器不属于本次分配，未使用、未同步或核查。

| sv2物理GPU | 任务 | 组/训练seed | UUID | EGL枚举索引 | 当前运行状态 |
| --- | --- | --- | --- | --- | --- |
| 0 | dmc_hopper_hop | logging / 0 | GPU-76344ce3-10a1-952d-68f5-7720a3cda667 | 0 | 未启动，配置声明缺项 |
| 1 | dmc_quadruped_run | logging / 0 | GPU-1e27fd5e-4231-926a-754f-e7d4c37d41ab | 1 | 未启动，配置声明缺项 |
| 2 | dmc_quadruped_walk | logging / 0 | GPU-24608f8d-a512-70b2-8578-45945b57bc35 | 2 | 未启动，配置声明缺项 |
| 3 | dmc_reacher_hard | logging / 0 | GPU-92630efa-be4e-7074-d29a-ccff0ad7f801 | 3 | 未启动，配置声明缺项 |

UTC13:42:38–13:43:06重新做单进程设备核查：GPU0/1/2/3的探针PID分别2774068/2774929/2775797/2776656。CUDA_VISIBLE_DEVICES使用对应UUID，CUDA_DEVICE_ORDER=PCI_BUS_ID，进程内仅cuda:0；MUJOCO_EGL_DEVICE_ID=0/1/2/3、MUJOCO_GL/PYOPENGL_PLATFORM=egl。各探针自身compute PID→UUID及pmon C+G只在对应物理卡，均通过；EGL枚举NV属性受CUDA mask影响，未将logical0误读为物理编号。仅小数组CUDA计算和简单XML模型64×64单帧render，**环境step/训练动作/agent更新均0**，不是baseline或size50m训练；探针自身prealloc=False不构成正式训练配置批准。

UTC13:47:52（北京时间21:47:52）再查：全部8卡0MiB、无compute/pmon进程，探针已释放四张指定卡；没有本轮仍运行的项目作业，没有训练目录。该状态是当时快照，不能代替未来启动核查。

## 版本、环境与证据目录

接手本机HEAD `40fd48dcf735ac2ad56c5ff86e86c9eeeb9b5985`，干净；sv2为干净 `08e2915f0a4f0d61947c7504992d81491c38fbb9`。差异仅AGENTS/CHATS/STATUS/正式准备报告四文档，不影响已验收运行路径。以Git bundle快进sv2至40fd48d后完成设备探针和CPU配置解析，未在服务器改源码。bundle SHA-256 `5c76f5c41d2655f4487ad0ac4c3a4f701d2414796a1c41bc6b69a70c13d502fd`，本机/服务器核对及verify通过。后续本轮状态文档的最终完整SHA另在交付回复记录；不能称为已启动正式训练。

pip check通过；freeze SHA-256 `cd50c7936690f9adbc5c35f99b3df2f7a977cef3222582f988979b11f65b6f22`，与依赖锁及前轮相同。Python3.11.16、JAX/jaxlib0.4.33、numpy1.26.4、dm-control1.0.48、MuJoCo3.15.0；dreamer继承base-py311，未升级或删除环境。

服务器证据 `P=/data/Policy_Discrepancy/runs/formal-baseline-preflight-20261009-40fd48d/` 保存bundle、初查原文、gpu*.json/log、外部探针/配置解析脚本及解析日志。探针由本机已测试文件复制到独立证据目录，源码SHA-256 `809cfc0b10840c6f0332e03178bb676faaaff89443aca1f207edb72e5b093759`；服务器仓库保持干净确定提交。P及备选配置已只读镜像至本机 `analysis/outputs/formal-baseline-preflight-20261009-40fd48d/` 和上述choices目录，不提交大产物，不改写旧证据。

## 已确认配置与后续完成标准

- 四任务clean视觉64×64 RGB，image=True/proprio=False、size50m、seed0、repeat1；batch16×64、replay_context1、16训练env/1串行评价env、train_ratio256、CUDA/bfloat16、policy/train logical0。完整其余值沿用03的defaults+m2_v1，不用debug。
- logging-only实际rep权重1；候选源w以alpha20计算。free_nats1、dyn/rep外层1/0.1、ac_grads=False、reward_grad/repval_grad=True。shuffle本次不启用，constant c尚未产生/不用工程c。
- 每任务1000000执行训练动作含预填充，排除reset和评价；初始化及每50000动作评价，含终点21点×10完整episode，不可变参数快照、采样动作和独立评价流。
- baseline全部完成/审计后，以成功更新开始动作步属于[100000,300000)且raw rep KL>1的有效活跃位置累计S/N，正常replay重采样保留，只处理同update ID重复，逐任务正式freeze。N=0/非法值停止，不填c1或改alpha；报告每任务c、来源、窗口、S/N、纳入更新、invalid及checksum。
- 全部四条baseline保存实际命令、完整config、SHA、seed流、环境、设备/主机、资源、退出/完成状态和21实际快照；不完整checkpoint拒绝续跑，保留失败，必要重跑新尝试ID，不覆盖或按return择优。
- 正式目录预案为 `/data/Policy_Discrepancy/runs/formal-m2-v1-seed0-20261009-plan01/<task>/logging`，当前根仍不存在；启动前确认新目录。已生成的两种备选配置使用此路径，不会创建runs。
- 生成新实际结果索引，再用 `analysis.return_metrics` 固定原始reward累计/点算术mean、梯形AUC/100万、tail80/85/90/95/100万五点、四任务baseline等权汇总。只有baseline来源，其他组指标和Dt差值应为空，不将16组缺失误当baseline已完整比较；逐任务baseline完整性单独验收。所有表/审计新建 `analysis/outputs/`，原始参数/replay留服务器。

当前交付是已完成资源/设备/版本/依赖核查和具体完整配置备选；正式训练、正式c及正式指标均未完成，未作科研效果结论。首选继续现有04，获得用户与03的执行设置声明后按本次已授权范围执行四baseline→c→指标，无需新chat或重复启动确认。代码修复不得改科学语义或放宽检查；不自动创建chat或发送消息。
