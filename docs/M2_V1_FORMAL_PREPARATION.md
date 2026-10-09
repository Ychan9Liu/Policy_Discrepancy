# 04 · M2 v1 正式配置与资源准备（未启动）

日期：2026-10-09，Asia/Shanghai。授权仅为准备完整配置、资源映射及16条新运行索引。**本轮没有训练agent、DMC训练交互、正式c或正式指标；未创建正式运行目录。** 资源探针只有小数组CUDA运算和简单XML模型的一帧64×64渲染，没有环境step或优化更新。既有工程证据继续见 `SERVER_RETURN_CUDA_AUDIT.md`，工程通过不表示M2有效。

**后续工作约定更新（2026-10-09）**：用户负责服务器/物理GPU分配，关键配置由用户与03共同声明；见 `AGENTS.md`、`docs/CHATS.md`。本报告及旧交接prompt的sv2 GPU0–3等保留为建议，不是用户分配；prealloc=False/jsonl等04提出的执行选择保留为候选，不因静态验收成为批准配置。已确认科学协议继续有效，原始配置/哈希/证据不改写；未来启动以用户实际分配与用户/03已声明配置为准。此次规则更新不新增正式训练授权。

## 版本、交付与复现

接手本机完整HEAD为 `12e2c7051af561394009fcec075d69ca62f00251`，工作区干净，等于交接版本，无回退。相对主要真实CUDA实测 `27261bf6df875939195879ab23e1dc161507fe5c`，交接版本仅有文档及工程supervisor的EGL设置/记录变化；相对绑定实测 `51ca42803e5c211d1be26cc164b9012f3388839d` 仅文档。

本轮本机新增 `scripts/prepare_m2_v1.py`：只解析配置和生成文件，不导入agent、runner或训练入口；唯一外部程序调用是只读git版本/工作区查询。首次解析验收为 `f1e4b29c54610f607010d31e15abeea170704237`；补充显式CUDA环境和绑定证据哈希后实测 `a3c699b16be8838e6b102788e086e377435e1cf0`；最后将准备审计声明限制为自身不导入训练代码，实测 `9b1c5bc9ee39c058c89ce46e49017753ec6ec64b`，完整解析及独立验收仍通过。设备探针执行时repo仍为12e2c705，探针本机编写、上传独立runs目录，未修改服务器源码。之后最终报告提交只改变文档，最终完整SHA由交付回复和最终索引的 `code_sha/execution_sha` 精确给出。**配置解析SHA不是新增size50m训练实测SHA。** agent、runner、模型配置预设、统计器、依赖和科学协议均未修改。

最终审阅包（本机和sv2 repo相同相对位置）：

- `analysis/outputs/formal-m2-v1-preparation-final/`：`planned-run-index.json`、16份完整flat配置（12份JSON+可执行配置YAML，4份constant pending JSON）、4份seed映射、配置差异审计、逐文件SHA-256清单。
- `analysis/outputs/formal-m2-v1-preparation-final-independent.json`：独立生产Flags/保存YAML往返、16条唯一索引、跨组/跨任务差异、正式目录和c不存在、840个评价环境seed核查。
- `analysis/outputs/formal-m2-v1-preparation-20261009-tools/`：三台只读核查原文、资源输入、原始探针/独立验收脚本、镜像探针证据和同步bundle；首版f1e4b29及a3c699b输出保留，不覆盖。
- sv2证据根 `P=/data/Policy_Discrepancy/runs/formal-preparation-20261009-12e2c70-device-checks/`：4份gpu*.json/log、resource-mapping.json、所有外部脚本/执行日志/同步包。小证据包及最终镜像完整性核对索引在本机 `analysis/outputs/formal-m2-v1-preparation-delivery.json`；大工程产物仍留原runs。

outputs均Git忽略；报告和可复现准备工具提交Git。下面命令**只生成审阅材料**，在干净确定版本的sv2 repo运行；output必须新目录。不会创建正式runs：

```sh
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu /data/Policy_Discrepancy/envs/dreamer/bin/python scripts/prepare_m2_v1.py \
  --resources /data/Policy_Discrepancy/runs/formal-preparation-20261009-12e2c70-device-checks/resource-mapping.json \
  --campaign formal-m2-v1-seed0-20261009-plan01 \
  --output analysis/outputs/NEW_FORMAL_PLAN
```

## 完整共同配置与唯一差异

从当前 `dreamerv3/configs.yaml` defaults+m2_v1经**生产elements.Flags**解析，不用debug覆盖。完整值在每份配置中，以下为关键核对索引：

| 项目 | 四组共同值 |
| --- | --- |
| 研究/任务 | M2；hopper_hop、quadruped_run、quadruped_walk、reacher_hard；clean，无干扰包装 |
| 观测 | 64×64 RGB，image=True、proprio=False、use_seed=True、repeat=1；camera=-1按已实现适配器映射：quadruped=2、hopper/reacher=0 |
| 模型/KL | size50m；RSSM deter4096/hidden512/classes32，encoder/decoder depth32、各units512；free_nats1、dyn/rep外层1/0.1 |
| 梯度/探针 | ac_grads=False、reward_grad=True、repval_grad=True、repval_loss=True；alpha20，训练seed0 |
| 数据/更新 | batch16×64、replay_context1、consec_train1；16训练env、1串行评价env；train_ratio256；replay5e6、chunk1024、online、uniform1/priority0/recency0 |
| 优化/精度 | 保留defaults的全部optimizer/heads/loss/normalization值；CUDA/bfloat16、jit=True；train/policy logical device=[0] |
| 正式预算/评价 | 1000000非reset训练动作（含预填充），排除评价；每50000动作，0至终点共21点；每点10完整episode，固定快照、采样动作 |
| 匹配/停止 | 成功更新开始动作步[100000,300000)，raw rep KL>1活跃位置累计S/N；正常replay重采样保留，仅同update ID重复处理；无活跃/非法值停止 |
| 初始化/恢复 | from_checkpoint为空，四组从头创建模型/optimizer/replay/env/counters；不分叉baseline，不续不完整checkpoint |
| 工程开关 | engineering_fixture=False，stop/actions trace/update trace/physics trace均0或False，run.debug/jax.debug=False；report_every_actions=0 |
| 共同执行选择 | prealloc=False、logger.outputs=[jsonl]；同前轮工程值，区别于defaults的True/[jsonl,scope]，四组一致，不改变科研参数 |

`run.steps=1e10`、`run.report_every=300`、`run.save_every=900`仍是defaults完整配置字段；protocol_v1终点由action_budget控制，报告由report_every_actions控制，主checkpoint在评价边界及完成时保存。不能把这些旧通用runner字段作为正式预算或评价节奏。priority.initial=Infinity为未启用优先采样的预设哨兵，JSON沿用manifest的Python序列化，YAML保留.inf；不等同于非法loss/诊断数值。

逐任务以logging为参照，dt/shuffle仅差 `agent.rep_probe.mode` 和独立logdir；constant另差本任务 `agent.rep_probe.c` 及 `run.frozen_c_file`。跨任务另差task、路径与对应资源/seed派生。移除这五个配置字段后16份配置完全相同。没有改变其他模型、梯度、优化器、replay、dtype或环境布局。

**四份constant配置除正式c数值外均确定，c=null明确为未绑定，launch_argv=null。** 它们是不可执行模板，不声称已产生完整可运行constant。对应来源固定为本任务新logging/c_frozen.json；不填工程c、c=1或其它数值。其余12份有完整解析配置和审阅命令，但授权状态仍为false，不自动运行。

各任务种子文件显式列出16训练env seed、replay seed及21×10评价episode seed和首次动作key；init key=[0,0]，训练policy/更新各自按counter以PCG64([0,counter])派生，评价用[0,EVAL命名空间,episode seed,counter]，shuffle PRNGKey(0)→fold_in(0x4454)→optimizer_step。跨组共用映射；各点更换episode初始情形。动态counter/key由实际运行记录，不能预先伪造完整训练随机轨迹；报告关闭，独立报告流规则保留源码。独立验收提取实际runner seed32函数核对840个评价环境seed及64个训练env seed，没有新增训练seed。

## 16条索引、设备与资源

新campaign为 `formal-m2-v1-seed0-20261009-plan01`，未来根 `F=/data/Policy_Discrepancy/runs/formal-m2-v1-seed0-20261009-plan01`。每任务下logging/dt/constant/shuffle四个独立目录，共16条；ID包含sv2/campaign/task/mode，seed0，status=not_started、authorized=false、正式AUC/tail=null。它是**计划索引**，不是已有结果索引。实际统计时镜像manifest/evaluations/config/execution/final/快照小metadata并另建可读directory索引，不直接用计划路径冒充本机输入。

| 任务（四组依次共用同一卡） | 服务器/物理GPU | CUDA_VISIBLE_DEVICES（UUID） | MUJOCO_EGL_DEVICE_ID | 探针PID |
| --- | --- | --- | --- | --- |
| dmc_hopper_hop | sv2/0 | GPU-76344ce3-10a1-952d-68f5-7720a3cda667 | 0 | 2763994 |
| dmc_quadruped_run | sv2/1 | GPU-1e27fd5e-4231-926a-754f-e7d4c37d41ab | 1 | 2764931 |
| dmc_quadruped_walk | sv2/2 | GPU-24608f8d-a512-70b2-8578-45945b57bc35 | 2 | 2765780 |
| dmc_reacher_hard | sv2/3 | GPU-92630efa-be4e-7074-d29a-ccff0ad7f801 | 3 | 2766630 |

现场探针UTC12:54:41–12:55:17，各次先要求目标UUID匹配且memory.used=0。EGL实际枚举8个device；CUDA mask后NV属性对可见卡返回logical0，其他槽属性查询失败，**未把此属性当物理编号**。独立依据nvidia-smi compute PID→UUID及pmon同PID C+G实际物理卡核对四个枚举索引，均与上表一致。单进程直接归属，无前轮worker父子树推断问题。每个探针训练动作/agent更新均0，进程退出后释放资源；该探针不替代已完成的size50m工程验收或百万步资源测试。

同时记录CUDA_DEVICE_ORDER=PCI_BUS_ID、JAX_PLATFORMS=cuda、XLA_PYTHON_CLIENT_PREALLOCATE=false、MUJOCO_GL/PYOPENGL_PLATFORM=egl、OMP/OPENBLAS/MKL threads=1、独立TMPDIR=run/tmp。JAX logical0由UUID mask映射，EGL使用全局枚举索引；以后不能仅写两个相同数字猜映射。启动前复查GPU UUID/枚举和所有compute/graphics PID，禁用遗留CPU/mock/MPS设置并记录完整有效环境。

UTC12:51:15只读资源快照（不是预约）：三台均A100-SXM4-80GB×8、driver570.158.01，pip check通过，freeze SHA-256 `cd50c7936690f9adbc5c35f99b3df2f7a977cef3222582f988979b11f65b6f22`。Python3.11.16/JAX+jaxlib0.4.33/numpy1.26.4/dm-control1.0.48/MuJoCo3.15.0与前轮相同，base-py311仍继承，未安装或删除依赖。

| host | 初查repo HEAD（工作区均干净） | 已有GPU作业 | /data可用bytes | root可用bytes | available RAM bytes |
| --- | --- | --- | --- | --- | --- |
| sv1/lyg2153 | 27261bf6df875939195879ab23e1dc161507fe5c | 全8卡，含GPU0多图形上下文 | 868964753408 | 106993647616 | 879401009152 |
| sv2/lyg0360 | 12e2c7051af561394009fcec075d69ca62f00251 | 8卡空闲 | 2148588937216 | 759923171328 | 1065258221568 |
| sv3/lyg0326 | d9be5574878d2c3f117cc6667be844f9a527eb84 | GPU0–2占用，3–7空闲 | 1250088497152 | 344510701568 | 914098073600 |

首选sv2四个任务槽，最多4训练并行：先四logging全预算，逐任务完整审计/正式freeze，再四dt→四constant→四shuffle，每卡同一时刻一运行。同任务四组共用同服务器/物理卡，减少硬件差异。sv1已有作业不使用；sv3保留备用，只有用户/04明确重新安排后，先验证其空闲卡EGL/UUID并同步同一确定提交、更新索引，再使用，不能后台自动替换。此次仅sv2执行探针/CPU准备并同步，sv1/sv3只读，不干扰已有作业。

容量建议沿用工程估算：预热24.24–24.64动作/s，单条训练100万约11.3–11.5小时；2episode评价11.32–15.16秒粗略×5再×21约20–27分钟，加编译/IO/长程波动暂按12–15小时/条、四波48–60小时量级。不是百万步实测或交付期限。短预算只绑定计算卡的旧峰值不能作为独占总显存；绑定hopper工程峰值8088MiB，规划每卡至少20GiB余量且独占整卡，不声称其它三任务完整峰值已测。

16条各预留60GB磁盘共960GB，sv2当前2.148TB可用；60GB/并行槽RAM共240GB，当前可用1.065TB。依据约0.513GB/参数快照，21评价快照加latest约11.3GB；100万64×64×3图像的未压缩量级12.3GB，另有reset/其他字段、replay压缩、ledger、日志及临时副本。60GB为余量规划，RAM为保守建议，未做百万步峰值测量。不得预删快照/replay来凑空间；每点评估记录盘/RAM/GPU增长并持续检查余量。四并行共享CPU/IO尚未长程压测，实际耗时可能更高。

## 启动依赖、恢复与验收边界

1. 用户另行明确授权正式运行；当前所有条目未授权。启动前本机/实际执行sv2干净且等于**最终索引execution_sha**，确认HEAD新增差异，不能悄悄改版本。只有实际使用服务器同步；服务端不改源码。
2. 再查全部作业/compute及graphics PID、目标UUID/EGL映射、依赖锁和实际freeze、CPU/RAM/盘余量；要求新run目录不存在。当前空闲不保证未来空闲，检查无法等同跨用户预约锁。
3. 按计划命令保存实际完整config、git SHA、seed映射、环境变量/版本、host/GPU UUID、命令、cwd、PID、起止/退出状态、资源采样到各新runs目录。当前工程launcher固定4098，**禁止拿它启动正式实验**；未来正式执行由04按已获授权的命令监督记录，不由本准备工具启动。
4. 四logging先完成100万动作、21不可变快照×10完整episode；从manifest/config/actual receipts重算窗口S/N、检查更新成功/顺序ID/非法值、final/matching一致，原始return/schema/快照及完成状态合格后正式freeze。不在30万窗口刚结束便启动下游。产物须有任务、正式来源SHA/config哈希、窗口/update ID列表、S/N/c/invalid0及checksum，基线与全部组同一确定版本。N=0/非法值停止该任务，记录故障交03，不填c1或改alpha。
5. c产生后只填本任务pending配置的c和来源文件，验证checksum、baseline ID/SHA/config、0<c<=1、task/seed0/alpha20/window/threshold1、engineering_fixture=False。本机保存新完整配置及哈希，重新用生产Flags解析，确认只差声明字段；再执行constant。dt/shuffle同样等待本任务baseline/c审计流程，从头训练。
6. 不完整checkpoint（包括0动作或终点评价不完整）拒绝续跑，保留原目录、故障/退出证据；需要重试时新ID/目录、相同冻结协议，披露全部失败，不按return筛选。完整运行仅同SHA/配置重开核对，不追加训练/评价。恢复限制不会因正式任务改变。

原始return及固定100万分母/指定五点/四任务等权继续冻结；score归一化暂缓且不阻塞。工程表正式指标仍空，未改旧统计/工程数据。真实渲染可能非逐位确定，不能要求独立新轨迹参数完全相等或事后放宽容差。长期episode/reset/replay周转、4并行争用、正式c数值和正式指标是未来验证项；无方法有效、跨seed稳定性或等效结论。

独立审阅已通过：16条唯一新ID/目录，12份生产CLI→完整配置→YAML往返相同，4份不可执行c模板；逐组及跨任务共同配置一致；正式目录/c不存在；840个评价环境seed和64个训练env seed吻合实际runner派生。已有output拒绝覆盖实测异常退出，原输出保留。哈希清单和最终提交重生成验收见最终交付索引，不能将这些静态检查写成正式训练完成。

首选下一任务由用户审阅包后**另行授权，继续现有04**，按baseline→逐任务正式c→其余三组执行。无需新chat或02重选指标；代码缺陷交现有03，协议变化须02及用户确认。当前准备范围已完成；未授权和未产生正式c均是明确启动依赖。

可复制交接prompt（最终完整SHA以交付回复为准，最终索引也固定该SHA）：

```text
你是现有04 | 运行与结果。项目根D:\program\PD\dmr3。
从交付最终完整SHA接手，核对HEAD差异并保留提交/修改，不回退。
必读根目录AGENTS.md、docs下CHATS.md、STATUS.md、EXPERIMENTS.md C/C.1、RESEARCH.md、
CODE_PROTOCOL_V1.md、CODE_SIZE50M_REPAIR.md、SERVER_INTEGRATION_SIZE50M.md、
SERVER_RETURN_CUDA_AUDIT.md、M2_V1_FORMAL_PREPARATION.md，以及analysis/README.md。
本轮仅完成配置/设备准备，无正式训练；size50m主CUDA实测27261bf6df875939195879ab23e1dc161507fe5c，
EGL绑定训练实测51ca42803e5c211d1be26cc164b9012f3388839d。
最终审阅包analysis/outputs/formal-m2-v1-preparation-final/及独立审计、formal-m2-v1-preparation-delivery.json；
16条全not_started/未授权，12份可解析配置、4份constant模板c=null，工程c禁止复用。
只有本条附带用户正式授权才启动，否则继续审阅/更新准备材料，不训练。
M2任务dmc_hopper_hop/dmc_quadruped_run/dmc_quadruped_walk/dmc_reacher_hard，clean仅视觉；
size50m、seed0、repeat1、alpha20、free_nats1、dyn/rep1/0.1、
ac_grads=false、reward_grad/repval_grad=true及四组不变；共同prealloc=false/jsonl。
预算100万非reset训练动作含预填充；0..100万每5万共21点，每点10完整episode，固定快照/采样动作/独立评价流。
先每任务完整logging baseline，从成功更新开始步[100000,300000)的raw rep KL>1活跃位置累计S/N，
正常replay重采样保留，仅同update ID重复处理；逐任务正式freeze后其它三组从头训练。
无活跃/非法值停止，不填c1、不改alpha、不增加seed或从baseline分叉；不完整checkpoint拒绝续跑，新ID保留失败。
原始接口reward完整episode累计（repeat已累计，不再乘），点均值→每seed梯形AUC/固定100万→任务seed均值→四任务等权；
tail严格80/85/90/95/100万，缺失不换分母/替点/重加权；归一化score暂缓且不阻塞。
首选sv2四任务GPU0/1/2/3（UUID/EGL映射见报告），同任务四组同卡；sv1忙，sv3备用未映射未同步。
每次启动前复查已有compute/graphics作业、实际UUID/EGL、依赖及磁盘/RAM；同时绑定并记录CUDA_VISIBLE_DEVICES与MUJOCO_EGL_DEVICE_ID。
repo=/data/Policy_Discrepancy/repo，env=/data/Policy_Discrepancy/envs/dreamer，不删除base-py311；
产物新根/data/Policy_Discrepancy/runs/formal-m2-v1-seed0-20261009-plan01（先核不存在），保存SHA/全config/seed/命令/资源/状态。
仅同步实际执行服务器，不在服务器改源码；工程launcher固定4098不能正式用，代码问题给03。
按报告启动依赖及验收收集全21快照/210episode、成功收据、正式c校验与完整终点；16条完成后新结果索引和固定口径统计。
表格/审计用analysis/outputs新目录，不提交大产物；不声称轨迹逐位确定或方法有效。
完成后更新报告/STATUS/EXPERIMENTS，给完整最终/实测SHA、差异/同步范围、证据、限制、下一步prompt。
不自动创建chat或发送消息。
```
