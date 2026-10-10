# 04 · PD_1 正式 constant / global shuffle

2026-10-10，Asia/Shanghai。用户要求运行剩余两组，沿用已分配sv2八张物理卡：constant GPU0–3、global shuffle GPU4–7，顺序hopper_hop、quadruped_run、quadruped_walk、reacher_hard。**当前运行中，尚未完成全程验收。** M2 v1科学协议不变；RESEARCH中DtLatch是未批准候选，不进入这16条实验。

## baseline / Dt终点与正式c

独立核查八条均完成1000000训练动作、249493成功更新、21点评价、210000评价动作；W&B均finished。原统计器核验21点×10完整episode、快照及原始return，八条AUC/tail完整。前三baseline的原exit code无法回收，仍null；其余五条exit0。训练源码均`48d71f1541721208322442f5a1d5977124f3e888`。

| 任务 | 正式c（完整精度） | S | N | baseline AUC / tail | Dt AUC / tail |
| --- | --- | --- | --- | --- | --- |
| hopper_hop | 0.8169140208772256 | 24831065.62310791 | 30396180 | 253.82870348475976 / 316.7513217655165 | 0.8596794502713874 / 1.013226058116634 |
| quadruped_run | 0.9287514026396865 | 47192358.03814697 | 50812691 | 467.4030915037747 / 810.9526664531352 | 600.1557759949192 / 846.082722903972 |
| quadruped_walk | 0.9492462655872804 | 47199733.726623535 | 49723381 | 638.158807614521 / 937.7926031842344 | 796.2536301200965 / 944.8860246267989 |
| reacher_hard | 0.9187557996879948 | 26834834.727371216 | 29207799 | 734.0725 / 926.82 | 730.2425 / 957.6600000000001 |

正式c来自本任务完整logging baseline：成功更新开始步`[100000,300000)`内有效活跃raw rep KL>1位置，S为alpha20候选w总和、N为位置数。正常replay重采样保留，只处理同update ID幂等重复。收据ID连续、invalid0，S/N精确重算，原freeze_c对既有artifact幂等复核成功，artifact/receipt/config哈希一致，engineering_fixture=false。每任务来源为B下`<task>/logging-attempt01/c_frozen.json`，新plan/index formal_c_source字段保存路径、SHA、c、S/N、receipt/source版本。

这些是单训练seed描述性结果。Dt hopper return很低，保留原值，未因此改alpha/梯度或择优重跑；科学解释待四组完整后由05审查，不自动判为代码故障，不写方法有效。

## 资源与启动

启动前11:47:37北京时间：sv2/lyg0360八卡均0MiB、XML全类型process为空、pmon无作业，无未知作业；`/data`可用1973805064192 bytes。constant/shuffle监督PID3147478/3147479，启动每张卡再检查，不使用其他服务器。

| 物理GPU | 任务 | 组 | 训练PID | CUDA_VISIBLE_DEVICES |
| --- | --- | --- | --- | --- |
| 0 | dmc_hopper_hop | constant | 3147915 | GPU-76344ce3-10a1-952d-68f5-7720a3cda667 |
| 1 | dmc_quadruped_run | constant | 3147945 | GPU-1e27fd5e-4231-926a-754f-e7d4c37d41ab |
| 2 | dmc_quadruped_walk | constant | 3148043 | GPU-24608f8d-a512-70b2-8578-45945b57bc35 |
| 3 | dmc_reacher_hard | constant | 3149213 | GPU-92630efa-be4e-7074-d29a-ccff0ad7f801 |
| 4 | dmc_hopper_hop | global shuffle | 3147912 | GPU-7bf636ec-fa03-aa3f-7314-dd803c1658be |
| 5 | dmc_quadruped_run | global shuffle | 3147942 | GPU-ac733835-08ee-7a17-a2fa-1e49d62d385b |
| 6 | dmc_quadruped_walk | global shuffle | 3148039 | GPU-548be1c8-86f0-b7a2-de92-5c455b12221b |
| 7 | dmc_reacher_hard | global shuffle | 3149215 | GPU-4cbf5b69-adb3-5210-2396-ca5d4e400f26 |

每进程CUDA_DEVICE_ORDER=PCI_BUS_ID、单logical CUDA0，MUJOCO_EGL_DEVICE_ID为表中物理编号，MUJOCO_GL/PYOPENGL_PLATFORM=egl。完整命令/允许列环境/UUID在execution-initial.json；真实计算/图形PID与子环境祖先关系保存startup audit。11:50:09快照八条存活、各卡2505MiB初始化，不能据此声称成功更新；现场首点评价/成功更新及设备结果另补充。旧baseline/Dt全部释放卡，新八条使用0–7，末次资源状态须重查。

**现场验收补充**：11:57:19新八条均W&B running，manifest与计划精确一致，已有1906–1998条连续成功收据、invalid0；136个计算/图形主/子PID的实际UUID与指定物理卡相符，无未知或错卡作业。每条0点评价恰好10个完整episode，episode ID 0..9、raw return有限、固定快照记录齐备。quadruped_run constant的W&B config数值显示c末位同样舍入，但argv/manifest/artifact精确一致，其他线上配置字段相等；该显示差异完整留痕，不改变训练或要求重跑。

12:00:54运行质量快照：八条最新成功update ID3335–3436、更新开始动作15372–15776，全部进程存活；12:00:31–32显存采样各8003MiB、磁盘可用约1.965TB。此时日志只有0点评价，尚未到首个5万点评价/训练指标flush；不能声称已经直接观察全部loss/诊断日志。成功更新收据的已审查runner路径先核对opt/loss与grad_norm有限后才写入，属于源码路径支持；其他诊断的直接数值检查仍待下一网格日志。资源峰值为采样值，最终完整资源审计待终点。

前八条实际总耗时10.94–11.01小时（含初始化、训练、21次评价）；据相同来源/规模/八卡并行，当前两组暂估10月10日22:45–23:30训练结束，之后CPU审计/上传。这是历史耗时外推，constant/shuffle成本、磁盘和共享CPU波动可影响实际时刻，不是保证或已完成状态。

## 完整配置、路径和版本

共同defaults+m2_v1：四任务clean仅视觉image=True/proprio=False、size50m、seed0、repeat1、alpha20、free_nats1、dyn/rep=1/0.1、ac_grads=False、reward_grad/repval_grad=True、repval_loss=True；batch16×64、replay_context1、16训练env/1评价env、train_ratio256、CUDA/bfloat16单逻辑卡。prealloc=False、JSONL+W&B project PD_1、无scope。constant用表中精确c；shuffle为冻结全局batch×time置换，独立PRNGKey(0)/fold_in(0x4454)/optimizer step流。

100万执行训练动作含预填充，排除reset/评价；0..100万间隔5万、21点×10完整episode、固定参数快照、采样动作、独立评价流。四组从头初始化model/optimizer/replay/env，不加载baseline checkpoint。按C/C.1累计原始接口reward、不再乘repeat，点算术均值、实际动作步梯形AUC/100万、末段80/85/90/95/100万五点、四任务等权。缺失/冲突标不完整，不替点/换分母/重加权；归一化score暂缓不阻塞。

完整Flags配置逐字段比较已完成baseline：shuffle仅mode/logdir，constant仅mode/logdir/c/frozen_c_file不同。c解析后与artifact精确相等，启动前再验证SHA，其他字段一致。

- B：`/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-seed0-20261009/`；D：`/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-dt-seed0-20261009/`。
- C：`/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-constant-seed0-20261010/<task>/constant-attempt01`。
- S：`/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-shuffle-seed0-20261010/<task>/shuffle-attempt01`。
- P：`/data/Policy_Discrepancy/runs/formal-remaining-preflight-20261010/`，本轮确定版本外部脚本/测试/核查/小证据。
- sv2 repo完整JSON/YAML/plan/index：`analysis/outputs/formal-constant-pd1-20261010-48d71f1/`、`formal-shuffle-pd1-20261010-48d71f1/`。实际状态索引C/S根RUN_INDEX.json，完整命令/环境/SHA在各execution-initial.json；config.yaml、manifest、receipts、evaluations、checkpoint、JSONL在各logdir。
- 本机小证据镜像`analysis/outputs/formal-constant-shuffle-pd1-20261010/`，大产物留sv2 runs，不提交。
- [W&B PD_1](https://wandb.ai/ychan9liu-tsinghua-university/PD_1)，entity `ychan9liu-tsinghua-university`。新run ID `formal-m2-v1-pd1-<constant或shuffle>-seed0-20261010-<task>-<mode>-01`；完整URL/config在index，不输出密钥。

全部16条训练源码保持干净`48d71f1541721208322442f5a1d5977124f3e888`。本轮外部监督/审计实际测试SHA `a8b62436827721c49a77af55663dbb5225e1ed39`，相对48仅操作脚本/测试/文档变更，agent/runner/模型preset/统计源/协议无差异。确定Git bundle SHA256 `2523beb4916116313ac4b6951485217effccdd06b8fb313672c7da02d3f411d1`，sv2 verify/fetch后从Git对象导出P，不修改服务器checkout。sv2 CPU9项边界测试及pip check通过；其他服务器未同步/使用，base-py311保留。最终文档版本与测试版本区别见交付。

最终CPU收尾脚本测试版本`d239c3fb7e5b77ddf205fe994c46c2e5a6acce6a`相比a8只增加完整配置精确JSON/hash字段，9项边界测试再次通过；新bundle SHA256 `4f097820ceb1b212287bbab19ee8ee8dce73c6d96067fe5c108dc6130d4f84c6`已verify/fetch并从Git对象导出P/audit_formal_pd1_with_config.py。运行中的监督仍为a8，不替换源文件；d239不改变任何训练路径。最终文档提交之后只有本轮文档差异，以交付完整SHA及P的交付收据为准。

## W&B收尾故障与修复

旧baseline收尾因summary浮点末位回读不完全相等失败；原B/baseline-completion-audit-failed.json、D/comparison-postprocess-failure.json、旧表格/日志保留。定位quadruped_run c的权威值`0.9287514026396865`，W&B数值回读`0.9287514026396864`；后两任务未发布。训练/c/原始return完整，不需要训练重跑。

新CPU审计`audit_formal_pd1.py`沿用原统计器、freeze_c和所有固定科学检查/容差。W&B保存数值显示镜像及规范化精确JSON字符串`formal/exact_values_json`/SHA256，回读逐字节相等、hash严格核对；非浮点字段仍相等，浮点显示必须存在且有限，所有显示差异保留证据。正式c始终取artifact，统计取权威表；不修改源数字、不放宽科学容差。测试覆盖占用拒绝、未声明梯度/c差异拒绝、精确payload损坏/缺字段拒绝、末位显示差异留痕。

旧八条修复审计新目录：sv2 repo `analysis/outputs/formal-baseline-dt-pd1-20261010-audit-attempt02/`，CPU PID3147480，原始独立复核P/independent-eight-audit.json。新八条完整终点、随机流来源、loss/诊断、实际快照/21点评价、资源及W&B最终完成仍待验收。本机其他chat已有未提交RESEARCH/STATUS候选归档保留，不混入本轮操作提交或运行协议。

**旧八条修复已完成**：11:50:17 attempt02/completion.json核验8条完整指标与精确W&B摘要，verified_run_count=8、full_four_groups_complete=false，三个数值显示差异留痕，科学源值/容差未改。表格SHA256 `25e4806cdec9d1113c724a4596aeca5f2e74fdbfb3442607d84ea24fc42175d7`。原失败仍保留，不需要训练重跑。

**16条依赖收尾已部署**：CPU PID3164661，P/all-sixteen-postprocess-launch.json保存完整命令；确认waiting.json存在且无failure，等待C/S的supervisor-final，随后新目录sv2 repo `analysis/outputs/formal-four-groups-pd1-20261010-final-attempt01/`汇总16条并用原CLI `--require-complete`、逐条终点/收据/c/精确配置与W&B摘要验证。输出只在完整时产生completion；失败保留failure及中间审计，不修改训练/重启/科学参数。这是服务器依赖后处理，不是Codex周期任务或新chat。P/all-sixteen-RUN_INDEX.json是本轮运行快照，正式终点索引由后处理重新读各根实际index构建。

小证据包P/delivery-evidence.tar.gz（820857 bytes，155文件），SHA256 `e6a10948ba992313a94425f4bf1d041300c57199030f47b6af3978326b694a9a`，本机镜像解包逐155项哈希校验通过。含旧终点/新manifest、完整配置/index/命令、正式c、启动/设备/首点评价、attempt02六表/原始审计、等待后处理及失败记录；不含agent/replay大产物或密钥。设备/依赖/资源证据分别为P/eight-startup-audit.json、initial-evaluation-audit.json、delivery-quality.json、before-launch.json、operations-tests*.log、pip-check.log、delivery-integrity.json。本机对应忽略outputs/evidence目录与mirror-validation.json。

## 下一步及可复制prompt

继续现有04监测已授权八条。结束后在新的analysis/outputs汇总16条index，以原统计器`--require-complete`核验四组/21点×10episode/快照/正式c及三种Dt差值，发布PD_1精确摘要并核查释放卡。不能无缝恢复不完整checkpoint/replay/env；失败保留旧目录，必要工程修复后同有效分配/冻结配置内新attempt从头，不择优。完整审计后由用户交现有05做单seed描述性分析，不自动创建chat/消息。

```text
继续现有04，项目D:\program\PD\dmr3；最终交接完整SHA见交付回复，16条训练源码48d71f1541721208322442f5a1d5977124f3e888，操作测试a8b62436827721c49a77af55663dbb5225e1ed39。核Git/修改并保留，不回退。
必读AGENTS.md、docs/CHATS.md、docs/STATUS.md、docs/EXPERIMENTS.md C/C.1、docs/CODE_PROTOCOL_V1.md、docs/FORMAL_CONSTANT_SHUFFLE_PD1.md、docs/FORMAL_BASELINE_DT_PD1.md、analysis/README.md。
baseline/Dt已结束，正式c见新报告；只用正式artifact，不用工程c/W&B数值镜像。constant sv2物理GPU0–3、global shuffle4–7已授权从头运行，顺序hopper/run/walk/reacher，prealloc=False、JSONL+W&B PD_1、无scope。M2/size50m/seed0/repeat1/alpha20/梯度/预算/评价/统计冻结不变，DtLatch未批准不得加入。
继续监测、必要工程修复和16条终点审计，不重复申请已授权启动；新启动先检查占用/CUDA/EGL，不换服务器/卡/增卡/共享。服务器repo=/data/Policy_Discrepancy/repo保持48且干净，确定操作脚本在runs/formal-remaining-preflight-20261010，环境dreamer/base-py311不得删除。
核查旧失败留痕、attempt02精确W&B传输、16条预算/收据/21点×10episode/快照/c及三种Dt差值，analysis/outputs新目录审计，完整时require-complete退出0。大产物留runs。报告资源/卡释放/限制，单seed不写方法有效；工程故障保留新attempt，不改协议/容差。
更新报告/STATUS，给最终/测试SHA、差异/同步范围、验收证据、未决项、下一步及可复制prompt，不自动创建chat/发送消息。
```
