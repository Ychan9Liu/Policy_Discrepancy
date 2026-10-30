# 03合并窗口 · Hopper seed1/2/3 baseline / Dt 正式补测

**2026-10-30终点核查（最新）**：六条均accepted、100万动作/249493更新/21×10，10月22日02:30已有完整completion与正式c/原始return表；本次表hash与线上六份精确摘要读回一致，W&B均finished，无本批训练进程。全批和版本/资源/证据见 `RUNS_STATUS_20261030.md`；下面“正在训练/尚未完成”是10月21日启动时记录，不代表当前状态。

日期：2026-10-21，Asia/Shanghai。**用户已明确“启动实验”“挂上wandb”；6条已启动，正在训练，尚未完成全预算验收。** 由当前03/04合并窗口继续实现、执行与审计，不再交原04。准备报告中“未启动/待授权”保留为启动前历史。

## 实际资源与启动状态

sv3/lyg0326，用户指定物理GPU1/3/4/5/6/7，每卡一条、不共享。启动监督PID **706010**，2026-10-21 **15:25:39** 启动；训练进程如下。15:36:30只读启动审计：6条存活、W&B均running，102个本项目计算/图形PID按训练父进程归属，均在本条指定物理卡；GPU0/2仍有4个未知归属作业，未使用、未干扰。资源状态仅代表该快照，不当作未来空闲保证。

| seed | 组 | sv3物理GPU | 训练PID | 已成功更新（15:36快照） | 最近更新开始动作步 | W&B |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | logging baseline | 1 | 706087 | 3070 | 14308 | [在线](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-hopper-seeds123-baseline-dt-20261021-hopper-seed1-logging-01) |
| 1 | Dt | 3 | 706096 | 3151 | 14632 | [在线](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-hopper-seeds123-baseline-dt-20261021-hopper-seed1-dt-01) |
| 2 | logging baseline | 4 | 706106 | 3104 | 14444 | [在线](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-hopper-seeds123-baseline-dt-20261021-hopper-seed2-logging-01) |
| 2 | Dt | 5 | 706123 | 3104 | 14444 | [在线](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-hopper-seeds123-baseline-dt-20261021-hopper-seed2-dt-01) |
| 3 | logging baseline | 6 | 706492 | 3154 | 14644 | [在线](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-hopper-seeds123-baseline-dt-20261021-hopper-seed3-logging-01) |
| 3 | Dt | 7 | 706850 | 3093 | 14400 | [在线](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-hopper-seeds123-baseline-dt-20261021-hopper-seed3-dt-01) |

动作列是读取收据时的最近**更新开始步**，不是读取瞬间的精确当前动作计数。表中更新数不能当作完整预算或收敛证据。各条0步评价均恰好10完整episode，首个成功更新开始动作2032，已写收据ID连续、invalid_count全0。尚未到完整匹配/训练/评价终点，三份正式c及最终AUC/tail仍待产生。

## 共同科学配置与设备

沿用已声明配置，详见 `docs/HOPPER_SEEDS123_PREPARATION.md`、实际bound plan及每条config.yaml/manifest：仅dmc_hopper_hop，seed1/2/3，各logging baseline与Dt；clean、64×64 RGB、image=True/proprio=False、size50m，CUDA/bfloat16，alpha20、free_nats1、dyn/rep=1/.1、ac_grads=False、reward_grad/repval_grad=True、repval_loss=True。batch16×64、replay_context1、16训练env、train_ratio256；从头训练，不继承seed0或baseline参数/回放。

每条100万训练动作含预填充、排除reset/评价、repeat1；0..100万每5万一次，共21点，每点10完整episode，固定快照、采样动作、独立评价随机流；原始return AUC/固定百万分母与80/85/90/95/100万尾部五点保持。baseline候选w同alpha20，实际权重1；Dt按原gate。每seed自身完整baseline按开始动作 `[100000,300000)` 的raw rep KL>1有效位置累计S/N，冻结c仅供本批审计；本批无constant/shuffle，Dt不等待c。原四任务seed0的16条数据不改、不混入新三seed主表。

共同prealloc=False、JSONL+W&B **ychan9liu-tsinghua-university/PD_1**、online、无scope、resume=never，每条独立W&B ID和新目录。每条实际config与准备配置/manifest/W&B full config逐字段一致。三个同seed对的294个初始参数/优化器/归一化数组哈希完全一致，三个seed组合哈希不同；这不是输入图像或后续真实轨迹逐位相同的证明。

设备实际验证：先在每进程CUDA UUID mask下无context枚举EGL_CUDA_DEVICE_NV，唯一可见logical CUDA0对应的EGL索引分别为1/3/4/5/6/7；再在本卡做小CUDA计算及非任务模型的单帧渲染，按实际PID→计算UUID与pmon图形物理卡双重核对。六个探针全部通过，训练动作/agent更新均0，退出释放探针后再次查无作业才启动正式进程。正式102个进程又核对实际归属，没有只按相同数字猜测映射。

UUID沿用准备报告的六张指定卡；CUDA_DEVICE_ORDER=PCI_BUS_ID、每进程仅logical CUDA0、MUJOCO_GL/PYOPENGL_PLATFORM=egl，MUJOCO_EGL_DEVICE_ID为探针实际值。监督进程仅CPU，子训练环境清除继承的CPU/XLA mock及MPS开关，显式绑定其CUDA/EGL卡。完整环境允许列在bound-plan及各execution-initial，不记录密钥。

## 实际提交、工程测试与已处理问题

- 本轮初始本机HEAD `97fe633335bec6eca97dc067f82181d1cf439ce4`；原RESEARCH/STATUS及docs/proposal候选修改保留，未混入提交。
- **6条实际训练、监督、63项CPU测试、CUDA/EGL探针和生产W&B验证的SHA均为 `743089f40662b031a5cf2b6bf988decb52ba0b28`。** 相比准备测试0b764bd，训练算法/runner/配置/梯度/RNG代码无变化，仅增加监督/审计脚本、测试及既有文档。没有服务器直接改源码。
- 新 `scripts/run_hopper_seeds123.py`：固定六条sv3范围校验、全部类型GPU进程/XML+pmon冲突拒绝、实际CUDA/EGL探针、生产logger验证、授权绑定、独立nohup监督、资源采样、逐运行收据/终点评价/c审计、六条结果统计与W&B exact JSON/hash收尾；不允许覆盖/续跑已有实验根。旧sv2/四任务/seed0监督器未套用。
- 新4项CPU边界测试覆盖GPU/seed/mode篡改拒绝、C/G/pmon占用拒绝、空闲1MiB不等于已有作业、子训练不继承CPU/MPS环境、未授权计划不启动进程；与原59项合计 **63项通过**。未放宽数值容差，没有改关键参数来宣布通过。
- SSH曾在握手前返回 `Exceeded MaxStartups`。用户确认sv3入口仍为原地址；成功后复用同一SSH会话部署/核查，未改地址、服务器或GPU。此问题发生在启动前，没有训练失败或重跑。
- W&B实际读取sv3已有凭据认证，未输出/拷贝密钥；生产make_logger在独立engineering run写入并回读标量和完整seed1配置，远端finished。该run及六个无动作探针不是训练重复，不纳入正式预算/指标。
- 启动审计attempt02直接读取 `protocol/eval_return` 时发现在线summary尚无该key：现有elements.WandBOutput调用 `wandb.log(metrics,step=...)`，显式step的首条历史在下一记录step或结束前暂缓存。没有改训练/日志节奏或重跑；attempt03给六个运行另写 `startup/exact_values_json`、SHA256及初始化评价的展示字段，逐个回读精确字符串/hash成功。它们是既有0点评价的审计注释，**不是新增评价或改写正式曲线**。初始化数值字段是展示镜像，精确JSON/原始evaluation记录是权威值。后续正式指标按原动作边界写入，终点监督按已有exact JSON/hash机制发布。
- TensorFlow profiler trace缺失提示仍为非致命日志，已有各3000余成功更新；不为此安装TensorFlow或改模型/训练配置。

本轮最终完整SHA见DELIVERY/最终回复；实际训练源码固定743089f。启动后的文档提交只在本机提交并向sv3同步Git对象/报告，不快进正在使用的训练checkout，不影响在跑源码。只同步sv3，未推送GitHub或同步sv1/sv2；最终SHA与训练/测试SHA的差异须在交付中明确。

## 产物、监督与后续验收

实际实验根：
`/data/Policy_Discrepancy/runs/formal-m2-hopper-seeds123-baseline-dt-20261021/`
下接 `seed{1,2,3}/{logging,dt}-attempt01/`，存config、manifest、checkpoint/replay、评价、更新收据和本地JSONL。根目录 `actual-RUN_INDEX.json`、`resources.jsonl`、`prelaunch.json`、`bound-plan.json` 是实际运行记录；准备计划不是真实结果。

启动证据根P：`/data/Policy_Discrepancy/runs/hopper-seeds123-start-20261021-743089f/`。
含startup-code.bundle、cpu-tests、logger-check、六个GPU绑定/探针日志、bound-plan、supervisor.log/PID、startup-audit-attempt01/03、attempt02读取失败说明、initial-parameter-audit及后续交付记录。本机小镜像/命令/哈希为 `analysis/outputs/hopper-seeds123-start-20261021/`，大checkpoint/replay留服务器。

监督启动命令（仅记录已执行，不重复运行）：

```sh
cd /data/Policy_Discrepancy/repo
nohup env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/Policy_Discrepancy/envs/dreamer/bin/python -u -m scripts.run_hopper_seeds123 run --plan /data/Policy_Discrepancy/runs/hopper-seeds123-start-20261021-743089f/bound-plan.json > /data/Policy_Discrepancy/runs/hopper-seeds123-start-20261021-743089f/supervisor.log 2>&1 < /dev/null &
```

监督每30秒保存资源和实际index，逐条退出后审计收据/manifest/config/final/evaluation及本seed c；六条均完成后使用 `analysis.return_metrics --profile hopper-seeds123 --require-complete`，写新 `final-audit-attempt01`，再把每seed指标/c/配对差值以精确JSON/hash发布W&B，写completion及supervisor-final。失败会保留状态和错误，不把未完成记accepted。若只收尾失败，在明确源码/失败原因后用新audit尝试，不重跑训练或覆盖旧审计。

**当前只完成启动验收**：6条实际size50m CUDA成功更新、0点各10完整episode、正确版本/配置/W&B/设备及初始化核查。尚未完成100万动作、后续20点评价、三份c、终点checkpoint/完整收据/最终raw-return统计。六卡仍有本批作业，尚无本批训练卡释放；探针和logger检查已结束。下一步在本合并窗口监测到训练结束并完成上述验收，之后交05分析；无需新chat或重复启动授权。普通工程故障在同一资源/协议内修复验证、保留失败后新attempt，不支持env/replay无缝续跑，不按return选择重跑。不得加入新组/seed/预算/奖励或DtLatch，不把本批重复性数据直接当作gate与输入微扰的因果裁决或M2方法有效性结论。
