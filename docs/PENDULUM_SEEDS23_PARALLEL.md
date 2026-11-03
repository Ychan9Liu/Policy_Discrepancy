# Pendulum seed2/3：PPD-KL改为GPU5/6并行

2026-11-04，用户明确补充“刚分配的其他的空闲gpu也可以用，可以并行的测试可以尝试”。本chat据此在原sv3 GPU1/4/5/6/7资源范围内，将尚未启动的两条PPD-KL安排到物理GPU5/6，与既有两baseline并行；GPU7保留。不增加任务、seed、组或科学参数，不重复seed1/hopper。旧同卡串行安排仅对尚未启动的Dt被本记录替代；[原启动报告](PENDULUM_SEEDS23_RUNS.md)和证据保留。

## 资源、配置与版本边界

|sv3物理GPU|任务/seed/组|本次动作|
|---:|---|---|
|1|pendulum_swingup seed2 baseline|保留PID3690581及原训练目录|
|4|pendulum_swingup seed3 baseline|保留PID3690588及原训练目录|
|5|pendulum_swingup seed2 PPD-KL / dt|通过新卡验收后启动|
|6|pendulum_swingup seed3 PPD-KL / dt|通过新卡验收后启动|
|7|—|保留，不安排额外实验|

00:59:31资源快照：GPU1/4各17个本项目C/G进程，5/6/7无进程；/data余336555589632字节。GPU5 UUID `GPU-79704918-e096-92b4-252c-e3244497a8b3`；GPU6 UUID `GPU-70db6c66-9917-e0e2-44db-0fc80fcf41b9`。实际CUDA/EGL映射尚待新卡探针；启动前重新核查，不把快照当预约。

全部四条agent继续执行同一个已验收的训练SHA `c5f86ca1a16648f57db3c3abfa2584ac560495ca`，路径`/data/Policy_Discrepancy/checkouts/pendulum-seeds23-c5f86ca1a166`，保持干净且不移动。新提交只增加控制器/测试及文档，部署到另一个独立checkout；控制器SHA和训练SHA分别记录。新控制器启动子进程时cwd明确指向原训练checkout，不把控制器版本冒充训练版本。

原配置不变：clean、64×64 RGB视觉、size50m、repeat1、batch16×64/context1/train_ratio256；seed2/3；baseline=logging实际rep权重1，PPD-KL=dt/alpha20，free nats1，dyn/rep=1/.1，ac_grads=false、reward_grad=true、repval_grad=true、repval_loss=true；原optimizer/RNG/奖励不变。每条百万训练动作含prefill排除reset/评价，0..100万每5万固定快照/10完整episode/原采样动作与独立评价随机流，CUDA/bfloat16/prealloc=False/JSONL+online PD_1。

PPD-KL的权重来自自身批次Dt，不读取baseline c，提前并行不改变训练方法。各baseline仍在百万动作终点按自身`[100000,300000)`成功更新、raw rep KL>1位置S/N完整冻结c；终点审计和四条完整指标仍依赖baseline全部产物。没有把工程c、部分窗口或旧seed c当正式c。统计继续严格`pendulum-seeds23`，未平滑原始return AUC/百万及80/85/90/95/100万五点均值，不自动混算seed0/1。

## 保留baseline并阻止重复启动

原监督器PID3690542保留；不发停止信号、不重启baseline。它继续持有两个baseline子进程，可真实记录退出码并完成原始收据/评价/快照/c审计。

新控制器只接管两个尚未启动的Dt槽位：先核原actual-RUN_INDEX身份及queued状态，再在原未使用的`seed2/dt-attempt01`和`seed3/dt-attempt01`目录建立唯一`SUPERSEDED_BY_PARALLEL.json`。marker记新run ID、目录、控制器/训练SHA及新合并索引，目录不含训练产物；若已经存在任何正式数据或不匹配marker，拒绝覆盖/启动。原监督器后续因目录已存在而拒绝该旧槽位，不能再额外启动Dt；此拒绝是已明确替代的排队项，不记成失败训练，也不纳入正式比较。原baseline目录与内容不修改。

新两条Dt使用新独立根`/data/Policy_Discrepancy/runs/formal-m2-pendulum-seeds23-parallel-dt-20261104-sv3/seed{2,3}/dt-attempt01`及新W&B ID；W&B仍PD_1，run group沿用原四条campaign。新根`actual-RUN_INDEX.json`作为四条合并索引，只包含原两baseline和新两Dt，保留各自真实路径/ID/SHA，不包含两个旧替代槽位。

新监督器读取原监督器的baseline验收结果，并自行记录新Dt真实退出码和逐条审计。四条均审计通过后才运行原始return完整统计及W&B精确JSON/hash收尾。旧根supervisor-final可能因两个替代槽位blocked而complete=false，不能据此否定已通过的baseline，也不能用新Dt单独完成冒充四条完整；四条最终验收以新合并索引和final-audit-attempt01/completion.json为准。

## 本次验收与输出

控制器`run_pendulum_parallel.py`及9项CPU测试验证：精确seed2/3、仅Dt、只用5/6；原训练SHA和科学配置保持；两个新卡绑定；已存在任务不可覆盖；marker阻止原launcher执行Popen；合并索引保留原baseline与唯一新Dt；容量不足和验收缺项不得启动。首次本机测试因导入TestCase导致重复发现旧冻结测试、遇最小venv缺portal；修正测试发现方式，9项新测试全部通过。旧冻结测试不跳过，服务器完整回归仍会执行。

原训练SHA已通过的129 CPU/三变体完整agent/在线logger证据复用；新控制器在服务器完成相关回归，并在GPU5/6各执行实际训练SHA的size50m CUDA工程短程（4098动作、517成功更新、3×1评价、2报告/评价隔离审计），绑定严格要求两个seed/卡及完整收据。新卡工程通过后才给原旧槽位写marker、启动两条正式Dt；工程数据不用于正式指标。重新使用保守240GB总剩余容量门槛，不删除原checkpoint/replay。

本机记录：`analysis/outputs/pendulum-seeds23-parallel-20261104/`；服务器工程根`/data/Policy_Discrepancy/runs/engineering-pendulum-seeds23-parallel-20261104-sv3/`。完整配置、命令、SHA、probe、CPU/CUDA验收、guard、新旧索引和W&B读回随后记录。当前文档初稿时两条新Dt尚未启动；不把准备计划写成运行完成。

## 实际部署、验收与启动（2026-11-04）

控制器实现及服务器实测SHA：`b7f42d46d16d4eda5cff5a798ec7b60cb27a3790`，独立checkout `/data/Policy_Discrepancy/checkouts/pendulum-parallel-b7f42d46d16d`。通过Git bundle校验/fetch/worktree部署，仅sv3同步；canonical checkout仍为`d8997f64a301bfdcfe08758faeae8957a7395766`，原训练checkout仍为c5f86ca且干净。逐Git diff确认dreamerv3/embodied、原指标及seed2/3入口/审计源码与训练SHA字节相同，新增文件是控制器和测试。

bundle SHA256 `e1b61d38c0f0f0f48d2ddbdbd53b7ae12dd55f2b056008c5a53f03c3dafdb425`。首次部署辅助脚本的手抄预期hash多一个字符，校验在git fetch/worktree之前拒绝执行；本机/服务器实际bundle哈希一致，保存`remote-deploy-failed-attempt01.py`后按实际64字符哈希修正，重新严格校验并部署。没有修改训练源码、跳过校验或影响baseline；不是失败训练尝试。

01:13服务器34项相关回归全部通过（21.078秒）：9项并行控制器、8项原seed2/3及指标/启动环境回归；旧冻结函数在完整环境真实执行通过。原129项训练SHA验收和零动作在线logger复用，未把34项说成重新跑完129项。01:17:31 GPU5/6两条真实CUDA短程各4098动作、517成功更新、3×1固定快照评价、2报告及隔离审计，全部退出0且通过；训练manifest为原c5f86ca，控制器为b7f42d4。

01:18:17启动前复查：GPU1/4各17个原本项目C/G进程，GPU5/6各2MiB但无作业，GPU7无作业；/data余330228621312字节，超过240GB保守门槛。新卡探针和正式绑定均使用单UUID，进程内CUDA0，EGL映射由实际探针获取，完整UUID/枚举/C/G证据在gpu5/gpu6-binding.json。

01:18:33新监督器PID3732048启动；绑定计划SHA256 `a4fc59303bd28635728376ec1f2797622ec76f60b0b54725dd7c45f3c188ee4d`。01:19:59实查两个监督器PID/创建时间/命令与四条agent命令均匹配存活，旧Dt槽位只有正确marker，原baseline PID未变：

|sv3物理GPU|seed|组|PID|状态|实际训练SHA|
|---:|---:|---|---:|---|---|
|1|2|baseline|3690581|运行中，原进程保留|c5f86ca1a16648f57db3c3abfa2584ac560495ca|
|4|3|baseline|3690588|运行中，原进程保留|c5f86ca1a16648f57db3c3abfa2584ac560495ca|
|5|2|PPD-KL|3732084|运行中，新启动|c5f86ca1a16648f57db3c3abfa2584ac560495ca|
|6|3|PPD-KL|3732097|运行中，新启动|c5f86ca1a16648f57db3c3abfa2584ac560495ca|

两baseline最新更新开始动作124956/124696，已完成0/5万/10万三点；10万点原始return均值704/200（各10episode）。这是训练阶段观察，完整AUC/末段指标尚未产生，不据此宣告训练稳定或跨seed效果。两PPD-KL当时正在零点评价，后续首批收据/线上配置启动验收另记。四条终点均尚未验收，GPU1/4/5/6仍被本项目使用，GPU7保留。

实际命令由同目录`start-checks.py`和`start-formal.py`调用。控制器命令为：

```text
/data/Policy_Discrepancy/envs/dreamer/bin/python -u -m scripts.run_pendulum_parallel run --plan /data/Policy_Discrepancy/runs/engineering-pendulum-seeds23-parallel-20261104-sv3/bound-plan.json --evidence /data/Policy_Discrepancy/runs/engineering-pendulum-seeds23-parallel-20261104-sv3
```

命令已经执行，不要重复启动。实际训练命令/完整配置/env在各execution-initial.json，子进程cwd固定原训练checkout。部署/辅助脚本和证据哈希保留本机输出目录。后续终点分析读取新根的四条合并索引，不读取旧两个被替代槽位，不把旧监督器预期的blocked当成新Dt训练失败。

### 01:22:34并行启动验收通过

四条均真实存活且仍执行c5f86ca：baseline seed2/3连续成功更新31751/31716，最新开始动作129032/128892；新Dt seed2/3为625/634更新、开始动作4528/4564，所有已读取收据invalid_count=0。各条零点10完整episode、环境seed/episode ID、0动作0更新固定快照、episode return均值、manifest完整配置正确；原两个baseline PID和创建时间未变，旧两个Dt目录只有匹配的替代marker。

四卡共68个C/G PID全部属于对应agent/子进程，进程内CUDA0，GPU1/4/5/6的实测EGL分别1/4/5/6；计算和渲染同卡、当次独占。四条线上均running/full config精确一致；仅为新Dt写入startup JSON/hash并精确读回，原baseline启动摘要未覆盖。Dt启动摘要hash分别`a3b7d56bfdf8b8eecbd2606a452c021e49d7141021a5434d23e320be1de8ee8f`、`2f73763539582feecbf12c8e5f141e5ae8e5aa0e30ee75e1868654313b39a4aa`。证据`parallel-startup-audit.json`，/data余328090333184字节。只核对评价episode累计值，未独立重放逐transition评价reward；启动验收不等于百万动作终点验收。

- [seed2 PPD-KL · PD_1](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-pendulum-seeds23-parallel-dt-20261104-sv3-pendulum-seed2-dt-01)
- [seed3 PPD-KL · PD_1](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-pendulum-seeds23-parallel-dt-20261104-sv3-pendulum-seed3-dt-01)

01:23:40采集132份小证据并在本机逐文件校验大小/SHA256通过；archive `small-evidence-20261103T172340Z.tar.gz`为1121671字节，SHA256 `101f3a6cbcef50e771e132709bf41ec93254e150c647859b23aa05e7769c56cb`；SOURCE_HASHES SHA256 `e4489acc9a81c8c52f016b396cea7bb14ff5cf32c39bb020c870426cd21785a4`。镜像在本机输出目录`mirror-startup/`，含新控制器/CPU/CUDA/设备/正式启动证据、原baseline与新Dt小产物、替代marker和合并索引。大checkpoint/replay未下载或删除，原129项证据继续引用原启动目录。

当前首选下一步是继续本chat监测四条至完整终点、逐seed复算并分析；不需创建新chat或重复申请已经批准的启动。正式c/完整AUC/末段均值、四条accepted以及科学解释仍待完成，目标保持进行中。已知问题仅为部署辅助脚本hash笔误并已修正，没有训练失败或重跑。最终交付版本、相对实测差异、资源快照及可复制prompt见本机输出目录DELIVERY.json/HANDOFF.md；活动两个checkout不随文档提交移动。

本次收尾提交自身只更新分析索引、实验索引、本报告和STATUS。相对本批实测控制器b7f42d4，HEAD还包含同期其他窗口的`run_vision20_seed1.py`/对应测试及存储清理文档；这些没有同步至本批活动checkout，也不宣称在本批34项验收中被测试。本批控制器/训练/审计/指标文件与实测版本的逐文件核对及完整差异清单写入DELIVERY.json，不把共享HEAD当成实际执行版本。

### 01:43:27运行观察与终点分析入口准备

两个监督器的PID/创建时间/命令与四条agent命令仍匹配，更新继续推进：baseline seed2/3的最新开始动作160100/159856、更新39518/39457；PPD-KL seed2/3为37160/37088、更新8783/8765，四条末条invalid_count均0。baseline已完成15万点评价，两均值745.2/119.5，各10episode；这些是阶段观察，不据此判断最终效果。GPU1/4/5/6各17个C/G进程，GPU7无进程，/data余326351060992字节；尚无四条终点completion。

本机新增`analysis/pendulum_seeds23_analysis.py`，复用冻结return指标和已有逐条收据/门控复核函数，提供严格四条终点采集、离线复算及逐seed学习曲线/门控图表入口；源码、命令、测试说明见`analysis/README.md`。旧两个Dt替代槽位、其他seed、工程数据、尚未accepted的运行及源hash冲突均不能通过该入口。完整c仍来自各baseline全窗口收据，分位数仅解释为日志批次分位数的均值；不新增科学阈值或归一化，不把平均权重解释成梯度完全匹配。

本次准备只改分析代码/测试/文档，不修改训练源码或服务器两个活动checkout，不写W&B。6项新增范围/hash/加权fixture与20项既有指标/分析测试的执行日志、实际版本/hash存`analysis/outputs/pendulum-seeds23-analysis-preparation-20261104/`；不能将这些离线测试写成四条真实终点验收。终点采集及完整图形验看尚待训练完成。

首选继续当前chat监测并完成四条终点审计和05分析；已完成新卡验收和并行启动，不重复申请或启动。新科学干预仍交02/用户，不自动建chat或发送消息。交付完整SHA/实测控制器SHA/实际训练SHA、差异与同步范围、实际资源状态、原始指标/限制及可复制prompt；本机其他chat草稿保留。

### 首个5万点与初始条件检查（2026-11-04）

01:54:07只读核对两条新Dt首个5万点：各10个完整1000动作episode，原始return均值seed2=38.2、seed3=0，JSONL均值一致，同seed baseline评价环境seed一致；快照实际50000动作/11993更新、metadata/done/payload存在，前11993成功收据连续且invalid0。门控日志有限且active_weight_available=1；两条首5万的实际活跃权重更新均值0.9869512231/0.9988607466。它们是早期描述，不作完整AUC、失败或方法效力判定；评价reward逐transition没有独立重放，该点参数payload仅查存在。证据在本次并行输出目录的monitor/partial-50k-20261104-015448.json和对应只读脚本。

01:58:18两监督器PID/创建时间/命令及四agent继续匹配，baseline seed2/3最新动作183440/183168、Dt为58672/58760，末条invalid0，/data余324.44GB，终点completion仍不存在。

02:01:43新增实际初始条件CPU检查：每seed baseline/Dt的294个0步参数数组及计数完全相同且有限，但reset图像已有623/301个坐标不同，reset观测上的策略输出有1/2个不同。完整更新前前缀的reward/episode边界一致，action/image不逐位相同；首次成功更新仍2032动作。本次实际解码并hash的是四条0步checkpoint，不能扩大为所有后续快照。详细计数、动作对齐语义与解释限制见docs/ANALYSIS_PENDULUM_CORRECTION.md新增小节。

分析/实测SHA58f6ff6710c62bb9871a260d59aba6813d6952a6，8项新分析测试及既有20项共28项通过；原始检查/源hash见analysis/outputs/pendulum-seeds23-initial-20261104/，未改训练源码或活动checkout。初始差异早于优化器更新，具体数值来源未裁决，不因此改协议/停训/重复启动，不把这批当输入逐位相同的机制实验；继续等待四条完整终点及逐seed结果。
