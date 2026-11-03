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

### 03:17匹配窗口前缀核查（尚非正式c冻结）

03:17:04两baseline均已写出30万点评价并继续训练；最新动作seed2/3=302044/301724，Dt=178840/179124。四条agent和两监督器的真实身份/命令均匹配，末条invalid0，GPU1/4/5/6各17个本批C/G进程，7无进程，/data余314.87GB；仍无终点completion。

03:17:15只读核对各baseline前74493条成功收据（update_id=0…74492）及30万点10完整episode、74493更新的快照metadata/done/payload存在。收据连续、开始动作单调且invalid0；[100000,300000)各纳入50000条，IDs24492…74491，300000动作的update_id74492正确排除。逐收据汇总如下，仅为窗口原始统计，**未计算/写入/冻结正式c**：

|seed|窗口S = match_sum|窗口N = match_count|窗口成功更新数|
|---:|---:|---:|---:|
|2|7396141.760381699|8363327|50000|
|3|7721404.809896469|8560619|50000|

两N均大于0；正式c仍由原监督器在完整百万动作/21×10/全部收据审计通过后冻结，并在终点独立复算再次比对本窗口S/N。当前检查不证明未来收据/全部终点评价完整，不代替其他快照的参数解码或评价逐reward重放。没有用部分窗口、工程c或默认值继续新对照，也未新增训练组。

脚本analysis/outputs/pendulum-seeds23-parallel-20261104/matching-window-prefix-audit.py，只经SSH stdin执行Python标准库只读检查；脚本SHA256 `242a7ea0c0aee840e778a9941ffbf30effd3028bda2c5e7d70fed566f7d558d8`。结果monitor/matching-window-prefix-20261104T031757217.json，SHA256 `52719abfeb1dd137b0b66cfdeceb759595267d816d118714fd07bb30b9b963b0`，包含原目录、实际训练SHA、固定前缀/评价行/metadata hash和边界IDs。大文件留sv3，未修改训练源码、活动checkout、W&B或c文件；正式终点0/4，继续原协议等待。指标代码/测试仍以58f6ff6及原冻结return_metrics为准，本次只有运行记录更新。

### 终点训练replay诊断入口与实际seed0回归（2026-11-04）

为判断原baseline的摆起失败是否在新seed重复，新增`analysis.pendulum_seeds23_analysis replay --input <终点输入镜像> --output <新目录>`。它再次要求权威四条全部accepted、supervisor/completion一致和原始小产物hash匹配，再以CPU只读完整训练replay的reward/action/episode边界。重用从seed0诊断中抽出的原函数，保持reset排除、前一行action对应当前reward、完整episode与尾段总和闭合等口径；不运行agent/环境/GPU，不下载大replay或改正式评价指标。输出逐seed的零回报episode、前10万正奖励次数、首次获奖动作位置，以及首动作reward=0/1分层的后来获奖比例和平均return。没有引入成功回报阈值或把训练episode当作训练seed。

实际分析/回归SHA **3591c8d2f54b0558badb2c98641c87f5e6edb52b**：10项seed2/3分析测试、4项原replay手算测试和20项既有指标/门控测试，共34项通过，unittest计时0.111秒。另对sv3已经完成的pendulum seed0 PPD-KL（训练d8997f64a301bfdcfe08758faeae8957a7395766）实际百万动作replay重新执行抽出函数：逐chunk标量hash、992个完整episode、16个尾段、全部汇总、20个5万区间、20个早期5千区间、prefill及初始标量hash与原inspection-entry/dt.json精确一致。正奖励动作699642、零回报完整episode49等均未改变。这是原seed0产物回归，不是新seed2/3终点验收。

本机证据`analysis/outputs/pendulum-seeds23-replay-preparation-20261104/`包含tests.txt/tests.json、实际远端只读脚本、seed0-replay.json、verification.json及命令。原参考JSON SHA256 `18ca0872092e577db802442796b11395f64c13dd9fdf78c9bf4467bc01093273`；本次实际回归JSON SHA256 `bfaf9f21a116186c65d7ca0b0eb971cd349e6ab563f6e10a98e311502aafb547`。未解析或hash全部图像；各chunk仅核所读标量数组hash与ZIP CRC。训练reward独立求和不能替代评价逐transition reward核验，首动作reward不能当成测得的reset角度。

03:31:18实时核查：原两baseline动作324376/324040，新两Dt均到200000动作、49493次成功更新，20万点评价当时尚未写出；两监督器及四agent真实身份/命令匹配、末条invalid0，四卡各17个本批C/G PID、GPU7无进程，/data余313.79GB。仍0/4终点验收，新replay入口尚未用于本批终点。只改本机分析代码/测试/文档，经stdin只读sv3，无服务器Git同步或训练/控制器checkout变更。下一步继续现有chat完成四条终点collect/analyze/replay及图形验看，再综合校正解释；整体目标保持进行中。

### seed3 PPD-KL 20万点零回报的阶段复核

03:32:58四条仍存活并继续训练，PPD-KL seed2/3已写出20万点，各10episode，均值807.4/0；baseline最新动作327000/326648，Dt202104/202380。03:34:16随后对seed3 Dt的15万/20万两点评价、JSONL和实际checkpoint作CPU只读检查：15万返回`[0,0,0,924,0,0,1000,0,0,0]`（均值192.4），20万10个return均为0，均为完整1000动作episode，与同点baseline评价环境seed匹配，原始episode值/重复数组/JSONL均值一致。

实际解码两份checkpoint，各294个参数数组均有限，全部294个数组在两点之间发生变化；更新计数分别36993/49493，与评价及snapshot metadata一致，payload/done/hash完整。所查10—15万和15—20万区间的门控标量日志无非有限值，日志中的梯度范数区间均值为20.710876/18.185102；这不是逐更新梯度轨迹，不能据均值排除瞬时尖峰。实际活跃权重均值0.984339/0.976759，free-nats活跃比例0.076268/0.074215；均为该阶段统计，不能代替整轮门控结论。

本次没有发现评价记录错误、快照未更新或已保存参数非有限的证据；不能仅凭20万点0回报宣布整轮工程失败、永久学习失败或方法无效，也不能据此证明优化没有问题。保留该点，继续原百万动作协议，不调参、删seed、重跑或插值。当前仅额外解码seed3 Dt这两个checkpoint，不扩大为全部21点参数或逐transition评价reward均已独立验证。

证据在并行输出目录`monitor/dt-seed3-200k-numerics.json`（SHA256 `0e1fc73acc28ab49f3eb27c6e82d5b62b7c4abb5c347e47ef0b1f879c63a5e18`）；只读脚本`inspect-dt-seed3-200k.py`（SHA256 `f20ef9ac4dc24c9bacd9a5ad713abb15f0cc7b232b56cd05f9835e008dde390e`）。脚本经stdin执行CPU/NumPy/pickle，不执行agent/环境或修改服务器源码；原始大文件未下载或覆盖，实际训练c5f86ca/控制器b7f42d4继续固定，正式终点仍0/4。

### 04:05现有28点评价复算：seed3 Dt的25万点重新出现正回报

04:04:55连续只读监测结束时，两监督器和四agent真实身份/命令匹配；baseline动作375460/374872，Dt250480/250792，末条invalid0，GPU1/4/5/6各17个本批C/G PID，7无进程，/data余309.64GB，终点completion仍不存在。两baseline35万点为818.2/875.6；两Dt25万点新写出800.6/472.2，随后继续训练。

04:05:33调用实际训练checkout c5f86ca1a16648f57db3c3abfa2584ac560495ca中的冻结return_metrics，以严格pendulum-seeds23范围只读复算已保存评价：baseline各0…35万共8点，Dt各0…25万共6点，合计28点/280完整episode。原始episode均值、来源/协议字段及固定快照小证据全部通过；无解析冲突、重复行或缺episode，检查期间评价文件hash未变。完整21点AUC及末段五点仍为null，未插值或用部分曲线冒充正式指标。

|训练seed|25万点baseline均值|25万点PPD-KL均值|PPD-KL该点原始10episode return|
|---:|---:|---:|---|
|2|772.5|800.6|549, 724, 719, 932, 705, 836, 829, 886, 885, 941|
|3|842.1|472.2|0, 929, 933, 0, 0, 0, 929, 1000, 931, 0|

seed3 Dt在20万点的10个0回报之后，25万点有5个0、5个正回报。这不支持把20万单点直接视为整轮永久失败；也不能将随后一个正回报点写成训练已稳定、已完全学会摆起或最终优于baseline。保留全部阶段点，继续原百万动作协议；当前没有新的重跑或调参依据。评价内部逐transition reward仍未独立重放，本次没有额外解码25万checkpoint参数；前述15万/20万参数检查的范围不扩大。

只读脚本`analysis/outputs/pendulum-seeds23-parallel-20261104/partial-return-250k.py`，SHA256 `76f4f9a45d2a57ac90bf433ce4a07c23537f5b63f6ce5322ac0602dff41172fb`；结果`monitor/partial-return-250k-20261104.json`，SHA256 `83e0899c4d8ce94821ee28e510169444259c51e5e3efa3b80e57987f1fbd19d7`，含原始评价记录、输入hash、点级核查及正式指标null。仅本机文档/小证据更新，无训练源码、科学配置、活动checkout、W&B或c文件变更；终点分析代码/34项测试仍3591c8d2f54b0558badb2c98641c87f5e6edb52b。继续当前chat完成四条终点collect/analyze/replay和校正解释，目标保持进行中。

### 04:38现有32点评价复算与跨评价点解释限制

04:38:20实际进程核查通过：baseline seed2/3动作426200/425456，Dt301200/301596，两监督器及四agent身份匹配、末条invalid0；GPU1/4/5/6各17个本批C/G、7无进程，/data305.95GB，终点0/4。baseline新40万点为705.1/839.8；Dt新30万点为835.7/91.6，各10episode，随后继续训练。

04:38:52复用未修改的`partial-return-250k.py`（它读取当前全部已写出点，不仅25万点），在c5f86ca冻结指标代码上复算baseline各0…40万9点、Dt各0…30万7点，合计32点/320完整episode；来源/协议/固定快照小证据通过，无重复或均值冲突。完整AUC与末段仍null。30万点同预算比较如下，仅为单点评价：

|训练seed|baseline均值|PPD-KL均值|PPD-KL原始10episode return|
|---:|---:|---:|---|
|2|817.2|835.7|1000, 842, 938, 615, 882, 984, 881, 616, 937, 662|
|3|686.8|91.6|0, 916, 0, 0, 0, 0, 0, 0, 0, 0|

进一步逐记录比较前次28点与本次32点：前次每条评价记录精确保留，新点追加。每seed两组在0…30万各共同点的10个评价环境seed完全相同，但不同评价点的环境seed集合各不相同，这是当前原协议的数据事实。seed3 Dt在20/25/30万点为0/472.2/91.6，不能称为已稳定恢复；也不能把跨点变化全部归为优化导致的参数退化或恢复，因为策略快照与评价初始条件同时变化。现有结果不能分离这两者的贡献，不修改评价协议或另行启动评价来补该缺口。完整曲线、训练replay与终点指标仍待完成。

新复算结果`monitor/partial-return-300k-20261104.json` SHA256 `e8355fe6368d2095d63a51d8105c7004975002a2f828c9b51275f80679dd724b`；本机记录一致性脚本`compare-observed-evaluations.py` SHA256 `15dce17e0dee5748f841675a2e99c35b5e1ef7c7bee2ad53535577cb76fae743`，结果`monitor/observed-record-consistency-300k.json` SHA256 `874c2d8ddf76538718ee370f3c5358b32565e6f7dfa516c03c6ed567e8b0f095`。均在本次并行输出目录。未新解码30万参数或重放评价逐reward；不把所查点完整说成四条终点完整。无代码/训练配置/服务器同步/W&B/c写入变更，继续当前chat至四条完整验收和校正分析完成。
