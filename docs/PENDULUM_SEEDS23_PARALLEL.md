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

### 05:44现有40点评价复算：四条继续原协议训练

05:43:57核对两监督器及四agent的真实身份/命令均匹配，baseline seed2/3最新动作525904/525148、PPD-KL为400556/401236，末条invalid_count均0。GPU1/4/5/6各17个本批C/G进程，GPU7无进程，/data余298.51GB；四条仍运行中，终点0/4。两条PPD-KL已写出40万点评价并继续训练，未发生本批训练失败或重跑。

05:44:08通过未修改的`partial-return-250k.py`和实际c5f86ca冻结指标代码，复算baseline各0…50万11点、PPD-KL各0…40万9点，共40点/400完整episode。全部现有点的原始episode均值、来源/协议与固定快照小证据通过；无解析冲突、重复评价行或episode缺项。前次已核的32条评价记录精确保留。完整21点AUC和末段五点均值仍为null，未插值、截短预算或用局部曲线代替正式指标。

|训练seed|40万点baseline均值|40万点PPD-KL均值|该点PPD-KL−baseline|PPD-KL原始10episode return|
|---:|---:|---:|---:|---|
|2|705.1|784.3|+79.2|651, 689, 937, 933, 635, 939, 889, 726, 799, 645|
|3|839.8|365.8|−474.0|0, 0, 0, 0, 0, 923, 922, 0, 896, 917|

新增35万点PPD-KL seed2/3为824.3/584.1；baseline45万点753.9/804.3、50万点775.7/788.0，也已纳入本次独立复算。每seed两组在0…40万共同点的评价环境seed一致，但不同点的环境seed集合仍各不相同。seed3 PPD-KL的30/35/40万均值91.6/584.1/365.8呈阶段波动；40万点6个零回报、4个正回报。不能凭这些点判定最终优劣、稳定恢复、永久失败，或把跨点变化完全归为参数变化；保留全部点，继续原百万动作协议。

本次仍仅核已记录的episode累计return，未独立重放评价逐transition reward，也未新增解码35万/40万checkpoint参数。既有四条0步和seed3 Dt的15万/20万参数检查范围不扩大。终点训练replay诊断、完整门控统计、正式c冻结及全部21×10审计仍待完成。

本机只读采集/一致性检查脚本`analysis/outputs/pendulum-seeds23-parallel-20261104/collect-observed-400k.py`，SHA256 `0774248495d10e32c512ae02e7c422885c2a6a6406b055ffbf9e97ac4b21c95d`；冻结指标读取脚本仍为`partial-return-250k.py`，SHA256 `76f4f9a45d2a57ac90bf433ce4a07c23537f5b63f6ce5322ac0602dff41172fb`。结果`monitor/partial-return-400k-20261104.json` SHA256 `5a8ba7804b005bb7dc734839ee7d41687adcb2941e8c4434bd98de80709cf200`；一致性结果`monitor/observed-record-consistency-400k.json` SHA256 `d4d3e483f298ed0aba25bce6dad7d3aaba6719f3b7e4df2f8bce4bffd3f1d03d`。执行命令：

```powershell
& 'D:/program/PD/dmr3/analysis/outputs/dmc16-preparation-20261022/venv/Scripts/python.exe' 'D:/program/PD/dmr3/analysis/outputs/pendulum-seeds23-parallel-20261104/collect-observed-400k.py'
```

该命令已执行成功，脚本拒绝覆盖既有证据；后续阶段使用新输出位置并保留原文件。仅本机小证据/运行记录更新，无训练源码、科学配置、服务器Git同步、W&B或c写入变更。实际训练SHA c5f86ca1a16648f57db3c3abfa2584ac560495ca、控制器/服务器验收SHA b7f42d46d16d4eda5cff5a798ec7b60cb27a3790保持；终点分析及34项已通过测试仍3591c8d2f54b0558badb2c98641c87f5e6edb52b。继续当前chat完成四条终点collect/analyze/replay、图形验看与校正解释，目标保持进行中，不自动建chat或发消息。

### 06:50现有48点评价复算与seed3零回报的算术分解

06:49:50实际核查：两监督器及四agent身份/命令匹配；baseline seed2/3最新更新开始动作626192/625316、PPD-KL为500496/501368，末条invalid0，全部继续训练。GPU1/4/5/6各17个本批C/G进程，GPU7无进程，/data余291.12GB；四条终点仍0/4。两条PPD-KL已完成50万点评价，没有新的训练失败或重跑。

06:50:03复用原c5f86ca冻结指标代码，只读复算baseline各0…60万13点、PPD-KL各0…50万11点，共48点/480完整episode。原始episode均值、来源/协议和固定快照小证据通过，无重复或均值冲突；前次40条评价记录精确保留。完整21点AUC和末段五点评价仍不齐，正式指标保持null，不插值或计算替代预算指标。

|训练seed|50万点baseline均值|50万点PPD-KL均值|该点PPD-KL−baseline|PPD-KL原始10episode return|
|---:|---:|---:|---:|---|
|2|775.7|791.1|+15.4|691, 1000, 886, 792, 939, 0, 802, 934, 933, 934|
|3|788.0|377.8|−410.2|0, 932, 0, 0, 934, 983, 0, 0, 0, 929|

新增45万点PPD-KL seed2/3为778.4/291.0，baseline55万点814.9/741.6、60万点800.3/762.1，也纳入本次复算。每seed两组在0…50万共同点的评价环境seed一致，跨点环境seed集合各不相同；跨点变化仍混合参数快照和评价初始条件变化。seed3 PPD-KL40/45/50万均值365.8/291.0/377.8，保持全部正式点，不据阶段结果调参、停训、重跑或宣告最终优劣。

50万点seed3的低均值可作直接的算术说明：PPD-KL有6个零回报episode，其相同评价环境seed下的baseline return为843/714/758/135/760/889，合计4099，对该点组间均值差贡献−409.9；其余4个PPD-KL正回报为932/934/983/929，对应baseline为935/931/996/919，差值合计−3，对均值差贡献−0.3。两部分相加为−410.2。因此该点评价差距主要来自6个零回报episode。这是按PPD-KL结果选取子集后的单点描述，不能据此证明两组保持能力等效、确定reset角度、区分是否完成摆起，或定位训练机制原因；完整训练replay和终点分析仍待完成。

本次还只读核对终点分析按seed/组分别汇总、按更新数加权，以及实际训练入口的数值检查。embodied/run/protocol_v1.py在c5f86ca和当前交付父版本ff1e76e的Git blob同为ddfa3ba8f00ff9a0b533ef369380bccf1fa10df1：写成功收据前检查opt/loss与opt/grad_norm有限，record_update在非有限时抛错。此代码事实不代替完整249493条收据终点核验，也不能排除有限数值的梯度尖峰或证明每个checkpoint参数均被独立解码。没有因代码阅读新增训练或改变验收阈值，既有34项分析测试不因文档更新重复执行。

本次没有额外解码45万/50万参数或重放评价逐transition reward；只核已记录的episode累计return，既有0步和seed3 Dt15/20万参数检查范围不扩大。正式c、完整门控统计、百万训练replay核验和四条终点验收均仍待完成。

只读采集/一致性脚本analysis/outputs/pendulum-seeds23-parallel-20261104/collect-observed-500k.py的SHA256为fed23b0b40d365e26910a0298be39f6bf5cd47f57dc44f8b3bac4a2cc3c6bd89，调用未修改的partial-return-250k.py（SHA256 76f4f9a45d2a57ac90bf433ce4a07c23537f5b63f6ce5322ac0602dff41172fb）。结果monitor/partial-return-500k-20261104.json的SHA256为58f15b4529e4f0062b6ffbd3377b2514d7e8bcb5b19a1d210ee78ae669c502df；一致性结果monitor/observed-record-consistency-500k.json为072aa18551a8d505677063aca60fb2d2520abc63d6d8cee72cfb5c3cdb33f2c3；单点分解monitor/paired-500k-seed3-decomposition.json为f1f751804fb0b95b4e1470d2aaee84552942e9d92d7ef7c9c689e1ffd02e6937。分解文件保留原episode ID、环境seed和两组return，按各子集的sum(PPD-KL−baseline)/10即可复算上述贡献。采集命令已成功执行，不能覆盖既有证据：

```powershell
& 'D:/program/PD/dmr3/analysis/outputs/dmc16-preparation-20261022/venv/Scripts/python.exe' 'D:/program/PD/dmr3/analysis/outputs/pendulum-seeds23-parallel-20261104/collect-observed-500k.py'
```

本轮仅本机小证据/辅助只读脚本和文档更新；无训练源码、科学配置、服务器Git同步、活动checkout、W&B或c写入变更。实际训练/指标SHA c5f86ca1a16648f57db3c3abfa2584ac560495ca、控制器/服务器验收/终点审计b7f42d46d16d4eda5cff5a798ec7b60cb27a3790，以及终点分析/既有34项测试3591c8d2f54b0558badb2c98641c87f5e6edb52b保持。继续当前chat完成四条终点collect/analyze/replay、图形验看和完整校正解释，目标保持active，不自动建chat或发消息。

### 07:56现有56点评价复算、门控日志与baseline低分复核

07:56:34实查两监督器及四agent身份/命令匹配，baseline seed2/3最新更新开始动作726976/726096、PPD-KL600832/601256，末条invalid0；GPU1/4/5/6各17个本批C/G进程，GPU7无进程，/data283.70GB。四条仍运行中，终点0/4，无新训练失败或重跑。

07:56:51在原c5f86ca冻结指标代码上复算baseline各0…70万15点、PPD-KL各0…60万13点，共56点/560完整episode；来源/协议、episode均值和固定快照小证据通过，无重复/均值冲突，旧48条记录精确保留。共同点两组环境seed一致、跨点集合不同。完整21点AUC与末段五点仍不齐，正式指标保持null，没有插值或替代预算。

|训练seed|60万点baseline均值|60万点PPD-KL均值|该点PPD-KL−baseline|PPD-KL原始10episode return|
|---:|---:|---:|---:|---|
|2|800.3|810.2|+9.9|790, 940, 932, 870, 679, 730, 762, 531, 929, 939|
|3|762.1|93.7|-668.4|0, 0, 0, 0, 0, 937, 0, 0, 0, 0|

新增55万点PPD-KL seed2/3为769.3/180.9；baseline65万点842.4/631.7、70万点810.2/205.3，也纳入本次复算。seed3 PPD-KL60万点9个0、1个937；不因该点低回报删记录、改参数、重跑或判定最终优劣。它与下述baseline70万点的回报分布不同，不能把两者直接当成同一现象或同预算组间比较。

07:34:51另行只读检查实际门控日志：baseline各至65万13个已完成区间、PPD-KL各至55万11个区间，共48区间；沿用既有FIELDS检查，所查门控及其他标量字段有限，invalid_D_frac和match_invalid_count均0，active_weight_available均1，区间更新均值与快照计数一致。baseline实际权重均值为1、超额损失减弱为0。Dt成功收据的匹配字段按实现固定写0，不能仅凭其invalid_count=0推断门控数值合法；本次直接核了真实诊断日志。这不是未来区间或完整终点验收。

同预算50万—55万区间（每条12500次更新）的实际日志统计如下；完整未舍入值及原始日志行/hash保存在gate-500k-550k-summary.json和gate-health-prefix-20261104.json：

|seed|组|Mean Dt|Dt精确零值比例|free-nats活跃比例|实际活跃权重均值|超额rep KL损失值减弱|
|---:|---|---:|---:|---:|---:|---:|
|2|baseline|0.074988|2.29%|9.61%|1.0000|0.00%|
|2|PPD-KL|0.085391|2.06%|10.78%|0.8531|25.14%|
|3|baseline|0.058801|2.77%|14.20%|1.0000|0.00%|
|3|PPD-KL|0.003377|0.94%|4.90%|0.9486|6.40%|

实际活跃权重是逐更新活跃位置均值再按更新取均值；超额rep KL损失值减弱为excess_reduction_mean/(weighted_excess_mean+excess_reduction_mean)，基于该区间累计均值之比。baseline权重日志的极小浮点偏差按表中精度显示为1，实际全位置权重严格为1。PPD-KL两个seed的本区间作用程度不同；这些值不是总梯度变化，也不能从不同轨迹/模型的统计反推回报变化原因、裁定强度足够或不足，或据此选择新的alpha。未新增效果阈值。

07:42:00针对baseline seed3从65万631.7到70万205.3的变化，CPU只读复核两点评价/JSONL及两份实际checkpoint。两点各10完整1000动作episode，原始return数组、均值、计数和快照metadata/done一致；各294参数数组全部有限，294数组均发生变化，更新计数161993/174493正确。60—65万和65—70万所查标量日志无非有限值；梯度范数区间均值12.5186427/12.0142815，实际rep权重均1，rep loss调整前后相同。区间均值不排除有限的瞬时尖峰，参数数值有限也不证明优化或策略表现正常。

baseline seed3的65万return为[689,840,759,763,88,657,657,940,124,800]；70万为[102,99,590,87,88,696,84,96,97,114]。两点均没有零回报episode；70万是8条84—114、另2条590/696。没有发现这些点的记录错误、快照停滞、非有限参数或门控组配置混用证据。两点环境seed集合不同，仍不能把均值下降全部归因于参数变化，也不能从累计return确定摆起/保持或reset角度；保留低分点并继续原协议。

当前独立参数解码范围仅四条0步、seed3 PPD-KL15/20万、seed3 baseline65/70万；不扩大为全部快照已解码。评价逐transition reward仍未独立重放。四条完整训练replay、正式c冻结、全程门控和完整指标均待终点，终点收据仍须全部249493条连续通过。

|本机相对并行输出目录的文件|SHA256|
|---|---|
|collect-observed-600k.py|b3db9aa65a1597a78db569c9e03b7f9cd7b850773948990b6b0ec93d373348e3|
|monitor/partial-return-600k-20261104.json|d258d87afad83c401c725e977fa364855904533fed8343021c80743b92cec92e|
|monitor/observed-record-consistency-600k.json|8776c139c4fcd0eb6b239b814cee09ffe009306d32a6c47de7800083bddae1fd|
|inspect-gate-prefix.py|900b11dc63b12a95e98842137f393868d7c4c9fe9e0d48a2351fcafaa1fc0eea|
|monitor/gate-health-prefix-20261104.json|80900db62951d574df152a898df041f00fdf00412049cd1c3b8a05f2d1447bbe|
|monitor/gate-500k-550k-summary.json|0fd91c18a2b2b4d7e6108baa0cc7c7c4dbaa80752f0dac94a21432cb41c2c3df|
|inspect-baseline-seed3-700k.py|48a0a4d5448986e43852eb06e79c26f40389f285fcf9ee22b885809054ac13fb|
|monitor/baseline-seed3-700k-numerics.json|e4824445c8f5084135250f5f1914024f33669da4df4385efc7e00304c4eda25d|

本机目录为analysis/outputs/pendulum-seeds23-parallel-20261104/。阶段采集执行命令如下，已经成功执行，脚本拒绝覆盖既有证据：

```powershell
& 'D:/program/PD/dmr3/analysis/outputs/dmc16-preparation-20261022/venv/Scripts/python.exe' 'D:/program/PD/dmr3/analysis/outputs/pendulum-seeds23-parallel-20261104/collect-observed-600k.py'
```

两个补查脚本均由本机经SSH stdin传给sv3的/data/Policy_Discrepancy/envs/dreamer/bin/python -，环境CUDA_VISIBLE_DEVICES为空、JAX_PLATFORMS=cpu、OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1；仅读取既有日志或可信checkpoint，输出上述本机JSON/stderr，未执行agent/环境或新增GPU作业，大文件留服务器。后续复读使用新的输出文件，不覆盖本次证据。

本次仅本机辅助只读脚本、小证据和文档；无训练源码/科学配置/服务器Git同步/活动checkout/W&B/c写入变更。训练及冻结指标SHA c5f86ca1a16648f57db3c3abfa2584ac560495ca、控制器/服务器验收/终点审计b7f42d46d16d4eda5cff5a798ec7b60cb27a3790、终点分析和既有34项测试3591c8d2f54b0558badb2c98641c87f5e6edb52b保持，未因文档重复测试。继续当前chat至四条终点collect/analyze/replay、图形验看及完整校正解释完成，目标保持active，不自动建chat或发消息。

### 08:19终点训练过程诊断入口准备（四条仍在运行）

为了在终点区分“固定快照评价低分”和“实际训练轨迹缺少奖励”，在现有replay入口补充20个5万动作区间的对齐输出。训练源、reward、配置、alpha、评价和正式指标均不变；该步骤只读取已通过终点范围/hash校验的镜像JSONL和标量replay结果。新增training_phases.json/csv，以及每seed的training_dynamics_seed2/3.png/svg；图中并列未平滑评价return、真实训练正奖励比例、想象奖励、Gaussian策略熵、优势幅度、实际活跃rep权重。

区间边界、完整动作数、原始episode均值和更新计数均显式核对；按冻结loss scales重建总loss，容差沿用原seed0诊断rel_tol=1e-7/abs_tol=1e-6，不为通过调整。真实奖励是动作均值，想象/策略日志是更新均值，两者的采样分布不同，不能取比值当奖励模型误差，也不据相关曲线宣告因果根因或选新参数。输出中仍按真实训练seed分开，不新增阈值或正式指标。

实际分析与测试SHA为5ccbd8c533107823fba558dd092cac41655efd8d，36项离线测试通过（含新增手算对齐，以及评价点缺失、乱序、计数、非有限值、均值冲突和损失系数的拒绝检查）。另对本机已保存、与原交付hash一致的seed0 baseline/PPD-KL两组实际数据回归，共40个区间，共有字段及动作/更新计数与既有分析完全一致；两组总loss重建最大误差5.565395433393405e-8/5.1054818150930714e-8，与原报告一致。这是历史真实数据回归，不是本批四条终点验收；本次没有再次远程读取历史replay。

证据目录analysis/outputs/pendulum-seeds23-dynamics-preparation-20261104/：tests.txt/tests.json、verification.json、historical_phase_rows.json、verify-preparation.py、COMMANDS.md。verification SHA256为376102459b3db3dcfd1738945214a88a0d390183541af84cd001c486348f6c47；测试日志SHA256为7d1854963b0aa866524cde3acd6ffb17b9e547a10ae57621281f09200d34f6bf；历史区间输出SHA256为acb86d2f234f3ec874a20dda7df5328440f54a7ed20997d64ea4fb58d859d871。训练过程图还未用本批终点产物生成，需四条验收后运行并目视检查。

08:19:02 sv3实查两监督器与四agent仍真实存活、身份/命令匹配；baseline seed2/3最新更新开始动作760156/759644，PPD-KL635792/636180，末条invalid0。GPU1/4/5/6各17个本批C/G、GPU7无进程，/data281.84GB，终点0/4。baseline新75万点原JSONL记录均值813.5/523.4尚未纳入本轮独立episode复算；不将这一点变化写成稳定恢复，先前56点评价验证范围保持。原始实查在并行输出目录monitor/20261104T081943171.json。

本次只修改本机分析代码/测试/README及文档，无训练源码、协议、服务器Git同步、活动checkout、W&B/c写入或新增GPU任务。实际训练仍c5f86ca1a16648f57db3c3abfa2584ac560495ca，控制器/服务器验收/终点审计仍b7f42d46d16d4eda5cff5a798ec7b60cb27a3790；旧3591c8d准备和600k交付是历史记录，后续终点分析采用本次已测试入口。继续当前chat至四条完整终点、collect/analyze/replay、图形验看和校正解释完成，目标active，不自动建chat或发消息。

### 09:06现有64点评价与60个门控区间复核

08:30:23从原始episode记录复算60点/600完整episode，baseline各至75万、PPD-KL各至65万；此前56条记录精确保留。09:04:24再次复算64点/640完整episode，baseline各0…80万17点、PPD-KL各0…70万15点，前60条精确保留。两次来源/协议/均值及固定快照小证据检查通过，无重复或均值冲突；共同点两组环境seed一致、跨点集合不同。冻结指标代码仍为实际训练checkout的c5f86ca；每条21点尚未齐，完整AUC与末段五点均保持null，无插值或部分预算替代。

|动作点|seed|baseline|PPD-KL|PPD-KL−baseline|
|---:|---:|---:|---:|---:|
|65万|2|842.4|860.8|+18.4|
|65万|3|631.7|185.5|-446.2|
|70万|2|810.2|827.2|+17.0|
|70万|3|205.3|93.4|-111.9|

seed3 PPD-KL65万点return为[0,0,0,0,0,0,0,930,925,0]，70万为[0,0,0,0,0,0,0,0,0,934]，分别8/9个零回报episode，全部保留。baseline75万点seed2/3=813.5/523.4，80万点=813.3/893.7，本次均已独立复算。baseline seed3的75万return为[889,795,0,0,805,0,939,806,1000,0]，80万为[970,711,940,940,1000,936,1000,672,886,882]；其75万也有4个零回报，80万没有。此前“65/70万没有零回报”的描述仅覆盖那两个点，不能扩大为baseline整条曲线没有零回报，或把两组各一个点的形态当成固定机制差异。跨点评价环境seed不同，回升/下降同时涉及快照与初始条件变化，不仅凭这些点认定稳定恢复、训练稳定性或唯一根因。

09:05:09读取实际门控JSONL，baseline各至80万16区间、PPD-KL各至70万14区间，共60个已完成区间；原48条区间记录及日志行hash逐项一致保留。所查FIELDS及其他数值标量有限，invalid_D_frac和match_invalid_count均0，active_weight_available均1，位置数1024，日志更新均值与快照计数一致；baseline全位置实际权重1、超额减弱0。此证据来自实际诊断字段，不能仅用Dt匹配收据固定为0的invalid字段替代。

同预算65—70万区间，每条12500次更新：

|seed|组|Mean Dt|Dt精确零比例|free-nats活跃比例|实际活跃权重均值|超额rep KL损失值减弱|
|---:|---|---:|---:|---:|---:|---:|
|2|baseline|0.067441|4.51%|11.08%|1.0000|0.00%|
|2|PPD-KL|0.067638|3.49%|9.28%|0.8846|11.17%|
|3|baseline|0.130551|5.68%|13.10%|1.0000|0.00%|
|3|PPD-KL|0.003440|1.31%|4.54%|0.9450|6.12%|

PPD-KL seed2/3的全位置实际权重均值另为0.9312643/0.9858510；表中活跃权重是逐更新活跃位置均值再取更新均值。损失减弱仍为excess_reduction_mean/(weighted_excess_mean+excess_reduction_mean)，不是总梯度变化；这些不同轨迹/模型上的统计不足以解释回报差、裁定门控强度足够/不足或选择alpha。区间均值也不排除有限的瞬时尖峰。本次没有新增参数payload解码或评价逐transition reward重放；既有八份参数解码范围保持，完整训练replay/c/全收据/终点指标仍待验收。

08:32:25另只读sv1既有Vision20 seed1索引：pendulum baseline/PPD-KL均queued、物理GPU3、实际计划SHA bef29480df12d07659c640b22218804dd1d69e63；已观察到监督器1609040存活，其cwd/argv与该checkout/计划相符。该卡先执行quadruped_walk一对，再到pendulum；本次没有重排、重复启动或把seed1纳入本批统计，也没有声称核查sv1 GPU空闲或完成其运行验收。

09:06:05 sv3两监督器及四agent身份/命令匹配，baseline最新更新开始动作831756/831420，PPD-KL705548/706052，末条invalid0。GPU1/4/5/6各17个本批C/G、GPU7无进程，/data276.30GB，四条仍运行、终点0/4。实查快照monitor/20261104T090646846.json；没有训练失败或重跑。

|相对并行输出目录的文件|SHA256|
|---|---|
|collect-observed-650k.py|ff1fc822b6458e7951055134c101e46b53acf247c380b1bd78ba8ef75465aaaf|
|monitor/partial-return-650k-20261104.json|84f41d2a60d0cf6c94297e89afb651416f2202b3a3b6f0d5fae724ea279751a2|
|monitor/observed-record-consistency-650k.json|a6b9ac0c552d2021b9222b3db1dbd0796ca08c72308a4d5a848d4c076799e3b7|
|collect-observed-700k.py|46f4f682621a971ac07c1925433950c504769f11fff66375b7919944fcbd9498|
|monitor/partial-return-700k-20261104.json|7a5a3c1894a53e284314e867f23dba509e60e35dd371b554bbed4b721305a34b|
|monitor/observed-record-consistency-700k.json|a3df1c3c55fa14133eb3175c7debed34a8c8aaf4a472cd0997896425106385df|
|collect-gates-700k.py|7fc91959758bae535fcaad15dd3c34bb07ca4a1c02bc0bb2fac781ef8c48b352|
|monitor/gate-health-prefix-700k-20261104.json|fa8821ee33b2c44cc410e810f1b158bfee577e09d55ad2673ae14c47f04a657b|
|monitor/gate-prefix-consistency-700k.json|243aba0cba7be07ad8c61a7564f4e66767cea4b15973ef0b3942030fc2ba7b1b|
|inspect-seed1-queue-20261104.py|4a512505356f63159996948f91a9af503fa05339990385c8f2a93c8a0fc639ce|
|monitor/seed1-queue-20261104.json|9e60a8598047f247845cb5cb5cdf879fef684928b1f6afa279ff339364825eee|

全部相对路径位于analysis/outputs/pendulum-seeds23-parallel-20261104/。使用原最小Python依次执行collect-observed-650k.py、collect-observed-700k.py、collect-gates-700k.py；均已成功执行且拒绝覆盖旧输出。前两者通过SSH stdin调用原只读partial-return-250k.py，后者调用原inspect-gate-prefix.py并隐藏CUDA、强制CPU/单线程读取日志；实际命令/源hash与逐行一致性保存在相应verification JSON。seed1队列补查同样仅SSH stdin/CPU读取，原脚本和结果已保留。

本阶段仅本机只读采集辅助脚本、小证据和文档，无训练源码/科学配置/服务器Git同步/活动checkout/W&B/c写入变更。四条实际训练及冻结指标SHA c5f86ca1a16648f57db3c3abfa2584ac560495ca，控制器/服务器验收/终点审计b7f42d46d16d4eda5cff5a798ec7b60cb27a3790，已准备终点分析和36项测试5ccbd8c533107823fba558dd092cac41655efd8d保持，未为文档重复测试。继续当前chat完成四条终点及完整collect/analyze/replay、图形验看和校正解释，目标active，不自动建chat或发消息。

### 10:17现有72点评价与68个门控区间复核

10:11:06在实际训练checkout的冻结c5f86ca指标代码上复算baseline各0…90万19点、PPD-KL各0…80万17点，共72点/720个完整1000动作episode。来源/协议、原始episode均值、固定快照小证据检查通过；原64条评价记录逐项精确保留。共同点两组评价环境seed一致、跨点集合不同。每条21点尚未齐，完整AUC和末段五点仍null，不插值，也不以部分预算结果替代正式指标。

|动作点|seed|baseline|PPD-KL|PPD-KL−baseline|
|---:|---:|---:|---:|---:|
|75万|2|813.5|811.8|-1.7|
|75万|3|523.4|193.6|-329.8|
|80万|2|813.3|837.4|+24.1|
|80万|3|893.7|292.6|-601.1|

seed3 PPD-KL75万原始return为[0,0,0,0,0,0,936,0,1000,0]，80万为[926,0,0,0,1000,0,1000,0,0,0]，分别8/7个零回报episode。seed2 PPD-KL80万为[933,844,722,842,726,654,891,938,932,892]。新增baseline85/90万两点均值：seed2为778.8/806.2，seed3为795.2/786.1，四个点均无零回报episode。本次保留所有点；这些事实不能确定reset角度、摆起或保持机制、跨seed稳定性，不能把跨点变化仅归因于模型参数，也不据阶段差异调参、停训或重跑。

10:11:06另读取实际门控JSONL，baseline各至90万18个区间、PPD-KL各至80万16个区间，共68个已完成区间；此前60区间的原始诊断、记录和行hash精确保留。所查FIELDS及其他数值标量有限，invalid_D_frac和match_invalid_count为0，active_weight_available为1，位置数1024，区间更新均值与快照更新计数一致。baseline全位置实际权重为1、超额减弱为0。Dt成功收据匹配字段按实现固定为0，门控合法性检查使用实际诊断字段，不仅依赖该收据字段。

同预算75—80万区间，每条12500次更新：

|seed|组|Mean Dt|Dt精确零比例|free-nats活跃比例|实际活跃权重均值|超额rep KL损失值减弱|
|---:|---|---:|---:|---:|---:|---:|
|2|baseline|0.058363|13.78%|8.34%|1.0000|0.00%|
|2|PPD-KL|0.078488|5.92%|8.95%|0.8779|11.37%|
|3|baseline|0.152580|15.55%|12.11%|1.0000|0.00%|
|3|PPD-KL|0.006507|1.16%|4.41%|0.9296|7.17%|

PPD-KL seed2/3的全位置权重均值为0.9218852/0.9780357。活跃权重是逐更新活跃位置均值再取更新均值；超额损失减弱仍为excess_reduction_mean/(weighted_excess_mean+excess_reduction_mean)，是区间累计均值之比，不是总梯度变化或回报差异的因果解释，也不能据此裁定强度足够/不足或选择alpha。所查区间均值不排除有限的瞬时梯度尖峰；全收据连续性、正式c和完整训练replay仍须终点检查。本次没有新增checkpoint参数解码或评价逐transition reward重放，既有八份参数检查范围保持。

10:17:10 sv3实际核查两监督器及四agent身份/命令匹配：baseline seed2/3最新更新开始动作938000/938276，PPD-KL811956/811988，末条invalid0。GPU1/4/5/6各17个本批C/G进程，GPU7无进程，/data可用266.22GB。四条仍运行、终点0/4；无新增训练失败或重跑。实查快照monitor/20261104T101751214.json。

|相对并行输出目录的文件|SHA256|
|---|---|
|collect-observed-800k.py|2f77266745218a840d1857299392d8cbb9c4a3d9fd4293b9aff5b5af3ce2d2a5|
|monitor/partial-return-800k-20261104.json|1e657226f59ec29d3af4a7b9290047ddd71aab585595a54900e3ab0ede4d49a9|
|monitor/observed-record-consistency-800k.json|ff1459bb2a89d47ab231abe9670631e3cd177e86f9095cc5d47a0766f4fbaae1|
|collect-gates-800k.py|719ce19eb9e932568242a53fa9b9d6ac08a371cfe5eaee83aff73fbe570166e5|
|monitor/gate-health-prefix-800k-20261104.json|12147b37fbac318ac995c66f422fe12a633216755bc762ec2ffa68aeae2315c4|
|monitor/gate-prefix-consistency-800k.json|c5c714a69ed686edb0341a751450c6b8c46816a2cbb84d8062e18c609aee6bd8|
|partial-return-250k.py|76f4f9a45d2a57ac90bf433ce4a07c23537f5b63f6ce5322ac0602dff41172fb|
|inspect-gate-prefix.py|900b11dc63b12a95e98842137f393868d7c4c9fe9e0d48a2351fcafaa1fc0eea|

文件位于analysis/outputs/pendulum-seeds23-parallel-20261104/。采集命令已执行成功；脚本拒绝覆盖旧证据，完整SSH命令、源hash、stdout/stderr和逐项一致性保存在上述文件：

```powershell
& 'D:/program/PD/dmr3/analysis/outputs/dmc16-preparation-20261022/venv/Scripts/python.exe' 'D:/program/PD/dmr3/analysis/outputs/pendulum-seeds23-parallel-20261104/collect-observed-800k.py'
& 'D:/program/PD/dmr3/analysis/outputs/dmc16-preparation-20261022/venv/Scripts/python.exe' 'D:/program/PD/dmr3/analysis/outputs/pendulum-seeds23-parallel-20261104/collect-gates-800k.py'
```

本阶段仅本机只读辅助脚本、小证据与文档，无训练源码/科学配置/服务器Git同步/活动checkout/W&B/c写入变更。实际训练及阶段冻结指标SHA c5f86ca1a16648f57db3c3abfa2584ac560495ca；控制器/服务器验收/终点审计b7f42d46d16d4eda5cff5a798ec7b60cb27a3790；准备终点分析及36项测试5ccbd8c533107823fba558dd092cac41655efd8d，均保持。文档变更不重复既有测试。继续当前chat完成四条终点collect/analyze/replay、真实图形验看和校正解释，目标active，不自动建chat或发消息。
