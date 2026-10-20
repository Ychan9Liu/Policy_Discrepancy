# 05 · Hopper M2 v1 低回报专项排查

**2026-10-20 03修复回执**：诊断语义版本2已正确记录坐标/因子/整向量比例，CPU冻结输入的损失、梯度及RNG与修复前精确一致；原16条正式指标不变。旧日志保留，真实整向量比例仍不可恢复。详见 `docs/CODE_MODE_DIAGNOSTICS_REPAIR.md`。下文是05发现时的历史证据及交接，不将诊断错误或修复当作hopper失败因果解释；下一科学复验由02与用户确认。

日期：2026-10-20，Asia/Shanghai。范围：已完成的hopper_hop四组seed0，原训练SHA `48d71f1541721208322442f5a1d5977124f3e888`。本轮只读原始训练产物、CPU统计和源码核对；没有训练、环境步进、模型推理、优化更新、GPU作业、调参或协议变更。

## 判断

**已经定位的是失败环节，而不是唯一根因：Dt在早期没有积累起持续的真实正奖励，模型想象奖励/优势信号长期偏小，策略保持高噪声、低回报行为。** 它不是仅评价出错，也不是先取得高回报再在后期崩溃。单seed以及首次训练前就存在的少量图像差异，使现有证据无法唯一裁决“Dt错误选择关键位置”与“输入微扰/后续探索路径分叉导致未发现奖励”。

此外确认一处**真实的诊断代码错误**：`mode_vector_diff_frac`实际是因子mode不一致率，`mode_factor_diff_frac`实际是one-hot坐标不一致率。整条latent向量的mode不一致率没有正确记录。它不参与训练损失，不能作为本次低回报的直接原因，但限制了对mode漏检和actor敏感性的解释，须交03修复与回归。本轮未改训练源码。

原16条预算/评价/c/指标验收与正式结果仍有效。**“产物完整、程序跑完”不等于“方法没有问题”，也不等于本轮已排除所有实现错误。** 下文区分已验证事实、解释推断和无法回答的条件。

## 新增证据来源及复算范围

本机分析/测试版本：`2427710a7a89f2a5c90a08634a74a7a11ee87139`；最终交付SHA在本轮HANDOFF/DELIVERY与回复中，后续交付相比此版本仅文档。本机原HEAD `210de9ecbd89c87f6818c29909ea581196002ce8`；原RESEARCH/STATUS修改和未跟踪docs/proposal保留，不混入提交。

服务器sv2仓库只读检查为干净训练SHA48d71f1。读取四条原始目录：

| 组 | 原始目录 | 原物理GPU |
| --- | --- | ---: |
| baseline | `/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-seed0-20261009/dmc_hopper_hop/logging-attempt01` | 0 |
| Dt | `/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-dt-seed0-20261009/dmc_hopper_hop/dt-attempt01` | 4 |
| constant | `/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-constant-seed0-20261010/dmc_hopper_hop/constant-attempt01` | 0 |
| shuffle | `/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-shuffle-seed0-20261010/dmc_hopper_hop/shuffle-attempt01` | 4 |

CPU NumPy/pickle只读了每组0步可信agent.pkl（各512902434 bytes，未复制大文件）、1280个replay chunk的reward/action/is_first/is_last/is_terminal数组、首次训练前图像，以及少量原始图像抽帧。总计5120个chunk，4×1001008条转移，其中训练动作4000000、reset4032。按chunk UUID/successor重建每组16条完整worker链，再按协议Driver的选worker规则重建动作顺序，检查总动作/reset数及链消费无剩余；链按首次创建时间对应callback顺序。每组992个完整训练episode，另有16个未完成尾部episode；**尾部不冒充完整episode，不改变正式评价指标。** 正奖励步/动作阶段统计包含所有实际动作；完整episode均值只用992条。

逐数组内容SHA256、原NPZ member CRC/长度、checkpoint文件/294数组hash、原目录与采集时刻在 `analysis/outputs/hopper-diagnosis-20261020-inspection01/remote_inspection.json`。逐字段小日志hash与10月12日输入镜像一致。采集时间以该文件created/completed_utc为准，本机CPU分析在analysis-final记录实际Git版本。

原始采集脚本保留为inspection01/remote_source.py，SHA见本轮DELIVERY；其执行时分析代码尚未提交。已提交采集器相比执行源仅增加manifest/非fixture断言、删除未执行死代码并增加独立抽帧入口，不改变数组读取/求和。不要将后续Git版本伪称为首次采集时已提交版本。

## 1. 没有证据支持配置错、初始化错或仅评价错

- 四组冻结配置除已声明mode/c/frozen_c_file/logdir外精确相同；训练版本、环境/随机流映射一致，正式c来源为本任务baseline的 `[100000,300000)`成功更新S/N，c=`0.8169140208772256`。Dt不依赖c，不能将其失败归因于constant校准。
- **0步全部294个参数/优化器/归一化状态数组逐位相同**，计数器均为0。组合hash为 `14001dc40a48070221b3bd94a6698baf664ce374becce24263a44b03f4a91194`。本次补齐了10月12日报告中未独立读取0步参数payload的证据缺口，但仅针对hopper四组。
- 首次优化更新开始于动作2032。此前四组2032个非reset动作及16条reset的stored action、reward和边界标志完全一致；相同6个正奖励步、累计真实reward `0.1687391723971814`。不是Dt抽到了不同初始seed或一开始就少了这6步奖励。
- 图像并不逐位相同：相对baseline，Dt/constant/shuffle各有25/32/30个uint8坐标相差1，比较总体为25165824个坐标；Dt最早差异出现在首条reset图像。**输入已存在组间差异，不能把后续全部分歧唯一归因于gate。** 成功的两个overlay对照也有差异，不能反过来把像素差异单独定为Dt失败原因。与既有EGL非逐位确定性证据相符，但本轮未取得物理状态轨迹作进一步归因。
- 原正式评价曲线仍是AUC B/D/C/S=`253.83/0.86/241.00/223.73`，末段=`316.75/1.01/314.92/300.43`。原始训练replay独立统计也显示Dt低回报，排除“只有评价环境/return汇总错，实际已经学会”这一解释。

## 2. 真正明显分叉发生在早期奖励积累

全训练数据的描述性统计，**这些是训练replay证据，不是新增正式主/次指标**：

| 组 | 992个完整训练episode平均return | 零return episode数 | 全百万动作正奖励步数 | 全百万动作零奖励比例 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 255.5581 | 49 | 372505 | 62.7495% |
| Dt | 0.7842 | 799 | 2974 | 99.7026% |
| constant | 242.7145 | 88 | 335197 | 66.4803% |
| shuffle | 222.7507 | 133 | 349066 | 65.0934% |

所有四组replay完整episode均1000个非reset动作，is_terminal总数0；低回报不来自异常提前终止。Dt 799/992≈80.54%的完整训练episode零return，说明训练分布本身长期缺少成功行为。不能把100万步执行完成误读为学到了行为。

5万—10万动作这一区间尤其清楚：

| 量 | baseline | Dt | 含义 |
| --- | ---: | ---: | --- |
| 真实正奖励动作数 | 9407 | 52 | 由原replay逐reward统计 |
| 真实reward累计和 | 2805.7952 | 2.2086 | 完全不同的奖励经验积累 |
| 模型想象reward均值 | 0.0133483 | 0.00000693362 | 模型内部量，不是真实reward |
| 想象归一化advantage绝对均值 | 0.0249394 | 0.0000263080 | 策略学习的代理信号相差约948倍 |
| 想象状态上的Gaussian动作熵 | 2.28246 | 5.66790 | Dt接近4维std≤1的最大值5.67575 |
| reward prediction loss | 0.0237341 | 0.000143512 | Dt更低不能解释为奖励模型更好 |

更细的5k动作区间表见early_5k.json：baseline在45k—50k获得累计9.08376真实reward，50k—55k增至60.08255，随后逐渐持续积累；Dt在45k—60k均为0，整个50k—100k仍只有2.20859。constant与shuffle前100k也较慢，但后续成功建立了高回报轨迹。**能定位到“早期探索到的稀少奖励没有被持续放大”，不能从5k/50k聚合数据定位某一次梯度更新为唯一转折点。**

低reward loss与低image loss不代表任务表示正确：大量reward=0时，“预测近0”本身就容易取得低loss。模型想象reward低与真实经验稀少一致；没有留出同分布预测校验，无法进一步断言head是忠实拟合还是漏掉稀少奖励。

源码 `imag_loss` 使用模型预测reward、归一化advantage和熵正则；reward量很小时策略任务信号也可很小，retnorm的scale下界1不会自动将它放大，advnorm为none。**“稀少正奖励→弱模型奖励/优势→高噪声策略→仍少正奖励”是符合现象的反馈解释，尚非经干预确认的唯一因果链。** 不能由两个标量大小直接宣布熵项在梯度向量上占主导，不据此自动改熵系数或归一化。

![真实奖励与模型/策略诊断](../analysis/outputs/hopper-diagnosis-20261020-final/learning_failure_diagnostics.png)

## 3. 行为证据：并非停止输出动作，抽查出现倒地/翻转

从原replay抽取chain0的第3、6、61个训练episode，各取0/100/250/500/750/1000步图像，没有创建环境或重新渲染。第61个episode中Dt return=0，抽帧可见倒地/翻转姿态；baseline/constant/shuffle对应episode return约317.89/283.06/304.07。四组该episode初始画面相似，随后行为不同。

![原始replay：训练episode61，四组逐行](../analysis/outputs/hopper-diagnosis-20261020-final/replay_episode_61.png)

这是特定chain/episode的可视样本，不拿6帧代表全部992条episode，也没有从图像估计精确躯干高度或速度。全局低回报由上节完整replay统计支持。原replay action非零且有限，Dt末50k的逐维std约1.23—1.25、原采样动作坐标约45.05%超出或等于[-1,1]边界，环境包装随后裁剪；这不是执行动作的std或熵，也不是动作越界工程故障。replay的action是本步obs之后生成、供下一步执行的输出，不能把它误当同一行reward的已执行动作；这里仅描述输出分布。它说明仍在持续输出强噪声动作，而非完全没有动作/更新。

检查实际安装的 `dm_control/suite/hopper.py`，hop reward为standing×hopping：standing要求躯干相对脚的高度落在[0.6,2]，hopping随向前速度达到2增大。倒地或没有正向速度均可能产生0奖励；没有原physics trace，不能把所有0reward都标签成“摔倒”，也不能声称reward函数错误。

## 4. gate作用没有消失，但均值不能定位关键错误

原free_nats、loss顺序和梯度隔离代码符合已声明公式：w=sg(1/(1+20D))，只乘free-nats之后的rep loss，原rep scale0.1只乘一次。复核80个训练区间按实际loss系数重组的总loss与opt/loss一致，记录的loss/gradient/update等标量均有限，未见“rep乘到了其他分支”、重复系数或数值爆炸的证据。已有完整agent fixture验证保留为工程证据，不冒充本次训练中每次梯度的独立验证。

Dt在前50k的活跃权重均值0.99524、超额KL减少约0.522%；直接减少的scaled rep loss标量约0.001110，只占同状态重组总loss约0.00941%。50k—100k活跃权重0.97261，全程0.95156、全程超额KL减少5.81%。**平均作用很小也可在少数位置或后续非线性学习中造成大差异；小loss比例不等于小梯度影响。** 不将“门控作用较弱”解释为可排除gate的因果作用。

constant全程超额KL减少18.31%、shuffle15.26%却能学会，这反对仅凭全程平均减弱幅度解释Dt失败；它们没有匹配Dt实际轨迹上相同位置的梯度，因此仍不能排除“Dt恰在关键少数时刻减弱了不该减弱的rep约束”。全局shuffle取自身batch、模型轨迹也不同，不能作为本次Dt逐点反事实。

另一个容易误读之处：**低D不等于actor完全不读取posterior信息。** 末50k中Dt的两路动作均值绝对差约0.03841，baseline约0.03657，前者并未更小；但Dt两路std差约0.000336，baseline约0.01439，且Dt保持高std。这是不同轨迹上的粗统计，不能据此宣布同一输入下两者同等敏感。Gaussian KL受均值差、方差与尺度共同影响；缺少逐位置分解，不能定量归因。不能把当前低D自动解释为“latent都相同”或“actor完全不敏感”，也不能只据低D上调alpha。

## 5. 确认的诊断错误：mode两个字段的归约轴/命名不对应

源码 `dreamerv3/rep_probe.py:122,161–162` 中qmode/pmode为 `[B,T,S,C]` 的one-hot：

```python
mode_diff = any(qmode != pmode, axis=-1)  # [B,T,S]
mode_factor_diff_frac = mean(qmode != pmode)  # one-hot坐标不同率
mode_vector_diff_frac = mean(mode_diff)       # 因子mode不同率
```

期望的因子mode不同率应先对C做any再mean；期望的整个latent向量不同率须再对S做any后mean。当前C=32，one-hot两个不同类别恰有2个坐标不同，因此日志后者=前者×16，本次4组×20区间全部精确符合该关系。

反例fixture：2个时刻×2因子×32类，仅一个时刻的一个因子mode不同，当前两日志值为0.015625和0.25；真实因子不同率0.25、整向量不同率0.5。新增单元测试通过，证明这几个概念不能互换。**不能用日志0.25声称“75%的整条latent向量mode相同”。** 也不能从聚合因子比例唯一恢复整向量比例。

影响范围：诊断解释与mode漏检证据缺失；不改变D、w、S/N、rep训练loss、正式c或原始return。当前无证据将它定为hopper失败原因。建议03在独立确定版本修复命名/归约、保留旧字段兼容说明和回归，不覆写旧运行日志；这是首选下一项可立即开展的工程补证，不自动将代码修复授权扩大到训练。

## 可排除与尚不能裁决

| 解释 | 当前判断 | 依据/缺口 |
| --- | --- | --- |
| 预算不足、漏评价、统计错 | 未发现；不支持作为这次失败解释 | 原完整审计与本轮训练replay再求和 |
| 初始化/训练seed配错 | 不支持 | 294初始数组、2032动作/奖励、配置/seed一致 |
| 只是评价差，训练实际学会 | 排除这一具体解释 | 799/992训练episode零回报，原图像抽查 |
| 后期突然崩溃、梯度爆炸 | 不符合日志现象 | 从早期就低回报，有限梯度/持续更新；无逐更新峰值日志，不能声称从未有瞬时尖峰 |
| 平均KL减弱过大 | 不能作为唯一标量解释 | 减弱更多的两个对照成功；关键位置作用仍可能有害 |
| gate完全没开 | 不成立 | 活跃位置实际权重<1、非零weighted excess减少 |
| D很小必然说明mode一样/actor无响应 | 不成立 | 归约诊断错误、均值差与Gaussian方差尺度关系 |
| Dt错误保护/错过关键时刻，损害学习 | 合理待验证假设 | 无逐位置任务价值/rep梯度反事实，无匹配初始图像的因果对照 |
| EGL微扰/探索路径导致单次失败 | 合理待验证假设 | 原prefill像素差异直接存在，但成功对照也有；单seed无法估计失败概率 |

不能把“稀疏奖励反馈失败”写成已证明的算法根因；它是当前最具体、证据最充分的失败链路描述。现在应补的不是立即改alpha，而是区分错误选择与单次路径放大。

## 交付、验证、下一步

代码：`analysis/inspect_hopper_v1.py`、`analysis/diagnose_hopper_v1.py`；20项统计/诊断测试通过。实际分析与测试SHA `2427710a7a89f2a5c90a08634a74a7a11ee87139`，最终文档提交只增加本报告、原分析补注和STATUS/README索引；最终完整SHA见HANDOFF/回复。本轮仅本机提交，未推送或同步服务器源码，服务器保持干净训练版本。没有实验变更或重跑，原16条科学参数和指标不变。

输入：`analysis/outputs/hopper-diagnosis-20261020-inspection01/`（JSON/首次采集源/SSH log）；正式抽帧输入`hopper-diagnosis-20261020-frames-final/`；正式派生结果`hopper-diagnosis-20261020-final/`（训练阶段真实/模型/策略表、early_5k、mode反例、4张PNG/1张SVG、validation和哈希）。attempt01为提交前成功分析，保留，不是额外训练尝试。完整命令、输入/输出哈希和交接记录放本轮final/COMMANDS.md、DELIVERY.json、HANDOFF.md。所有大checkpoint/replay仍在sv2，Git只提交代码和文档。

复算：

```powershell
cd D:\program\PD\dmr3
$py = 'C:/Users/97370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$env:PYTHONPATH = 'D:/program/PD/dmr3/analysis/outputs/m2-v1-seed0-20261012/plot_deps'
& $py -m analysis.diagnose_hopper_v1 --inspection analysis/outputs/hopper-diagnosis-20261020-inspection01/remote_inspection.json --mirror analysis/outputs/m2-v1-seed0-20261012/inputs --frames analysis/outputs/hopper-diagnosis-20261020-frames-final/replay_frames.npz --output analysis/outputs/NEW_HOPPER_RECALC
& $py -m unittest tests.test_return_metrics tests.test_m2_analysis -v
# 如需从原始服务器重新收集：CPU只读，使用不存在的目录
& $py -m analysis.inspect_hopper_v1 --frames --output analysis/outputs/NEW_HOPPER_INPUT
```

本轮无训练或GPU使用，也未查当前GPU占用，不沿用10月12日卡状态作为10月20日状态。采集器不调用agent/环境/JAX执行；受信任项目pickle只加载原数组。缺少评价逐reward轨迹的限制不变；本轮独立求和的是训练replay。

**首选下一任务：继续现有03做mode诊断修复与冻结输入补证设计。** 可立即做源码/测试和已有产物的CPU只读核查；完成标准为正确的因子/整向量统计、旧日志语义映射、确认训练目标/RNG未改变，并明确哪些因果问题仍需02制定协议。多seed复验、同输入反事实或新诊断训练的具体范围/seed/预算/资源仍依赖用户与02/03决定，不自动执行；已确认v1无需重开。若03只修日志，不能把修复后新数据冒充原16条记录。

可直接复制给03（最终交付SHA取HANDOFF/回复）：

```text
继续现有03｜代码与测试，项目D:\program\PD\dmr3。
交接版本取本轮HANDOFF/最终回复完整SHA；实际分析/测试2427710a7a89f2a5c90a08634a74a7a11ee87139；原16条训练48d71f1541721208322442f5a1d5977124f3e888。
必读AGENTS、docs/CHATS.md、STATUS.md、ANALYSIS_HOPPER_M2_V1.md、ANALYSIS_M2_V1.md、CODE_PHASE2.md、CODE_SIZE50M_REPAIR.md、METHOD_SPEC.md、EXPERIMENTS.md及analysis/README.md。
原16条有效，科学协议与指标不变。hopper Dt训练replay也很低：992完整episode中799零return，99.7026%动作reward=0；5万—10万仅52正奖励步，baseline9407。四组0步294数组和首次2032动作/奖励相同，原图像有25/32/30个1级坐标差异。根因尚不能在Dt关键位置作用与路径微扰之间裁决。
已确认诊断错误：rep_probe.py的mode_vector_diff_frac其实是因子mode不同率，mode_factor_diff_frac是one-hot坐标不同率；整向量未正确记录。请在用户授权范围内修复/制定最小修复，提供因子与整向量反例和兼容字段说明，保留旧日志。不得把此日志错误说成训练失败根因。
允许源码核查、CPU只读已有产物、必要诊断字段代码/测试；不启动训练/环境仿真/GPU作业，不改变D/w/free-nats/梯度/随机流/alpha/c/seed/预算/评价/冻结v1，不覆写原产物。新的因果训练或多seed协议交02和用户确认。
证据在analysis/outputs/hopper-diagnosis-20261020-inspection01、-frames-final、-final。输出代码/测试及独立outputs，更新必要工程报告/STATUS，明确未解决因果条件和交02的科学问题。
完成按AGENTS给最终/实际测试完整SHA、差异/同步范围、验收限制、下一步及可复制prompt，不自动建chat或发消息。
```
