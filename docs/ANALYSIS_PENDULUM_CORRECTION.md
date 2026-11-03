# Pendulum / Hopper 实验校正：只读诊断与下一步方案

**后续授权更新（2026-11-03）**：用户已明确确认“按此方案准备并启动四条复验”，本批启动限制已解除。执行配置、严格入口、sv3 GPU1/4同卡串行安排和后续实际状态见`docs/PENDULUM_SEEDS23_RUNS.md`。下文的“待确认/先不启动”记录保留为本次诊断交付时的历史，不再阻止已授权工作；诊断结果与科学解释不变。

日期：2026-11-03。当前完成的是校正工作的诊断阶段，没有启动新训练、GPU探针或等待空闲自动启动的队列。用户最新资源指令是sv3物理GPU **1/4/5/6/7，先不启动**；继续执行校正工作的目标未被解释为自动撤销这条限制。下面的新seed方案为待确认建议，不是已批准科学协议。

## 本次判断

pendulum baseline seed0确实没有建立持续的高回报控制。完整训练replay也显示失败，不能归为仅评价波动：992个完整episode中927个回报为零，而PPD-KL为49个。两组首次获得奖励的时间相同，但5万—10万动作的正奖励次数为53对6775，随后模型想象奖励、优势与策略噪声出现明显分化。

进一步按第一个非reset动作的奖励分层后，差异更具体：两组各40个episode起步就在奖励区间，baseline平均return902.25；其余952个起步无奖励的episode，baseline只有25个后来进入过奖励区间，PPD-KL有903个。**数据更符合baseline学到部分保持直立，却未普遍学会摆起**，而非完全不会控制。这里“起步”严格指第一个动作之后，没有从图像反算精确初始角度。

伴随现象是稀少的成功经验、长期高噪声策略及较弱优势信号。这仍是失败链路，**不是已证明的唯一根因**。没有发现动作预算、reset排除、reward累计、初始化参数、关键配置或损失系数错误；尚不能裁决奖励模型的状态辨别能力、策略优化、门控或输入微扰分别贡献了多少。

新发现：两组294个初始保存数组完全相同，但在**任何优化器更新之前**，初始图像和部分动作已经不同。因此不能声称这是一对“完全相同输入、只有gate变化”的因果对照。结果仍可作为原协议下的探索性seed0性能比较，不能据此删除失败baseline，也不能把全部差值归给位置选择机制。

hopper已有seed1/2/3补测，不需要因未查看旧结果而再跑一套相同实验。三条PPD-KL均未重复seed0近零失败，其中两条低于baseline；旧mode诊断命名错误已修复并有损失/梯度/RNG不变回归，不能再次把该日志问题说成训练故障。

## 来源及复核范围

|组|实际训练SHA|原服务器/物理GPU|原目录（runs根下）|
|---|---|---|---|
|baseline|dbd28b6703d51d61e5d3485dbfea6c413075ea85|sv2 / 1|formal-m2-dmc16-remaining8-serial-20261024-sv2/dmc_pendulum_swingup/logging-attempt01|
|PPD-KL|d8997f64a301bfdcfe08758faeae8957a7395766|sv3 / 4|formal-m2-dmc16-missing-dt-20261030-sv3/dmc_pendulum_swingup/dt-attempt02|

原根为`/data/Policy_Discrepancy/runs`。运行ID、配置、旧验收状态、文件hash和完整来源在本轮`inspection/{baseline,dt}.json`；W&B和此前收据/固定快照审计见`docs/ANALYSIS_DMC15_V1.md`及其run_manifest。baseline旧索引仍为run_audited_pending_group_and_WB，Dt为accepted，不在本轮改写旧批次收尾状态。

本轮从服务器只读所有replay chunk的reward/action/reset/episode边界数组；每条16条保存链、1001008条记录，其中1000000训练动作、1008个reset、992个完整1000动作episode及16条未完成尾段，terminal计数0。完整episode与尾段的动作数、reward总和互相闭合，reset reward全0；所有reward及动作有限。逐chunk保存所读数组的SHA256、ZIP成员CRC/字节数和后继关系，没有宣称对全部大图像/replay压缩文件做了完整文件hash。

按首chunk创建时间恢复worker顺序，再按该训练SHA的Driver顺序、5万动作边界重建全局动作位置；检查无漏chunk、无重复访问、无多余/缺失行。奖励来自当前观测，动作来自该观测产生的下一步输出，本轮动作统计使用每条链前一行action对齐真实transition，不将同一行action误作当前reward的原因。

完整checkpoint留服务器，只在CPU反序列化受信任的0步checkpoint，读取数组/计数/hash；未执行agent、环境、JAX或渲染。只复制了每环境首128行的图像/标量用于核对，共2032个非reset动作，不复制完整checkpoint/replay。小证据文件hash与11月2日采集一致。

再次从21×10完整episode的return_raw复算均值、AUC与末段。没有评价逐transition reward轨迹，**独立核验内部reward求和的是训练replay**；评价内部求和语义继续复用既有执行器审计，不将训练replay当评价录像或固定参数结果。

## 原始奖励与学习过程

|指标|baseline|PPD-KL|
|---|---:|---:|
|评价AUC / 100万|32.5325|691.4650|
|末段五点评价均值|101.48|787.80|
|完整训练episode零return数 / 992|927|49|
|完整训练episode平均return|53.7409|700.1099|
|全部百万动作正奖励次数|55198|699642|
|正奖励动作比例|5.5198%|69.9642%|
|前10万动作正奖励次数|128|6865|
|首次正奖励的训练动作位置|16013|16013|

两台机器安装的pendulum.py、pendulum.xml与rewards.py内容hash一致，源码保存于inspection JSON。pendulum奖励为杆偏离竖直向上不超过8度时1，否则0（tolerance的margin默认为0）；没有接近目标的连续塑形奖励，也没有单独的速度或控制惩罚。XML的步长0.02秒、时限20秒，原repeat1对应1000动作episode。**全部训练原始reward确实只有0和1**，与这份实现一致。此项是安装源码与产物核查，未重跑环境来还原每步角度。

两组的完整episode再按第一非reset动作reward分层，分层不涉及回报高低阈值或择样：

|训练episode类别|baseline|PPD-KL|
|---|---:|---:|
|第一动作reward=1的episode数|40|40|
|上述episode平均return|902.25|952.95|
|第一动作reward=0的episode数|952|952|
|上述episode之后至少获得一次奖励的数量|25（2.63%）|903（94.85%）|
|上述952个episode的平均return|18.09|689.49|

baseline不是在所有起点都无法保持：第一动作已获奖励时return中位数1000。主要差距在从奖励区间外进入并持续获得奖励。第一动作reward=0并不保证离直立很远，后来一次获奖也不等于完整掌握摆起；因此不将25/952或903/952命名为任务成功率，也不推断准确摆起角度。这是同一训练seed的replay轨迹描述，不能当独立训练重复。原始逐episode首次/末次正奖励步、连续正奖励段长度保存在`inspection-entry`，汇总在`entry-final/summary.json`。

|训练动作区间|baseline正奖励次数|PPD-KL正奖励次数|baseline想象奖励|PPD-KL想象奖励|baseline策略熵|PPD-KL策略熵|
|---|---:|---:|---:|---:|---:|---:|
|0—50000|75|90|0.000485|0.000388|1.4104|1.4131|
|50000—100000|53|6775|0.001146|0.019025|1.4101|0.8842|
|100000—150000|3358|33764|0.011762|0.184843|1.4013|0.5945|
|150000—200000|625|37215|0.023825|0.344760|1.4085|0.4536|

真实奖励按动作累计，想象奖励/策略熵是对应50k区间的训练更新均值，两种采样分布不同，不可把两者相除当奖励模型误差。PPD-KL在5万后逐步出现更多正奖励，到9万—9.5万区间有3214个；baseline前10万只有少数区间获得奖励。

baseline的想象Gaussian策略熵一直约1.40—1.415，接近该一维、maxstd=1策略的上界1.41894；PPD-KL在5万—10万降至0.884。baseline到15万—20万的优势绝对均值为0.000289，PPD-KL为0.011165；55万—60万为0.000120对0.006349。这支持baseline长期缺乏有效策略改进信号的描述，**高熵不等于有效探索**，但不能从熵直接推出应该降低熵系数。

baseline并非全程从未碰到奖励，也不是模型预测始终为零：后期想象奖励约0.05；它仍不能稳定完成任务。末期baseline reward loss约0.036、PPD-KL约0.464，不应据此说baseline模型更准，因为两组训练分布、正样本比例与损失基线不同。现有均值无法回答“奖励头对同一批正/负样本是否正确辨别”，也无法定位reward head、latent或actor中的因果故障。

40条区间日志均有限，按原loss_scales重建总loss，最大绝对误差约5.6e-8，使用原分析容差而非为通过放宽。baseline未出现日志可见的梯度爆炸。PPD-KL在15万—20万的裁剪前gradient norm区间均值约2512（50万—55万约138），需保留这个异常；其对应update RMS约8.12e-6、成功更新连续，之后仍维持高回报。不能写成“两组从未出现梯度尖峰”，也不能单凭该值宣告执行失败。日志没有逐更新峰值、分模块梯度或奖励误差切片。

## 门控及首次更新前的差异

|更新加权诊断|baseline|PPD-KL|
|---|---:|---:|
|Mean D|0.002788|0.167724|
|D=0位置比例|3.3654%|3.3810%|
|raw rep KL > free nats的位置比例|6.6168%|15.7041%|
|活跃位置实际权重均值|1（日志舍入约0.999999993）|0.895541|
|累计超额KL减弱比例|0|9.9964%|

baseline正式c为0.9839062759074624，来源是自身[100000,300000)成功更新有效位置S/N；本轮沿用已验证冻结来源，未运行constant。

PPD-KL前5万的活跃权重0.997705、超额KL减弱约0.0883%；5万—10万为0.977530、2.5486%。门控早期较弱，后期确有作用，既不能把这次成功解释为强门控从第一步就控制了过程，也不能说gate没开。平均权重和超额loss减弱都不是梯度范数减弱比例。

首次更新收据两组均update_id=0、start_action=2032。0步294数组和计数、训练/评价/replay seed派生、包版本及依赖锁一致。10个关键文件（agent/rep_probe/RSSM/runner/Driver/wrappers/DMC/optimizer/scope/依赖锁）在两个实际训练SHA之间逐字节一致；完整SHA仍分别记录，不说整仓版本相同。

然而16环境第一次reset的196608个图像坐标中有314个不同，最大差1灰度级；此时2/16个采样动作不同，最大绝对差0.0006103515625。整段128行前缀中1638/2048个动作不同，最大差0.0041351318359375；图像差异随轨迹推进扩大。两组此时尚未优化，故这些初始差异不能由“gate改变训练梯度”解释。结果与跨设备渲染/数值路径扰动相容，但本轮没有单独隔离渲染与数值源，也不能证明微扰就是最终输赢的原因。

同一卡串行可以减少硬件条件差异，仍不保证图像/轨迹逐位一致。严格判断位置选择价值还需要constant/shuffle或经确认的固定输入反事实；当前两组单seed不具备该因果识别能力，10个评价episode不是10个独立训练seed。

## Hopper已有复核与校正顺序

复用`docs/ANALYSIS_LOW_RETURN_FOLLOWUP.md`的6条已accepted原始评价复算，不在本轮重复全量扫描hopper replay或声称新验收。历史seed0与补seed1/2/3分列，不混成一个同批次主分析。

|hopper训练seed|PPD-KL−baseline AUC|PPD-KL−baseline末段|判断|
|---|---:|---:|---|
|0，历史批次|约−252.97|约−315.74|PPD-KL近零学习失败|
|1，补测|+58.13|+141.45|两指标较高|
|2，补测|−9.36|−94.87|两指标较低，已学到非零回报|
|3，补测|−54.30|−59.58|两指标较低，已学到非零回报|

补测训练SHA743089f40662b031a5cf2b6bf988decb52ba0b28。原mode字段修复的代码/CPU证据在`docs/CODE_MODE_DIAGNOSTICS_REPAIR.md`：冻结输入四模式loss/梯度及RNG逐位不变，属于工程回归，不是实际任务有效性证明。无需再将已修复的日志命名问题作为待修训练错误。

推荐顺序：

1. **已完成**：pendulum全训练reward/episode/动作对齐、初始化及早期输入、日志/门控、原始评价复算；复核hopper既有补seed和历史工程修复边界。
2. **下一首选、待用户确认**：原科学配置补pendulum seed2/3各baseline与PPD-KL，共4条；seed1复用已安排的vision20计划，不重复建任务。保留seed0失败，所有新seed按预声明范围报告，不按return决定删、续跑或补seed。用途是观察失败是否重复，不将四条补测当位置机制的因果检验。
3. **视结果再定、未批准**：如果baseline仍系统性失败，优先区分摆起与保持阶段，并设计同一批真实正/负样本的奖励预测与条件辨别诊断；若关注gate作用，再由02确认constant/shuffle或固定输入反事实。任何新初态分层评价须另立诊断协议，不替换正式评价分布。没有足够日志依据直接改alpha、free nats、熵系数、reward或训练预算。

## 待确认四条复验的具体建议

当前代码只支持原seed0、hopper限定seed1/2/3及vision20限定seed1，**尚无pendulum seed2/3正式入口**。以下是待确认范围；若获授权，工程实现应新增严格独立scope及回归，不放宽旧scope、不冒用hopper入口。

2026-11-03追加CPU预检：在ca0a62d7c0df1b86fea8284e18f3afb548cf2cfb上实际执行16个范围案例，原pendulum seed0和vision20 seed1允许，dmc16、vision20 seed1和hopper三个入口下pendulum seed2/3按预期拒绝；四份候选配置与原组配置除seed和未分配目录外完全一致。证据为`analysis/outputs/pendulum-correction-preflight-20261103/`，含完整候选配置/hash、preflight.json和可重新执行的verify_preflight.py。它们保留旧script以显式暴露未实现条件，目录含UNASSIGNED、authorized_to_train=false、launch_argv=null，**不是可运行计划或授权**。该预检不执行agent/env/GPU，不改训练入口；下一步仍待上一条范围/启动问题的明确回复。

|服务器/物理GPU|顺序|训练seed|任务/组|状态|
|---|---|---:|---|---|
|sv3 / 1|先baseline，完整验收后PPD-KL|2|pendulum_swingup，两条|建议，未启动/未排队|
|sv3 / 4|先baseline，完整验收后PPD-KL|3|pendulum_swingup，两条|建议，未启动/未排队|
|sv3 / 5/6/7|保留空闲|—|当前没有额外必要正式任务|未占用|

选择同seed同卡串行以减少硬件条件差异；使用已授权资源的子集，避免为了填满卡而增加没有定义的实验。若用户另定并行安排，应记录实际配置与限制。

建议关键配置完全沿用原pendulum：clean、视觉64×64 RGB、size50m、repeat1、每条100万训练动作含预填充不含reset/评价；batch16×64、context1、16环境、train_ratio256；baseline logging实际rep权重1，PPD-KL alpha20，free nats1，dyn/rep系数1/.1，ac_grads=false、reward_grad/repval_grad=true；原优化器/动作采样不变；21个固定快照、每点10完整episode、独立评价随机流；JSONL+W&B PD_1、prealloc=False、CUDA/bfloat16。每seed baseline自身[100000,300000)成功更新S/N冻结c，仅审计不新增constant；没有c或收据非法时停止报告，不填默认值。

新scope、目录、run ID、确定训练SHA、完整配置hash、CUDA/EGL绑定和真实短程工程验收均在获准后落实；不会从失败seed0 checkpoint续跑，不覆盖原尝试，不改变评价策略。工程通过仅表示可执行；正式每条须验证百万动作、连续成功更新、21×10及checkpoint/c证据，最后才比较原始return AUC和末段五点。

23:07:19北京时间再次只读核查sv3：物理GPU1/4/5/6/7均A100-SXM4-80GB、已用显存0MiB、逐卡XML及pmon无C/G进程；GPU0/2/3有其他作业，不干扰。/data余349467992064字节（349.47GB）。按现有60GB/条门槛，4条预留240GB后约余109.47GB，仍需启动前核查工程产物及系统余量。未删除、迁移旧产物或降低存储门槛。CUDA/EGL未实际测试，空闲快照不是预约。

23:07:20 sv1的vision20-seed1正式actual-RUN_INDEX尚不存在；该计划文档已安排pendulum seed1在sv1第二轮，并记录等待空间。它不能当作已完成/已启动结果，也不能据本chat的sv3授权自行迁移或重复执行。后续读取该批实际记录再纳入，不依赖聊天摘要。

## 交付、版本与复算

代码`analysis/diagnose_pendulum_v1.py`；4项手算replay边界/动作对齐/奖励进入时点测试及原指标/统计测试共24项通过。真实两条replay另通过完整计数、reward闭合、源hash、初始数组与实际前缀比较；fixture通过和真实产物复核分别记录，不混同。

本轮完整采集与首次分析SHA **82c7e3a82e6eb22ccee6feab92015d768702644a**；前缀量化及初次复算SHA **dc95e434b8271ecfacc46a4786f53d72c5c8763d**；奖励进入时点完整再采集/最终分析及24项测试SHA **e90e972ee1f795c157581eef40da8847336e6a53**。这些提交仅分析代码变化，不改训练源码；最终交付SHA见`analysis/outputs/pendulum-diagnosis-20261103/DELIVERY.json`及最终回复。该测试SHA之后仅新增报告/索引。只在本机提交，没有GitHub推送或服务器源码同步；原大产物留服务器，服务器只读CPU脚本经stdin执行。

证据根`analysis/outputs/pendulum-diagnosis-20261103/`：inspection及inspection-entry为两轮完整小证据/实际远端脚本，prefix为受限原始前缀/比较及首收据，entry-final为最终含奖励进入分层的JSON/CSV/PNG/SVG，final/verified保留早期分析，另有source_comparison、sv3资源快照、seed1重叠检查。图`entry-final/diagnostics.png`与已目视核查的前版逐字节相同；评价曲线未平滑，其余均标出动作/更新统计口径。

```powershell
cd D:\program\PD\dmr3
$py = 'C:/Users/97370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$env:PYTHONPATH = 'D:/program/PD/dmr3/analysis/outputs/m2-v1-seed0-20261012/plot_deps'
& $py -m analysis.diagnose_pendulum_v1 collect --source analysis/outputs/dmc15-analysis-20261102-final --output analysis/outputs/NEW_PENDULUM_INPUT
& $py -m analysis.diagnose_pendulum_v1 analyze --input analysis/outputs/NEW_PENDULUM_INPUT --output analysis/outputs/NEW_PENDULUM_ANALYSIS
& $py -m analysis.diagnose_pendulum_v1 prefix --input analysis/outputs/NEW_PENDULUM_INPUT --output analysis/outputs/NEW_PENDULUM_PREFIX
& $py -m unittest tests.test_pendulum_diagnosis tests.test_return_metrics tests.test_m2_analysis -v
```

分析可以直接使用已保存inspection离线复算，输出须新目录，不能Python -O绕开校验。补充核对脚本、命令和文件hash见证据根COMMANDS.md/DELIVERY.json。

首选继续当前校正chat处理范围确认；需要科学协议审阅时由用户转给现有02，获准的工程与启动由03职责承接。当前总目标尚未完成，新训练与后续验收未做；不得以本报告代替全部校正完成。不自动创建chat或发送消息。

可复制交接prompt：

```text
项目D:\program\PD\dmr3。继续实验校正工作；版本取analysis/outputs/pendulum-diagnosis-20261103/DELIVERY.json，实际分析/测试e90e972ee1f795c157581eef40da8847336e6a53。
必读AGENTS.md、STATUS.md、ANALYSIS_PENDULUM_CORRECTION.md、ANALYSIS_LOW_RETURN_FOLLOWUP.md、ANALYSIS_DMC15_V1.md、CODE_MODE_DIAGNOSTICS_REPAIR.md、VISION20_SEED1_RUNS.md及现有冻结协议。
pendulum baseline完整训练992episode中927零return，PPD-KL49；前10万正奖励128/6865。两组40个第一动作有奖励的episode中baseline平均902.25；其余952个起步无奖励episode，后来获奖25/903，更符合摆起能力未建立而非完全不会保持。初始294数组/seed相同，但任何更新前314初始图像坐标相差1、2个动作不同；不能把最终差值全归gate。没有发现可直接判为工程失败的计数/配置/累计错误。hopper补seed1/2/3已完成，三条PPD-KL未重复近零失败但2/3低于baseline，保留seed0。
已确认资源sv3物理GPU1/4/5/6/7；用户仍明确先不启动。23:07:19五卡空闲、/data349.47GB只是快照。不得训练、GPU探针、自动排队或改参数，除非用户后续明确恢复。可继续只读CPU准备。
待确认建议：pendulum seed2/3各baseline和PPD-KL四条，原科学参数/百万动作/21x10/原始return指标不变；seed2在GPU1两组串行，seed3在GPU4串行，5/6/7保留。seed1另有vision20计划，不重复或迁移；目前没有pendulum seed2/3入口，需要获准后新增受限scope、测试、确定SHA、存储/设备/真实工程验收，不放宽旧入口。不新增constant/shuffle/奖励塑形/调alpha等候选干预。
输出独立analysis/outputs及正式运行目录，保留全部失败/原始产物。验收百万训练动作、连续成功更新、21点每点10完整episode、固定快照、c完整S/N和设备/W&B证据，最后逐seed比较，不筛选坏seed、不宣称显著性或机制已证。完成报告最终/实际测试/实际训练SHA、同步范围、资源时间、未决问题和后续prompt；不自动建chat或发消息。
```
