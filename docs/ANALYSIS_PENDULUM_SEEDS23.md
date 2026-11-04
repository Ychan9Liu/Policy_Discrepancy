# Pendulum seed2/3 原参数复验：终点分析

日期：2026-11-04。四条已全部验收并纳入分析，独立复算与真实训练回放检查完成。用户授权的并行执行及启动证据见 [PENDULUM_SEEDS23_PARALLEL.md](PENDULUM_SEEDS23_PARALLEL.md)，历史 seed0 诊断见 [ANALYSIS_PENDULUM_CORRECTION.md](ANALYSIS_PENDULUM_CORRECTION.md)。本报告只汇总新 seed2/3，不自动并入 seed0/1 或 hopper。

## 结果与判断

**这次没有复现“baseline 一定学不好、PPD-KL 一定能解决”的结果。** seed2 两组取得接近的原始 return，PPD-KL 的 AUC 差为 +2.8850；seed3 则是 PPD-KL 明显落后，AUC 差为 −421.6675。这里“接近”是描述数值，不是预先定义的等效检验。原 seed0 baseline 的失败保留，但不能将它作为 baseline 实现普遍损坏的证据。

seed3 PPD-KL 的低分也存在于训练回放：992个完整训练 episode 中820个零回报，baseline 为134个。第一动作无奖励时，后来至少一次获奖的比例为135/955（14.14%），baseline 为820/954（85.95%）；第一动作已有奖励的 PPD-KL 轨迹平均 return 为919.49。**更符合未普遍建立从奖励区间外进入的控制能力，而非完全不能保持直立。** 第一动作奖励发生在动作之后，不是测得的 reset 角度；这是一条训练轨迹的分层描述，不是任务成功率或独立重复。

四条的预算、更新收据、评价、配置和正式 c 均通过检查，未发现需要按工程无效重跑的错误。它们仍存在学习失败与明显的 seed 差异；“验收完成”不等于“方法有效”。两条新增训练 seed 不足以确定失败概率、跨 seed 显著性或训练稳定性，也不能证明唯一根因。

## 正式评价指标

每点先对10个完整 episode 的 `return_raw` 求均值；主指标为0、5万……100万共21点梯形 AUC / 1000000；次指标为80、85、90、95、100万五点均值的算术平均。使用未平滑值，保留全部0点，不插值、不新增归一化或效果阈值。

|训练seed|baseline AUC / 1m|PPD-KL AUC / 1m|PPD-KL − baseline|baseline末段|PPD-KL末段|PPD-KL − baseline|
|---|---:|---:|---:|---:|---:|---:|
|2|728.3500|731.2350|+2.8850|792.38|810.08|+17.70|
|3|607.6225|185.9550|−421.6675|752.48|115.66|−636.82|
|本批两seed等权平均|667.98625|458.59500|−209.39125|772.43|462.87|−309.56|

末点评价分别为 seed2 baseline/PPD-KL 804.4/817.8，seed3 480.7/0。seed3 PPD-KL 在20、90、100万点均为10个完整 episode 全0，原样保留；它在其他阶段也有正回报，不能把任一0点解释成从此永久失败。seed3 baseline 的百万点也包含3个0回报，不能写成 baseline 全程稳定。

同一训练seed、同一预算点的两组使用相同10个评价环境seed，42对评价点全部核对一致；每条21个预算点的环境seed集合各不相同。因此跨点评价变化混合了参数快照与初始条件变化，不能单凭一个点的下降/回升定位参数退化/恢复。10个评价 episode 不是10个训练seed。

学习曲线：[PNG](../analysis/outputs/pendulum-seeds23-final-20261104/learning_curves.png) / [SVG](../analysis/outputs/pendulum-seeds23-final-20261104/learning_curves.svg)。完整数值见该目录 `return-audit/tables.json` 及CSV，逐seed与两seed均值均保留。

## 运行身份与终点验收

实际主机为 sv3 / `lyg0326`；四条训练 SHA 均为 `c5f86ca1a16648f57db3c3abfa2584ac560495ca`。baseline 最初单条审计实际使用 c5f86ca；两条 PPD-KL 单条审计及最终四条重审实际使用 `b7f42d46d16d4eda5cff5a798ec7b60cb27a3790`。相关七个审计/指标文件的 Git blob 一致，进程工作目录、完整提交与命令的实际证据见并行报告；不把训练、审计、分析版本混写成一个版本。

以下目录均位于 `/data/Policy_Discrepancy/runs/`；B根为 `formal-m2-pendulum-seeds23-20261103-sv3`，D根为 `formal-m2-pendulum-seeds23-parallel-dt-20261104-sv3`。

|seed|组|物理GPU|正式尝试目录|最终状态|W&B|
|---:|---|---:|---|---|---|
|2|baseline logging，实际权重1|1|B根/seed2/logging-attempt01|accepted|[seed2 baseline](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-pendulum-seeds23-20261103-sv3-pendulum-seed2-logging-01)|
|2|PPD-KL，dt alpha20|5|D根/seed2/dt-attempt01|accepted|[seed2 PPD-KL](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-pendulum-seeds23-parallel-dt-20261104-sv3-pendulum-seed2-dt-01)|
|3|baseline logging，实际权重1|4|B根/seed3/logging-attempt01|accepted|[seed3 baseline](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-pendulum-seeds23-20261103-sv3-pendulum-seed3-logging-01)|
|3|PPD-KL，dt alpha20|6|D根/seed3/dt-attempt01|accepted|[seed3 PPD-KL](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-pendulum-seeds23-parallel-dt-20261104-sv3-pendulum-seed3-dt-01)|

完整运行ID、逐字段配置、UUID/CUDA/EGL绑定、环境、启动命令和路径在最终输出 `run_manifest.json`、输入镜像的 `protocol_manifest.json`、`execution-initial.json` 及启动报告。关键配置：pendulum_swingup、clean、仅64×64 RGB视觉、size50m、repeat1、16环境、batch16×64/context1/train_ratio256，free nats1，dyn/rep损失系数1/0.1，ac_grads=false、reward_grad/repval_grad=true、repval_loss=true；原优化器、采样动作和独立评价随机流不变；CUDA/bfloat16、prealloc=False、JSONL+线上PD_1、禁止恢复。两组同seed的配置只差 logdir 与 rep_probe.mode；没有新增 constant/shuffle 训练。

最终权威索引是 D根 `actual-RUN_INDEX.json`。2026-11-04 12:23:05.597北京时间产生 `final-audit-attempt01/completion.json`，verified_run_count=4；`supervisor-final.json` complete=true，四条均accepted、退出码0，四份W&B精确传输核对均为true。W&B仅作交叉检查，不作为原始指标来源。

每条恰好1000000训练动作（含预填充、排除reset/评价）、1008 reset、249493连续成功更新与相同数量batch key、21×10完整1000动作评价episode、210000评价动作。独立分析共复核84点/840 episode、997972条更新收据；300个显式采集的小产物全部hash一致，84个快照metadata/完成标记/参数文件存在性匹配。此前已核80点评价、76个门控区间及原始行hash全部精确保留。

旧B根的两个串行Dt槽位只有 `SUPERSEDED_BY_PARALLEL.json`，其 `FileExistsError`/blocked 是防重复启动的预期结果，没有训练数据，未纳入本批。旧监督器的complete=false不代表新合并四条失败；未删除这些证据，也未再启动旧槽位。

12:23:14北京时间，两个监督器与四训练PID均已退出，sv3物理GPU1/4/5/6/7均无C/G进程，/data余253.31GB。GPU7未用于新增任务。后续最新资源快照见本次交付记录；只读CPU分析不占GPU。12:26:24另查，原vision20计划的pendulum seed1仍在sv1 GPU3队列、监督器身份匹配；未迁移、重复或并入本批。

## 训练reward与学习过程

在sv3以CPU重读四条完整replay的reward/action/episode边界，重建每条16条worker链、1000000动作、1008 reset、992完整1000动作episode与16尾段。动作与下一行reward对齐、有限性、全部reward取值0/1、完整episode/尾段/总和闭合和读取前后文件清单不变均通过。只保存标量数组hash及ZIP CRC；大图像、checkpoint和replay留服务器。

|训练统计|seed2 baseline|seed2 PPD-KL|seed3 baseline|seed3 PPD-KL|
|---|---:|---:|---:|---:|
|零回报完整episode / 992|30|40|134|820|
|完整episode平均原始return|724.5897|739.9274|658.1361|151.0030|
|百万动作正奖励次数|723802|739417|658436|150273|
|前10万正奖励次数|28648|20128|202|273|
|首次正奖励全局动作位置|13|13|16010|16010|
|首动作无奖励episode数|949|949|954|955|
|上述轨迹后来获奖数|919|909|820|135|
|上述轨迹平均return|715.0832|730.7292|649.9874|121.2293|
|首动作有奖励episode数|43|43|38|37|
|上述轨迹平均return|934.3953|942.9302|862.7105|919.4865|

seed3 PPD-KL 并非起初完全遇不到奖励：前10万正奖励273次，baseline只有202次。明显分化出现在15—20万区间：baseline 正奖励26796次、策略Gaussian熵0.8353、优势绝对均值0.01045；PPD-KL 为7806次、1.3298、0.00295。最后5万分别为40614/7043次正奖励，想象奖励0.6632/0.1516，熵0.8519/1.3805，优势0.00426/0.00071。它支持“PPD-KL seed3未形成持续高获奖轨迹、保持较高策略噪声且改进信号较弱”的描述，不能确定因果方向，也不能推出应调低熵系数。

seed3 PPD-KL 百万点评价为0，但末5万训练仍有7043个正奖励动作；训练使用变化中的参数，评价固定在一个快照，二者初始条件及采样对象不同。不能把训练reward当该固定checkpoint的评价，也不能认为百万点评价0意味着全部训练reward为0。

80个5万动作区间全部与JSONL/评价更新计数对齐，第一段11993更新、其余每段12500；按冻结loss scales重建总loss，最大绝对误差1.234016e-7，沿用原容差。日志标量有限，四条最大梯度范数**区间均值**均在首段（约107.03、107.32、98.46、98.55）；没有逐更新峰值证据，不能据区间均值排除瞬时尖峰或宣告优化完全正常。较低reward loss也不能跨不同采样分布解释成模型更准确。

训练过程图（已实际查看）：[seed2 PNG](../analysis/outputs/pendulum-seeds23-final-replay-20261104/training_dynamics_seed2.png)、[seed3 PNG](../analysis/outputs/pendulum-seeds23-final-replay-20261104/training_dynamics_seed3.png)，同目录有SVG和 `training_phases.json/csv`。真实reward按动作平均，想象奖励、熵、优势按更新平均；二者分布不同，不取比值当奖励模型误差。无平滑、额外评价或新增正式指标。

## 门控实际作用

冻结权重为 `stop_gradient(1 / (1 + 20 D))`，baseline仅记录D、实际乘权为1。以下为全部249493更新加权的日志统计；活跃位置定义为raw rep KL > free nats1。活跃权重为每次更新活跃位置权重均值再按更新平均，不等于把全部活跃位置合并后的S/N。

|诊断|seed2 baseline|seed2 PPD-KL|seed3 baseline|seed3 PPD-KL|
|---|---:|---:|---:|---:|
|Mean D|0.061670|0.064758|0.074574|0.004678|
|精确D=0比例|5.5444%|5.4671%|6.0589%|1.6628%|
|free-nats活跃比例|12.4480%|13.3367%|14.1450%|6.4299%|
|全部位置实际权重|1|0.929065|1|0.985355|
|活跃位置实际权重|约1|0.884296|约1|0.948149|
|累计超额KL损失减弱比例|0|12.2814%|0|12.6620%|
|rep损失值整体减弱比例|0|7.8074%|0|2.3785%|

门控确实改变了损失，不能称为“未开启”；但没有冻结的充分作用阈值，也不能仅凭这些均值说作用足够。seed3 PPD-KL 前10万活跃权重0.99345、最后20万0.90990，早期较弱、后期增强；seed2对应0.96601/0.89351。seed3 的平均权重更接近1、整体rep损失值减弱更少，超额KL的相对减弱却与seed2相近：统计分母、活跃位置分布及其与KL大小的对应均会影响结果。平均权重不等于损失作用，更不等于总梯度减弱比例。

`D_p10/p50/p90`仅是日志中各批次分位数的更新均值。PPD-KL seed2/3的p50均值为3.6482e-5/1.8819e-6，p90均值0.019438/0.000885；未保存逐位置样本，不能重建全训练合并分布或把这些值当合并分位数。D小、高熵、低回报的共现也不能说明D小是失败原因。

两条baseline的正式c由各自[100000,300000)窗口50000次成功更新（IDs24492…74491）复算，N为完整成功更新中有效活跃位置数；含上界300000的更新被排除。全部收据、冻结文件/sidecar、S/N、配置hash和审计绑定一致：

|seed|S|N|正式c|
|---:|---:|---:|---:|
|2|7396141.760381699|8363327|0.8843540089227288|
|3|7721404.809896469|8560619|0.9019680480928387|

这两条c仅按既定协议冻结与审计，没有据此启动constant。**本批只有baseline与PPD-KL，不能分离整体减弱rep KL和权重与位置对应的价值**；也不能因为某个seed输给baseline就直接宣布位置门控无效。constant/global shuffle若需新增，须由02与用户制定并确认下一协议。

门控图：[PNG](../analysis/outputs/pendulum-seeds23-final-20261104/gate_diagnostics.png) / [SVG](../analysis/outputs/pendulum-seeds23-final-20261104/gate_diagnostics.svg)。完整逐区间和阶段表为 `gate_intervals.json/csv`、`gate_summary.json/csv`。

## 证据边界与下一步

同seed两组294个初始参数数组一致，但任何优化器更新之前的reset图像和部分策略输出已不同，具体来源未裁决。因此这批可以作为原协议下的探索性性能复验，不能当成完全相同输入的门控因果实验。训练核心文件与原seed0对应文件的逐blob一致性、入口差异和配置比较见启动报告 `source-equivalence.json`。

终点采集检查了全部快照metadata/完成标记/文件存在性；本轮此前实际解码的参数payload共10份（四个0步、Dt seed3的15/20/85/90万、baseline seed3的65/70万），只在该范围内确认参数有限/变化。没有解码全部84份参数，也没有逐transition重放评价reward。独立reward求和覆盖的是训练回放，评价内部累计语义复用执行器审计。保留这些限制，不把“没有发现某类错误”写成所有实现无误。

**首选下一任务：交现有02制定固定输入的离线checkpoint诊断协议，再决定是否需要新的训练干预。** 理由是两个新增seed未给出一致优势，失败组又复现了原seed0低回报伴随的高熵/弱优势/进入奖励区间困难，继续仅看训练均值或直接调alpha无法区分奖励表示、策略改进和输入数值差异。建议02预先定义共同观测序列、正/负奖励样本及checkpoint选择，分别检查奖励预测、prior/posterior策略输出与门控作用；这些仍是候选设计，不是本报告批准的新实验。

可立即开展：02审阅本报告及已有原始产物，冻结要检验的问题、样本与输出。依赖后续确认：任何新增agent/环境执行、GPU诊断、训练、constant/shuffle、参数或评价协议变化。已排队seed1由原执行队列继续，完成后另审计，不重复启动、不在本批提前汇总。hopper已有seed1/2/3六条验收，无需再跑相同补测。继续现有02即可，由用户手动转交；不自动建chat或发送消息。

## 复算、版本与交付

实际终点采集/分析/replay及36项离线测试版本均为 `5ccbd8c533107823fba558dd092cac41655efd8d`。本机独立detached工作树固定此提交；五个分析依赖文件已与Git blob逐字节核对、工作树/暂存区干净，实际Python3.12.14、NumPy2.4.2、Matplotlib3.10.8及导入路径留档。初次CRLF checkout造成精确hash检查失败后，只在新工作树恢复匹配的Git blob字节并刷新索引；无语义修改或放宽检查。36项测试与旧seed0真实40区间回归通过不能替代本批真实结果，本批已另实际运行全部终点分析。

|产物|本机 `analysis/outputs/` 下目录|
|---|---|
|300个源文件/逐文件hash/完整配置与终点/W&B证据|pendulum-seeds23-final-input-20261104/|
|正式指标、四条清单、全部收据/c、门控和两张图|pendulum-seeds23-final-20261104/|
|400万训练动作标量复核、分层、80区间与两张过程图|pendulum-seeds23-final-replay-20261104/|
|固定源码/环境、命令、旧记录一致性、图形验看、最终交付绑定|pendulum-seeds23-terminal-preparation-20261104/|
|启动/过程/失败保留/设备/资源快照|pendulum-seeds23-parallel-20261104/|

输入压缩包 SHA256：`ebc531133f05057cde423e38239bce1f0392445c339e635aba1960293835bedb`；COLLECTION：`82849336a4b3600e37c259bd7dcfd6ab5f7e6547f7b36eebfbfb4b9f3fcd0ae3`；训练replay小证据 inspection：`901a0a4506f15aad1d078520085a3d2cc715b6357f3b4ed2332a00dbfa744c18`；80区间训练表：`45f26113bc97a5900501ad04c634f39add215cf6dca82038517d224dac0a72b0`。原始大小文件未覆盖，所有生成图均实际查看，无平滑、缺点填补或择seed。

完整PowerShell命令在准备目录 `COMMANDS.md`。从固定工作树运行 `python -m analysis.pendulum_seeds23_analysis collect/analyze/replay`，输出必须新目录；analyze可从已保存输入镜像离线复算，replay需只读sv3原标量产物。`verify_precedents.py`将此前80点评价和76个门控区间绑定到本次终点；`terminal_precedents_verification.json`记录逐条相等及行hash证据，不执行训练/评价。

最终文档完整SHA及图形/输出hash见准备目录 `DELIVERY.json` 与 `HANDOFF_02.txt`；实际分析/测试SHA保持上述5ccbd8c。本次终点收尾仅在本机提交分析文档和索引，无服务器源码同步或GitHub推送，不改变训练源码/配置。原四条执行已经使用前述固定训练/控制器checkout，未回退共享HEAD；其他chat的提交和未提交工作保留。
