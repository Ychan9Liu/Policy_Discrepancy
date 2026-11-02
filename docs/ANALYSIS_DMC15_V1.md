# 05 · 新一轮DMC任务分析（排除acrobot_swingup）

日期：2026-11-02。当前分析版本 `a0b66181586194066fe0c7bdd34ae75847a5ddcc`。用户明确排除acrobot；本轮新增15任务×baseline/PPD-KL（代码名dt）×seed0共30条均有完整评价和成功更新证据，纳入逐任务分析。历史四任务仅在单独一节作为参照，不混算新的19任务总分；hopper补seed另属批次，本轮没有合并。

## 结论

新增15任务的主指标AUC有7项正向、8项负向；末段五点均值有11项正向、4项负向。这是数值正负计数，不是显著性胜率。最强正向信号是pendulum_swingup与walker_run；cartpole两种swingup也正向，普通swingup主要加快学习。finger_turn_easy主次均下降；多数接近高回报平台的任务差异较小。本轮支持“任务依赖的性能改善线索”，尚不支持普遍收益或M2位置选择价值的因果结论。

## 来源、纳入和验收边界

只读SSH获取sv1/sv2/sv3正式根的actual-RUN_INDEX、manifest、final_state、evaluations.jsonl、metrics.jsonl、c冻结产物；在服务器逐行检查成功更新收据并记录原文件SHA256，不复制checkpoint/replay、不改远程文件。完整源数据、配置、索引、服务器/GPU、run/attempt ID和W&B链接保存在sv1.json/sv2.json/sv3.json；简表run_manifest.json记录逐条版本/原路径/哈希。

- 30条均1000000训练动作、249493连续成功更新且invalid=0，合计7484790更新；630个固定评价点、6300个完整episode，均为0..100万每5万、每点10条。训练动作口径复用已验收runner，final评价动作计数另行核对。
- 所有评价snapshot ID、动作步、update ID、训练SHA与小元数据一致；检查checkpoint存在及done标记。本轮未反序列化大checkpoint重算参数hash，固定参数快照语义仍复用03源码/既有验收。
- 从每条episode的return_raw独立求均值，并与JSONL精确均值核对；再复算21点梯形AUC/1000000及80/85/90/95/100万五点均值。没有逐transition评价reward轨迹，不能声称独立验证了每条episode内部reward求和。
- 每任务两组完整配置仅agent.rep_probe.mode及logdir不同。seed0、repeat1、alpha20及原科学协议一致。依赖锁和manifest包版本在30条间一致；五个关键文件agent.py、rep_probe.py、rssm.py、protocol_v1.py、dmc.py在五个实际训练SHA间逐字节一致（training_source_comparison.json）。这不等于跨服务器、不同GPU和异步执行逐位可复现。
- baseline冻结c逐条与[100000,300000)完整成功收据的S/N、receipt hash及冻结记录一致；此轮只跑baseline/Dt，c是对照准备/审计证据，不是已运行constant组。
- 索引中14条accepted，16条仍run_audited_pending_group_and_WB；后者已有逐条run_audit，本轮已独立复核上述关键产物，但未补写旧批次completion或W&B摘要。因此可完成用户指定15任务的原始return分析，不能宣布原32条全批收尾完成。未读取或分析acrobot结果。
- 失败/未启动尝试按缺少final或run_audit排除并留清单，30条无重复task/group。finger_turn_easy使用工程故障后成功的retry03，不按return择优；sv1 GPU7历史短时外来并发和跨机数值限制继续保留，见DMC8_REMAINING_RUNS.md。

原始输入采集时间见各server JSON；本次无GPU资源核查、训练/推理/环境仿真。服务器仅CPU读取文件，当前卡占用不由本报告断言。

## 新增15任务逐项结果

主指标为原始return AUC/百万；末段为最后五点评价均值。Δ均为PPD-KL减baseline；四舍五入仅用于展示。

|任务|baseline AUC|PPD-KL AUC|ΔAUC|baseline末段|PPD-KL末段|Δ末段|
|---|---:|---:|---:|---:|---:|---:|
|cartpole_balance|962.69|952.24|-10.45|993.58|991.17|-2.41|
|cartpole_balance_sparse|970.14|914.58|-55.57|975.36|999.12|23.76|
|cartpole_swingup|665.51|774.50|108.99|853.42|859.37|5.96|
|cartpole_swingup_sparse|423.81|502.68|78.86|738.08|761.72|23.64|
|cheetah_run|703.70|686.38|-17.32|831.21|831.33|0.12|
|cup_catch|935.53|947.41|11.88|970.08|974.84|4.76|
|finger_spin|471.56|468.39|-3.17|496.82|566.80|69.98|
|finger_turn_easy|754.00|666.91|-87.09|890.36|787.60|-102.76|
|finger_turn_hard|589.06|633.13|44.07|802.64|816.50|13.86|
|hopper_stand|782.10|795.09|12.99|904.39|913.71|9.32|
|pendulum_swingup|32.53|691.47|658.93|101.48|787.80|686.32|
|reacher_easy|857.58|845.05|-12.52|933.76|971.54|37.78|
|walker_run|301.76|692.31|390.56|349.59|804.83|455.24|
|walker_stand|947.14|942.74|-4.40|975.85|975.78|-0.07|
|walker_walk|892.08|890.83|-1.25|966.03|960.19|-5.84|

## 曲线解释

- **pendulum_swingup**：baseline前50万评价仍为0，PPD-KL在10万已528.8、30万803.7，末段787.8对101.48。差异是本seed下是否进入高回报学习路径；尚不能定位为门控保护了何种信息，也不能保证换seed重复。
- **walker_run**：10万为406.2对190.9，30万724.8对310.9，终点813.0对353.2；大幅差距跨多个快照持续存在，是本轮最清晰的性能信号之一。walker_walk/stand已经高回报，未呈现同样提升，不能推广成walker域普遍有效。
- **cartpole_swingup**：AUC提高108.99，末段仅提高5.96，主要体现更快学会；swingup_sparse两指标分别+78.86/+23.64。balance_sparse反而AUC−55.57、末段+23.76，不能统一称“稀疏奖励都受益”。
- **finger_turn_easy**：AUC−87.09、末段−102.76，中后期多个点偏低。finger_turn_hard主次正向，域内也不一致；不能用一条难任务提升解释所有finger。
- **finger_spin**：AUC−3.17而末段+69.98。PPD-KL最后五点为468.5、480.8、480.7、477.2、926.8；baseline为495.2、496.5、495.8、497.9、498.7。末段优势依赖最后一次跳升，不擅自删除该点，也不把它描述为已验证的持续改进。
- **接近平台的任务**：cartpole_balance、cup_catch、walker_stand/walk已较高回报，改进空间有限；小差异仍保留数值，不设置新等效范围或宣布“等效”。reacher_easy末段高而AUC低，早期学习与后期水平要分开看。

## 门控实际作用

以下为完整训练更新加权统计。活跃比例指raw rep KL超过free nats的位置；活跃权重是每更新活跃位置均值再按更新数平均。超额KL减弱比例为累计excess_reduction/(weighted_excess+excess_reduction)，不等于参数梯度范数下降比例。

|任务|PPD-KL Mean Dt|D=0比例|活跃比例|活跃权重|超额KL减弱|baseline正式c|
|---|---:|---:|---:|---:|---:|---:|
|cartpole_balance|0.0141|0.19%|15.96%|0.869|15.43%|0.825114|
|cartpole_balance_sparse|0.0169|0.77%|21.43%|0.917|10.00%|0.870690|
|cartpole_swingup|0.0105|1.18%|22.57%|0.875|12.56%|0.867524|
|cartpole_swingup_sparse|0.0269|0.77%|36.42%|0.919|8.76%|0.934292|
|cheetah_run|0.0055|0.63%|48.62%|0.926|9.82%|0.917876|
|cup_catch|0.0454|1.69%|31.46%|0.865|15.78%|0.927760|
|finger_spin|0.3587|1.46%|61.84%|0.626|45.90%|0.723453|
|finger_turn_easy|0.0663|0.74%|43.23%|0.846|18.16%|0.836720|
|finger_turn_hard|0.1078|1.12%|49.91%|0.787|24.73%|0.808270|
|hopper_stand|0.0222|0.27%|46.74%|0.874|19.21%|0.814044|
|pendulum_swingup|0.1677|3.38%|15.70%|0.896|10.00%|0.983906|
|reacher_easy|0.0365|1.26%|28.93%|0.896|10.00%|0.914984|
|walker_run|0.0780|0.01%|98.98%|0.676|38.62%|0.489779|
|walker_stand|0.0502|0.02%|94.78%|0.828|25.82%|0.800742|
|walker_walk|0.0604|0.01%|99.04%|0.760|29.81%|0.631686|

所有任务都存在实际门控：活跃权重约0.626—0.926、超额KL减弱约8.8%—45.9%，不存在全程退化为baseline权重1的情形。baseline实际权重为1。walker_run活跃比例约99%、活跃权重0.676、超额KL减弱38.6%；finger_spin减弱更强（45.9%）却没有全程AUC提升；walker_walk减弱29.8%仍无收益。减弱强度不能单独解释任务收益。

pendulum的活跃权重从前10万约0.987变为后20万0.861；walker_run从0.684到0.731。门控强弱随训练变化，并与已学策略/数据反馈耦合；这些日志不能确定谁是因、谁是果。阶段统计及精确零比例见gate_summary.csv；不把均值当作完整分布，不从旧mode诊断错误字段推理。

本轮没有constant/global shuffle，无法区分“统一减小rep KL也能做到”和“权重与具体位置的对应关系有价值”。即使baseline正式c已计算，也不能当作constant实验结果或损失/梯度作用完全匹配。

## 历史四任务参照（本轮未重跑）

复用ANALYSIS_M2_V1.md的已验收seed0结果；只作单列参照，不与本轮统计混称新实验。原始表见analysis/outputs/m2-v1-seed0-20261012/analysis-final/return-audit/tables.json。

|任务|ΔAUC|Δ末段|
|---|---:|---:|
|hopper_hop|−252.97|−315.74|
|quadruped_run|+132.75|+35.13|
|quadruped_walk|+158.09|+7.09|
|reacher_hard|−3.83|+30.84|

因此全任务视野仍需同时保留hopper_hop负结果；本轮hopper_stand为正不能推翻hop失败。原四组对照中walk同时超过constant/shuffle，run未超过constant的边界也保持。hopper新seed六条属于另一批次，不自动用来替换原seed0或加入此轮seed0总表。

## 实际版本

|完整训练SHA|条数|服务器|
|---|---:|---|
|18b85c4d85ff456e6822626f17cb14cc0f44d2b8|9|sv1|
|6e468ad4a8252f21401eab6a9761626b33659f0e|4|sv1|
|c7689d50f4caed0fef89912c65280b8d2558b126|6|sv2, sv3|
|d8997f64a301bfdcfe08758faeae8957a7395766|4|sv3|
|dbd28b6703d51d61e5d3485dbfea6c413075ea85|7|sv2|

本次实际分析/验证SHA为a0b66181586194066fe0c7bdd34ae75847a5ddcc，随后最终提交仅新增报告/README/STATUS索引，最终SHA见输出DELIVERY.json。只本机提交、不推送或同步服务器源码；保留原AGENTS/CHATS/RESEARCH/STATUS/proposal及根TeX草稿。

## 产物与复现

目录：analysis/outputs/dmc15-analysis-20261102-final/。
- task_metrics.csv/json：逐任务原始指标；learning_points.csv/json：630点评价均值；gate_summary.csv/json：门控与阶段统计；run_manifest.json/config_differences.json：版本/配置/原路径；sv1/2/3.json：原始episode、日志、审计和收据汇总；verification.json及training_source_comparison.json：哈希/核对。
- learning_curves.png/svg：15任务未平滑曲线；differences.png/svg：主次指标差值。单seed不绘跨seed误差带；10条评价episode不能作10个训练seed。

```powershell
$env:PYTHONPATH='D:/program/PD/dmr3/analysis/outputs/m2-v1-seed0-20261012/plot_deps'
# 首次采集只读服务器；NEW_DIR必须不存在
python -m analysis.analyze_dmc15_v1 collect --output analysis/outputs/NEW_DIR
python -m analysis.analyze_dmc15_v1 analyze --output analysis/outputs/NEW_DIR
# 只复算现有镜像时，仅执行analyze；输入sv1/2/3.json不被改写。
```

本机实际Python为C:/Users/97370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe。脚本中assert校验要求正常Python启动，不使用-O。collector仅生成本机证据，不运行03带freeze写操作的审计函数。图形已检查标题、任务范围及曲线。

## 下一步建议与交接

首选继续现有02，制定有代表性的多seed复验协议：优先覆盖pendulum_swingup、walker_run两种大正向信号，以及finger_turn_easy负向信号；若重点仍为M2选择性，需设计constant/shuffle对照。任务/seed/预算/判据/资源仍待02与用户确认，此处是分析建议，不是批准实验。不按本轮结果直接调alpha或改奖励。完成标准是明确复验矩阵、主要指标和判据，并交03在资源授权后执行。

03可立即并行补旧批次最终来源汇总与线上精确摘要；这不阻止本报告逐任务原始return分析，但全批完成声明需03补齐。双方都不自动创建chat或发消息。

可复制给02：

```text
项目D:\program\PD\dmr3。你是02。必读AGENTS.md、docs/ANALYSIS_DMC15_V1.md、docs/METHOD_SPEC.md、docs/EXPERIMENTS.md、docs/RUNS_STATUS_20261102.md；交接最终SHA见analysis/outputs/dmc15-analysis-20261102-final/DELIVERY.json，实际分析SHA a0b66181586194066fe0c7bdd34ae75847a5ddcc。
用户要求排除acrobot。新增15任务30条seed0原始产物可分析，AUC正向7/15、末段11/15；pendulum/walker_run大幅正向，finger_turn_easy负向，finger_spin末段提升依赖最后一点。16条旧索引仍待批次/W&B收尾，不宣称原32条全批完成。实际门控存在，但没有新constant/shuffle，不能宣称位置选择因果或跨seed显著性。
下一任务：制定多seed复验与必要对照的候选协议，优先覆盖两项强正向和一项负向；具体任务/seed/预算/判据/资源由用户确认。只允许协议分析与文档，不训练、不自行调参/改奖励或覆盖冻结v1。输出协议文档和STATUS，说明批准项/未决项与验收标准，完成后给用户可交03的prompt；不自动发送。
```
