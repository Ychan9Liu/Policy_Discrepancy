# 项目状态

更新日期：2026-11-03。

## 05 最新：PPD-KL无明显优势与参数强度解释（2026-11-03）

- 已只读关联新增15任务seed0的return指标与门控统计，核对30条alpha20/rep系数0.1：finger_spin超额KL减弱45.90%但AUC差−3.17，walker_walk减弱29.81%但AUC差−1.25；pendulum减弱10.00%却AUC差+658.93。不能统一归为alpha太小，也未证明alpha20最优；loss减弱不等于总梯度范数减弱。
- 报告docs/ANALYSIS_PPD_KL_STRENGTH.md区分高分空间、学习速度/末段、阶段作用不足、选择机制与单seed路径；首选先核对现有seed1重复趋势，新的双向alpha/匹配对照交02与用户确认，不自行选参数。修正DMC15旧报告pendulum前期为0的简写，原正式指标不变。
- 核查脚本、源hash、未舍入表及版本在analysis/outputs/ppd-kl-strength-20261103/。本次仅文档和只读数据核查，无训练代码/协议变更、无新训练/GPU探针/资源核查或服务器源码同步；sv3 GPU1/4/5/6/7“先不启动”保持，不自动建chat/发消息。

## 05 最新：pendulum摆起失败链路复核，校正复验尚未启动（2026-11-03）

- 两条seed0全replay独立复核：各100万动作/1008 reset/992完整episode；baseline零return 927、PPD-KL49。两组第一动作即获奖励的40episode中baseline均值902.25；其余952episode后来获奖仅25对903，更符合保持能力部分存在、普遍摆起未建立，不等于已证明代码故障。
- 294初始数组/seed/10个关键文件一致，但首次更新前初始图像已有314个坐标相差1、2/16动作不同，不能把最终差值全归gate。原始评价32.5325/691.465 AUC、末段101.48/787.8不变；实际门控存在，保留日志聚合/因果/单seed限制。hopper补seed已有结果与mode修复证据已复核，无新增重跑理由。
- 新代码analysis/diagnose_pendulum_v1.py，24项测试与真实产物验证通过；最终分析/测试SHA e90e972ee1f795c157581eef40da8847336e6a53。报告docs/ANALYSIS_PENDULUM_CORRECTION.md，完整输入/hash/脚本/表图/命令在analysis/outputs/pendulum-diagnosis-20261103/。只读CPU、本机分析代码/文档；无训练源码修改、GPU探针、服务器同步或推送。
- 用户仍要求先不启动。23:07:19 sv3物理GPU1/4/5/6/7均空闲，/data349.47GB；新任务0、队列0。建议下一步待确认补pendulum seed2/3两组共4条，同seed同卡串行GPU1/4、其余保留；seed1复用既有vision20计划，不重复/迁移。新seed入口未实现、启动限制未解除，整体校正未完成；不自动建chat/发消息。

## 03合并窗口最新：Vision20 seed1，sv2七baseline已启动，sv1待空间（2026-11-03）

- 用户授权20视觉任务×baseline/PPD-KL、原配置、seed1，明确复用已有hopper seed1一对，其他19任务新跑38条；资源sv1 GPU2—7、sv2 GPU0—5/7，不用sv2 GPU6。PPD-KL映射既有dt，baseline映射logging，原seed0/补seed结果不改。保持clean视觉/size50m/alpha20/free_nats1/原KL梯度/百万动作含prefill/21×10固定快照采样评价/JSONL+online PD_1/raw-return口径，独立scope/profile，不自动并seed0或加constant/shuffle。
- 22:42指定卡空闲，但sv1 /data余1.23GB；用户明确自己释放/扩容、sv2可先跑，23:01 sv1仍仅1,236,582,400字节，六卡24条不启动/不部署源码/不删旧数据。sv2七任务14条每卡baseline→终点审计→Dt：GPU0 balance_sparse、1 swingup、2 swingup_sparse、3 cheetah_run、4 finger_spin、5 finger_turn_easy、7 finger_turn_hard（均dmc前缀）。前驱失败/非法匹配或占用/容量不足阻断，不外来空闲重试、共享或换卡。
- 实现/实际验收与运行SHA `77699380851701f7a03a9d13985c5cc0de5d0702`：独立20任务seed1入口、c审计范围、指标profile、38条准备/受限串行控制器，不改agent/rep probe/梯度/协议runner RNG。38配置与原hopper同组seed1逐字段只差script/task/logdir；原hopper两条源码743089f40662b031a5cf2b6bf988decb52ba0b28的百万动作/21×10/线上finished/config/精确摘要hash只读复核通过。本机39项回归和最终7项新模块通过；sv2同SHA121项CPU/三变体float32完整agent精确一致、七卡实际CUDA/EGL和seed1 online logger通过；14条真实size50m CUDA短程各4098动作/517更新/3×1/2报告，7238更新/42点评价/28报告只读隔离审计通过，工程不是正式c或收益。
- 23:00 sv2实际启动监督器2504096，七baseline agent2505601/2505611/2505629/2505915/2506475/2506749/2506761。23:03:57启动审计各0点10完整episode/固定0动作快照、182—325连续成功更新invalid0；119个C/G归属/落卡一致、七条W&B running/full config和startup exact JSON/hash读回一致，/data余约603GB。七条Dt队列存在，依赖自身baseline审计，百万步/c/正式指标未完成；sv2七卡仍有本项目作业，无释放，GPU6未使用。
- 当前40记录为2复用已完成/7运行/7排队/24待sv1空间；不称全部完成或方法有效。只向sv2以校验Git bundle部署独立干净worktree，未改canonical checkout，无sv1源码/GitHub推送、无依赖变更或放宽检查。本机用户及其他chat草稿保留。本轮最终提交自身仅文档；相对实测另含同期其他chat的pendulum CPU分析脚本/测试及sv3只读资源文档，本批训练/控制器/统计源码逐文件与实测一致、不移运行checkout，差异详见报告和DELIVERY。
- 报告`docs/VISION20_SEED1_RUNS.md`；完整计划/复用来源/配置diff/测试/同步/工程与启动证据及DELIVERY/prompt在`analysis/outputs/vision20-seed1-preparation-20261103/`。首选继续当前03监测sv2 baseline→Dt，在用户解决空间后按原sv1六卡验收/启动24条；终点要求百万动作/21×10/收据/快照/合法c/raw-return/W&B及全40来源一致性，缺项不重加权。新科学选择仍交02与用户；不自动建chat/发消息，原科研/复现限制保留。

## 05 最新：sv3资源更正及只读校对，先不启动（2026-11-03）

- 用户将分配更正为sv3物理GPU1/4/5/6/7，明确先不启动。22:50:46北京时间逐卡XML与pmon均无C/G进程、已用显存0MiB；五卡均A100-SXM4-80GB。本轮无训练/GPU探针/后台队列，本批占用0；GPU2不属于新分配，不再作为本批卡冲突。
- /data余349.5GB；六条按既有60GB/条保守预留需360GB，未来启动前落实存储核算。独立vision20 seed1计划已包含pendulum seed1，须核实际状态避免重复。CUDA/EGL绑定未测试，空闲快照非预约；用户恢复启动前不自动执行。
- 报告docs/SV3_RESOURCE_CHECK_20261103.md；证据/实际脚本/版本在analysis/outputs/pendulum-resource-check-20261103/。仅只读sv3/本机文档，无训练代码/协议变更或服务器同步/推送。后续可继续现有窗口只读准备，启动依赖用户恢复指令；不自动建chat/发消息。

## 05 最新：两条低回报运行判断与hopper补seed读回（2026-11-03）

- pendulum baseline seed0未建立持续倒立控制；0至70万仅5万点1.4，其余为0，末段101.48。hopper PPD-KL seed0为更严重的近零学习失败，既有全replay证据支持。两者不能仅凭回报判为实现故障，也不能用工程验收通过否认学习失败。
- 只读sv3补seed1/2/3六条accepted原始21×10评价并复算：PPD-KL末段457.71/291.81/272.37，baseline316.26/386.68/331.94。三条均未重复seed0近零失败，但2/3低于baseline；批次/版本差异保留，不剔除seed0、不并总体、不作显著性结论。
- 报告docs/ANALYSIS_LOW_RETURN_FOLLOWUP.md，证据analysis/outputs/hopper-pendulum-followup-20261103/。仅只读CPU/本机文档，无训练代码/协议修改、GPU检查、服务器同步或推送。首选继续05只读补pendulum训练奖励/模型信号/策略诊断；根因未裁决，新实验待02/用户，不自动建chat或发消息。

## 05 最新：四任务指定步数附近对比GIF（2026-11-02）

- 已制作cartpole_swingup 300k、finger_spin/pendulum_swingup/walker_run 1M对比GIF及MP4，左baseline右PPD-KL。沿用第一保存链、目标前最后完整episode规则，无回报择样；实际动作范围272001—287985、其余976001—991985，均完整episode原速循环。属于训练replay，不是固定checkpoint评价。
- 输出 `analysis/outputs/dmc-four-gifs-20261102/final/`，时长分别10/20/20/25秒；报告/复现/交接prompt `docs/ANALYSIS_DMC_FOUR_GIFS.md`。八条源帧hash/reward求和、四GIF逐帧像素与时长及MP4完整解码通过，接触图已目视检查。
- 实际采集/编码/验证SHA 575e9916077e4e9c8b731453d3d284d489599acf，后续仅文档；只读sv1/sv2/sv3文件、本机提交，无训练/推理/环境/GPU/服务器源码同步或推送，未核GPU现状；既有草稿保留。后续版式继续现有05；严格checkpoint评价补录由03在用户资源授权后处理，不自动建chat/发消息。

## 05 最新：新增15任务分析完成，排除acrobot（2026-11-02）

- 用户明确排除acrobot_swingup；新增15任务baseline/PPD-KL共30条seed0，逐条复核百万动作/249493连续成功更新/21×10评价/快照小证据/正式c与配置，复算原始return AUC及末段五点均值。AUC正向7/15、末段正向11/15，非显著性胜率；pendulum_swingup和walker_run最大正向，finger_turn_easy主次负向，finger_spin末段优势依赖最后一点跳升。
- 门控实际作用存在，活跃权重约0.626—0.926；缺新constant/shuffle与多seed，不能归因位置选择价值/跨seed稳定性。14条索引accepted，16条逐条审计通过但仍待批次/W&B收尾；本次完成指定子集分析，不宣称原32条全批完成。历史四任务单列参照，不混入新总体，hopper补seed另批。
- 报告 `docs/ANALYSIS_DMC15_V1.md`，代码 `analysis/analyze_dmc15_v1.py`，原始小证据/哈希/逐任务与诊断表/学习曲线和差值图在 `analysis/outputs/dmc15-analysis-20261102-final/`。实际分析/核查SHA a0b66181586194066fe0c7bdd34ae75847a5ddcc，后续仅文档；三服务器仅只读CPU文件，无训练/GPU/服务器源码同步或推送，未核GPU现状；既有草稿保留。
- 首选继续02制定覆盖强正向与负向任务的多seed及必要对照协议，配置/资源待用户确认；03可并行补批次/线上收尾。报告附可复制prompt，不自动建chat或发送消息。

## 03合并窗口最新：31/32训练结束，仅acrobot在sv2 GPU1运行（2026-11-02）

- 12:07—08核查：昨晚六条Dt全部accepted/退出0，各百万动作、249493更新、21点×10完整episode，分片表hash/completion一致；线上六条finished/full config/原始return精确JSON/hash与源表一致。因此新增32条为31训练结束/1运行，无排队或失败，未完成全32条最终汇总；旧17条所属批次收尾仍未自动完成。原21条计数继承11月1日索引，本轮重点核新六条、原补四条及最后运行，不冒充旧21条全量重审。
- acrobot在cup完整终点审计后于10:34:50自动串行启动，sv2物理GPU1/agent1791869；12:07:41有34893连续成功更新、最近开始动作141600，invalid累计0，0/5万/10万三点评价各10完整episode。manifest/配置/SHA、CUDA/EGL同卡和17个C/G归属核对通过，W&B running/full config一致。旧queue-progress的training_started=false是释放前快照，实际目录/queue-released/收据已证明运行，不再报告排队。
- sv2 GPU3/5/7和sv3 GPU1/3及原4—7均由本批释放、当时无C/G PID，sv2 GPU1仍运行且无外来共享；sv2 GPU6从未使用，sv3 GPU0/2外来作业不干扰。/data余sv2约677GB、sv3约118GB；sv1本轮未核资源。资源快照非未来预约。
- 实际运行/算法测试版本仍c7689d50f4caed0fef89912c65280b8d2558b126，队列及17项测试2cccd5e006a639f10699907ad3f5b2f49c2c0b56，早先补四条d8997f64a301bfdcfe08758faeae8957a7395766。本轮只读核查/本机状态文档，无代码/测试/训练变更、服务器Git同步或GitHub推送，保留用户草稿/原日志和冻结科学协议。
- 报告`docs/RUNS_STATUS_20261102.md`，小证据/DELIVERY及复制prompt `analysis/outputs/run-status-20261102/`。首选继续现有03完成最后acrobot终点和全32条来源/原始return/线上审计；当前可监测，完整汇总依赖终点与全体核查。工程/数量不当方法有效结论，原科研/复现限制保持，不自动建chat/发消息。

## 03合并窗口最新：25/32训练结束，六条实际补启动、一条依赖排队（2026-11-01）

- 23:32—38重新核查：新增16任务×baseline/Dt共32条，25条训练结束、6条运行、1条排队。旧21条索引仍为accepted或run_audited_pending_group_and_WB、退出0；新增sv3四条于10月31日01:30全部accepted，各百万动作/249493更新/21×10，终点表hash及线上finished/full config/正式精确JSON与源表读回一致。旧17条仍待所属批次汇总，不把25条统称全32条验收完成。hopper补seed六条仍全部accepted、单列；新增38槽位为31结束/6运行/1排队。
- 上轮后七条只完成工程验收，正式启动/后台队列没有执行，不能将中断当持续运行。本次按已有授权补启动，不重复确认、不重跑已完成四条、不用sv2 GPU6。冻结seed0/clean纯视觉/size50m/alpha20/原KL梯度/百万动作/21×10/原始return/在线PD_1均不变。
- 23:34启动sv2 GPU1 cup_catch、3 cartpole_balance_sparse、5 cartpole_swingup_sparse、7 hopper_stand，监督器1647284；sv3 GPU1 cartpole_balance、3 cartpole_swingup，监督器875315。23:37:57六条启动审计通过：各0点10完整episode/固定0步快照、289—378连续更新invalid0、102个C/G PID正确同卡/对应agent子进程、W&B running/full config/startup精确JSON/hash读回一致；百万动作尚未完成。
- acrobot在sv2 GPU1串行等待cup完整成功及终点审计，23:35 CPU队列1650585启动、queue-progress明确training_started=false/等待本项目前驱或其审计，正式根不存在、无agent。生产用独立queue_missing_dt辅助器覆盖agent退出但监督器尚审计窗口；原c768内置PID等待不直接用于长时队列。释放后仍核空闲与空间，不共享/等候外来作业或自动恢复。
- 实际旧四条SHA `d8997f64a301bfdcfe08758faeae8957a7395766`；新七条训练/CPU114及CUDA工程实测 `c7689d50f4caed0fef89912c65280b8d2558b126`；独立CPU队列及sv2实际17项验收 `2cccd5e006a639f10699907ad3f5b2f49c2c0b56`。c768两机各114项/三变体float32完整agent、7条真实size50m CUDA短程各4098动作/517更新/3×1/2报告已通过，保留旧算法梯度回归。未改训练源码/科学协议/容差；最终相对2ccc仅文档，相对c768另有CPU队列辅助器及其测试。
- 23:33—34六张指定卡空闲并重新CUDA/EGL实测通过，/data余sv2约764GB、sv3约237GB；启动审计时当前六卡仅本批作业、仍运行未释放，原sv3 GPU4—7已释放。sv3 GPU0/2外来作业不干扰，sv1本轮只读索引、未核GPU。两机canonical repo仍d899，独立训练工作树c768，sv2独立队列工作树2ccc；Git校验同步仅sv2/sv3，无sv1源码/GitHub推送，不改活动checkout或删除base-py311/产物。
- 报告`docs/RUNS_STATUS_20261101.md`，准备/控制器报告`docs/MISSING_DT_REALLOCATION_20261030.md`，小证据/hash/DELIVERY及可复制prompt `analysis/outputs/missing-dt-start-20261101/`。首选继续当前03监测六条/队列并完成七条终点和全32条原始return/快照/收据/W&B审计；结束前不重加权或称方法有效。旧跨机逐位/真实环境、hopper因果及GPU7并发限制保留，不自动建chat/发消息；用户既有草稿保留。

## 03合并窗口最新：补十一条Dt，sv3四条启动验收通过、sv2七条资源阻塞（2026-10-30）

- 用户新分配sv2七条GPU0—5/7（不使用sv2 GPU6）、sv3四条GPU4—7。仅补新增16环境中从未启动的11条Dt，seed0/clean纯视觉/size50m/alpha20/free_nats1/dyn-rep=1-.1/原梯度开关/百万动作/21×10/采样评价/独立RNG均不变，挂PD_1在线W&B和JSONL。不重跑原baseline/已完成Dt，不新增constant/shuffle或科学协议。
- 本机新增受限`run_missing_dt.py`及7项测试；53项本机、两服务器各107项CPU回归、CPU三变体float32完整agent精确一致、11张分配卡CUDA/EGL实测、两台零动作online logger全部通过。11条真实size50m CUDA/bfloat16短程各4098动作/517更新/3×1评价/2报告，5687更新/22报告/33点评价隔离审计通过；Dt活跃位置和非1实际权重已记录，工程作用不代表收益。实际代码/测试版本`d8997f64a301bfdcfe08758faeae8957a7395766`，Git bundle同步sv2/sv3干净源码，无sv1/GitHub同步。
- sv3于14:21:10启动四条Dt：GPU4 pendulum_swingup、5 reacher_easy、6 walker_run、7 walker_walk；监督器618156，agent618571/618580/618594/618917。14:25启动审计各零点10完整episode/0动作快照，316—437连续成功更新invalid0，68个C/G PID均为对应agent及子进程、四卡无外来，四条W&B running/full config及startup exact JSON/hash一致。本批新根`formal-m2-dmc16-missing-dt-20261030-sv3`，每任务dt-attempt02；正式百万动作尚未验收。
- sv2七条工程/配置/绑定齐全，但14:20启动前GPU2/4出现外来计算作业，14:25再核GPU0也新增计算/图形作业，正式根未创建、七条均未启动。旧blocked和新plan保留；已报告并请用户协调或重新分配，未共享/停对方/自行换卡/自动等待空闲启动。GPU1/3/5/7当时无C/G PID，不是预约；sv2 GPU6全程不使用，sv3 GPU6属于已分配范围。实际容量及冲突证据见工程根，14:25 /data余sv2约806GB、sv3约706GB。
- 14:29:36复核：sv3四条仍运行、2030—2153更新/最近开始动作10148—10640，invalid累计0、各0点10完整episode，四卡仅本项目进程；sv2 GPU0/2/4冲突仍在、正式根未创建。sv2本项目工程进程全部退出，sv3四张训练卡尚未释放；不能把本项目释放与卡空闲混同。因此新增32条为21条已结束/4条运行/7条未启动；加hopper六条为新增38槽位27条已结束/4条运行/7条未启动。既有10条完整指标/W&B收尾及17条待所属批次收尾的边界不变。新7/4条Dt分片结束时的shard_complete不等于全32条cohort_complete；仍需全来源原始return/快照/收据和线上最终审计，缺项不重加权，不作方法有效或跨seed结论。
- 配置/实现/完整命令、提交/同步、测试、逐卡PID/设备及限制见`docs/MISSING_DT_RUNS_20261030.md`；小证据镜像/哈希、DELIVERY和可复制prompt在`analysis/outputs/missing-dt-start-20261030/`。首选继续当前03监测sv3、在用户解决sv2资源条件后按原配置补齐七条并完成全32条收尾；可立即做只读监测，阻塞卡启动依赖明确资源。未自动建chat/发消息；本机其他AGENTS/CHATS/RESEARCH/STATUS/proposal/根TeX草稿保留，仅本段进入提交。

## 03合并窗口最新：全批27条已训练结束、11条Dt从未启动（2026-10-30）

- 13:40资源/索引及13:43—13:45线上只读核查：新增16环境×baseline/Dt共32条，21条已结束且有逐条audit，11条Dt从未启动；另hopper seed1/2/3六条全部accepted。新增38槽位共27条训练结束、11条未启动、运行中0。所有已结束条均百万动作/249493更新/21×10、退出0，线上27条finished且full config一致；本轮核对已有audit/终点评价/尾收据，不冒充新的一次全量原始产物重审计。
- 前八16条为8 baseline+cheetah_run/finger_spin两个Dt已结束，余6条Dt缺失；后八16条为8 baseline+finger_turn_easy/hard/walker_stand三个Dt已结束，余5条Dt缺失。缺失11条目录/manifest/评价均未创建，blocked不会自动重试。两个finger单卡4条和hopper六条已有最终统计/精确线上摘要，10份JSON/hash读回一致；其余17条仍run_audited_pending_group_and_WB，所属批次不完整，不能称全部验收完成。
- 历史原四任务四组16条completion/full_four_groups_complete=true及原表hash本轮只读复核一致；若仅盘点历史16+新增38，为54槽位43训练结束/11未启动，不是合并科研指标或方法有效结论。冻结协议、原始return和缺项不重加权保持；hopper六条可供05独立分析，不自动并seed0或新任务总体。
- 本轮训练/监督进程均已退出，原分配卡已由本批释放；13:40实际外来C/G卡为sv1 GPU0/1、sv2 GPU0/6、sv3 GPU0/2，项目未识别者不猜测、不干扰。sv1 GPU2—7、sv2 GPU1—5/7及sv3原六卡当时无C/G PID，但不是未来预约。sv1/data仅52,328,312,832字节，低于原单条60GB门槛；sv2约821GB、sv3约725GB。补11条须落实有效资源/空间方案，不自行迁移/共享或删原checkpoint/replay。
- 实际训练及既有测试版本：前八 `18b85c4d85ff456e6822626f17cb14cc0f44d2b8`；后八sv1 `6e468ad4a8252f21401eab6a9761626b33659f0e`、sv2 `dbd28b6703d51d61e5d3485dbfea6c413075ea85`；hopper `743089f40662b031a5cf2b6bf988decb52ba0b28`；额外上下文测试 `5c6dd4fae0b35ac9d769dd00bc7d2845dfd226eb`。本轮无代码/测试/训练操作，只本机状态文档提交，无服务器源码/Git同步或GitHub推送。当前各服务器仍各自确定运行SHA且干净，原AGENTS/CHATS/RESEARCH/STATUS/proposal/TeX草稿保留。
- 总报告 `docs/RUNS_STATUS_20261030.md`；小证据/配置比对/完成标记与表hash/DELIVERY及可复制prompt `analysis/outputs/all-runs-status-20261030-1340/`。首选继续现有03落实11条Dt资源和空间后补齐、完成全体收尾；可立即做CPU准备和只读复核，实际GPU启动依赖执行条件。已齐全hopper六条可供现有05分析；无新增科学协议选择，不自动建chat/发消息，不以交接替代缺失工作。

## 05 最新：hopper六段动作任务示意GIF（2026-10-26）

- 用户确认六段hopper动作示意，不作方法比较。已交付512×768、三行两列、横纵间隙/边框0、无文字、25 fps/6秒循环GIF；源为sv2既有seed0 baseline前六条保存链的末个完整episode各2—8秒片段，非六个训练seed或固定checkpoint评价。
- GIF在`analysis/outputs/hopper-demo-gif-20261026/final/`，同字节副本`docs/proposal/hopper_hop_任务示意.gif`；脚本`analysis/hopper_demo_gif.py`，报告`docs/ANALYSIS_HOPPER_GIF.md`。源hash/训练reward求和、量化前分格像素、全部150帧解码/时长/循环/无间隙及目视检查通过；GIF保留256色量化限制，无损参考帧留存，不作方法有效性结论。
- 实际采集/编码/检查SHA `fa5b54a7bf678717bc090cf8121da43700034998`，后续仅文档。本机提交，不推送/同步或写服务器，无训练/推理/环境/GPU，未核GPU现状。既有候选和TeX文件保留，素材副本不入Git；后续版式继续现有05，完整prompt在产物HANDOFF，不建chat/发消息。

## 03合并窗口最新：后八baseline全部逐条通过，Dt三运行五占用阻塞（2026-10-25）

- 14:10北京时间只读核查：八条baseline均100万动作、249493成功更新、21点×10完整episode、退出0，已有逐条audit/c；索引仍为run_audited_pending_group_and_WB，整批与线上汇总收尾尚未完成。本轮读取并交叉检查已有审计/收据/评价，不冒充重新全量终点复算。
- Dt实际运行三条：sv1 GPU2 finger_turn_easy最近更新开始动作310828（已7点评价）、GPU7 finger_turn_hard373088（8点）；sv2 GPU4 walker_stand394300（8点）。进程/监督器存活，最近收据invalid0、已完成点各10完整episode；14:11线上W&B三条running且完整配置一致。不是三条Dt已完成，不作方法有效或强M2结论。
- sv2 GPU0/1/2/3/5对应hopper_stand、pendulum_swingup、reacher_easy、walker_run、walker_walk的Dt在09:50—09:53交接时因外来/未核实GPU进程被门禁阻止，**从未启动**。14:10这五卡仍有root用户JaxGCRL目录作业，PGID1338124/1338125/1338126/1338127/1338129，非本批session；本批已释放五卡，但卡不空闲。没有终止外来、共享或换卡；原卡释放确认或用户重新分配前不补开。blocked不会自动变queued，不能向用户承诺自动空闲重试。
- 固定v1/后八范围/seed0/size50m/alpha20/百万预算/21×10/梯度/c与原始return协议均不变；baseline正式c已记录，未用工程c或新增constant/shuffle。三条Dt仍用原三卡，前八旧六条Dt缺项也保留，不挪本批资源、不自动汇成20任务完成。
- 实际运行与原完整agent/CPU/CUDA测试版本仍sv1 `6e468ad4a8252f21401eab6a9761626b33659f0e`、sv2 `dbd28b6703d51d61e5d3485dbfea6c413075ea85`；额外上下文测试 `5c6dd4fae0b35ac9d769dd00bc7d2845dfd226eb` 不变。两台工作区干净。本次无代码/测试/科学协议/实验启动或停止变更，仅服务器/W&B只读核查和本机文档提交，未同步服务器或推送GitHub；原未提交草稿保留。
- 详情 `docs/DMC8_REMAINING_RUNS.md` 最新段；小证据/版本/继续prompt `analysis/outputs/dmc8-remaining-status-20261025-1410/`。首选继续当前03监测三条、协调sv2五张原卡后补开缺失Dt，保留占用/故障/恢复限制；16条全量审计、正式c和raw-return/W&B收尾通过再供05。无需新chat，不自动创建或发送消息。

## 03合并窗口最新：后八任务八条baseline已启动，八条Dt排队（2026-10-24）

- 用户授权sv1物理GPU2/7、sv2 GPU0–5跑DMC16剩余后八任务、baseline→审计→Dt共16条。同卡分配：sv1/2=finger_turn_easy、sv1/7=finger_turn_hard；sv2/0..5依次hopper_stand、pendulum_swingup、reacher_easy、walker_run、walker_stand、walker_walk。冻结v1保持：clean纯视觉/size50m/seed0/alpha20、free_nats1、dyn/rep1/.1、原梯度开关、repeat1、百万动作含预填充、21点评价各10完整episode/采样/隔离流；正式baseline按原窗口S/N冻结c作审计，不跑constant/shuffle、不用工程c、不调alpha。
- sv2代码/实际测试/运行SHA `dbd28b6703d51d61e5d3485dbfea6c413075ea85`：本机49项、两服务器各93项CPU回归和各3条显式float32完整agent通过。sv1单卡分片代码/实际测试/运行SHA `6e468ad4a8252f21401eab6a9761626b33659f0e`：本机50项、sv1 94项/3条float32 agent通过，原共同参数hash不变；相对sv2只控制器分片及测试不同。补充仅测试SHA `5c6dd4fae0b35ac9d769dd00bc7d2845dfd226eb` 的2项启动环境回归在本机/两服务器通过，服务器执行已提交blob、不移动活动checkout。算法/梯度/协议runner/RNG源码未改，后续交付仅文档。
- 最终16条真实size50m CUDA/bfloat16工程短程通过：每条4098动作/517更新/3×1完整episode/2报告，共8272更新、32报告与48点评价指纹隔离；logging活跃非零/实际权重1，Dt实际权重非平凡。8卡实际CUDA UUID/EGL落卡及线上W&B生产logger验证通过；工程证据不代表正式百万步完整或方法有效。
- sv2六条baseline22:53启动；sv1 GPU7 hard于23:04启动；GPU2 easy两次在零动作JAX编译期SIGSEGV，旧目录全部保留，使用已验证的干净非交互父环境后原卡/原配置retry03于23:45启动，23:51通过0点10完整episode/1229成功更新/invalid0/17个同卡C/G PID/W&B完整配置与exact摘要读回。未安装依赖、未放宽检查或改科学配置；上下文恢复解决当前故障，尚未逐变量证明具体动态库内部根因。
- 23:52—23:54 sv1 GPU7后进入外来cgy PID2235998/2236001（PGID2235891），独占检查未通过，已报告、未停任何作业或换卡。**23:58:59 sv1 / 23:59:02 sv2复核外来进程已退出，八卡当时只发现136个本批C/G PID**，资源询问因冲突解除而无需调整；不把未答复当共享批准。八条baseline仍running、八条Dt queued，sv2约9.8万更新开始动作、GPU7约8.1万、GPU2约1.8万，全部收据连续invalid0，已完成评价点均恰好10完整episode，尚无训练卡释放。旧冲突证据保留，短时并发的性能/逐位数值影响未单独量化，最终有效性审查不得写全程独占；后继Dt仍逐卡require_free，旧快照不是当前预约。
- 当前正式根：sv2 `formal-m2-dmc16-remaining8-serial-20261024-sv2/`；sv1 hard `...-20261024-sv1-gpu7/`、easy `...-20261024-retry03-sv1-gpu2/`，均在 `/data/Policy_Discrepancy/runs/`。CPU监督器sv2=1251084、sv1 hard=2181514、easy=2232259。W&B在线 `ychan9liu-tsinghua-university/PD_1`，JSONL保留；完整百万步/21×10/c与终点指标尚未验收。前八旧批实际10条逐条audit、6条Dt未启动，整体不完整，不挪本批卡补跑、不自动拼总体。
- 报告 `docs/DMC8_REMAINING_RUNS.md`，本机小证据/命令/bundle/镜像/DELIVERY及继续prompt `analysis/outputs/dmc8-remaining-start-20261024/`。仅本机修改提交、校验Git bundle同步sv1/sv2；运行checkout固定上述两SHA，后续测试/文档只fetch对象，无GitHub推送/sv3访问或服务器直接改源码。原RESEARCH/STATUS/proposal/根TeX草稿保留，STATUS仅本节及日期进入本次提交。
- 首选继续当前03监测/修复/逐条审计同批16条，保留GPU7短时冲突限制，完成所有百万动作/21×10/收据/快照/正式c/raw-return与W&B后供05。可继续当前已授权八卡原队列；前八旧六条Dt仍依赖用户新分配，不挪本批卡，科学协议不重新等待确认。无需新建chat，不自动创建或发送消息；完整复制prompt见本轮HANDOFF-current03.txt。

## 03合并窗口最新：前八任务16条串行队列已启动（2026-10-22）

- 用户恢复并缩小为前八任务×baseline/Dt，总16次；sv1物理GPU0–7依次acrobot_swingup、cup_catch、cartpole_balance、cartpole_balance_sparse、cartpole_swingup、cartpole_swingup_sparse、cheetah_run、finger_spin，每卡baseline完成审计后Dt。其余八任务仍不启动，旧32条暂停准备记录保留。
- 冻结v1沿用：clean纯视觉、size50m、seed0、alpha20、free_nats1、dyn/rep1/.1、原梯度开关、repeat1、百万动作含预填充、21点评价各10完整episode/采样动作/隔离流；自身logging baseline按 `[100000,300000)` 活跃位置S/N冻结c作审计，本批不跑constant/shuffle，不填默认c、不调alpha。
- 实际代码/测试/启动SHA `18b85c4d85ff456e6822626f17cb14cc0f44d2b8`。sv1 85项CPU回归、CPU/float32完整agent和16条真实size50m CUDA/bfloat16短程通过；每条4098动作、517更新、3×1评价、2报告，共8272更新、32报告/48点评价隔离通过。logging实际权重1，Dt活跃门控非平凡；这些工程证据不代表方法有效或正式指标完成。
- 20:39:20北京时间八卡无C/G PID、/data约1.07TB可用；20:39:41启动CPU监督器3990345。20:44:22核查八条baseline均运行、0点各10完整episode、470—701连续更新、非法值0；136个项目C/G PID映射正确，无未知作业，八卡仍由本批baseline占用。八条Dt排队，未启动。线上W&B PD_1完整配置及独立startup摘要读回通过；JSONL保留。
- 正式根 `/data/Policy_Discrepancy/runs/formal-m2-dmc16-first8-serial-20261022/`；工程/探针/依赖/启动审计 `/data/Policy_Discrepancy/runs/engineering-dmc8-serial-20261022/`；本机小证据镜像 `analysis/outputs/dmc8-first8-serial-start-20261022/`。报告 `docs/DMC8_SERIAL_RUNS.md`，包含完整配置、命令、UUID/EGL与恢复限制。
- 仅以校验bundle向sv1同步干净确定提交；10个W&B依赖新增、旧版本未变、pip check通过。无GitHub推送、sv2/sv3同步、服务器源码手工修改或原运行覆盖。后续交付仅文档，运行checkout固定18b85c4；本机已有RESEARCH/STATUS/proposal/TeX草稿保留。
- 首选继续当前03合并窗口监测同批，完成各baseline审计后衔接Dt，全部百万动作/21×10/正式c/收据/快照/原始return与W&B终点审计后交05分析；当前没有完整正式结果。失败保留独立尝试，不把环境/replay不完整恢复称为无缝；不扩大任务/seed/组、不自行改变科学协议、不自动建chat或发消息。

## 05 最新：run视频两行11帧对比图（2026-10-22）

- 从用户指定的run完整MP4按walk最终版式生成2×11图，0/2/…/18/约20秒，末帧实际19.96秒标`≈20 s`；上baseline下PPD-KL，同列同刻。保留quadruped_run标题和480k—496k训练步数说明，无seed0/页脚，横向间隙0，行间24像素。
- 当前图`analysis/outputs/run-500k-eleven-frames-20261022/`，4442×954 PNG/SVG、22个MP4裁剪PNG、sampling/verification/命令/交接；报告`docs/ANALYSIS_RUN_MEDIA_V1.md`。视频hash、成图像素、SVG内嵌PNG/文字/同刻/间隙和目视检查通过，原视频未改，不伪造20秒帧；仍是480001—495985动作训练replay，不是固定50万步评价。
- 脚本`analysis/video_comparison_grid.py`新增受限run视频/源元数据支持，实际生成/检查SHA `55cae2ecf1142e29f51c79824338bd2966d723b4`，后续仅文档。本机提交，无服务器访问/同步/推送、训练/推理/环境/GPU，未核GPU现状，既有候选和根TeX文件保留。后续版式继续现有05，完整prompt在产物HANDOFF，不建chat/发消息。

## 05 最新：walk对比图改为11帧无横向间隙（2026-10-22）

- 按用户最终修订生成2×11对比图，0/2/…/18/约20秒；末帧实际19.96秒标`≈20 s`。上baseline、下PPD-KL，保留quadruped_walk标题和训练步数说明，只删除seed0及底部说明；横向间隙0，原24像素行间距保留。旧10帧图与无标题草图不覆盖。
- 当前图在`analysis/outputs/walk-250k-eleven-frames-20261022-v2/`，4442×954 PNG/SVG、22个原视频裁剪PNG、sampling/verification及交接；报告`docs/ANALYSIS_WALK_MEDIA_V1.md`。源视频hash/逐块像素/SVG内嵌PNG/同刻对应/标题文字/无横向间隙与目视检查通过，不插值/伪造20秒帧，仍为原训练replay证据。
- 实际生成/检查SHA `f76b3254270de1d04d0f363a3ca40b213bbb86a6`，后续仅文档；本机提交，无服务器访问/同步/推送、训练/推理/环境/GPU，未核GPU现状。既有候选和根TeX文件保留；图可直接用，版式后续继续现有05，完整prompt在当前产物HANDOFF，不建chat/发消息。

## 05 最新：walk视频同刻10帧对比图（2026-10-22）

- 按用户指定的完整MP4从首帧至末帧均匀抽10个实际视频帧，2行×10列，上baseline、下PPD-KL；每列严格来自同一帧，时间0.00—19.96秒。继承224001—239985动作训练replay的来源，不作为固定25万步评价录像。
- 脚本`analysis/video_comparison_grid.py`；图/20个裁剪PNG/时间与hash在`analysis/outputs/walk-250k-ten-frames-20261022/`，提供4148×1010 PNG及内嵌PNG的SVG。源MP4哈希匹配原manifest，20块成图像素与解码裁剪一致，PNG目视检查通过，不插值/挑结果/改原产物。报告索引见`docs/ANALYSIS_WALK_MEDIA_V1.md`。
- 实际生成/检查SHA `ffe82b95d18bdbf85758e2a0ecc4cdf53343053a`，后续仅文档；只本机提交，无服务器访问/同步/推送、训练/推理/环境/GPU，未核GPU现状，既有候选和根TeX文件保留。图可直接使用；如需版式调整继续现有05，按同一来源/时间/像素规则验收，完整可复制prompt在产物HANDOFF，不建chat/发消息。

## 03合并窗口最新：新增DMC16准备完成，实测暂停（2026-10-22）

- 用户指定除原四任务外的其余16个DMC视觉任务、logging baseline/Dt共32条，sv1物理GPU0–7，全部沿用原v1（seed0/size50m/alpha20/百万动作/21×10评价）；现场八卡已有Ray未知作业后，用户明确暂停实测、仅做准备。暂停优先，无训练/真实环境/GPU探针或空闲自动启动，不停止其他作业；原hopper六条不由本次任务操作。
- 新独立scope/preset/入口 `M2-v1-dmc16-seed0-baseline-dt` / `m2_dmc16` / `protocol_dmc16` 及 `--profile dmc16`，仅新16任务、seed0、baseline/Dt。原四任务四组与hopper补seed范围保持。接球成绩ID `dmc_ball_in_cup_catch` 显式映射为当前wrapper运行ID `dmc_cup_catch`。
- 代码及实际本机CPU测试SHA `e83a012a8dde255b8cb9df0631243eddab2454db`：33项通过，无跳过；32配置、16随机流映射、八卡各4条队列、暂停/范围/真实入口前置校验、c边界/旧产物兼容与分层统计fixture通过。Windows c写出修为LF，严格hash校验保留；原loss/agent/RSSM/probe及runner训练/评价/RNG逻辑未改。CPU函数fixture不是完整agent/CUDA或新任务验收，原算法测试保留未修改，本轮未重跑梯度数值验收。
- 原四任务16运行真实小镜像CPU复算require-complete退出0，3360 episode，tables.json逐字节不变。新32条无任何真实结果/c；没有constant/shuffle，不自动合并历史四任务或hopper新seed，不把baseline/Dt两组差异当强M2选择性归因。
- 资源唯一只读快照为14:52:17北京时间：sv1实际host lyg2153、八卡都有未知Ray计算进程、当时仓库313532571f0a0e39759b9229264361c0fd9bd48f干净；暂停后没有再访问/写服务器、安装或同步源码。此快照不代表未来资源状态，本批占用GPU为0。代码/测试/后续报告只本机提交，不推送GitHub，原RESEARCH/STATUS/proposal/根TeX文件保留。
- 报告 `docs/DMC16_PREPARATION.md`；本机证据/最终32配置/随机映射/队列/bundle/交付 `analysis/outputs/dmc16-preparation-20261022/`，最终交付SHA绑定plan-final。未来服务器实验根计划为 `/data/Policy_Discrepancy/runs/formal-m2-dmc16-baseline-dt-20261022/`，本轮未创建；所有authorized=false、launch_argv=null、EGL未绑定。
- 首选继续当前03合并窗口等待用户恢复实测；恢复后刷新资源/依赖/确定SHA、实际CUDA/EGL及16任务接口、完整agent/梯度/W&B验收，再做启动绑定和受限队列监督。完成百万动作、21×10、S/N/c、收据/快照、原始return/W&B审计后交05；不自动建chat/发消息，不因资源空闲自行启动。

## 05 最新：run 50万步附近训练录像（2026-10-22）

- 按walk相同的预先固定规则提取第一条保存链在50万动作前最后完整episode，两组实际480001—495985动作，原始return508.234756/582.275476；已交付带来源标签的完整20秒、固定8—18秒短片与接触图。PPD-KL前段翻转后恢复，完整视频保留该过程，不把短片当全程稳定性或固定50万步评价。
- 50万步正式10 episode均值baseline442.028357、PPD-KL725.627475，差+283.599118；逐episode/原seed映射/源hash/快照与已验收表一致。原评价视频未保存，两组50万步checkpoint/完成标记仍在。源帧hash、训练reward独立求和、MP4全解码及旧walk输入兼容检查通过。
- 脚本`analysis/walk_replay_media_v1.py`新增受限run/500000入口；报告`docs/ANALYSIS_RUN_MEDIA_V1.md`；产物`analysis/outputs/run-500k-media-20261022/`。实际采集/编码/检查SHA `aaff5e863d7a8941762c10a2ea19651f9e1d68fe`，最终后续仅文档。本机提交，无训练/模型推理/创建环境/GPU或服务器源码同步/推送，未核GPU现状，既有候选修改保留。
- 本次截取完成；如需严格50万步固定策略配对视频，首选用户继续现有03做只读checkpoint推理补录，GPU执行依赖用户资源分配；不得改协议或覆盖正式结果。完整prompt在产物HANDOFF，不建chat/发消息。

## 05 最新：walk既有训练录像提取与25万步核查（2026-10-21）

- 原25万步seed0评价均值baseline366.208158、PPD-KL854.627989，差+488.419831；原10个相同环境seed及source hash/快照核对一致。baseline有4个约725—779分episode，不能概括为每次不会走。
- 原评价无保存视频/逐步轨迹；两组25万步checkpoint及完成标记仍在。只读CPU抽取第一条保存链在25万动作前最后完整训练episode，实际224001—239985动作，片段return500.288589/940.728044。PPD-KL行走、baseline后段失稳；已交付带来源标签的完整20秒、8—18秒短片和接触图，不伪装成固定25万步评价。
- 脚本 `analysis/walk_replay_media_v1.py`；报告 `docs/ANALYSIS_WALK_MEDIA_V1.md`；产物 `analysis/outputs/walk-250k-media-20261021/`。源帧hash/reward求和、原评价交叉检查、MP4完整解码及目视检查通过，实际编码/检查SHA `200d7c74b5201a5921aaf228b661e827e00007a7`；后续交付仅文档。本机提交，不训练/推理/创建环境/使用GPU，不同步服务器或推送，未核GPU现状；既有候选修改保留。
- 首选用户继续现有合并窗口03，准备25万步checkpoint同原10个eval seed的配对推理录像；不得改协议/训练或覆盖原结果。GPU执行依赖用户资源分配，未由本轮指定；可先准备/审查方案。完整prompt在产物HANDOFF，不建chat/发消息。

## 05 最新：M2 v1三任务PPD-KL展示图（2026-10-21）

- 根据用户要求从原seed0复算表重绘三张图：学习曲线三任务一行且仅baseline/PPD-KL；主次指标差值保留三个对照；门控仅Mean Dt和Actual active weight。省略hopper_hop仅适用于本组展示图，原dt组仅改图例名，原四任务16条结果与总体指标不变。
- 新脚本 `analysis/plot_m2_v1_ppd_kl.py` 校验原输入哈希、21×10评价/点均值/主次差值/20个诊断区间；生成PNG/SVG并逐张目视检查，保存126个学习点、9组主次差值、240个门控区间及来源/版本/输出哈希。无平滑、插值、三任务总体或新显著性结论。
- 图表在 `analysis/outputs/m2-v1-ppd-kl-figures-20261021/`，说明及复现命令见 `docs/FIGURES_M2_V1_PPD_KL.md`，原报告已加索引。实际绘图与最终SHA记在产物manifest/HANDOFF；本机提交，无训练/GPU作业、服务器同步或GitHub推送，未核GPU现状。原未提交候选内容保留，首选继续现有05核对版式，不自动建chat/发消息。

## 03合并窗口最新：hopper六条补测已启动并挂W&B（2026-10-21）

- 用户已明确“启动实验”“挂上wandb”；15:25:39在sv3启动监督PID706010。seed1 baseline/Dt=GPU1/3、seed2=4/5、seed3=6/7，训练PID依次706087/706096、706106/706123、706492/706850；size50m/clean纯视觉/alpha20/KL与梯度/百万动作/21×10评价沿用已声明配置。本批无constant/shuffle，c仍由各自完整baseline按原窗口S/N冻结，未产生终点结果。
- 全部实际训练/监督/63项CPU测试/六卡CUDA-EGL探针/生产logger验证SHA `743089f40662b031a5cf2b6bf988decb52ba0b28`；本机修改提交、bundle同步sv3确定版本，未在服务器改源码。启动后仅文档本机提交并同步Git对象，server checkout保持实际训练版本；原RESEARCH/STATUS及docs/proposal候选修改保留。无GitHub推送或sv1/sv2同步。
- 15:36:30快照：六条均running，已成功更新3070/3151、3104/3104、3154/3093；每条首更新开始动作2032、收据连续invalid0、0点评价各10完整episode。各seed的两组294初始数组哈希一致、三个seed不同。102个本项目计算/图形PID均在指定卡；GPU0/2为未知作业不使用；本批六卡仍运行，尚无训练卡释放。快照不是后续实时状态。
- W&B `ychan9liu-tsinghua-university/PD_1` 六条在线，full config与manifest/计划精确一致。现有显式step日志首条暂缓存至下一step/结束，原节奏不改；独立startup精确JSON/hash与0点评价展示摘要已逐条在线写读验证，不是新增评价。SSH MaxStartups和首条summary缓存读取失败证据保留，均未造成训练重跑；TensorFlow profiler缺失为既有非致命提示，不改依赖/配置处理。
- 新监督器 `scripts/run_hopper_seeds123.py` 后台每30秒记录资源，结束后自动逐条及六条整体审计/c冻结/raw-return统计/W&B精确摘要；失败留错误与产物，不静默续跑/覆盖。启动已验收，完整百万步及终点产物尚未验收，不写成方法有效或hopper根因裁决。
- 报告 `docs/HOPPER_SEEDS123_RUN.md`。实际根 `/data/Policy_Discrepancy/runs/formal-m2-hopper-seeds123-baseline-dt-20261021/`；启动证据P `/data/Policy_Discrepancy/runs/hopper-seeds123-start-20261021-743089f/`；本机小镜像 `analysis/outputs/hopper-seeds123-start-20261021/`。首选本窗口继续监测/工程修复/结束审计，完成后交05；不再交原04、不建chat/发消息，保持已授权资源/协议。

## 03 最新：hopper seed1/2/3补测准备完成，未启动（2026-10-21）

- 用户确认只准备hopper的logging baseline/Dt共6条；sv3物理GPU seed1=1/3、seed2=4/5、seed3=6/7；每seed自身baseline按原[100000,300000)活跃位置S/N冻结c，本批不跑constant/shuffle、不复用seed0的c。仅准备授权，训练/GPU探针未启动。
- 新受限scope/preset/script只允许本任务、seed1/2/3、logging/Dt；原v1仍限seed0，科学模型/损失/梯度/RNG/预算/评价不改。新统计profile=hopper-seeds123保留逐seed、三seed均值与配对差值，不自动混入seed0或报告四任务总体。
- 实现/实际测试SHA `0b764bd91e0bc145f8b0c4234c9fefc719ad96eb`：本机27项、sv3两CPU设备59项测试通过；六份生产配置和随机映射已解析核对。原16条正式return表复算require-complete退出0且逐字节不变。最终交付后续仅文档，最终计划固定最终SHA；这是CPU准备，不是新训练或CUDA验收。
- sv3北京时间14:12:42只读快照：指定六卡无计算/图形作业，GPU0/2已有未知作业不使用；EGL实际映射仍待验证，计划egl/launch_argv为空，不能用旧快照当预约。W&B0.30.0按既有锁补齐10包，已有包版本全部保持、pip check通过；在线认证/上传未测试，无训练run。
- 报告 `docs/HOPPER_SEEDS123_PREPARATION.md`；本机证据 `analysis/outputs/hopper-seeds123-preparation-20261021/`；sv3准备证据 `/data/Policy_Discrepancy/runs/hopper-seeds123-preparation-20261021-cpu/`，最终配置包在repo `analysis/outputs/hopper-seeds123-preparation-20261021-final/`。本机修改/提交、Git bundle快进只同步sv3，无服务器直接改源码、无GitHub推送；原RESEARCH/STATUS候选与docs/proposal保留隔离。
- 首选用户明确启动后继续现有04；启动前核资源/EGL/账号、按六份配置监督并审计真实产物。原hopper失败/16条结果保留；补seed不自动解决gate/输入微扰因果问题，不写成方法有效，不自动建chat/发消息。

## 05 最新：官方任务/奖励归因分析（2026-10-20）

- 官方论文与dm_control1.0.48源码核对：hopper_hop实际reward=`1[.6<=躯干相对脚高度<=2]*clip(向前速度/2,0,1)`，高度项无margin，是硬门槛；站着不前进仍0，速度不必达到2才开始有奖。quadruped姿态项连续，完全直立静止已有.5reward。hopper倒地不触发失败终止，20秒/.02秒控制步截断1000动作，符合原产物。
- 上述任务结构能解释Dt长期零reward为何难以脱离，但不能单独解释为何本seed的Dt进入失败路径。原99.7026%零reward尚无法分成“高度没过”与“高度合格但速度非正”，因为缺逐步physics trace。不是奖励函数已被证明错误，不自行加站立奖励或改阈值/alpha。
- 报告 `docs/ANALYSIS_HOPPER_REWARD_OFFICIAL.md`；官方五文件与sv2安装hash一致，72个hopper+32个quadruped纯数值核对通过，证据在 `analysis/outputs/hopper-official-reward-20261020/`。实际分析HEAD `313532571f0a0e39759b9229264361c0fd9bd48f`，最终仅文档；本机提交，不推送或服务器源码同步，无训练/环境仿真/GPU作业，未核GPU现状。原未提交内容保留。
- mode修复已由03完成（下节），首选继续现有02制定高度/速度奖励入口条件与D/w联合诊断的协议或缺项清单；具体采样/checkpoint/新增运行/seed/预算/资源尚未批准，不重开冻结v1，不自动创建chat或发消息。

## 03 最新：mode诊断归约修复完成（2026-10-20）

- 实现/实际测试SHA `1ccde38c042d2264b5d5d1e6c93c658d170a9d88`：新增mode诊断语义版本2与坐标比例，因子/整向量按正确轴归约。旧无版本vector→真实因子、factor→真实坐标，旧整向量仍为null，不覆写旧日志或拼接同名跨版本曲线。
- 本机23项、sv1两虚拟CPU设备37项测试通过，无失败/跳过；四模式冻结输入D/w、损失/梯度、其他诊断及前后Ninjax key与修复前逐位相同。没有训练、环境仿真、优化器更新、GPU作业或新完整agent/CUDA验收，未查GPU现状。
- 只读补证16运行×20诊断区间，原1161个镜像文件hash保持；正式return统计require-complete退出0，3360 episode/336点/16条指标，tables.json与原05复算逐字节相同。未改变冻结v1、alpha/c/梯度/seed/预算及正式指标。
- 报告 `docs/CODE_MODE_DIAGNOSTICS_REPAIR.md`；新证据 `analysis/outputs/mode-diagnostics-repair-20261020/`，sv1 CPU日志/bundle在 `/data/Policy_Discrepancy/runs/mode-diagnostics-repair-20261020-cpu/`。本机提交、Git bundle快进sv1，无服务器源码直接编辑；不推送GitHub、不访问或同步sv2/sv3。原RESEARCH/STATUS修改和docs/proposal隔离保留。
- 本修复不是hopper失败根因结论；原整向量缺口及gate/输入微扰因果问题仍未解决。首选用户继续现有02制定/确认下一项辨别复验，具体干预/seed/预算/判据/资源尚未授权；不自动建chat/发消息或启动训练。

## 05 最新：hopper低回报专项排查（2026-10-20）

- 只读CPU检查sv2原hopper四组0步可信checkpoint及5120个replay chunk：四组294参数/状态数组逐位相同，首次2032动作/奖励/边界相同；图像相对baseline有25/32/30个1级uint8坐标差异。原16条预算/评价/c/指标完整性不变，另新增mode诊断语义缺口；不将原像素差异或Dt单独定为唯一根因。
- 训练replay独立累计：Dt 992完整训练episode平均return0.7842、799条零return，99.7026%实际动作reward=0；baseline/constant/shuffle均值255.56/242.71/222.75。5万—10万动作baseline有9407正奖励步、Dt仅52，后者模型奖励/优势长期低且Gaussian策略熵高，支持早期奖励积累/策略学习反馈未建立的失败链路；具体为何进入该路径仍未裁决。原始抽帧可见选定Dt episode倒地/翻转，不外推全部episode。
- 确认诊断字段问题：rep_probe的mode_vector_diff_frac实际是因子mode不同率，mode_factor_diff_frac是one-hot坐标不同率；整向量mode不同率缺失。本轮只记录/反例验证，不改训练源码，不把诊断错误当低回报根因。首选继续03修复诊断/核对语义；新的因果训练或多seed方案由02/用户决定。
- 报告 `docs/ANALYSIS_HOPPER_M2_V1.md`；输入/数组hash/训练阶段表/抽帧/反例/复算在 `analysis/outputs/hopper-diagnosis-20261020-inspection01`、`-frames-final`、`-final`。实际分析/20项测试SHA `2427710a7a89f2a5c90a08634a74a7a11ee87139`，后续交付仅文档；本机提交，不推送/同步服务器。保留原RESEARCH/STATUS修改及docs/proposal，不混入提交。本轮无训练/环境步进/模型推理/GPU作业，未核当前GPU占用，不沿用旧卡状态。

## 05 最新：M2 v1 seed0 分析完成（2026-10-12）

- 已只读核对sv2的16条最终审计（2026-10-10 22:59），逐条确认实际训练SHA `48d71f1541721208322442f5a1d5977124f3e888`、外部最终审计SHA `d239c3fb7e5b77ddf205fe994c46c2e5a6acce6a`及部署脚本hash；旧baseline/Dt attempt02源a8b6243另核对。16条全部验收完成并纳入，各100万训练动作、249493成功更新、21点×10完整episode，四个正式c按原始收据S/N复算一致，无工程数据或训练重跑。下面“仍运行/待终点”均为历史快照。
- 主指标Dt−baseline/constant/shuffle四任务等权分别 `+8.51/+5.21/+10.79`；末段分别 `−60.67/−74.40/−50.42`。walk两指标胜三个对照；run不胜constant；reacher排序混合；hopper Dt回报很低。单seed结果不支持一致的四任务M2主张，不作跨seed显著性/稳定性结论。
- 门控在活跃位置有实际减弱，Dt全程超额KL减少约5.81%/16.87%/20.75%/10.40%（hopper/run/walk/reacher）；constant窗口系数匹配不等于全程损失/梯度匹配。两quadruped组间0步评价return小差异保留；已知EGL像素边界不能直接当本轮因果解释。未保存逐reward/像素/实际梯度轨迹，不能声称逐transition累计独立复算。
- 交付报告 `docs/ANALYSIS_M2_V1.md`；输入/运行清单/哈希/六表/门控表/三张PNG与SVG/命令/测试在 `analysis/outputs/m2-v1-seed0-20261012/`，有效结果仅analysis-final。实际分析和19项测试SHA `ad325482a46b0c9dd3c3709cccca6a63bea9ba5d`，真实16条require-complete退出0，任务/总体/差值与04完全一致。最终交付相比分析SHA仅文档；本轮只本机提交，无训练源码变更、不推送或同步服务器，原RESEARCH/STATUS未提交归档隔离保留。
- 2026-10-12 12:43:45资源只读快照：原16条PID均退出，M2已释放sv2物理GPU0–7；八卡目前为其他目录作业占用（未知项目归属），不能称空闲，本轮未启动任何GPU作业。完整快照CURRENT_RESOURCE.json。
- 首选由用户继续现有02制定原四组多训练seed复验协议，检验walk正向与hopper负向可重复性；具体seed/任务覆盖/预算/判据/资源尚未批准，条件shuffle及DtLatch也不自动实施。完整02及可选04只读补证prompt见分析报告，不自动创建chat或发送消息。

## 04 最新：PD_1剩余constant / shuffle已启动（2026-10-10）

- baseline/Dt八条均结束，100万训练动作、249493成功更新、21点×10完整episode；四任务正式c与原始return指标已独立复核。原W&B收尾浮点末位失败保留，新的attempt02以精确JSON/hash审计，不改科学容差或重跑训练。
- 用户授权剩余两组，沿用sv2物理GPU0–3跑constant、4–7跑global shuffle，顺序hopper/run/walk/reacher；11:47:37启动前八卡空闲、无未知作业。八条新attempt01从头启动，科学配置不变，prealloc=False、JSONL+W&B PD_1、无scope，正式c不取工程来源。
- 全部训练源码48d71f1541721208322442f5a1d5977124f3e888；启动监督a8b62436827721c49a77af55663dbb5225e1ed39、最终CPU收尾测试d239c3fb7e5b77ddf205fe994c46c2e5a6acce6a，仅同步sv2确定Git对象/外部操作脚本，checkout保持48且干净，9项CPU边界测试及pip check通过。RESEARCH中的DtLatch未批准，不进入v1。
- 11:57:19新八条W&B运行、manifest精确一致、136个计算/渲染PID落卡正确；0点评价各10完整episode，收据连续invalid0。12:00:54均有3300余成功更新，0–7仍有本项目作业，无释放新运行卡；旧baseline/Dt全释放。旧八条attempt02已核验完整指标/精确W&B摘要，原失败保留无重跑；16条CPU依赖收尾PID3164661等待终点，不声称已验收新八条或方法有效。
- 最新配置/设备/PID/c/指标/目录/证据/修复和可复制交接见 `docs/FORMAL_CONSTANT_SHUFFLE_PD1.md`。首选继续现有04监测和16条终点审计，八条新运行尚未完成验收；下面资源/授权段均为历史快照，不自动创建chat或发消息。

## 04 最新：PD_1八条正式训练运行中（2026-10-09 22:55快照）

- sv2 GPU0–3四logging baseline、GPU4–7四Dt已按用户授权启动；全部prealloc=False、JSONL+W&B PD_1、无scope，冻结科学参数不变，同任务完整配置仅mode/logdir不同。八条训练来源SHA48d71f1541721208322442f5a1d5977124f3e888，服务器checkout保持该版本，外部监督/收尾脚本从确定Git对象导出，不改运行中源码。
- 22:55八条进程与W&B均running、无未知或错卡作业；已有成功更新/10完整episode评价，前三baseline已完成50000点。正式c及完整指标尚未产生；不能宣称全程验收/方法有效。八卡均仍有本项目作业，没有释放；constant/shuffle仍未授权。
- 正式现场报告/版本差异/修复与限制/命令配置索引/后续prompt见 `docs/FORMAL_BASELINE_DT_PD1.md`。CPU依赖收尾已部署，仍待真实终点触发；首选继续现有04监测和完整审计，不重复要求启动批准。

## 04 当前执行：baseline + Dt（2026-10-09）

- 用户已声明prealloc=False、JSONL+W&B PD_1、无scope，已启动sv2 GPU0–3四任务正式logging baseline；22:40:19均有成功更新及0点评价10完整episode，累计收据ID连续/invalid0，尚未完成。
- 用户追加授权四条Dt在sv2 GPU4–7并行直接启动，冻结科学参数不变、从头训练、不依赖正式c；22:43:41指定4–7无作业，启动准备中。constant/shuffle仍未授权。
- 正式训练源固定 `48d71f1541721208322442f5a1d5977124f3e888`；监督进程故障仅外部修复/接管，无已有训练重跑。详见 `docs/FORMAL_BASELINES.md` 最新段及独立outputs证据。baseline的正式c/最终指标、Dt最终指标/比较均待完整运行后审计，不能把启动验收写成全程通过。

## 04 正式baseline执行设置已明确（2026-10-09，当前）

- 用户声明prealloc=False、JSONL+W&B、暂不使用scope；新建project PD_1并纳入本次四条正式logging baseline。sv2物理GPU0–3已有分配与启动授权，冻结科学协议不变；前轮配置缺项已解除，准备验证日志并启动，不重复请求批准。
- 22:10:49（北京时间）sv2八卡均空闲、无compute/pmon作业；启动前重查，不将旧快照当当前状态。仅sv2 dreamer安装wandb==0.30.0及新增依赖，约束已有包不升级、pip check通过，base-py311保留；其他服务器不安装/同步。
- 详见 `docs/FORMAL_BASELINES.md` 当前执行声明；完整依赖/失败安装证据在sv2 runs/wandb-pd1-setup-20261009。正式c与指标仍待完整baseline后产生，其他三组不在本轮授权范围。

## 04 W&B登录只读核查（2026-10-09 21:58，当前）

- sv1/sv2/sv3当前SSH用户的netrc凭据均通过api.wandb.ai只读viewer在线校验，HTTP200，账号均 `ychan9liu`。三台项目dreamer环境未安装wandb SDK、当前PATH无CLI；有效凭据不等于已验证logger/run上传或具体项目写权限。
- 详见 `docs/WANDB_STATUS.md`，脱敏证据在 `analysis/outputs/wandb-login-check-20261009/`。无密钥输出/保存、登录修改、依赖安装、GPU使用或实验启动；当前baseline执行配置待声明的状态不变。

## 04 四任务正式baseline授权与启动核查（2026-10-09，当前）

- 用户已指定sv2物理GPU0–3并授权四任务logging-only baseline、正式c及原始return指标；Dt/constant/shuffle不在本轮范围。初查21:38:12与复查21:47:52（北京时间）八卡均空闲无进程，依赖检查通过；只同步实际执行核查的sv2至 `40fd48dcf735ac2ad56c5ff86e86c9eeeb9b5985`，其他服务器未使用。
- 四张指定卡重新逐PID验证CUDA UUID、logical0与EGL枚举实际落卡通过；探针无环境step/agent训练，已结束释放。四baseline仍未启动：仅prealloc/logger执行设置待用户与03声明；已确认科学配置不重选，现有启动授权不重复请求。已按生产Flags保存两方案各四份完整配置，只有jax.prealloc/logger.outputs不同。
- 详情/完整配置/证据/后续验收见 `docs/FORMAL_BASELINES.md`。当前无正式c或指标，实际资源需启动时重查；设置明确后现有04继续本次已授权任务，不用工程c、不改变协议、不自动创建chat或发送消息。

## 04 职责与启动边界更新（2026-10-09，当前）

- 用户负责服务器/物理GPU分配，关键实验配置由用户与03共同声明并落入文档；04负责执行、工程处理、检查和结果记录。详见根 `AGENTS.md` 与 `docs/CHATS.md` 当前规则。已确认科学协议不变，此次无训练授权、无运行或GPU核查。
- 下方正式准备报告/16条索引保留为历史审阅材料：sv2 GPU0–3是04的资源建议，尚非用户分配；prealloc=False/jsonl等04提出的执行选择须经用户与03共同声明后才可采用，不因已有静态验收成为批准配置。资源快照不代表当前占用，原证据/产物不改写。
- 后续用户明确启动任务且资源/配置明确时04自主执行，不重复确认；指定卡冲突等待用户调整，普通工程故障自行修复验证，保留失败且新尝试重跑；不改变关键配置或放宽检查。当前首选继续现有04，依据用户后续分配与用户/03声明安排已授权任务，不自动创建chat或发送消息。

## 04 正式配置与资源准备（2026-10-09，当前）

- 从干净的 `12e2c7051af561394009fcec075d69ca62f00251` 接手，未回退。新增只生成文件的 `scripts/prepare_m2_v1.py`，最后工具CPU配置解析/独立验收 `9b1c5bc9ee39c058c89ce46e49017753ec6ec64b`；agent/runner/模型预设/统计/依赖/科学协议未变。最终文档提交后新目录重生成的SHA以最终索引和交付回复为准，不算新增训练实测。
- 最终审阅包 `analysis/outputs/formal-m2-v1-preparation-final/`：16条新ID/目录计划，全not_started/未授权；12份完整配置、4份正式c未绑定且不可执行的constant模板、840个评价seed及完整随机流/命令/恢复依赖。统一prealloc=False/jsonl，四组仅声明字段不同；工程c不复用，正式指标仍空。
- 当前首选sv2四任务GPU0–3，同任务四组顺序共用卡；12e2c705上的单帧CUDA/EGL设备探针逐PID验证实际UUID与EGL枚举映射，训练动作/agent更新为0。sv1全卡已有作业，sv3 GPU0–2已有作业，未干扰；仅实际执行准备的sv2同步，其他两台只读。资源是快照，启动前须复查，4并行/百万步资源仍未长程实测。
- 详见 `docs/M2_V1_FORMAL_PREPARATION.md`、最终独立审计和delivery索引。当前准备任务完成，**未启动任何正式训练**；首选用户审阅后另行授权并继续现有04，先四正式logging完整完成及逐任务c冻结，再Dt/constant/global shuffle从头训练。无需02重选原始return，不自动创建chat或发送消息。

## 04 原始return复核与size50m CUDA补验（2026-10-09，当前）

- 接手/主要实测 SHA `27261bf6df875939195879ab23e1dc161507fe5c`，工作区干净。旧统计六张CSV及JSON表独立复算逐字节相同；102个输入/6项摘要产物哈希正确，三台实时96个源小文件与48个实际快照参数/counter重新核对一致。旧工程96episode/48点评价保留，正式AUC/tail/总体/差值仍null，require-complete生成审计后退出2。
- sv2四任务新增logging真实size50m CUDA：4098动作/517成功更新，report2049、3点评价×2完整episode。24条新schema v2记录/12个实际快照、8次报告和12次评价fingerprint交叉审计、四条完成重开通过；均独立目录，不合并旧baseline择优。工程c从本轮各自收据冻结并校验，正式freeze拒绝工程来源；未新跑constant或启动正式比较。
- 同提交冻结视觉输入的size50m CUDA两进程4次更新完全一致（容差0）；2049动作工程中断后拒绝续跑，关键checkpoint/日志哈希不变；真实收据的同ID幂等/冲突、重复持久化ID和零ledger偏离拒绝通过。零ledger项为小fixture，不伪装成原零动作故障目录续跑。独立CPU两逻辑设备39项回归通过。旧图像非逐位确定性、旧return无逐transition轨迹、真实长期训练边界保留。
- 资源核查发现EGL需单独绑定，执行包装器本机提交 `51ca42803e5c211d1be26cc164b9012f3388839d`，只新增MUJOCO_EGL_DEVICE_ID设置/记录，未改agent/runner/统计/协议。独立2048动作CUDA绑定补验正常完成，pmon主PID计算/图形和观察到的16个G进程均在sv2 GPU1。全卡断言因其他卡新计算进程失败的原始证据保留，按执行PID/图形归属核查；未放宽数值容差。其他服务器/卡映射仍须启动前确认。
- 详细报告/限制/可复制prompt见 `docs/SERVER_RETURN_CUDA_AUDIT.md`，原报告已加当前索引。新R为 `/data/Policy_Discrepancy/runs/engineering-return-cuda-20261009-27261bf/`；新增CUDA只在sv2，sv1/sv3只读审计未同步新提交、不触碰已有作业。表格/审计在analysis/outputs新目录，参数/replay留服务器，最终完整SHA及同步范围见完成回复。
- 本次有限工程补验完成，没有新agent/runner/统计故障或方法效果结论。原始return已确认，score不再阻塞；首选继续现有04准备正式完整配置、计算/EGL落卡和资源/新目录索引，正式百万步训练须用户另行授权。不复用工程c、不新增训练seed、不自动创建chat或发送消息。

## 03 原始 return 统计与结果表验收（2026-10-09，当前）

- 从干净的 `22fe4b3bf712bb43019f6e88001e4df6353cf324` 接手，按已冻结C/C.1实现原始return累加器、兼容评价schema v2及离线分层结果表；不改变算法或科研协议。代码首次提交 `6eef3696dbdfff74a6f9ede61c28186ecd0d93f7`，统计与相关回归最后实测 `1f5b5a385346da01e1108c90114fed721c8bbaf2`；完整CPU agent fixture实际 `c4de170928e9314559be793634209320d97fe0b4`，后续仅离线审计/测试/使用说明变化。详情与命令见 `docs/CODE_PROTOCOL_V1.md` 最新节、`analysis/README.md`。
- 本机15项手算统计测试、sv1 CPU两逻辑设备39项相关测试通过；连续动作完整agent的报告/评价/并行变体保留既有训练参数摘要与收据一致性，新schema的40条完整episode/20源快照审计通过。这是工程fixture；本轮没有新的真实任务训练、正式百万步运行或方法结论。
- 只读核查既有16个真实size50m主运行：三台服务器96个源文件哈希与本机小证据镜像一致；48个工程评价点/96条episode保留原始return、长度、seed和快照映射，源配置完整性核对通过。所有正式AUC/末段/总体/差值均null且标不完整，保留预算4098、每点2episode及缺失正式网格的原因；不换分母或替点。旧score完整性有已审查源码与快照审计依据，但缺少逐reward/terminal原始轨迹，限制明确记录。
- 产物在忽略的 `analysis/outputs/return-v1-engineering-audit-1f5b5a3/`、`return-v1-hand-fixture-1f5b5a3/` 及 `return-v1-validation-summary-1f5b5a3.json`；源镜像/实时哈希在 `return-v1-source-20261009/`。CPU agent产物在sv1 repo的 `analysis/outputs/return-v1-agent-fixtures-c4de170/`，测试/bundle记录在 `/data/Policy_Discrepancy/runs/statistics-raw-return-v1-20261009-c4de170/`，没有大产物入Git。
- 本次03任务已完成；首选用户手动交给**现有04**核查表格与真实源快照、完成前轮修复的size50m工程补验和完整配置/资源质量检查。原始return协议已确认，不再等待02选择score。正式实验依赖工程闭环及用户授权；正式c仍未产生，不用工程c替代，不自动创建chat或发消息。

## 02 原始 return 口径与 03/04 交接（2026-10-09，当前）

- 用户明确继续沿用冻结的原始 return 指标，02不重开全部协议。原始完整episode reward累计和（repeat已由接口累计）→评价点算术mean→每训练seed独立梯形AUC/固定100万预算→任务内seed平均→四任务等权；末段按实际步/快照选80/85/90/95/100万五点，不取最后五条日志。边界或冻结评价点缺失时标不完整，不缩区间、换分母或替点。完整定义和结果表要求见 `docs/EXPERIMENTS.md` C/C.1。
- `score/scores` 字段若实际累计原始环境reward，其语义明确为return；名称不表示归一化。暂缓score/归一化讨论，它不再是工程验收或正式准备的阻塞项。下方历史报告的“02口径待确认”已被本轮决定替代，其他工程事实与限制不变。
- 仅训练seed0仍有效，评价episode不是独立训练重复；无最小增益/等效阈值、跨seed稳定性或统计等效结论。本次说明不代表工程验收通过、不新增正式百万步实验启动授权。
- 本轮只读核对本机 `696fa9e0ded3a8da0864beadfb850af05668e1c3`：protocol_v1 已累计环境reward到scores并对其求mean；DMC ActionRepeat已累计reward，现有工程分析脚本尚无完整正式AUC/指定五点/跨任务汇总。只说明源码路径，未执行统计fixture或服务器测试。本轮只维护RESEARCH/EXPERIMENTS/STATUS，不改代码；03/04仍需按口径落实统计与结果表，核查现有产物。
- 首选下一任务交**现有03 chat**：补齐/核对统计代码、语义映射、完整性标记和结果表，用手算fixture验证，再由现有04对既有工程产物核查。立即可做本机实现/fixture及既有日志只读审计；正式运行依赖剩余工程验收、完整配置/资源核验和后续用户授权。由用户手动复制交接prompt，不自动创建chat或发送消息。

## 03 修复及真实 size50m CUDA 验收（2026-10-09，当前）

- 03 已修复 CUDA `report_on_batch` 设备数组切片的 transfer guard 异常：复用训练消费的主机 batch，主机切片后显式 sharded 传输；报告使用独立 seed，不重新抽 replay。已有不完整 checkpoint（包括动作数0）拒绝续跑，先核对 checkpoint 与持久化收据/评价；重复持久化 ID 拒绝，完成运行重开不追加。最后代码/测试实测提交 `20ab3486bfaa067cc7950d3f5c8408fe2ed92ec3`，22项相关 unittest 通过。详细提交、配置、命令、结果与边界见 `docs/CODE_SIZE50M_REPAIR.md`。
- sv1 真实 hopper size50m CUDA/bfloat16 报告开启运行：原4098动作/2049报告节奏在 `98225e...` 子进程完成2次报告；最新代码在独立2048动作工程 fixture 上完成报告与2点评价，并逐项审计报告/评价前后训练参数、训练计数、replay RNG/metrics 等未变。完成运行同配置重开退出0，关键产物哈希不变。原04零动作故障目录只读审计确认 checkpoint 更新0而收据0..4，明确拒绝。新证据目录 `/data/Policy_Discrepancy/runs/engineering-size50m-repair-20261009-98225e1/`。
- 真实同配置 hopper 的首个差异已定位为**视觉图像**：两次2048动作追踪中物理状态、动作、奖励逐转移相同，6条图像不同，第一条在动作0/reset；首批训练图像输入已不同、参数从首次更新后开始分歧。独立 DMC/EGL 固定物理轨迹复现单像素差异；冻结相同输入的真实 size50m CUDA 四次更新完全一致。该环境下不可要求独立渲染轨迹逐位一致；不以放宽容差掩盖差异。审计支持评价/报告未改变训练 RNG 来源，但不声称完整环境/replay 崩溃后无缝恢复。
- 03 当前任务工程修复已完成。该修复的下一步是由用户交回**现有04 chat**做确定提交的工程补验与质量核查；不自动发送。本轮02已明确原始return口径，score不再阻塞准备；新增统计落实先交03，正式百万步比较仍需工程闭环和用户授权。工程 c 不能用于正式运行；没有正式效果结论。

## 04 真实 size50m 集成试跑（2026-10-09，修复前记录）

- 本轮四任务×四模式 16 个真实 size50m 工程主运行全部完成；实际 SHA `1393548fb6e46d28f92a834204f7b09495a3eab1`。CUDA/bfloat16、seed0、alpha20、默认 batch16×64/16环境/train_ratio256；工程预算4098、网格0/2049/4098、每点2完整episode、窗口 `[2200,3600)`。每运行4098训练动作/517成功更新，评价与reset分开计数；工程c均有隔离来源/校验，正式freeze拒绝工程来源。真实快照/收据/窗口重算/重复ID拒绝冲突/完成重开检查通过。
- **整体验收未闭环**：sv1 hopper 开启 CUDA report 在 `report_on_batch` 切片触发 transfer guard 异常；独立评价补验及同配置重跑的最终参数不逐位相同，最早收据分歧在update2/动作2040。初始化294数组、训练随机counter一致，但不能因同配置自身差异就认定评价RNG泄漏。补验 SHA `63d36a111d247aa8cd05af173f74e43e4e35d2e1`，相对主矩阵仅外部脚本变化，算法/runner不变。03修复与定位后再由04补验，不在服务器直接改代码。
- 每任务工程c约0.99993–0.99995，活跃位置有非零但很弱的减弱；未改alpha，不把工程通过写成方法有效。正式c尚无、正式16条百万步比较未启动。用户本轮告知return/score正由02重审，旧v1指标记录保留依据，最终口径须由02确认并落入文档。
- 主运行采样峰值6.48–9.31GiB、预热吞吐约24.5–24.9动作/s；百万动作及210episode资源线性估算约11.6–11.7小时/条，不是长程实测。完整配置、资源、边界、故障与可复制03修复prompt见 `docs/SERVER_INTEGRATION_SIZE50M.md`；小证据索引含完整配置见 `docs/size50m_integration_evidence.json`。执行/分析脚本在 `scripts/run_size50m_integration.py`、`scripts/analyze_size50m_integration.py`。
- 产物根目录（三台各有其承担任务）：`/data/Policy_Discrepancy/runs/engineering-size50m-20261009-1393548/`。sv1承担hopper/quad_run，sv2承担quad_walk，sv3承担reacher；完整checkpoint/replay留服务器，失败目录保留。中断非零checkpoint明确拒绝续跑，未声称无缝恢复。补验曾与sv3后来新增GPU0作业短暂共享，污染资源数据已排除，supervisor已新增每次启动前占用拒绝检查。
- 本次sv3根分区已有约321–323GiB可用，旧“根分区满”不再是当前事实；没有清理基础环境/其他作业。正式启动条件是03修复和补验闭环、02最终指标确认、正式完整配置/资源核验以及后续正式运行授权。首选下一步交现有03 chat，由用户手动发送本报告prompt；不自动派发。

## 科研交接状态

- 01 · Idea 讨论已结束，02 · 方法与机制已接手并完成首轮源码只读核对；研究记录见 `docs/RESEARCH.md`。
- 用户已确认以《DT方法迁移研究说明.docx》为参考，不采用读取该文档前助手提出的假设。原文件路径、参考设计、工作假设、反例和未决问题均已记录在 `docs/RESEARCH.md`。
- M2 主问题、首版参考方法与四组已确定；四个 clean 仅视觉 DMC 任务统一 size50m，free-nats=1、dyn/rep=1/0.1，ac_grads=false、reward_grad=true、repval_grad=true。Dt/shuffle alpha=20；只用训练 seed0，不另设校准训练。baseline复用、半开匹配窗口/统计规则、100万动作步预算、21点评价与原始return指标继续冻结，指标语义见 `docs/EXPERIMENTS.md` 当前补充。03已实现协议链路并有缩小预算fixture/真实短跑证据；正式长程与03修复后的04工程补验不能因本次指标说明自动视为通过，统计代码/结果表仍须核查。
- 既有 DMC 适配与工程烟测、03 logging/overlay fixture 验收保留为工程证据，详见下文及 CODE_PHASE1/2；本轮协议链路验收见 `docs/CODE_PROTOCOL_V1.md`。正式 M2 比较尚未启动，没有效果结论。

## 已验证

- 本机代码仓库：`D:\program\PD\dmr3`；`origin` 为 `Ychan9Liu/Policy_Discrepancy`，`upstream` 为官方 DreamerV3。
- `sv1`、`sv2`、`sv3` 的代码位于 `/data/Policy_Discrepancy/repo`，Python 环境位于 `/data/Policy_Discrepancy/envs/dreamer`。
- 三台机器在提交 `e3f02248693a79dc8b0ebd62c93683888ddaccfe` 上通过了 `debug` 假环境的 CUDA 训练短跑，产生训练损失、日志和检查点。测试目录分别为 `/data/Policy_Discrepancy/runs/smoke-verified-sv1`、`smoke-verified-sv2`、`smoke-verified-sv3`。
- 三台机器已安装 `dm-control==1.0.48`、`mujoco==3.15.0`，`pip freeze` 哈希一致且 `pip check` 通过；依赖快照见 `env/requirements-dreamer.lock`。
- 三台机器的四个 DMC 任务均通过真实环境创建、reset、零动作步进及 64×64 RGB/EGL 渲染。动作维度依次为 4、12、12、2；仅视觉观测未暴露 proprio。
- 提交 `58cae2a` 在四个视觉 DMC 任务上均完成 1200 环境步的极小 `debug` 模型 CUDA 训练短跑，日志含训练损失；加载各自训练后检查点的独立 `eval_only` 均完成一条 1001 步 episode。运行目录见 `docs/EXPERIMENTS.md`。这是工程验收，不是 baseline 性能或 Dt 证据。
- `quadruped_run` 同一检查点与评估 seed 17 的两次独立评估得到完全相同的 episode return。`make_env` 入口四个任务同 seed/index 的首次物理状态重复、不同 index 不同；仍未证明完整训练轨迹或不同服务器之间逐位一致。
- `sv3` 的 DNS 已写入持久 Netplan 配置，GitHub 和 PyPI 访问已验证；尚未通过重启验证。

## 尚未决定或验证

- v1参数和原始return统计口径已明确，不再因score讨论等待02重审；当前只作描述性探索，不设最小增益/等效阈值，不作跨seed稳定性或统计等效结论。完整协议链路已有缩小预算fixture/主矩阵及03修复证据，04补验与统计实现/结果表核查仍须记录完成范围；正式c尚未产生，不由03自选，不反算alpha或新增训练seed。
- 更强 M2 所需条件打乱的分层/区间/边界与退化处理协议仍未定；基础四组及首版验收要求已确定，不受其阻塞。03 在已确定范围内制定并记录配置字段、测试容差、数值处理和 RNG/全局跨 shard 置换等工程方案。
- 真实任务的小模型训练/独立评估短跑及dummy协议fixture已有记录；本轮size50m四任务短预算资源/吞吐、独立评价和四模式主链路记录见上方，不能升级为百万步长程验收。真实训练重跑与隔离缺口需03/04闭环。v1只作单seed0探索，跨seed确认属于后续工作；短跑不代表正式任务效果。
- 源码审计发现原 DMC 适配器未接收和传递环境 seed；现已补上 `DMC(seed)`、视觉预设的 `use_seed` 及 `train_eval` 评估环境索引偏移。`make_env` 的环境初始物理状态重复性已验证，但 `train_eval` 合并路径及完整训练重跑未验证；直接 DMC 渲染有极少量 1 级像素差异。
- 正式实验启动、结果汇总和跨服务器分配脚本尚未实现。
- 历史记录：`sv3` 曾根分区满，用户当时决定跳过处理；2026-10-09 的04核查见本文件当前章节，已有约321–323GiB可用，产物仍放 `/data`。

## 02 首轮方法核对进展（2026-09-27 历史记录）

- 核对提交：`f0236c7699a87895a8cfde8e13a0d1c198024f75`。接手时仅研究、状态文档有未提交修改；本次保留并增补这两份文档，未提交或推送。
- 已读取原始 DOCX 全文，并只读核对 latent、actor、KL/free-nats、replay/reduction、初始化及梯度路径；具体源码定位见研究文档。这些是源码事实及数学分析，不是运行验证。
- 确认源码中的 bounded_normal 是均值经过 tanh 的普通 Gaussian，执行动作另有裁剪；参考 Dt 对应裁剪前分布。rep KL 当前对 replay context 处理后的 `[B,T]` 直接求均值，没有 padding loss mask。
- 明确 gate 分支梯度隔离不意味着 rep 对 history 的总梯度为零，也不意味着共享参数更新后的 prior 轨迹不变。
- 研究文档新增科学主张层级、常数匹配与条件打乱的证据边界、alpha 校准退化处理、机制反例和 03 候选验收清单；均标明助手建议及未决项，尚未成为用户确认的算法或实验协议。
- 未修改算法代码，未启动 pilot、训练或实验，未运行算法数值/梯度测试。`docs/EXPERIMENTS.md` 继续保持协议待确定。

## 2026-10-08 方法选择与规范（v1 前历史记录）

下述当时未决状态及后续工程阶段记录保留；本轮 size50m、KL/梯度设置、alpha=20 和 seed0 的选择已更新，以文末 v1 状态为准。

- 已确认：M2 为主问题；首版采用参考定义，仅连续动作与标准 RSSM；至少 baseline、Dt、预先校准固定常数、全局打乱四组，更强 M2 再加条件打乱。
- 已确认：mode 漏检、早期不敏感、干扰误敏感和延迟线索列为局限并诊断，暂不叠加修复。
- 已整理 `docs/METHOD_SPEC.md` v0.1：数学/梯度路径、对照语义、诊断义务以及回退、free-nats、随机流等验收要求；研究决策和实验边界已同步到相应文档。
- 方法实现规范已可交接 03，实现可配置 overlay 与验收测试；任务、观测、校准规则、主指标、训练 seed、预算等实验字段继续按用户指定顺序固定，随后确定 alpha/c 与效果判据，不阻塞可配置方法的实现。
- 2026-10-08 先前检查现有任务列表及 Codex 归档时未找到 03 接收任务，交接材料当时已准备但未发送；指定或创建接收任务属于交接安排，不表示方法或验收仍待确认。本次一致性核对不派发实现任务。
- 先前源码核对基准为 `f0236c7699a87895a8cfde8e13a0d1c198024f75`。本次保留原有未提交文档内容并修正歧义，仅提交上述四份文档、推送 main；实际提交 SHA 和 Git 状态以完成报告及 Git 记录为准。没有算法修改或实验运行，验收条件尚未实际通过。

当时下一步是选择模型规模并补齐校准与评价；本轮模型规模已固定为 size50m，剩余协议和资源核验继续推进。正式比较不因已有方法规范或 fixture 验收自动启动。

## 03 第一阶段 logging-only 工程验收（2026-10-08）

以下两阶段保留当时的测试、同步状态及推进建议；“多 seed 比较”等旧建议由文末已确认的单 seed0 v1 替代，历史推进建议不构成本轮实验授权。

已在交接基准 `e935ff7` 之上实现可配置的只读 Dt 探针，并在 sv1 的 CPU/float32 与 A100 CUDA/bfloat16 完成未改动源码、off、logging-only 三模式的完整连续动作 agent 比较。各后端 618 个共享数组逐位相同，覆盖初始化、原始梯度、采样特征、分支 loss、训练状态和一次更新；五项解析 KL/配置/梯度 unittest 通过。诊断包含 Dt、候选 w、raw rep KL、free-nats 活跃比例及活跃区权重；训练目标仍为 baseline。代码验收提交 `a8ca727`，详细配置、命令、容差、环境、结果、性能样本和风险见 `docs/CODE_PHASE1.md`。这是极小工程 fixture，没有四个真实任务的正式效果结论；第二阶段 Dt/constant/shuffle 尚未启动。

本机和 sv1 的 `/data/Policy_Discrepancy/repo` 已通过 Git bundle 同步到同一提交；截至本记录，`origin/main` 仍停在 `e935ff7`。本机 GitHub HTTPS 不可达，sv1 的 HTTPS 推送缺少凭据且随后连接超时；远端推送待网络/认证恢复后完成。验收产物仅在 sv1 的 `/data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/`。

## 03 第二阶段 overlay 工程验收（2026-10-08）

用户已验收第一阶段，第二阶段完成 Dt、训练前校准后固定的 constant 配置入口和全局 shuffle；条件 shuffle 未实现，等待更强 M2 协议。算法提交 `9e0fb8e`，后续测试脚本提交和服务器实际运行提交详见 `docs/CODE_PHASE2.md` 与本机 Git 历史。gate 只乘在 free-nats 后、原 reduction 前的 rep KL；probe 与 shuffle 不消耗原 Ninjax 随机流。所有 alpha/c/seed 均为工程 fixture，未定科研协议没有被填为正式决定。

sv1 的 CPU/float32、A100 CUDA/bfloat16 完整连续动作 agent fixture 已验证：off、logging、Dt alpha=0、constant c=1、shuffle 全 1 均与未改动 `e935ff7` 基准的 789 个共享数组逐位相同；有效三模式各有 490 个未改动数组逐位相同，活跃 rep loss 与原始梯度按预期变化。9 项 unittest 通过，包括两卡跨 shard 全局置换；两卡完整 shuffle agent 也完成初始化、loss、梯度和一次更新。补充 replay context 开关、内部 reset、不同 imag_last、连续 chunk，以及 free-nats 全不活跃诊断。测试目录为 `/data/Policy_Discrepancy/runs/verify-overlay-phase2-64459ae/`。此验收只证明工程语义，不提供四个真实 DMC 任务上的性能或 M2 结论。

下一步可在已确认四个 clean 仅视觉任务上做**工程集成试跑**，逐次记录确定提交、完整配置、seed、服务器/GPU、环境版本、独立产物目录及 Dt/权重/活跃性/活跃位置作用；先核查运行路径和资源。正式多 seed 比较仍须冻结模型规模、baseline、校准规则及 c 匹配口径、主指标、训练/评估 seed 列表、预算和效果判据。`origin/main` 的推送状态以最新 Git/网络核验为准。

本阶段本机 `git push origin main` 因 GitHub HTTPS connection reset 失败，远端仍为 `e935ff7`；本机提交保留，sv1 已用校验过的 Git bundle 快进至相同确定提交。bundle 路径、SHA-256 和运行提交详见 `docs/CODE_PHASE2.md`；网络恢复后再推送，不在服务器直接改项目源码。

## M2 实验协议 v1 草案（2026-10-08，确认前历史）

本节保留上一轮草案与当时未决状态；下节为本轮确认后的 03 交接状态。此处“待确认”不再适用于已冻结的 v1。

- 已确认设置见本文件开头及 `docs/EXPERIMENTS.md`；候选 baseline 复用与 c 匹配、窗口/预算/评价/指标均明确标为待确认，没有新增 c 数值或效果结论。
- 草案：每任务一次 seed0 logging-only baseline 同时作正式参照和 c 数据来源；10万至30万动作步实际训练 batch 上按活跃位置统计和/计数匹配本任务 c。constant 前冻结，不从 checkpoint 分叉，不随 return 调窗口/alpha/c。该方案仍须用户确认。
- 草案：每组100万训练动作步，repeat=1；初始化和每5万步含终点评估10完整 episode，统一采样；主指标梯形 return AUC/预算，次指标末20%评价点均值，报告三项差值及四任务等权汇总。窗口/端点/计步/随机性/指标规则未自动成为决定。
- 单训练 seed0 只支持探索性判断；episode/时间点不是独立训练重复。D=0、w=1 正常；报告实际作用不足，不自动改 alpha，工程非有限值另报。
- 04 实施前需补齐/核验：训练更新级 source_active_weight_sum/active_count 和动作步 stamp；排除 reset 的动作计数与精确终点；独立训练/评估/report/shuffle RNG；按动作步评价而非墙钟；恰好10完整 episode 的不可变参数评估；完整配置/提交、size50m 资源、checkpoint/统计去重恢复。具体源码事实与要求见协议工程表。
- 当前本机核对 HEAD `4b8b00f54e7f1f5bbaa959391d0c18beba2952f2`，工作区原先干净，main 相对本地 origin/main 记录 ahead25；本轮未核验远端或服务器同步状态，也未推送这批既有提交。仅修改三份指定文档，算法及 METHOD_SPEC 未修改，未启动运行或实验。

下一步由用户确认候选窗口、预算、评价和匹配/汇总规则，再由 03/04 给出并验收工程方案；候选未冻结或工程条件不满足时不启动正式比较。

## M2 实验协议 v1 确认与 03 交接（2026-10-08，本轮）

- 用户已确认：仅训练 seed=0；每任务一次 logging-only baseline，同时是正式 baseline 与 c 匹配来源，不另设校准训练，实际 rep 权重1。
- 窗口为 `[100000,300000)`，按优化更新开始时的累计训练动作步归属；统计每次成功优化更新的有效 loss 位置。raw rep KL>1 时累计 S=sum(active*w)、N=sum(active)，w=1/(1+20D)，c=S/N。正常 replay 重采样保留计数，仅恢复/日志重写导致的同一 update ID 重复记录去重。
- 四任务各计算并冻结 c，constant 运行前冻结；窗口无活跃位置或出现非法值时停止并报告，不填 c=1、不自动调整 alpha=20，不随 return 改窗口/参数。
- 四组从头训练；每组100万训练动作步，包含 replay 预填充、排除 reset 和评估，repeat=1。评价固定为0、5万……100万共21点，每点恰好10完整 episode，固定参数快照、采样动作和独立评价随机流。
- 主指标为梯形 return AUC/100万；末段为80万、85万、90万、95万、100万五点的算术平均。报告 Dt−baseline、Dt−constant、Dt−shuffle，四任务原始 return 等权汇总并保留逐任务。只作单 seed 描述性探索，不设最小增益或等效阈值，不作跨 seed 稳定性或统计等效结论。
- 03 按已确认协议落实统计收集、成功更新/ID/恢复去重、c 冻结产物、非 reset 动作计数、精确网格、固定快照评价、独立 RNG 与完整配置，验收条件见 `docs/EXPERIMENTS.md`。剩余事项是工程实现/验收及运行产物，不是研究参数待选。方法假设仍无正式效果证据。
- 本轮沿用源码核对基准 `4b8b00f54e7f1f5bbaa959391d0c18beba2952f2`，保留上一轮三份未提交草案并纳入本次文档提交；未改算法或 METHOD_SPEC，不推送、不启动实验。交接提交 SHA、提交范围和 Git 状态以完成报告为准，不在文档中写自引用 SHA。

下一步由 03 按本版本实现和验收工程条件；04 核实同提交、完整配置、随机流、环境/GPU 和产物记录后，依据后续运行指令实施。已确认 v1 无需再次等待窗口、预算、指标或阈值选择；本轮不启动实验。

## 03 协议 v1 工程链路验收（2026-10-09）

代码与测试验收到 `b20ffeda07fe741deb1b9ee285174a3e34bcaaf8`。本机实现 `protocol_v1`：预填充也计入的非 reset 训练动作计数、多环境精确停止、动作步评价网格和固定参数快照、独立评价与报告随机流、每成功优化更新的窗口 S/N 收据、冻结 c 产物与检查点审计。正式预设固定 size50m、纯视觉 DMC、seed0、alpha20、100万动作、21 点各 10 完整 episode；缩小预算 fixture 不改变正式口径。具体提交、命令、容差和运行目录见 `docs/CODE_PROTOCOL_V1.md`。

sv1 的 17 项算法/协议单元测试通过。缩小预算连续动作完整 agent 中，报告开关、评价 episode 数及并行环境改变后，baseline 最终参数、训练动作/更新计数和匹配收据逐位一致；Dt/constant/shuffle 也完成动作网格与预算。CPU/float32 与 CUDA/bfloat16 八模式对未改动基准的完整 agent 回归均通过；结果与产物索引见工程报告。中途 checkpoint 能保存累计量和计数，但环境/replay 随机内部状态无法完整恢复，runner 明确拒绝不一致的训练续跑。没有执行正式四任务百万步训练，没有正式 c 或方法效果结论。

下一步由 04 在确定提交、干净服务器工作区及独立运行目录上检查四个真实 size50m DMC 任务的显存、吞吐、日志、作用诊断、评价快照和磁盘，再按已冻结 v1 执行正式比较；中途失败不伪装无缝续跑。本轮 GitHub 远端初次查询曾出现 HTTPS connection reset，重试后确认 `origin/main` 为交接基准 `e935ff7`，随后成功快进推送；服务器也通过可校验 Git bundle 快进同步。最终提交及 bundle 哈希见本轮交接报告。
