# 03合并窗口 · Vision-task20 seed1 baseline/PPD-KL（2026-11-03）

**2026-11-04续启**：用户明确接受sv1初始容量缺口并继续清理。01:48在sv1 GPU2—7实际启动六条baseline，18条排队；24条真实CUDA短程及01:56启动审计通过。sv2原七baseline/七排队不变。容量例外、实际运行bef2948、首批任务/收据/线上与剩余条件见`VISION20_SEED1_SV1_RUNS.md`；下方sv1未部署/空间阻塞为11月3日历史状态。

## 用户确认与范围

用户要求vision-task20、原配置、训练seed1、baseline与PPD-KL全部测试。用户进一步确认：资源为sv1物理GPU2—7、sv2物理GPU0—5/7（不用sv2 GPU6）；**复用已完成hopper_hop seed1 baseline/PPD-KL一对，其余19任务新跑38条**。最终20任务×两组40条记录中两条保留原运行身份，不重训hopper、不把旧源改写成新提交。用户同意sv2先跑、自己释放或扩容sv1 /data。

PPD-KL为现有`agent.rep_probe.mode=dt`的展示名，baseline为`logging`、训练实际rep权重1。本批不增加constant/shuffle，不实施候选DtLatch/条件shuffle，不改变已有seed0或hopper seed1/2/3的结果。此前05分析排除acrobot是那份分析范围；本次用户明确vision20全任务，所以本批包含acrobot。

20个运行task（均`dmc_`前缀）：hopper_hop、quadruped_run、quadruped_walk、reacher_hard、acrobot_swingup、cup_catch、cartpole_balance、cartpole_balance_sparse、cartpole_swingup、cartpole_swingup_sparse、cheetah_run、finger_spin、finger_turn_easy、finger_turn_hard、hopper_stand、pendulum_swingup、reacher_easy、walker_run、walker_stand、walker_walk。仓库score清单的ball_in_cup_catch对应运行cup_catch，保持原别名，不额外增加任务。

## 原配置声明

独立范围`M2-v1-vision20-seed1-baseline-dt`，preset `m2_vision20_seed1`，script `protocol_vision20_seed1`，仅上述20任务/训练seed1/logging或dt。继承原m2_v1：clean、仅64×64 RGB视觉，size50m，alpha20、free_nats1，dyn/rep外层系数1/.1，ac_grads=false、reward_grad=true、repval_grad=true、repval_loss=true；batch16×64、replay_context1、16训练env、train_ratio256、repeat1、CUDA/bfloat16、prealloc=False，正式report关闭。JSONL+在线W&B `ychan9liu-tsinghua-university/PD_1`，独立run ID、resume=never。

每条100万训练动作包含replay预填充、排除reset/评价；0..100万每5万评价，共21点，每点固定参数快照、采样动作、独立评价随机流、恰好10完整episode。各任务seed1自身logging baseline提供c审计来源，按成功更新开始动作`[100000,300000)`归属，raw rep KL>1有效loss位置累计S/N，候选权重1/(1+20D)，logging训练实际权重1；正常replay重采样保留，仅同update ID重写去重，无活跃/非法停止报告，不填c=1或调alpha。Dt不需要c，不从baseline checkpoint分叉。

原始reward完整episode求和不折扣/不除长度/repeat不重复累计，点评价取10episode算术平均；按每训练seed实际动作x梯形AUC/固定100万，末段只取80/85/90/95/100万指定五点。`vision20-seed1`统计profile独立20任务/两组/seed1，缺失不重加权；复用hopper只接纳原hopper范围seed1，保留源manifest，不接入seed0/2/3或其他任务。全20任务汇总为本次用户确认范围，不自动与seed0池化，不从两组单seed比较推出强M2选择价值、显著性或跨seed稳定性。

## 资源与队列

22:42北京时间：sv1指定六卡、sv2指定七卡均无C/G PID；sv1 GPU0/1为未知项目Ray作业不干扰，sv2 GPU6即使空闲也不使用。磁盘sv1 /data余**1,234,591,744字节**，无法运行；sv2余640,337,616,896字节。用户明确sv2可先跑、sv1自己释放/扩容，未授权删除旧checkpoint/replay或base-py311。sv1保持准备状态、未部署/启动；启动前须重核GPU/容量并完成真实CUDA验收。

每任务一张固定卡、baseline→审计成功→PPD-KL串行，同卡下一任务再依序；前驱失败/审计失败时阻断后继，不等待外来作业空闲重试、不共享或替换卡。新尝试独立目录，不称环境/replay可无缝恢复。初始容量门槛为并行卡数×60GB，后续每次新训练再检查至少60GB；这是工程门禁，不改科学预算/replay大小，容量不足则阻断并报告，不删数据。

| sv2物理GPU | task（先baseline后PPD-KL） |
| --- | --- |
| 0 | cartpole_balance_sparse |
| 1 | cartpole_swingup |
| 2 | cartpole_swingup_sparse |
| 3 | cheetah_run |
| 4 | finger_spin |
| 5 | finger_turn_easy |
| 7 | finger_turn_hard |

sv2首批七任务14条；sv1待空间的12任务24条按GPU2—7两轮：第一轮quadruped_run、quadruped_walk、reacher_hard、acrobot_swingup、cup_catch、cartpole_balance；第二轮hopper_stand、pendulum_swingup、reacher_easy、walker_run、walker_stand、walker_walk，逐卡各自baseline→dt→下一任务baseline→dt。每次前驱完成后重新核实际C/G占用，不把旧空闲当预约。

## 实现与验收

代码提交`77699380851701f7a03a9d13985c5cc0de5d0702`：

- `protocol_scope.py/configs.yaml/main.py`增加独立20任务seed1入口，旧v1/DMC16/补seed范围继续严格关闭；训练runner函数/计步/评价/RNG代码不改。
- `freeze_c.py`扩展原S/N冻结入口的显式scope，原数值计算/去重/精度不变；`analysis/return_metrics.py`增加独立profile和严格复用身份检查，AUC/末段公式、均值容差`atol1e-9,rtol1e-12`不变，配置/离散边界/hash精确校验。
- `prepare_vision20_seed1.py`生成38条完整配置、13卡队列与两条hopper来源，不导入agent/环境，不使用GPU，launch_argv为空、授权false；`run_vision20_seed1.py`受限分片、真实设备/日志/任务验收绑定和串行监督，原GPU门禁/环境清理复用。按任务结束审计百万动作/21×10/原始收据/快照/正式c，不把进程退出0当通过。每服务器partial表保持cohort_complete=false；全20及线上精确收尾待全部来源齐全后独立完成。
- `tests/test_vision20_seed1.py`覆盖全部40配置、旧scope不放宽、指定卡/seed/模型参数篡改拒绝、输出不覆盖、baseline审计前不可启动dt、前驱失败阻断dt；手算CLI生成20×2×21×10=8400episode的fixture，两个总体AUC9.5/11.5和缺失hopper不重加权符合预期。fixture没有训练或效果证据。

本机原协议/return/scope回归39项通过（21.146秒），最后新增队列检查后vision20模块7项通过（10.723秒），均对应上述代码文件内容。此前37项首轮也通过。命令/完整日志在本机outputs，不以本机CPU替代真实CUDA；服务器同提交算法/梯度/完整agent/CUDA/EGL/logger/实际任务短程与正式启动证据待实测记录。

sv2同7769938提交121项CPU回归通过（78.031秒）；三条CPU/float32完整agent比较report开关与评价episode数量，训练参数/计数/batch key/成功更新收据、匹配累计和共同评价逐位一致。七张分配卡实际CUDA/EGL探针同卡通过；零动作seed1 logger线上finished、写读成功，训练动作/agent更新均0。14条真实size50m CUDA/bfloat16工程验收执行中，各工程预算4098、评价间隔2049/每点1episode、匹配窗口[2200,3600)、报告间隔2049；这些仅工程参数，正式百万/21×10配置不变。

复用hopper两条源代码均为`743089f40662b031a5cf2b6bf988decb52ba0b28`，本次只读核实accepted/百万动作/249493更新/21×10、W&B finished/完整配置/精确JSON摘要hash一致。38条新配置与其对应组逐字段对比，仅差script/task/logdir（包括模型、优化器、随机流设置全部其余字段相同），完整diff证据在本机outputs。复用源未修改，保留原训练日期和seed1身份。

```powershell
.\analysis\outputs\dmc16-preparation-20261022\venv\Scripts\python.exe -m unittest -v tests.test_vision20_seed1 tests.test_dmc16_preparation tests.test_hopper_repeat_scope tests.test_return_metrics
```

代码通过Git bundle同步**仅sv2**，独立干净工作树`/data/Policy_Discrepancy/checkouts/vision20-seed1-776993808517`，未切换canonical repo或旧训练工作树。bundle SHA256 `1037390001e0472665bbe1cb846f67b4b4e4eed291a58b7330727ee603cbfd9f`，verify/fetch/实际HEAD校验记录在工程根；无sv1源码同步或GitHub推送。没有服务器直接修改源码、依赖安装、全局关闭transfer guard或放宽容差。保留本机其他chat的AGENTS/CHATS/RESEARCH/STATUS/proposal/根TeX草稿。

## 产物与当前执行状态

新正式根`/data/Policy_Discrepancy/runs/formal-m2-vision20-seed1-20261103-{sv1,sv2}/`，各task下`logging-attempt01`/`dt-attempt01`；实际启动后保留bound-plan/actual-RUN_INDEX、PID/退出码、stdout、manifest/RNG/收据、评价快照、c_frozen及原始return审计。工程根`/data/Policy_Discrepancy/runs/engineering-vision20-seed1-20261103-sv2/`保存同提交CPU/完整agent、设备映射、online logger、14条真实size50m CUDA/bfloat16短程及后续启动记录。复用hopper源位于sv3 `formal-m2-hopper-seeds123-baseline-dt-20261021/seed1/`，源actual index在本机保存且保留原SHA/审计身份。

本机`analysis/outputs/vision20-seed1-preparation-20261103/`存资源快照、原hopper index、全部38配置/队列/两复用源、测试、代码bundle/同步收据及最终交付/prompt。所有大产物留服务器runs、不提交checkpoint/replay/凭据。

### sv2真实验收与实际启动

14条真实size50m CUDA/bfloat16工程全部passed，每条4098动作/517更新/3点各1完整episode/2报告，合计7238成功更新、42点评价、28次报告；持久化收据/快照/动作预算和报告/评价只读指纹审计通过。logging窗口活跃位置非零、实际权重精确1，Dt权重非1及活跃位置作用另存diagnostic-summary，工程短程不是正式收益或c来源。

服务器UTC 15:00:00（北京时间23:00）启动正式CPU监督器2504096，实际代码7769938；启动前逐卡刷新七卡无C/G、同UUID并留完整收据。agent为GPU0=2505601、1=2505611、2=2505629、3=2505915、4=2506475、5=2506749、7=2506761，对应上表七任务baseline；七条Dt状态queued，依赖自身baseline终点和审计。独立launch-pipeline 2494388仅等待本项目工程通过后启动，非外来GPU空闲重试，失败证据保留，完整argv/父环境白名单见formal-supervisor-command.json。

**23:03:57启动审计passed**：七条0点各10完整episode，固定快照动作0/update0，成功更新182—325次/最近开始动作2756—3328，连续收据invalid累计0。七份manifest/完整配置/SHA/seed1精确一致，119个C/G PID归属各agent/子进程且CUDA UUID/EGL/进程内cuda:0同物理卡、无外来共享；七条W&B running/full config/startup精确JSON/hash读回一致。/data余603,021,078,528字节。此为实际启动检查，不是百万动作终点、正式c或收益。

14条工程中Dt平均实际权重0.9989862680—0.9999315517，活跃位置非零；logging实际权重精确1，窗口活跃计数分别358400/358400/358400/323370/358400/358400/358400。保持诊断存在作用与没有收益的区分，不将短程数值移作正式c。stdout保留继承原配置的JAX profiler提示缺少tensorflow profiler trace；提示后仍连续成功更新，未安装TF/改变profiling或训练参数来消除提示。

23:01复核sv1 /data仍仅1,236,582,400字节，六卡未启动、未同步源码，24条继续等用户释放/扩容；没有迁移或删除原产物。当前合计2复用已完成+7运行baseline+7排队Dt+24空间阻塞=40记录，新增38条尚未完成。sv2七卡仍有本项目作业、无释放；sv2 GPU6未使用。本次最终提交自身仅运行文档/STATUS/实验索引；同期其他chat已提交pendulum CPU分析脚本`analysis/diagnose_pendulum_v1.py`、对应测试及sv3只读资源文档。最终相对7769938另含这些分析/测试，未以本次121项声称验收其内容；逐文件确认dreamerv3/embodied/本批两控制器/return_metrics及本批测试与实测提交一致，不影响本批执行，不切换运行worktree，完整最终SHA/差异/同步范围见DELIVERY。

23:07:04资源/进度复核：七baseline均running、连续1350—1457成功更新、最近更新开始动作7428—7856、invalid累计0，七卡仍仅对应本批C/G，七Dt queued，/data余603,013,611,520字节。最近开始动作不是精确实时总动作，快照不保证未来始终独占。本机292文件小镜像SHA256 `27f23cff3684cf58d1f00a1ae1ebb9d0a0636ff84319df7ff9aad5daad6eed9e`，逐文件hash通过，仅含配置/日志/工程与启动小证据，不含checkpoint/replay；运行JSONL仅当时完整行前缀，不能当完整终点。

首选继续当前03：监测已授权sv2 baseline→Dt十四条，持续核容量/前驱审计/故障与版本，sv1待用户解决空间后按原六卡验证/启动24条；不重复请求启动授权。全38条新训练及hopper原一对证据齐全后，按vision20-seed1统一40记录完整性/原始return/W&B，保留全部来源和旧科研/复现限制，再由用户交现有05分析。其他窗口当前有pendulum seed1独立计划且明确先不启动（SV3_RESOURCE_CHECK_20261103），sv1后续启动前须读最新文档/真实索引核是否新增重复，不把准备记录算另一份已完成source。不自动建chat/发消息，不以交接替代已授权可执行工作；新科学协议仍交02与用户。
