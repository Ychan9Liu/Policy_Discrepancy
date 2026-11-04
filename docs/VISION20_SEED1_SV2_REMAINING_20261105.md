# Vision20 seed1：七条阻塞迁至sv2（2026-11-05）

用户指令：“排队的四条继续等待，七条阻塞挂到sv2，该次任务确保实验全部正常运行。”保留四个原排队身份/位置/配置，不迁移、提前启动或追加组。七条均为从未启动的原blocked槽位，跨服务器替代，不增加原38次新训练/40条记录数量。

## 分配、容量和配置

01:33:51核查sv2 GPU0—5/7全无C/G进程，/data余327263158272字节；不用GPU6。沿用用户已分配的sv2资源，四卡初始4×60GB=240GB，避免将不足七卡420GB的容量宣称通过。采用四卡串行队列、七条全纳入：

| sv2物理GPU | 第一条 | 后继 |
| --- | --- | --- |
| 0 | pendulum_swingup logging baseline | 同任务Dt |
| 1 | walker_walk logging baseline | 同任务Dt |
| 2 | walker_stand logging baseline | 同任务Dt |
| 3 | reacher_easy Dt | 无 |

task有dmc_前缀。每卡同时仅一条，三条Dt等待本任务baseline完整单条审计；reacher_easy Dt无需读取c，允许独立运行，其自身seed1 baseline仍按用户要求在sv1原队列等待。四条sv1原队列为reacher_hard Dt、reacher_easy baseline、hopper_stand Dt、walker_run Dt；正常前驱结束/完整审计后按原控制器执行，不人为暂停监督器。

所有科学配置继承m2_vision20_seed1/protocol_vision20_seed1、M2-v1-vision20-seed1-baseline-dt/profile vision20-seed1：clean、仅64RGB视觉、size50m、seed1、alpha20、free nats1、dyn/rep1/.1、ac_grads=false、reward_grad/repval_grad/repval_loss=true；batch16×64/replay_context1/16env/train_ratio256/repeat1、CUDA/bfloat16/preallocFalse、正式report关闭，JSONL+线上ychan9liu-tsinghua-university/PD_1、新ID/resume=never。七份完整config仅logdir改变；host/GPU/EGL/W&B身份为执行元数据。

百万训练动作含prefill、排除reset/评价；0..百万每5万21点、每点固定快照恰好10完整episode、采样动作/独立评价RNG。各任务自身seed1 logging按成功更新开始动作[100000,300000)有效loss位置S/N冻结c，正常replay重采样保留、同update ID重写去重；非法或无活跃停止，不填c1/调alpha。Dt不读取c。完整原始环境reward return、固定百万分母实际x梯形AUC、指定80/85/90/95/100万尾五点和缺项不重加权按EXPERIMENTS C/C.1。

## 版本、跨服务器防重复和验收

控制器/CPU测试提交`87975bb4733660bcccd7453cd09a3f043335260b`，新增scripts/run_vision20_sv2_remaining.py与tests/test_vision20_sv2_remaining.py。实际训练/真实CUDA固定原`bef29480df12d07659c640b22218804dd1d69e63`；dreamerv3/、embodied/、复用启动/审计代码与控制器版本逐Git diff相同。sv2独立控制器checkout `/data/Policy_Discrepancy/checkouts/vision20-sv2-seven-87975bb47336`，训练checkout `/data/Policy_Discrepancy/checkouts/vision20-seed1-bef29480df12`；旧7769938及canonical d8997f64a301bfdcfe08758faeae8957a7395766不移动，旧14条原始产物保留。

01:36:06在sv1重新核对七个source_run_id仍blocked、无PID、无旧目录/训练产物、config一致，排他创建原正式根reallocation-claims/task-mode.json。映射到新moved-sv2-20261105 run ID及独立moved01目录。原监督器从不自动重试blocked，不修改其持续写入的索引。源快照SHA256 `6b7e7e101e7a4e78738efa12cd0a3c8032a82590c14eec1b067d6df12f4f7a36`与七份claim/target root的完整性共同验证；缺/重复source、运行中source、SHA或配置变化拒绝。

Git bundle SHA256 `58592b51d27f8b315f2b1b21690e76249c0dc7badb0b9fadb7dcc50c011e2f26`已verify/fetch，仅sv2增加确定提交工作树；sv1仅写迁移收据和claims，不同步新代码或移动活动checkout，sv3无操作/无GitHub推送/无服务器源码手改。Python环境仍/data/Policy_Discrepancy/envs/dreamer，不安装依赖或删除base-py311。

本机六项新增/旧调度测试通过（1.010秒）：固定七槽/四队列与仅logdir差异、live/缺项/参数变更拒绝、seal精确source ID集合/hash/target，以及同卡无重叠/前驱失败阻断、旧目录防覆盖。01:36:49验收pipeline2996283在sv2启动，先保留126项并增加3项CPU回归，再四卡实际CUDA/EGL同卡探针；首条四任务实际size50m CUDA短程各4098动作/517更新/3×1/2报告，原严格收据/快照/报告与评价隔离审计不放宽。短程通过后重核资源与240GB门槛，启动四run/三queued；启动后另验零点各10episode/连续更新/实际manifest与W&B读回。工程fixture不是百万步终点或方法效果。

01:38—39，87975bb上的129项CPU回归及四卡实际CUDA/EGL同卡探针通过，真实CUDA工程继续使用bef2948。01:40检查来源时发现克隆的baseline_c_source旧标签仍指向旧目录：冻结器和统计审计实际按row.directory读取c_frozen.json，训练目标、RNG和结果没有使用该标签，但它会误导后续来源追溯。新增修复`666ce9031760f37e6f3fbd604f526621e9cffa43`，仅让logging标签指新目录、Dt为None及相应断言；本机六项补测通过（0.953秒），服务器同六项随后通过，不把129项旧提交验收改写成新提交全部重跑。既有sv1原始记录保留，其旧标签需按实际目录/claim解析，不能照抄为新c来源。

为保证修复先于任何正式启动，01:41:21只暂停本批CPU pipeline2996283，保留其已启动、仍运行的工程子进程2998882（创建时间1793813957.53）；没有停止GPU训练或其他作业。新finalizer3003110在干净666ce9独立控制器工作树执行六项回归，按已验证的本批子进程身份/完整acceptance等待并采用同训练SHA的既有四卡短程证据。原工程计划/日志不覆盖；正式用新plan-v2.json，source claim的target、source集合和hash不变，源收据中的87975bb保留为当时版本，后续运行用666ce9另记。原暂停CPU helper只在工程结束后关闭，不把该控制交接称失败重跑或恢复训练。

修复Git包SHA256 `11e042aba482cb8ebc92bb7f5b21d27e4782acad74ae182229e47b3e0c999c0c`已verify/fetch，仅sv2创建`/data/Policy_Discrepancy/checkouts/vision20-sv2-seven-666ce9031760`；旧控制器与训练checkout不移动。环境复核Python3.11.16、JAX/jaxlib0.4.33、numpy1.26.4、dm-control1.0.48、mujoco3.15.0、W&B0.30.0、elements3.22.2、ninjax3.6.3；pip check通过，无安装/升级。

## 原队列限制与产物

01:28—33 sv1仅余74—76GB，距60GB后继底线很近；GPU4自11月4日19:10有外来PID2327009与运行中的hopper_stand baseline同卡。GPU3/7新外来占用已阻断pendulum/walker_walk，七槽因此迁移。四条排队依指令保留原位，但不能保证GPU4的后继在外来占用仍在时启动；不停止外来进程、不共享启动、不擅自删产物。已向用户集中提出协调释放GPU4/继续清理或保持等待的资源问题，答复与现场变化另记；不能把“确保正常”解释为放宽资源门禁或隐瞒冲突。

正式根 `/data/Policy_Discrepancy/runs/formal-m2-vision20-seed1-remaining-20261105-sv2/`；工程根 `/data/Policy_Discrepancy/runs/engineering-vision20-seed1-remaining-20261105-sv2/`；sv1迁移收据 `/data/Policy_Discrepancy/runs/migration-vision20-seven-20261105-sv1/`。本机证据/计划/代码bundle/DELIVERY在analysis/outputs/vision20-sv2-remaining-20261105/，不提交模型/replay/凭据。实际启动、完整验收和最终SHA在后续段，不把pipeline或Popen称全部已正常运行。

全40汇总先按原六个sv1替代和本次七个跨服务器claim去掉源blocked槽，再纳入替代身份，严格40唯一task/mode，保留hopper复用两源；三个host/分片partial不当全20完成。首选当前03连续监测四卡七条及原队列、按同update ID/完整评价/正式c/raw-return/W&B精确终点收尾；新协议交02与用户，完成全批审计后由用户交05，不创建chat或发消息。

## 实际启动与持续核查

01:47四卡实际size50m CUDA/bfloat16短程完整通过，均4098动作/517更新/3×1完整episode/2报告，共2068次成功更新、12点评价、8次报告。新finalizer只关闭了原暂停的CPU helper2996283，采用原工程子进程全部成功的证据，没有停止或重跑GPU训练。经PD_1 viewer/project访问检查、四卡fresh无C/G/UUID及240GB初始容量核查后启动。

正式监督器3005895；01:47:48—56首批四条启动：GPU0 pendulum logging PID3005949、GPU1 walker_walk logging PID3005966、GPU2 walker_stand logging PID3005983、GPU3 reacher_easy Dt PID3006270。三个同任务Dt queued；实际控制器666ce9、训练bef2948，实际计划plan-v2.json，不能使用旧plan.json启动第二次。01:48:35 sv2余318688616448字节。

01:44对sv1原四个运行做CPU只读核验：完整成功更新前缀ID连续/无重复、invalid全0且匹配sum有限；已有评价为冻结网格前缀、每点10完整episode；manifest SHA/完整config与W&B配置一致、线上均running。外来占用单列，收据健康不等于资源独占通过，不将外来作业定为结果差异根因，也不据此直接宣称结果有效/无效。

01:45—48，cup_catch Dt、quadruped_run Dt相继百万动作及单条审计通过，原控制器自动接续GPU1 reacher_easy baseline PID2473985、GPU0 reacher_hard Dt PID2475750；不是迁移四排队或人工提前执行。walker_run baseline在GPU5继续，hopper_stand baseline在GPU4达到百万动作后仍待最终评价/审计，后继Dt保留。01:48:37 sv1余69820948480字节，GPU4外来占用条件仍待用户协调，不称全部无阻塞或存储问题已解决。

**01:51:53 sv2正式启动审计通过**：四条各0动作/0更新固定快照、恰好10完整episode，连续更新444/326/365/425，invalid均0；实际manifest SHA/完整config精确一致，全部C/G归属各自agent树，CUDA/EGL同卡独占，四个PD_1 run running且完整config、精确startup JSON/SHA256写入读回一致。记录formal-startup-audit.json；本批仍4运行/3按依赖排队，没有正式百万步验收。

**01:52:50 sv1原安排核查**：hopper_stand baseline已完整单审，但其Dt在启动前被GPU4外来PID2327009阻断，未创建新训练或挤占共享。另两条原排队已自然接续：GPU0 reacher_hard Dt连续1337更新/7376动作，GPU1 reacher_easy baseline连续1961更新/9872动作；GPU5 walker_run baseline45588更新/184380动作，三条均invalid全0、评价冻结网格前缀且每点10完整episode、manifest/config/W&B running一致、实际C/G同卡独占。walker_run Dt继续queued。原四排队目前为2运行/1排队/1外来占用阻塞，不假装四条仍全queued，也不称全批已全部正常。

01:53:20 sv2余314414604288字节、GPU0—3均为本项目，4/5/6/7空闲（不用6）；sv1最新余69322973184字节，GPU0/1/5为本项目、GPU4已释放本项目baseline但仍有外来作业。用户资源答复未到时保留原位等待，不自动移动hopper Dt/停止外来作业/删除产物；需要用户协调释放GPU4及继续清理sv1磁盘。其余现有7条实际运行均已通过当前健康核查；运行健康不是终点完成或方法有效。

全40在该快照为28训练结束（含hopper旧2复用、26新训练已单审但仍待整批/线上正式指标收尾）/7运行/4queued/1资源阻塞。七条新迁移均已纳入确定队列，四卡并发是容量约束内的安排，不宣称七条同时运行或全批无阻塞。

本轮122小文件镜像逐项hash验证通过，包167124字节，SHA256 `2808901d5a6854b4bf97cff5d07113f1fd6e81cde6e92ed199affff2e9c5f111`。只含小JSON/完整行前缀JSONL/config/stdout，不含checkpoint/replay/eval_snapshots/wandb二进制，不是无缝恢复备份。源码包、修复包、完整命令/配置/UUID-EGL/环境/源claims与版本差异见本机输出；工程命令为pipeline.py中129模块验收与原严格run_dmc8_serial.integration，补测命令python -m unittest -v tests.test_vision20_sv2_remaining tests.test_vision20_reallocated。正式argv见formal-supervisor-command.json，队列已启动，禁止重复执行。
