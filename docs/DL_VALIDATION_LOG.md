# DL 验证日志与证据索引

## 2026-10-11 18:07首个反馈尝试失败与工程r2

- 实际确定测试/运行SHA `bd0af3a47ac0861d932a4a08af2dd1ce84754dbe`，已push/远端独立核验，remote main仍 `595e982f1a3939775879728009e2105ee516e86d`，未main合并；只同步sv3 clean detached `/data/Policy_Discrepancy/dl-worktrees/bd0af3a`，canonical repo干净d9be557保持。git bundle SHA256 `36342d166a2e3a671dd335b7acdbf8506b8802843185ec322f0c406d37dd3721`。
- clean server CPU27项全部pass（statistics9/model2/physics6/replay4/supervisor6），全流程含Linux实际own-group timeout/failed-leader-descendant清理、pip check、source七runtime版本/32artifact hashes/只解码8calibration。证据 `/data/Policy_Discrepancy/runs/dl-feedback-20261011-bd0af3a/cpu/summary.json` 及五logs，本机镜像 `analysis/outputs/dl-feedback/bd0af3a/cpu/`。独立auditor r1 SHA256 `923d998acd9856790db11a109a265761772090a3e427d1735785bb8137ad6cfe`，server五基础/三完整fixture/24重新hash篡改自检exit0；不执行真实model/render/physics，不能称科学产物已验。
- 启动前18:05:55北京时间完整query/XML/pmon记录4–7空闲；CUDA/EGL probe物理GPU4、UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23`，进程内cuda:0、`MUJOCO_EGL_DEVICE_ID=4`。任务为300k baseline seed0/size50m/vision64/propriofalse/camera2/repeat1，private20261018/stat20261019，8cal×4固定位置×4paired RNG×q/p×H100，科学DL-feedback-r1，zero optimizer/gate，父预算3600s；没有alpha/c/shuffle新训练设置。
- attempt01完整目录 `/data/Policy_Discrepancy/runs/dl-feedback-20261011-bd0af3a/cp300000-attempt01/`，launcher SHA256 `258b20396eb0ca97853e307d5e601b327d333764a875aeca23267b2fdb0038ae`，18:06:29–18:07:30北京时间；worker1828665/38.720526秒/exit1/非超时/terminal/own-group cleanup requested。GPU采样37点峰2501MiB，非连续显存上界。budget SHA256 `c7501b2695b2b745ddc449cef8b70f93ec6c3dd437079136f6c6a116b093522e`。所有小文件、日志、XML/资源/samples已镜像 `analysis/outputs/dl-feedback/bd0af3a/cp300000-attempt01/`。
- 32个actual size50m源策略oracle全部byte-exact通过，plan收据齐全。单帧source数组索引返回NumPy scalar reward/flags，而物理初始化严格要求0D ndarray；首次fresh render和反馈前报production reward错误。无feedback NPZ/F64效用/result/complete，最终参数检查未到达；仅有限源policy事实可验，不发布utility裁决。
- 必要工程r2只把source_observation索引包为 `np.asarray(...).copy()`，保留dtype/shape/bytes与实际 `_obs` B1模型输入、所有科学/key/精确门不变。runner SHA256 `255bb4edfede73608636973902c79caf15c26ab83311665655207014f9a3f3a5` 经两Agent只读复核。新增真实quad/wrapper源数组→helper→prepare接口反例，物理7项本机pass2.077秒；仅stub render/model，无真实反馈验收。独立auditor r2 SHA256 `b3a18990f3e9c29606e60216ea06fa5d022169ea1c8cae3f3bc39f39c47c6cbc` 只更新工程版本，同一自检pass5.07秒，原r1不可变副本保留。
- 18:07:40北京时间fresh完整核查4–7各0MiB无compute，本项目GPU4释放，0/2未知作业不碰。下一步新确定SHA/clean server CPU28项及auditor r2自检后fresh核GPU4，同科学r1/同预算用attempt02重跑；不覆盖失败、不追加窗口/样本/RNG/checkpoint或放宽容差。最低1独占A100串行即可，完整研究费用未冻结；独立信号、gate/压缩、替代解释与条件训练仍未完成。

## 2026-10-11 正常反馈协议冻结与CPU实施

**17:56代码独立审查更新**：额度恢复后runner和replay独立审阅完成，runner SHA256 `eb57fb0e00bf6f3d9befad85f09558dbdd383c556615e0111a2c39c5e500f71b`，未见阻断。source/runtime七版本及实际F64bounds前置强校验补齐。独立反例发现缓存RGB可被callback修改、samebytes raw reshape可被放行；root补obs/key隔离和carry/obs/key hash/byte守卫、raw.shape，4项/9subtests独立复核通过0.03秒；本机0.013秒。未改变RNG／容差／收益或科学门，旧3项通过只为此前范围。artifact审计接线仍进行，GPU未启动；下述中断是历史阶段，不能作为当前永久阻塞。

科学版本 `DL-feedback-r1`／工程版本 `DL-feedback-engineering-r1`，详见 [DL_FEEDBACK_DIAGNOSTIC.md](DL_FEEDBACK_DIAGNOSTIC.md) 第9节及RESEARCH/DL-16。本机基底 `cf6aef82b0296d25fdb5a98f336c1da5372a44d5` 加新增三脚本／五测试；没有改Dreamer算法、既有gate或正式配置。模型adapt只在trace选first prior，实际生产core/cast/采样保留；真实quad/wrapper记录raw/applied/F64/F32与状态，cached-RGB policy和stored-applied physics各自复放。父监督限定自己新建Linux group，失败或超时清理descendant，不操作未知作业。

本机必要CPU：统计9项0.078秒；actual tiny BF16模型2项最终5.690秒；真实quad/wrapper物理6项1.220秒（model/RGB为stub）；synthetic cached-policy replay3项0.009秒；监督6项0.404秒，Windows4passed/Linux2skip。py_compile及diff检查通过。Windows原venv redirector导致Popen PID不同，原失败保留tool输出；pure stdlib fixture改用直接base executable，实际server Linux生产解释器另验，未降低父PID/硬budget检查。finite F64输入累加溢出的反例已fail closed；实际任务reward无溢出结论未由fixture推出。

分工Agent完成模型／物理与统计检查，但完整runner审阅及独立artifact审计器schema未完成时额度中断。ignored `analysis/outputs/dl-feedback/independent_feedback_audit.py` 独立NumPy/stdlib统计和key基础五项synthetic通过，`audit_artifacts()`显式NotImplementedError；不可声称ready或运行独立实际产物审计。根Agent继续完成synthetic replay及failed-replay证据保存，**这些后续修改仍待独立审查**。32actual size50m source oracle/CUDA/EGL/F64主reward/完整反馈尚未运行，不启动GPU或短训练。已有source完成仍是f2实际SHA；当前代码确定提交与server clean CPU随后记录，不能把dirty测试基底称已部署最终版本。

13:43:49北京时间只读sv3query确认GPU4–7均0MiB无compute作业、GPU0/2未知作业不干扰；本阶段仅CPU，不消耗分配GPU。最低1张独占A100串行、2张可并行独立环节、四张授权够；曾测source2585MiB/440.14秒、完整单步6675MiB/668秒仅为不同单项，完整剩余费用未知。反馈拟用GPU4单卡3600秒固定预算，启动仍需独立审查／确定版本检查／fresh完整占用与映射。

## 2026-10-11 动作接口审查、来源采集与资源验证

**13:09后实际结果**：source科学版DL-source-r1/工程版r2，实际采集/clean source5项测试完整SHA `f2d24da1831da2ffca22d313e7f534c96fc0d3eb`，服务器五项10.805秒，原action helper/test与bb78相同。已push/ls-remote核验、未main，仅sv3隔离worktree同步；原repo干净d9be557保持。13:00:04 fresh query GPU4–7空闲；launcher自身另存fresh完整库存/pmon，实际CUDA/EGL physical4/UUID9c070d5a通过。GPU4 attempt02于13:00:42–13:08:22北京时间完成，source父监督440.142416秒/exit0，worker PID1752617 terminal，1秒采样426点峰值2585MiB。不是GPU总预算或连续峰值上界。

32完整episode/32000实际动作、512实际source policy/key/carry字节重演、51200 reference physics步全部通过。独立NumPy/stdlib＋ml_dtypes（仅注册BF16 dtype）artifact审计与官方CPU physics128window/12800步审计passed；后者15.014750秒，声明envseed初始/restore/transition完整state/task/RNG/flags/reward检查误差0。两个独立审计只打开8calibration，blind24仅hash/工程收据；未计算D/K/C/U/return统计/新gate或训练。独立审计未运行模型/render，不能称完整闭环独立复现。完整实际版本/配置/资源映射/命令/receipt哈希及失败镜像见 [DL_PRODUCTION_SOURCE_RESULTS.md](DL_PRODUCTION_SOURCE_RESULTS.md)。GPU4已释放，13:09:36 fresh query/pmon确认4–7均0MiB无作业；GPU0/2未知作业未碰，是时间快照。

本次正确source约0.12226GPU小时，CPUphysics无需卡；当前最低1卡串行、2卡可并行独立任务，4卡足够。新 [DL_FEEDBACK_DIAGNOSTIC.md](DL_FEEDBACK_DIAGNOSTIC.md) 仍为未冻结候选，须先actual B1/key/F64奖励复放语义与工程验收；没有新方法结果或短训练启动。完整剩余GPU小时尚不能可靠冻结，旧20.4外推不应用。下方12:56及更早为历史证据，不作为当前pending状态。

**12:56后执行更新**：确定实现提交bb78e8c625c141d862293d26fc32d5ca64a598cb已push/ls-remote核验；仅sv3隔离worktree部署，原repo干净d9be557不变。clean CPU action3项8.971秒/source4项4.154秒及pip check通过。12:52:37 fresh完整库存/pmon：GPU4–7均0MiB无作业，GPU0/2未知作业不碰。新CUDA/EGL probe核验physical4/UUID9c070d5a；source attempt01于12:53:35启动、12:54:46结束exit1，父监督记录worker50.438587秒/PID1749300。GPU4释放（12:56:00 query为0MiB）；没有source episode artifact/complete或科研结果。

失败来自冻结RGB/policy重演时将已保存的host BF16 carry直接交给公共policy `_stack`，触发原JAX implicit transfer guard。保留目录 `/data/Policy_Discrepancy/runs/dl-production-source-20261011-bb78e8c/cp300000-attempt01/`、完整source.log/supervisor/budget/resource/采样，不把source首episode或exit0局部步骤当整体验收。必要修复 `DL-source-engineering-r2`：按原local policy_mirrored sharding显式device_put原carry，dtype/结构/数值/key/counter检查不变；未放宽guard、容差或科学协议。新增actual public tiny Agent/BF16/strict disallow下host carry重演，source本机最终5项通过6.722秒。提交后clean服务器source验收、新attempt02同seed/分割/预算/资源再核查，未改变source-r1对象；新反馈仍未冻结，不启动训练。

旧DL采集/外部后果绕过生产动作映射；实际quadruped float64 bounds不是全±1。新CPU动作检查8固定case与实际wrapper逐字节相等、完整物理/task/counters exact，legacy raw有真实qpos/qvel差异；实际tiny Agent raw sample/carry/下一RSSM输入相等，3项本机测试通过5.78秒。测试基底dd7da2ef4b99447e7fa867bec48fe6235655c8ae加新helper/tests；ignored证据 `analysis/outputs/dl-action-contract/cpu-audit-r1-final.json` SHA25648f74482d22fa8f4549b30b21ff4f5c02a8b2acd604c160ebfa0c9057df456ca。详细证据范围/必要修复见 [DL_ACTION_CONTRACT.md](DL_ACTION_CONTRACT.md)；不覆盖旧数值、原门与失败。

新source实现实际DMC+main.wrap_env、raw/applied独立、actualprecarry/evalkey来源、suite+FromDM._done保存/恢复、冻结RGB实际policy/aftercarry重演及H100每transition的F32 reward/完整物理核对。本机source CPU最初3项通过2.426秒；独立审查指出NaN放行及cooperative deadline不足，已修finite/dtype/shape拒绝与完整父监督硬终止、末帧action_executed mask和环境provenance。最终4项通过2.582秒，含真实sleep子进程.15秒期限后终止124。模型/RGB为stub，DMC/wrapper/physics为实际对象，包含1001帧/16×100步与近终止/隐藏_done反例；不是size50m科研通过。两套新检查共7项；确定提交服务器clean验收与实际采集仍待执行。

授权内Agent已选择source-r1，实际配置见RESEARCH/DL-14：仅300k、seed20261020/32ep8+24/16预定H100位置、GPU4/3600秒、0optimizer/0gate。独立代码审查、clean CPU及fresh资源/真实CUDA-EGL通过后才启动。新source不能靠修未来动作或inverse-map旧raw得到；旧科学结果限制于legacy_raw_suite，旧完整模型旁路/梯度和统计验收保持。正常后续观测反馈仍未冻结，完整方法目标未完成。本段没有新GPU启动或现场占用核查；旧GPU快照不得当当前状态。

版本 **DL-engineering-r8 / DL-protocol-r1 / DL-opportunity-r1 / DL-predictive-r1 / DL-predictive-MC-r1**，2026-10-10。整体目标**未完成**：真实size50m五组单步工程验收及独立审计已通过；r1/H100信号与幅度未获支持，有限固定控制下mode后验特异预测技能也未支持。后续一次有限MC是授权内排除诊断，未采用为gate或正式方法。进程exit0、文件完整或工程fixture通过均不等于方法有效。

本日志区分历史证据和当前状态。运行状态依据本轮根 Agent 已核查的收据；本次H100启动前资源快照为 **2026-10-10 18:35:43 北京时间**，GPU4–7各0MiB且无可见作业，不代表持续空闲。GPU0/2有未知其他作业，未触碰。科研定义、冻结分割与筛选标准仍以 [DL_PREVALIDATION.md](DL_PREVALIDATION.md) 和 [DL_PREVALIDATION_REVIEW.md](DL_PREVALIDATION_REVIEW.md) 为准。

## 版本和同步边界

项目根 `D:\program\PD\dmr3`；本机隔离工作树 `C:\Users\97370\.codex\worktrees\1f9a\dmr3`，分支 `codex/dl-prevalidation`。DL-r0 归档提交 `2b9641d`；合并基准完整 SHA `ae8372e42a815824d2e4ec3a527bfb985a81972b`，含 main `595e982f1a3939775879728009e2105ee516e86d`。

| 确定版本 | 范围与实际验证 |
| --- | --- |
| `9e6da39e59ad347f53db93af649dc8e8b63425ad` | DL-code-r1 方法实现、诊断和 runner；clean 提交上完成本机完整 tiny Agent、回归与 CPU 物理复原；真实 pilot/采集使用此方法代码 |
| `1280bea7bf409e9cc6fa975704bcb370b8880a0a` | DL-engineering-r2，仅改资源空闲判定及测试；完整 XML/pmon/UUID 作业库存判占用，显存作为真实 telemetry 保留；clean 提交上资源/runner 26项通过 |
| `3721103a081e6e667ca2329a0b221cf8d0aeb5ef` | DL-engineering-r3，仅改 size50m 梯度验收 helper 及其测试；对 probe 实际访问的参数逐叶检查，未读取状态明确列为不依赖；clean 提交上2项 helper 测试通过 |
| `55a527c620fb14339ee00f76f045a5dfddc81068` | DL-engineering-r4，将CUDA prior/v动态传入独立局部梯度检查，避免闭包常量触发transfer guard；没有改变stop-gradient、均值、free-nats或容差；本机helper2项通过，真实GPU7 attempt03五worker执行完成但paired整体验收失败 |
| `5765c893a419a59db46e5a3e3197af20b9059757` | 原r1裁决与H100机会协议/字段/独立分析；原r1覆盖充分而幅度不足分类修正为not_supported，数值和阈值未改。sv3 clean CPU 14项diagnostic及6项opportunity检查通过；84项真实H100恢复/prefix检查通过。新seed20261012 GPU4/5采集/评分使用此确定版本 |
| `bd8bd4dd11fc5f251435487d810bc8d277c5152f` | r5 logging独立精简前向；sv3 clean CPU 13项dt_latch（含完整9个Agent快照、2个真实逻辑CPU设备）、logging1项两次完整更新、helper1项、rep_probe5项均通过；GPU7 attempt04在logging特征exact处失败，off/alpha0/rho0完成 |
| `ed2afde0274694bace1c42086f3ceb2d3c4530af` | r6只改helper失败证据记录，不改生产模型；sv3 clean helper1项通过14.384秒。GPU7 logging-features-attempt05完成观察及单步重放，但exact失败仍exit1；日志对应训练更新已与r5 off严格相等，重算recurrent特征不相等 |
| `597a3a6374955d9bb31277c1c4678abd709d7369` | r7完整独立loss-and-gradient日志collector、严格只读score；新增calibration-only DL-predictive-r1。sv3 clean CPU 24项通过：dt_latch13项159.810秒、logging1项50.967秒、helper1项16.927秒、rep_probe5项3.537秒、predictive4项7.234秒；pip check通过。21:38后GPU7 attempt06及GPU4/300k预测诊断启动，结果仍待审计 |
| `be9e32aa018d8b61c0b2931fe44f72a6f480d4b5` | 输入契约修复，源raw Gaussian样本保留；sv3 clean predictive CPU4项7.399秒。GPU4/300k attempt02 exit0、128校准位置原数组审计通过，必要预测方向未支持，停止1m/代理/短训练 |
| `ab665a850466f0948edc877d5f9a977709905c33` | r8仅修helper嵌套对象hash，生产代码仍r7；sv3 clean helper CPU2项17.539秒及pip check通过。GPU7 attempt07真实五组完整通过，四日志leaf bytes相等、off/logging完整update相等、probe梯度零、N局部缩放及非rep梯度通过；独立summary复核通过 |
| `bb78e8c625c141d862293d26fc32d5ca64a598cb` | action helper/production source-r1；sv3 clean action3项8.971秒/source4项4.154秒/pip check通过；GPU4 source attempt01因host BF16 carry implicit transfer guard失败，无episode/complete，原目录保留 |
| `f2d24da1831da2ffca22d313e7f534c96fc0d3eb` | source-engineering-r2显式放置原carry；sv3 clean source5项10.805秒，actual公共tiny BF16 strictguard通过；GPU4 production source attempt02真实32ep及独立artifact/CPUphysics审计通过，440.142416秒/2585MiB；仍无第二信号/gate或方法结果 |

历史attempt03实际SHA为 `55a527c620fb14339ee00f76f045a5dfddc81068`；后续r5/r6/r7的实际版本如上表，不混同。9e之后至55的三个提交未改 `dreamerv3/` 方法代码，也未改科学配置、筛选阈值或精确比较条件。r7改变诊断执行图而保持原optimizer执行图及科学gate；CPU通过不替代CUDA精确状态/特征验收，不能将文档提交或当前HEAD写成所有旧运行的执行版本。

sv3 已部署上述确定版本，含be9e32a/ab665a8，位置 `/data/Policy_Discrepancy/dl-worktrees/<版本>/`。原 `/data/Policy_Discrepancy/repo` 在21:33:39核验仍保持干净d9be557，不修改原作业checkout；SV2仅只读取得checkpoint，没有代码同步或GPU运行。DL分支**未合入main**；前序连接失败保留，22:01 push恢复且ls-remote核验 `origin/codex/dl-prevalidation=ab665a850466f0948edc877d5f9a977709905c33`。不由该次push推定后续提交已推送，不覆盖canonical main既有本地修改。

## r7 / predictive启动及CPU证据（21:39快照）

实际提交均为 `597a3a6374955d9bb31277c1c4678abd709d7369`。CPU收据在sv3 `/data/Policy_Discrepancy/runs/dl-engineering-r7-597a3a6-cpu/`，本机镜像 `analysis/outputs/dl-prevalidation/validation-597a3a6/`；5套检查总24项，含真实2个逻辑CPU设备、完整Agent退化/隔离、两次完整日志更新、生产写入拒绝及18个真实物理固定控制重复。CPU fixture不等于真实size50m或方法有效。

启动前21:38:28北京时间，GPU4–7均0MiB、compute/pmon库存无PID；GPU0/2未知作业不触碰。每阶段launcher另生成fresh资源proof及实际CUDA/EGL probe，物理GPU4 UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23`、GPU7 UUID `GPU-c9a4f076-5f8a-8136-9535-7e74e14e895f`，进程内均CUDA0、EGL分别物理4/7。

- GPU7：`R/size50m-cp1000000-attempt06/`，off/alpha0/rho0/logging/N，B16×T64/context1，真实1m checkpoint，原free-nats1、dyn/rep1/.1、ac_grads=False、alpha20/rho.5。完整精确比较条件不变；允许工程单步和同源重放，不是短训练。21:39:33 PID1431993在GPU7初始化/编译，尚未验收。
- GPU4：`/data/Policy_Discrepancy/runs/dl-predictive-20261010-597a3a6/cp300000-attempt01/`，size50m/vision64/repeat1；8 calibration×16预定位置，6固定控制×H10×真实双重复，seed20261012、统计seed20261016，硬预算3600秒、optimizer0。21:39:33 PID1431893在GPU4初始化/编译，尚未验收。1m是否启动依冻结必要技能条件决定，GPU5/6本轮未用。

上述进程状态均是带时间快照；结束后另记exit、峰值、hash、独立审计和释放卡，不由启动成功推定通过。此前2.5GiB诊断/6.5GiB工程峰值和条件20.4GPU小时仍是旧实测/外推，r7额外梯度图的新成本待测。

**21:43:26更新**：GPU4 predictive attempt01在21:39:15–21:39:49执行后exit1，未产生预测技能裁决，GPU4已释放。错误是新入口假定记录动作必须[-1,1]；源实际为未裁剪Gaussian样本。八条calibration仅文件经hash核验、均有限、范围[-5.10,5.33]，blind数组未读；小证据在 `analysis/outputs/dl-predictive/597a3a6/recorded-action-audit.json`。`DL-predictive-engineering-r2`移除错误样本域假定，仍严格要求10步/float32/有限、原动作保留及真实复原；五固定控制和评分/停止条件不改。dirty CPU4项通过4.475秒，包含[-3,3]原动作真实复原及编码精确，新提交需clean CPU后新attempt，不覆盖attempt01。GPU7 off/alpha0已完成，其余仍运行。

## 21:50完成快照与必要工程修复

`be9e32aa018d8b61c0b2931fe44f72a6f480d4b5` sv3 clean CPU4项通过7.399秒。GPU4预测attempt02于21:45:40–21:46:48完成exit0/2511MiB；独立原数组审计通过，但主必要技能未支持，停止1m/新代理/短训练。真实控制敏感68点/8ep、复原0；见 [DL_PREDICTIVE_RESULTS.md](DL_PREDICTIVE_RESULTS.md)，不写成reward head全部无技能。

GPU7/r7 attempt06于21:39:15–21:47:20结束exit1/6675MiB；off/alpha0/rho0/logging单步完成、N未执行。四特征逐元素exact且误差0，但helper对嵌套repfeat dict产生object标量并hash内存地址，导致假失败。旧exit1保留；数值相等不能证明signed-zero/完整byte。根独立summary审计确认597a3a6 off/logging全部initial/updated state（含optimizer/norm）、gradient/othergradient、原抽样、RNG、raw/context、branch/total loss严格相等；证据 `analysis/outputs/dl-prevalidation/validation-597a3a6/attempt06/independent-update-audit-v1.json`。不替代新完整paired验收。

**DL-engineering-r8**只修helper为四真实数值leaf的dtype/shape/bytes hash并拒绝object，不改生产代码/配置/容差。另分配同值字典/单叶改变/object拒绝反例及tiny完整helper共CPU2项通过10.190秒；确定提交clean CPU及GPU新attempt07待执行。21:50:06核查GPU4–7无本项目compute作业，已释放，GPU0/2未知作业未触碰；非持续资源状态。

## r8完整工程验收（22:14:53快照）

确定执行SHA `ab665a850466f0948edc877d5f9a977709905c33`。sv3 GPU7 `R/size50m-cp1000000-attempt07/` 于22:02:04–22:13:12运行，668秒、exit0、1秒采样峰值6675MiB；原科学配置、checkpoint、B16×T64/context1、精确state/gradient/RNG及local rtol2e-6/atol1e-8不变。新helper对象修复不是降低标准，旧attempt01–06失败完整保留。

五组off/alpha0/rho0/logging/N全部完整，`paired_check.json checks_passed=true`。根独立审计重读五summary：三个旁路组相对off的全部initial/updated state（含optimizer/norm）、gradient/othergradient、sampled latent、RNG、raw/context、branch/total loss严格相等；日志tokens/deter/logit/stoch的dtype/shape/bytes四leaf hash严格相等；每组detached实际读取参数梯度严格0且原checkpoint单步重放精确。N非rep梯度相同、rep梯度实际改变，局部缩放最大绝对误差2.3283e-10。小证据 [dl_engineering_r8_independent_summary.json](dl_engineering_r8_independent_summary.json)，ignored原始镜像 `analysis/outputs/dl-prevalidation/validation-ab665a8/attempt07/`。

预热单步off/alpha0/rho0/logging/N分别.11254/.10833/.10841/.24479/.13430秒。logging额外完整gradient图有实际开销；这些是同checkpoint单次重放、不是整段训练吞吐，旧20.4GPU小时不得直接当修订后预算。最低1卡串行、2卡并行诊断/工程；当前4卡分配足够，不需占满或多卡模型。

22:14:53 fresh queryGPU4–7均0MiB、无compute作业，本项目所有卡已释放；GPU0/2未知作业未干扰。这是时间快照。工程通过只证明真实单步旁路/压缩语义，不证明长期训练复现、held-out信号/幅度、S/P/W或性能有效；整体目标仍未完成。

## 有限MC冻结与启动前检查

协议 [DL_PREDICTIVE_MC.md](DL_PREDICTIVE_MC.md)；代码/测试新增，生产训练代码不改。本机CPU5项通过8.16秒：实际vision64小Agent、全11步实际分布抽样、unimix0/.4及逐key参考、q=p共噪声、seed/分块/未来prefix、状态与counter不变/写入拒绝、来源仅8校准episode、概率混合反例和不稳/缺覆盖停止。测试基底ab665a8加未提交新文件；确定提交后的sv3 clean CPU及真实来源验收仍待执行，不能称真实size50m/科学验收。

必要入口修复：旧d.load_agent为取spaces调用make_env；MC直接按冻结DMC/wrapper spaces构造同配置Agent，避免新增环境构造或physics/render。固定源checkpoint/config哈希在pickle前校验，实际载入后仍需完整parameter digest相等；raw动作不裁剪。额外mode replay保存原128点logits精确性作为归因限制，不新增接受容差或改变主裁决。

22:25:31北京时间只读查询：GPU4–7均0MiB、compute/pmon无作业，GPU0/2未知作业不触碰；原repo干净d9be557。GPU4计划只运行一次300k MC-r1（pooled128/chunk8/H10/六控制），包含源验证/加载/编译/评分的3600秒上限；GPU5–7本项不使用。启动仍须新fresh库存及CUDA/EGL proof。本节尚无MC GPU结果，不构成新gate或短训练依据。

**22:33运行快照**：实际执行提交 `4fb7e8947f71dce8faf96096e700ab2b1987f8e3`，已push并ls-remote核验，未合入main；只部署sv3隔离worktree，原repo干净d9be557保持。确定提交上sv3 clean CPU5项15.831秒/pip check及真实来源校验通过：只打开8校准episode/128预定位置，checkpoint/config/八来源与modal产物hash一致、控制敏感68点/8ep；没有模型/GPU/环境构造。CPU收据 `/data/Policy_Discrepancy/runs/dl-predictive-mc-4fb7e89-cpu/`。

22:31:52 fresh完整GPU inventory/pmon确认4–7均0MiB无作业；新CUDA/EGL probe及单UUID binding通过。GPU4真实MC于22:32:50开始，目录 `/data/Policy_Discrepancy/runs/dl-predictive-mc-20261010-4fb7e89/cp300000-attempt01/`，PID1463837，进程内CUDA0，物理UUID9c070d5a；22:33:40已完成首5位置，2508MiB，实际参数digest与冻结源一致。GPU5–7不使用；GPU0/2未知作业不触碰。该行仅运行快照，尚无完整数值/科学裁决及独立原始chunk审计，不称方法已通过。

## MC完成及独立审计（22:42后）

4fb7e89真实MC于22:35:37结束exit0，167秒/2519MiB；原128位置/128路径/六控制H10全部完成。模型参数/counters收据不变，原mode128点重演logits字节精确、最大差0。科学裁决MC-inconclusive：两个主epsilon .00140737/.00125471超过事前.001，均值−.00537181/−.01277616及区间跨0，无必要方向支持。按事前门停止此分支，不提高L或追加key、换H/控制/阈值，不运行1m/新gate/18条短训练。完整结果 [DL_PREDICTIVE_MC_RESULTS.md](DL_PREDICTIVE_MC_RESULTS.md)。

NumPy/stdlib独立审计37.483秒passed，2056个NPZ原字节hash、全部1,081,344命名key、8 source/modal来源、控制/reward、概率先混合CE、episode bootstrap/epsilon/裁决全部核对；最大概率/统计差2.4869e-14/9.7145e-17，无科学阈值调整。小证据 [dl_predictive_mc_independent_summary.json](dl_predictive_mc_independent_summary.json)。原数组留服务器、完整收据镜像 `analysis/outputs/dl-predictive-mc/4fb7e89/attempt01/`；收据tar SHA256c16cc82fce4c38c2923c7547c21e885fafb40fab75d4d50a65ff814547f195a9本机/服务器相同。SSH个别连接关闭只影响收据复制，未重跑模型、覆盖产物或改写失败。

22:42:35 fresh query/pmon确认GPU4–7均0MiB、无本项目作业，GPU0/2未知作业未触碰；GPU4已释放，5–7本项未使用。当前验证最低1卡串行、2卡并行即可，现有4卡足够；训练费用尚不能冻结，旧20.4GPU小时外推不应用于修订方法。整体目标仍未完成。闭环一次观测省略是首选下一研究建议（见RESEARCH/DL-13），尚未冻结/执行，不能靠一个新公式宣布可正式实验。

## 环境与本地证据

隔离环境为 `%LOCALAPPDATA%/Codex/dl-validation-env`，Windows/Python3.12.14、JAX/jaxlib0.4.33、NumPy1.26.4、elements3.22.2、ninjax3.6.3、optax0.2.5、chex0.1.90、portal3.8.1、dm-control1.0.48、MuJoCo3.15.0；关键版本对齐服务器 lock，`pip check`通过。Windows Python3.12 与实际 sv3 Linux Python3.11.16 仍有差异，本机验收不能替代服务器验收。未升级服务器环境；进程 DLL 路径补充只存在本机外部隔离环境。

确定提交的本机证据根（Git ignored）：`analysis/outputs/dl-prevalidation/validation-9e6da39/`。目录名保留最初验证版本，其中后续 r2/r3 日志实际 SHA 见表，不把目录名当执行版本。

| 证据与实际版本 | 已观察到的结果 | 限制 |
| --- | --- | --- |
| `full-agent.log`，clean `9e6da39` | 完整 tiny Agent 测试1项通过，77.71秒；验证 off/alpha0/rho0/logging/strengthκ0 旁路的参数、梯度、loss、latent 与优化更新精确一致，非法有效 reward 明确失败 | tiny 模型，非真实 size50m、非方法有效性 |
| `regression.log`，clean `9e6da39` | 92 passed、1 skipped、13 subtests passed，20.20秒；覆盖探针、TwoHot、mask/terminal、梯度、置换、诊断和 runner 等工程路径 | 保留 skipped 状态；跨 shard fixture 不是多 GPU 方法需求或真实任务验收 |
| `physics.json`，clean `9e6da39` | 3 seeds×4位置×7候选，共84项 H10重复检查，完整积分状态/任务/RNG/counter一致，最大误差0；float32 H1标签一致 | 实际 CPU 物理分支，禁止渲染；不验 segmentation、冻结 size50m 信号或 gate/return收益 |
| `resource-r2-regression.log`，clean `1280bea` | 资源/runner 26项通过，3.15秒；包含真实 `dl_train.run` 控制流 fixture，以及1MiB无作业边界、非空作业库存拒绝 | 资源查询使用 mock；flow 使用真实 serial Driver/ProtocolState/EpisodeReturn/Checkpoint保存，但 agent/replay为double；Windows fixture关闭 checkpoint清理，未验 Linux清理或真实训练 |
| `helper-r3.log`，clean `3721103` | helper 2项通过，5.50秒；小型生产 Agent 上实际读参数的 detached probe 梯度严格为0、状态不变 | 仅验实际访问参数；不将未读参数声称为直接梯度实测；真实 size50m r3待验 |
| `helper-r4.log`，确定 `55a527c` | helper2项通过，5.43秒；代码已提交，其他dirty变化为文档 | 未完整执行真实worker局部梯度路径；由GPU attempt03补验，不能据本项宣布整体验收 |

初步 dirty 工作树证据保留在 `analysis/outputs/dl-prevalidation/diagnostic-cpu-r1/`、`runner-flow-20261010T073136Z/`、`runner-resource-20261010T074442Z/`。前者含 `tests-frozen.log`、`physics-final.json`、`cpu-validation-frozen.json`；旧文件哈希、失败和未提交标识不覆盖。当前验收优先引用上表的确定提交证据。

### DL 已发现的工程失败和修复

- 静态 gate=1 仍执行乘法会产生后续 optimizer 末位差；改为真正的 alpha0/rho0/strengthκ0 旁路，没有放宽精确比较。
- JAX ordered callback 在 Dreamer transfer guard 下创建 host token；改用 unordered invalid-signal callback，实际非法训练 fixture 仍明确失败。
- 物理 float64 reward 与采集 float32 标签误作同精度比较；分开物理重复误差与 float32 编码一致性。
- 按诊断 batch4 的 clean carry推进会使历史随机流受选点影响；每帧统一 batch1 advance，诊断 carry丢弃并实测 invariance。
- P 索引移动可能只是互换相等权重；新增实际权重变化和释放量检查，未改变门槛求通过。
- Windows regression 发现旧 `freeze_c.py` 的 JSON LF/CRLF 转换导致旁边 checksum 不等。在 main `595e982` 的独立源码归档复现同一失败；r1 改为写入实际被 hash 的 UTF-8 bytes。未改变 c 公式、统计、容差或 Linux 字节；确定提交回归已完成，不能再记为待重验。
- **DL-engineering-r2 资源误报**：17:07北京时间现场 GPU4/6/7各1MiB，但完整 XML/pmon 无 PID；GPU5有实际 collect PID1293785、2585MiB。原 `memory.used==0` 条件误阻健康卡。对齐既有正式监督器的作业库存语义，仍拒绝任何计算/图形/未知 PID、库存不可用和 UUID/绑定异常；照实记录1MiB，不写成0，不放宽科学或数值判据。预检不是原子资源锁，启动和初始化仍须核查实际映射。
- **DL-engineering-r3 size50m helper 失败**：GPU7 的 `size50m-cp1000000-attempt01` 在 detached probe 梯度检查中失败，Ninjax 目标集合含该 objective 未读取的参数，尚未执行 optimizer update。改为先追踪实际访问，再以完整叶名检查所有被读共享参数，记录未访问状态，并禁止 probe 创建/修改状态。保留 attempt01 原日志；r3 用新 attempt02重跑，不删除失败或放宽梯度为0的判据宣布通过。

## 冻结 checkpoint 来源

可审计清单：[dl_checkpoint_sources.json](dl_checkpoint_sources.json)。这是只读 sv2/lyg0360 的 source provenance 复核，不是新的 GPU 推理或方法证据。源为 `dmc_quadruped_walk`、baseline logging、同一训练 seed0、clean vision64、size50m，原训练 SHA `48d71f1541721208322442f5a1d5977124f3e888`。固定选择300000与1000000训练动作阶段，相隔700000动作，没有按 return搜索更好 checkpoint。

| 实际训练动作阶段 | checkpoint SHA256 | 成功 optimizer更新 |
| --- | --- | --- |
| 300000 | `2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9` | 74493 |
| 1000000 | `c9c893081485c8ad22468611d046bc5bf9fbe463d55996fda15cf85b2a0323fe` | 249493 |

清单核对各513050095 bytes、checkpoint hash、done/latest、evaluation metadata、更新counter、源 manifest/config，终点参数与 final_state一致。源原始进程 exit code不可获得，不能伪造exit0；完整产物与传输核查另记。`pickle.counters.actions` 是训练 policy 调用计数，不是训练环境动作预算；阶段依据 `evaluation_snapshot.json.actual_action_step`。

**两个阶段不是两个独立训练 replicate**；独立采集 episode 仅支持各冻结模型内的诊断区间，不能当跨训练 seed 泛化。正式源审计也不等于四组 M2 全部正式验收。清单只提交小型路径、哈希和元数据，参数文件不入 Git。

## GPU 和真实方法验收

实际运行根：`/data/Policy_Discrepancy/runs/dl-prevalidation-20261010-9e6da39/`，以下简称 `R`。运行目录名保留 r1，不替代每条 manifest/config 的实际 SHA。每条新 attempt保存 launcher/hash、开始结束时间、exit-code、真实资源与 CUDA/EGL probe、GPU采样和原始日志；失败不覆盖。

| 工作/实际物理 GPU | 状态与证据位置 | 验收范围 |
| --- | --- | --- |
| 300k checkpoint pilot采集 / GPU4 | exit0，`R/pilot-cp300000-attempt01/`，实际方法/执行 `9e6da39` | 2完整 episode，仅1 calibration/1 blind，非32-episode Stage-C |
| 上述 pilot score / GPU6 | exit0，`R/pilot-cp300000-score-attempt01/`，实际执行 `9e6da39` | 每episode4位置、H10，共8位置；工程/数值诊断，不能做完整筛选裁决 |
| 300k checkpoint完整采集 / GPU4 | exit0，`R/cp300000-collection-attempt01/` | 32完整episode、8 calibration/24 blind；仍待完整独立视觉与结果审计 |
| 1m checkpoint完整采集 / GPU5 | exit0，`R/cp1000000-collection-attempt01/` | 同样32完整episode、8/24分割；仍待完整独立视觉与结果审计 |
| 1m checkpoint完整score / GPU5 | exit0，`R/cp1000000-score-attempt01/`，实际执行 `1280bea`、方法代码 `9e6da39` | 每episode32位置、H10；参数冻结，完整分析及独立数组审计完成，科研筛选未通过 |
| 300k checkpoint完整score / GPU4 | exit0，`R/cp300000-score-attempt01/`，实际执行 `1280bea` | 对应32-episode完整采集/32位置，独立分析与审计完成；不以1m或pilot替代 |
| 1m checkpoint真实size50m paired / GPU7，attempt01 | 失败，`R/size50m-cp1000000-attempt01/`，实际执行 `1280bea` | helper工程失败、无optimizer update；不是方法失败或size50m通过 |
| 同一真实size50m paired / GPU7，attempt02 | 失败exit1，实际 `3721103` | probe读取参数梯度严格零已执行；局部梯度编译捕获CUDA数组违反transfer guard，无optimizer update，不放松guard |
| 同一真实size50m paired / GPU7，attempt03 | 失败exit1，实际 `55a527c`；五worker各自complete | off/alpha0/rho0状态、logging独立梯度/loss/latent、N非rep梯度等核查完成；各自单步可精确重放。但logging两个optimizer二阶状态hash不等，paired整体不通过 |

完整采集的具体执行 SHA 以各自 manifest为准；`9e6da39` 和 `1280bea` 的采集/score方法实现一致，资源 r2不重命名为科学修订。尚无 DL 18条短训练启动；size50m单步 replayed update验收即使后来通过，也不等于短训练或正式性能实验。

### r1真实裁决、视觉及工程失败

完整科学裁决与精确汇总见 [DL_R1_RESULTS.md](DL_R1_RESULTS.md)、[dl_r1_scientific_summary.json](dl_r1_scientific_summary.json)。两stage原analysis与全部64个score文件保持不变；当前代码只纠正“充分活跃但释放不足”误记inconclusive的分类，不改阈值。保护幅度未满足、label/cue不足、P对照不足分别记录，不能由修复代码宣布科学通过。

attempt03唯一logging差异为 `opt/state/2/1/rew/head/logits/bias` 和 `opt/state/2/1/val/head/logits/bias`，实际模型参数及独立总梯度完全相同。日志附加图影响优化器融合舍入是**待检验的工程解释**，尚不能作已证因果；不允许强制round状态或放宽严格hash。必要修复将独立编译只读logging诊断与训练核心分离，须证明参数/抽样/随机流不变，再以新attempt复验。

校准视觉证据在ignored `operational-evidence-r1/`：两stage各固定episode4/6/10的frame50/500/900、NPZ与来源hash、两张montage及逐像素JSON。root已实际查看18帧：背景仅替换天空，geometry/ground/边缘/影子保持；protected RGB精确相等。灰body保留姿态轮廓，实际移除外观纹理而非全部运动学线索。此有限QA不能担保全数据语义或排除所有背景尾部误保护。

同目录 `run-resources.json`（小型镜像 [dl_run_resources_r1.json](dl_run_resources_r1.json)）保存实际start/end/exit-code及资源证据hash、1秒GPU采样：300k/1m的32episode采集walltime分别341/335秒、score198/178秒，各采集峰值2585MiB、score2511MiB；attempt03峰值6665MiB、09:54:19–10:06:22 UTC，exit1。峰值为1秒采样观察值，不是连续上界。完整GPU数据、config/环境与UUID映射留服务器原目录。

### 独立长后果机会分支

协议见 [DL_OPPORTUNITY.md](DL_OPPORTUNITY.md)，`DL-opportunity-r1`；新seed20261012、每stage32episode/16预定位置、H100累计主后果，原H1信号/gate不变。只确认一次动作持续机会，不称闭环policy价值或新方法通过，任何结果均不启动短训练。原r1已公开结果不用作新未见数据。

precommit真实CPU H100审计84候选×独立短prefix，完整记录动作100步float32逐step对照采集精确、完整integration/task/counter重复最大误差0；这支持物理机制。20项诊断/opportunity CPU测试与6subtests通过（含真实tiny冻结scorer），实际来源55a527c加prefix/analyzer补丁；确定提交后须另存实测版本。新GPU采集/评分及机会结论尚待实际执行。

### DL pilot 已核查的小型证据

本机镜像为 `analysis/outputs/dl-prevalidation/bootstrap/`：`pilot-collection-complete.json`、`pilot-independent-integrity.json`、`pilot-independent-numeric-summary.json`、`pilot-mask-check.json`、`pilot-mask-montage.png` 和 `pilot-episode000.npz`。

- pilot两 episode各1001帧，collection参数不变；score/collection checkpoint与参数一致，plan/complete及两个score artifact哈希核对通过。prior logits、prior action/std和reward支持跨当前观测变体精确相同。
- 独立 NumPy float64复算对保存的float32信号最大差：`e`约8.09e-7，TwoHot entropy约5.36e-8；有限字段检查通过，物理重复最大误差0。记录的是实际误差，不新增或放宽验收容差。
- episode000 的 frame50/500/900背景替换中，被保护RGB逐像素相同；可变背景面积分别15.625%、12.5%、10.9375%。这是三帧pilot像素证据，不自动证明全数据语义保留，完整32-episode视觉审计仍待完成。
- pilot出现背景新增保护和mode扰动影响的具体示例：episode0/position1的背景A新增保护约0.03534；小KL mode扰动下 clean `e`最大变化约0.04011、保护量最大变化约0.00587。它们提示完整诊断应保留背景与mode反例；8位置不足以裁决整体筛选门槛，也不据此调参后继续称盲测。

### DL 真实设备绑定与资源历史

sv3 初期多次 SSH 在认证前关闭，失败证据 `bootstrap/sv3-ssh-20261010-145416.txt` 保留。15:30北京时间恢复；15:31:14快照实际 hostname=lyg0326、GPU4–7均A100-SXM4-80GB/81920MiB、0MiB/无计算图形进程。原 checkout干净、Python3.11.16、pip check通过、/data约1.1TiB可用；这是当时环境快照。

15:36:53–56北京时间在 sv3 GPU4 / UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23` 完成既有 `scripts/probe_formal_devices.py`。helper从 `ae8372e42a815824d2e4ec3a527bfb985a81972b` 导出，SHA256 `3954d6051287141c7bf27b6ce68da8a813f1b798357c216f44a1680bbaf54194`；PID1248755 的CUDA UUID/pmon C+G均只在GPU4，exit0，训练动作/Agent更新为0，probe随后结束。证据：服务器 `/data/Policy_Discrepancy/runs/dl-prevalidation-bootstrap-20261010-ae8372e/gpu4-device-probe.json`，本机 `bootstrap/gpu4-device-probe.json`。这只是首次绑定探针，不替代后续运行证据。

后续每条实际任务都另做该任务版本的 CUDA/EGL probe，绑定单个 `CUDA_VISIBLE_DEVICES` UUID、`CUDA_DEVICE_ORDER=PCI_BUS_ID`、物理 `MUJOCO_EGL_DEVICE_ID`、`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`；进程内CUDA0。不能沿用bootstrap旧空闲时间，或把GPU4绑定证据外推GPU5–7。r2 fresh resource记录以完整XML+pmon+UUID证明当时无可见作业，实际1MiB保留。17:39北京时间 sv3 GPU4–7均0MiB，是运行结束后的瞬时快照，不构成下一启动核查；GPU0/2未知其他作业未触碰。

XLA报告driver CUDA12.8比PTX编译器12.9.86旧，关闭并行编译；不改驱动/环境，实际耗时包含该影响。最低一张独占A100可串行推进，四张用于独立任务并行，无需多卡模型。六组×3seed×100k=180万动作，旧吞吐外推约20.4 GPU小时纯训练；诊断、编译、评价及新探针开销另计，尚非本轮实测预算或完成时刻承诺。

## 新H100机会诊断实际执行

确定运行SHA `5765c893a419a59db46e5a3e3197af20b9059757`，clean sv3 worktree `/data/Policy_Discrepancy/dl-worktrees/5765c89`。服务器CPU预检exit0：`test_dl_diagnostic.py` 14项通过、9.264秒（含真实FrozenScorer），`test_dl_opportunity.py` 6项通过、2.422秒；真实MuJoCo H100 84项通过、12.382秒，完整参考回放、候选终态/任务/RNG/counter及独立prefix重复最大误差0。证据根 `O=/data/Policy_Discrepancy/runs/dl-opportunity-20261010-5765c89`，分别为 `diagnostic-cpu.log`、`opportunity-cpu.log`、`physics-cpu.json`。这些是实施证据，不是模型机会或方法有效证据。

18:35:43北京时间资源预检后启动：GPU4负责300k、GPU5负责1m，各单GPU size50m/vision64/repeat1、32完整episode（8校准/24盲测），seed20261012；评分16预定位置/H100并共用H1/10/50/100 prefix。alpha20/rho.5、原H1 gate不变，不更新optimizer，无c匹配或训练评价。本轮每阶段collection+score硬预算3600秒；launcher `launch-opportunity-5765c89.sh` 存在外部bootstrap目录，实际hash、资源及CUDA/EGL核验每phase独立写入 `O/cp<stage>-<phase>-attempt01/`。两阶段均exit0，32×16位置齐全。20:30:47北京时间再次核查GPU4–7各0MiB且无可见作业。CPU分析使用同clean SHA，输出 `O/cp<stage>-analysis-attempt01.json`，均为limited_opportunity：300k positive139/negative176/useful-cue150/joint93；1m为49/205/62/27，分母各384。300k C AUROC .508、C−D区间跨0。新结果不覆盖原r1；小数组/收据包SHA256 `405df188f0d431d7e7e2022eafde9a88b0911c8a2d58b94b103e021b7591a2b8`，本机镜像在 `analysis/outputs/dl-opportunity/5765c89/`，独立复算仍待完成。

DL-engineering-r5将logging放入独立只读JIT，在相同pre-update参数/batch/seed上重现实际loss的抽样key，完成诊断读取后才允许optimizer donation；默认off和非logging weighting图保留。两层copied context禁止任何状态创建/写入，独立代码审查未发现阻断问题。dirty CPU小模型两次完整更新验证已完成；新增实际helper抽样/只读fixture1项通过8.012秒。helper先前新增audit漏传nested RNG的失败保留在helper-r5-precommit.log，修正为与生产wrapper相同的顶层pure+显式seed，未改生产随机流。真实CUDA验收尚待确定提交后的新attempt04，不能把此项写成DL-B已通过。

## 待完成清单

1. 完成必要logging分离修复及确定提交测试，在GPU7新attempt严格重验size50m B16/T64/context1、完整optimizer/模型状态、抽样/随机流、probe零梯度、局部缩放与单步重放，保留所有失败。
2. 新H100机会分支完成确定SHA、CPU完整参考/前缀复验，再在fresh GPU4/5完成各32新episode采集和16位置评分；审计float64真实序列、源hash和独立机会交集，不借旧r1数据。
3. 按H100唯一主窗逐stage裁决；不足停止该长窗对象，必要改进须另立有因果对象和独立数据的版本。核心gate、对照和压缩判据未通过前不能进入短训练。
4. 若DL-B/C及对照有效性满足，再在确定SHA、真实校准文件和完整冻结配置上启动18条100k短训练；否则保存失败/证据不足，论证必要修订并换版重验。尚无短训练或正式性能结论。
5. 逐项审计原五个目标，形成证据支持的正式候选、适用条件和剩余局限；缺真实gate/替代解释证据时不写“可正式实验准备就绪”。最后核对main/GitHub同步，报告最终文档SHA与各实际测试/运行SHA之间差异。

继续本次已授权任务，无需新建聊天或重新申请已有算力权限；不自动创建或发送其他 chat。模型来源核查、进程完成、工程通过和科研结论分别记录。
