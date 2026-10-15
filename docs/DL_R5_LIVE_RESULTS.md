# DL-r5 live 随机诊断工程记录

2026-10-15。规范为[DL_R5_LIVE_TRIAL.md](DL_R5_LIVE_TRIAL.md)。
**当前终态（2026-10-15 18:42:31）**：原attempt01因GPU4资源守卫中止，parent exit1，
全对象失败/未独审、无科学裁决；不是方法阴性。19:00:24 fresh4–7空闲，原本项目PID均已退出。
下一已采用[工程r2一次完整重试规范](DL_R5_LIVE_RETRY.md)，新GPU5/12600秒、科学内容不变，尚未启动。
Agent在既有完整预验证授权内采用有限对象，不称用户逐项确认的新算法；原R4失败、M2及旧停止门不变。

## 原attempt01终态、失败封存与证据限制

外部terminal时间18:42:31.635740北京时间，parent returncode1，完整外入口4044.184685104秒，
按整段独占计费约1.123384635GPUh，probe另记。supervisor接纳13完整family/208unit，
磁盘223个unit complete包含family13未闭合的15个；没有generation_complete/完整统计或独审。
family13 terminal.failure明确为`Unknown work appeared on assigned GPU4; stop only this owned family`，
parent仅清本family组，original parent301006/worker327217/shadow327230等29个原身份已核不存在。
原7200秒没有耗尽，不将资源中止当超时、模型错误、第二信号阴性或成功。

**原记录缺口**：资源guard先判unknown再写telemetry，故触发sample没有持久化；
最后保存快照18:42:15.656729仍只有本worker。稍后18:50见330353 C/330414 C+G、proc AccessDenied，
只能标当时未知作业，不能反推它们就是触发PID或断言作业归属，未干扰/终止它们。
family13 worker_returncode=None是原supervisor异常前赋值缺口，不能编成worker exit0或任一实际退出码；
cleanup/主exit1及原PID消失另有证据。r2将先写冲突sample再fail，并在cleanup后写实际returncode，不放宽守卫。

18:58:12.745663完成0GPU失败封存，capture入口到结果7.698292秒/900秒，工具确认exit0：
`/data/Policy_Discrepancy/runs/dl-r5-failure-review-20261015-f845de0/attempt02/`。
原run全部1715文件/2,915,484,739字节双遍原始SHA一致，原N1M/config pins及f845的226项真实Git blobs前后匹配，
29个原parent/worker/shadow身份均不存在；未解trusted pickle、未读cal系数或D/e/Y、未物理/模型复放。
capture.json 620,367字节，SHA256 `50c47bf96a2b6330fce97ef3a3688623213a700450824c7be4b90021ba0315ae`；
取证脚本rawSHA `c8156b94b7ad323f082c8c2dd6964f74f4f3d44a5ea4c1075f3e3bf06beae3a2`，
小镜像在ignored `analysis/outputs/dl-r5-failure-review-f845de0/attempt02/`。
这是原失败证据稳定性封存，不是完整R5独审、unit有效性/科学统计或另一正常来源验收。
首次取证准备误将outer inputs目录当checkpoint目录，在首次inventory前FileNotFound退出；
保留旧脚本/attempt01空目录，attempt02仅更正为原r4-utility-inputs模型路径，不改原run或任何验收门。

19:00:24 fresh完整XML+pmon物理4–7均无进程/各0MiB，原parent不存在；
外部记录resource-post-failure-110024.json/XML。此前未知GPU4作业已自行退出，未归属或干预；不是持续空闲保证。
下方运行中记录均为带时间历史，不能据其继续等待已终态的handle53150或重启/拼接原对象。

## 实际启动与运行快照（2026-10-15）

实际源码保持f845，17:32:20只读核sv3 clean、原N1M/config完整SHA匹配，/data可用583,993,733,120字节，
物理4–7完整XML+pmon空闲。17:34零动作CUDA/EGL probe exit0，外4.269332秒，成本另记；
compute和render均物理4/UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23`，0训练动作/0更新。
probe后17:34:06 fresh idle记录，随后17:35:07.451831北京时间主入口启动。

主run：`/data/Policy_Discrepancy/runs/dl-r5-live-20261015-f845de0/attempt01/`。
外部probe/resource/launcher/stdout/terminal记录放在严格主产物图之外：
`/data/Policy_Discrepancy/runs/dl-r5-live-inputs-20261015-f845de0/`。
parent PID301006、PGID301006、start_ticks407927773；首family worker301084。
外部启动脚本launch-attempt01.py SHA256
`3be3a8894596d9b12bd6ddc373d8f30b75cade7c526abe7ce15120f3884be8f9`，
完整argv/环境/原parent身份在launch-attempt01.json，parent自身budget/process/资源也留档。

实际命令为f845 clean代码下的`/data/Policy_Discrepancy/envs/dreamer/bin/python scripts/dl_live_trial.py`，
`--expected-git f845de0cce297443c28761421ab91a7955b62553`，原已审inputs目录agent.pkl/config.yaml，
output为上述attempt01，resource-record为新inputs/resource-attempt01.json。
环境CUDA_DEVICE_ORDER=PCI_BUS_ID、CUDA_VISIBLE_DEVICES为物理4 UUID、MUJOCO_EGL_DEVICE_ID=4、
MUJOCO_GL/PYOPENGL_PLATFORM=egl、JAX_PLATFORMS=cuda。模型/协议字段沿规范冻结，不训练、不重算c/调α或损失梯度设置。

17:39:08原parent身份仍匹配且活跃，首family已有13个unit complete；cal128未封口、held未开始、
没有generation_complete/supervisor_terminal。实际完整load/save state SHA
`4e0d1a0e1de806bf8e4863278cabd171ed0d208dddc1fea18b8b54d68ca6383f`及params/counters认证通过，
真实worker CUDA/EGL核验同GPU4；当时采样2587MiB、GPU5–7各4MiB无本对象作业。
这是生产自检和进度，不是完整512或独审验收；不读取中间D/e/Y或据此修改设计。
驱动CUDA12.8/编译器12.9.86导致XLA关闭parallel compilation的既有warning保留，不改依赖或预算。
观察超时不当失败，不重启原handle；继续监督同一parent至权威终态，再独审。

18:12:37北京时间原parent301006/start407927773仍匹配，124unit/7family完整；cal未封口、held未开始、
无generation_complete或terminal。fresh完整XML+pmon当前worker315926在物理4 C+G，5–7无进程，
快照保存在外部inputs/resource-question-101237.json及同名XML。未读取中间D/e/Y。
按已观测吞吐，完整512有触及7200秒上限风险；此运行预算不改，超时对象不能拼接为完整成功。

18:24:23北京时间原parent身份继续匹配，162unit/10family完整；cal已封存、held已开始，
无generation_complete或terminal。fresh完整XML+pmon本worker322060仅物理4 C+G/2587MiB，
5–7无进程/各1MiB；外部快照resource-cpu-complete-102423.json/XML。没有读取中途科研值。

## 实际版本与工程证据

实际/tested源码SHA `f845de0cce297443c28761421ab91a7955b62553`。
新增policy、CPU shadow、family producer、stdlib父入口、统计、独立CPU审计和各自测试；
不改Dreamer算法、原R4或正式M2代码。只将确定版本部署至sv3新隔离clean目录：
`/data/Policy_Discrepancy/dl-r5-code-20261015-f845de0`；canonical、sv1和sv2代码不移动。
完整bundle29,170,569字节、SHA256
`3ace3e75334ff6c383de19ffe545d151d3789d109f6ee80e6aab651b9c71637e`。

本机Windows47项46pass/1 Linux进程组skip，15.896秒；同提交clean Linux47项全部通过，
unittest25.092秒/测试外入口25.458905933秒/含clone与source绑定全入口26.345918268秒，exit0。
结果时间17:28:02.763113北京时间，0GPU、owned测试进程组清理，checkout前后clean。
命令：`/data/Policy_Discrepancy/envs/dreamer/bin/python -m unittest
tests.test_dl_live_trial tests.test_dl_live_family tests.test_dl_live_statistics
tests.test_dl_live_policy tests.test_dl_live_shadow tests.test_dl_live_audit tests.test_dl_r4_statistics -v`。
环境CUDA_VISIBLE_DEVICES空、JAX_PLATFORMS=cpu、MUJOCO_GL=disable。
父runner与独审分别从真实Git blobs产生的完整source manifest逐值一致，先于测试时未导入JAX。

实际证据目录：
`/data/Policy_Discrepancy/runs/dl-r5-engineering-20261015-f845de0/attempt02/`，
包含linux-tests.json/log、source-binding.json、clone/checkout stdout/stderr；
本机小收据镜像在ignored `analysis/outputs/dl-r5-preflight/`。
第一次工程准备的bundle只含HEAD，错误指定branch导致clone exit128、没有建成checkout；
证据`setup-failure-attempt01.json`保留。第二次用HEAD bundle clone后明确checkout完整SHA，
只修部署命令、未改源码或科学配置、未产生GPU/科研运行，不冒称失败尝试通过。

## 已验收范围与限制

真实tiny BF16 policy/public oracle及官方CPU physics/shadow的有限接口通过；
producer/独审实际单unit包含完整prefix、H100和stdio CPU shadow。
该独审fixture虽用N1M tensor形状，NN/H1/oracle和RGB均显式fake，不能称N1M/真实渲染结果。
单ULP F64而F32不变、动作/key/carry、原r_t来源、写盘先封口再label、篡改、真terminal早停、
cal128封口先于held、Linux父/worker/shadow进程组清理和固定统计反例已有有限测试。
Windows coarse monotonic仅在Windows fixture中使用真实perf_counter；Linux真实strict<未放宽。

以下为CPU阶段的验收边界：当时未执行真实512/N1M/GL。上述新实际运行仍未完整生成/独审，
未执行压缩或保护学习干预，没有第二信号有效或方法性能结论。完整512审计入口实现不等于已验收真实512。

## 资源和下一步

原attempt01使用sv3物理GPU4独占一张A100-SXM4-80GB，32fresh family串行，5–7未预留。
完整入口7200秒，**单卡最多2小时/≤2GPUh**，包含初始化/编译/512units/统计与清理；不是成功ETA或整个预验证预算。
完整CPU独审另1800秒/0GPU；工程/统计/replay核验0GPU。
历史同规格1M完整训练41421.0822秒/11.505856GPUh仅作吞吐参考，不表示每个后续对象均需重训或获新增训练授权。
完整预验证还缺正常active压缩/选择性保护/有效替代对照，数量和预算尚未固定，不能虚报总小时。

17:29:31北京时间fresh XML+pmon确认sv3物理4–7各0%/4MiB、无compute/graphics进程，
本项目无新GPU作业；不是持续空闲保证。ignored资源快照
`analysis/outputs/dl-r5-preflight/resource-final-20261015.txt` SHA256
`6df899e4a69d816b27f32eaa108e7ea8f7055642a7c64f0eb61fe290eea62f0b`。

继续当前chat即可：按[新r2规范](DL_R5_LIVE_RETRY.md)完成确定源码/相关测试与Linux验收后，
fresh核GPU5及CUDA/EGL映射，从头执行新完整attempt；不改原科学样本/门或拼接旧对象。
只有新完整退出才独立CPU全审并裁决；原partial只封存，不称旧失败被新预算修成通过。
R5执行checkout保持f845，运行后不修改其生产文件；研究分支另新增独立CPU私有replay工程，
actual/tested2ae18a5详见[DL_NORMAL_REPLAY_ENGINEERING.md](DL_NORMAL_REPLAY_ENGINEERING.md)，不把该分支HEAD称为R5实际源码。
最终GitHub分支状态见完成报告/最新Git收据。
