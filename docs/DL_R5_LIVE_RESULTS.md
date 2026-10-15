# DL-r5 live 随机诊断工程记录

2026-10-15。规范为[DL_R5_LIVE_TRIAL.md](DL_R5_LIVE_TRIAL.md)。
**当前：工程CPU验收完成，真实N1M/GPU诊断未启动；完整方法目标仍active。**
Agent在既有完整预验证授权内采用有限对象，不称用户逐项确认的新算法；原R4失败、M2及旧停止门不变。

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

未执行真实512完整对象、N1M/GL、实际效用预测、压缩或保护学习干预，
没有第二信号有效或方法性能结论。完整512审计入口实现不等于已验收真实512。

## 资源和下一步

当前对象只需sv3物理GPU4独占一张A100-SXM4-80GB，32fresh family串行，5–7不预留。
完整入口7200秒，**单卡最多2小时/≤2GPUh**，包含初始化/编译/512units/统计与清理；不是成功ETA或整个预验证预算。
完整CPU独审另1800秒/0GPU；工程/统计/replay核验0GPU。
历史同规格1M完整训练41421.0822秒/11.505856GPUh仅作吞吐参考，不表示每个后续对象均需重训或获新增训练授权。
完整预验证还缺正常active压缩/选择性保护/有效替代对照，数量和预算尚未固定，不能虚报总小时。

17:29:31北京时间fresh XML+pmon确认sv3物理4–7各0%/4MiB、无compute/graphics进程，
本项目无新GPU作业；不是持续空闲保证。ignored资源快照
`analysis/outputs/dl-r5-preflight/resource-final-20261015.txt` SHA256
`6df899e4a69d816b27f32eaa108e7ea8f7055642a7c64f0eb61fe290eea62f0b`。

继续当前chat即可：按f845源版本重新核真实占用、零动作CUDA/EGL映射后，在新attempt执行已冻结单卡对象，
不扩unit/预算、改门、拼接或把工程证据升级科研支持；完整退出后独立CPU审计，再据结果确定下一机制对象。
最终文档提交相对f845只记证据/资源和交接，执行checkout保持f845；最终GitHub分支状态见完成报告/最新Git收据。
