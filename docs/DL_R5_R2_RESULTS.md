# DL-r5工程r2运行记录

2026-10-15，科学版本DL-live-utility-r1，工程版本DL-r5-engineering-r2。
这是Agent在既有完整预验证授权、sv3物理4–7范围内采用的
[一次工程重试](DL_R5_LIVE_RETRY.md)，不是用户逐项确认或正式训练。
**已启动、尚未完整生成/独审，无信号增量、保护或方法性能结论。**
本页时刻均为服务器UTC记录换算北京时间；不同主机有钟差，耗时用对应monotonic，不跨主机相减。

## 实际版本与有限工程验收

实际/测试/运行Git **1584ef31c322fc3371ea9ce220c06fe7e654b23a**，新clean隔离代码
`/data/Policy_Discrepancy/dl-r5-r2-code-20261015-1584ef3`。
只同步sv3该目录；原f845执行代码、原attempt及canonical均不移动，sv1/sv2无本次源码部署。
最终报告文档提交另记，不能冒充生产测试SHA。

本机有限R5组合38项37pass/1 Linux-only skip（15.490秒）；新确定提交的sv3 clean Linux
12模块**71项全部通过、无skip**（unittest41.019秒、测试进程41.371104424秒、全入口42.499175002秒/300秒）。
71项包括原R5七模块及normal-learning/readout/replay/replay-probe/inventory五模块，0GPU。
234源码真实Git blob在测试前后相同，且父source_binding与独审source_manifest逐值一致；
新retry及三个normal规范额外逐一核真实blob，owned清单为空。
retry规范blob SHA256 `5875b9d7cc41c82b2d9783654b608e8acc4e731e66988dfea638e5e6b20ed13e`。

两个独立源码审查及root重核确认：run_unit AST与原f845一致；policy/shadow/statistics、
R4 policy/physics、feedback physics、Dreamer agent/rssm/dt_latch和原科学规范字节不变。
改变仅工程12600秒/物理5、规范blob身份链、冲突sample先write/flush/fsync后拒绝、cleanup后真实返回码。
有限NN测试仍为tiny CPU fixture，不能当真实512、N1M学习或方法有效验收。

工程证据：`/data/Policy_Discrepancy/runs/dl-r5-r2-cpu-20261015-1584ef3/attempt01/`；
本机小镜像ignored `analysis/outputs/dl-r5-r2-cpu-1584ef3/attempt01/`。
source-binding.json SHA `69e04f825aad51fe6c9e09479cd454d044c28f4e414a4bdab7f6bb4152a586ee`，
terminal.json SHA `bf940e478f9292ae83fce4a91e3acf09f9196eda4ff93f0c5a3425ec65c7b8e5`，
linux-tests.log SHA `2bb7f4c53444ce8998105d3afc64386eee9a265bd80bca5e0ac057d17e2beb3b`。
bundle SHA `1559755eb4059b514d6ba9e66748daa7c6f8fb429444c8ffd26d43b4b60b764c`；采用Git bundle与原blob，不用archive转换字节。

## 输入、配置与一次实际启动

新完整run：`/data/Policy_Discrepancy/runs/dl-r5-live-r2-20261015-1584ef3/attempt01/`。
外部launcher/probe/环境/占用/日志：`/data/Policy_Discrepancy/runs/dl-r5-live-r2-inputs-20261015-1584ef3/`，
与严格run图分离，不向原attempt添加新文件。

checkpoint/config仍是原`/data/Policy_Discrepancy/runs/dl-r4-utility-inputs-20261015-0655/`中的
agent.pkl（513050093字节，SHA `c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24`）及
config.yaml（4927字节，SHA `877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377`）。
原N1M seed0/size50m/BF16/vision64/camera2/action-repeat1，冻结模型无optimizer更新或正式训练。
32family×16流=512，env8192+f/eval0x52350000+u、时间/label/root子key、H1及H100、
8cal先封口/24held、ridge.1/coverage/主辅比较/bootstrap/sham及所有停止门沿原规范，不复用旧partial。
alpha20/rho.5原信号定义保留，free-nats1/原loss与梯度配置无训练更新；不涉及新c匹配、shuffle或训练评价。

19:19:08完整XML+pmon物理4–7均无作业；19:19:22–25零action CUDA/EGL probe通过，
19:19:26 fresh资源记录证明物理5仍空/独占。probe controller全入口4.915630197秒（单独记录，不是主预算）。
实际物理5 UUID `GPU-79704918-e096-92b4-252c-e3244497a8b3`，
CUDA_VISIBLE_DEVICES为该UUID，进程内CUDA0，CUDA_DEVICE_ORDER=PCI_BUS_ID，
MUJOCO_EGL_DEVICE_ID=5、MUJOCO_GL/PYOPENGL_PLATFORM=egl、JAX_PLATFORMS=cuda。
mapping probe SHA `418bd503728fd3b7f6ec0d1ae016ae7657500a0779713eb07b64976aa61e0f5e`。

19:19:36.321984一次启动parent **341486**，PGID341486、start_ticks408554660；outer parent341483。
完整argv/env/启动身份见launch-attempt01.json。外部launcher SHA
`7ed189e6dab5faf25180ff7d06100a051c6806c4c52b49e1f1a787e7916c2e16`。
整个主入口**12600秒/最多3.5占卡小时**，outer watchdog12610秒，CPU独立全审仍1800秒/0GPU。
不把上限当成功ETA，超时/冲突保留新失败，不自动第三次延时或拼接。

19:20:08原parent身份匹配、无generation_complete或supervisor_terminal；首worker341564。
完整loaded state SHA `4e0d1a0e1de806bf8e4863278cabd171ed0d208dddc1fea18b8b54d68ca6383f`，
params SHA `cffc13c0dcdfbd0c8b0e5c503c9d6251f4b1550c08b85a8742a638655d66d372`，
原actions62563/batches=updates249493/eval_actions1001/eval_seed1289196729认证通过。
worker实际CUDA/EGL均在物理5、C+G约2585MiB；物理4有未知PID340914（C），未识别/干预，
6–7无可见作业/各1MiB。只说明当时快照，本项目未占4/6/7。
首快照progress-first.json的unit_files字段只计错误层级的`family_*/unit_*.json`，不用于判断完成数量；
后续必须数实际`family_*/unit_*/complete.json`，不读取中间D/e/Y。

19:22:16.077742复核同一parent身份匹配；按正确完整路径已**8unit complete、0family complete**，
cal未封口、无generation_complete/权威终态。GPU5仍仅本worker341564 C+G/2587MiB，
GPU4未知340867 C及340914 C+G（后者另有GPU0 graphics），不猜其归属或干预；6–7无作业/各1MiB。
新progress-review.json/XML位于外部inputs，unit只数文件，不读取中间D/e/Y、cal系数或科学统计。

19:39:40.144530同一parent341486/start408554660仍活跃，65unit/4family complete、cal未封口、
无generation或权威终态。GPU5仅本worker351730 C+G/2587MiB，4/6/7各1MiB无可见作业，
此前GPU4未知作业自行退出未干预。新外部watch-20261015T113940144530.json/XML只读进程/计数/资源，
不读中间科研值；没有重启、改预算、选取partial或新GPU任务。

19:51:25.915495仍同parent341486/start408554660，105unit/6family complete、cal未封口、
无generation/终态。GPU5仅本worker358271 C+G/2587MiB，4/6/7各4MiB无可见作业；
快照watch-20261015T115125915495.json/XML，不读中间D/e/Y，继续原12600秒任务。

19:55:47.579963同parent身份持续匹配，118unit/7family complete、cal未封口，无全生成/终态；
GPU5本worker368376 C+G/2587MiB，4/6/7各1MiB无可见作业。
watch-20261015T115547579963.json/XML，不读科学值；正常C前置510339的0GPU验收没有移动该1584运行代码。

19:58:30.833197同parent341486/start408554660仍活跃，128unit/8family完整、cal封口存在，
无generation/终态。物理5处于family切换时无GPU进程/1MiB，不称整个任务释放或插入新C；
4未知370131 C+G（另有GPU0 graphics），6–7无作业/1MiB，不干预。
watch-20261015T115830833197.json/XML；只核封口存在，不读cal系数/科学数组，继续原R5。

21:00:21.473575同parent341486/start408554660仍活跃，**332unit/20family完整、cal已封口**，
无generation/内外terminal。GPU5仅本worker412107 C+G/2587MiB，4/6/7无可见作业/各1MiB；
此前GPU4未知作业自行退出未干预。外部watch-20261015T130021473575.json/XML只核计数和资源，未读科学值。
中间20:13/20:23/20:35/20:47快照分别176/209/252/288unit，完整原始收据留外部inputs。
并行完成的NM1新8d63代码142项CPU验收和e704原数据迁移均0GPU，未移动此1584运行代码或插入新C。

环境prelaunch：Python3.11.16、jax/jaxlib0.4.33、numpy1.26.4、dm-control1.0.48、MuJoCo3.15.0、
driver570.158.01，详细版本与pip freeze保存在外部inputs。驱动/编译器既有parallel-compile warning不改环境求过门。

## 资源答复及下一步

**21:37:07更新**：原parent341486/start408554660身份匹配、452unit/28family完整，cal封口，
无全生成/内外terminal；GPU5仅本worker438660 C+G/2587MiB，4/6/7无可见作业/各1MiB。
只核计数和资源，没有读取中间科学值或增加GPU作业。独立的正常特征CPU窄接口五项通过，
不读取真实R5且不认证source/N1M；实际1584/12600秒/科学门不改。

**21:26:46服务器只读快照**：原parent341486/start408554660身份匹配、418unit/26family完整，
cal封口，无generation/内外terminal；GPU5仅本worker432246 C+G/2587MiB，4/6/7无可见作业/各1MiB。
未读中间D/e/Y、统计或cal系数，原12600秒/科学门不改，无新增GPU任务。

独审工程前置已用实际新提交 **7c4affc49d4535b83804cb443471141645fbe658** 完成sv3 clean Linux
55项无skip、25.775878075秒完整入口/0GPU；[外部退出及组身份修复](DL_R5_AUDIT_ENGINEERING.md)
不改正在运行的1584、统计或物理复放。原run_git与新audit_git分别记录，不能互换。

当前需要**一张独占A100-SXM4-80GB串行**，授权4–7不是四卡并占。
旧同规格诊断采样峰2587MiB、历史完整训练峰8086MiB，只是实测参考，不能保证较小显存卡可用。
CPU库存/私有replay/tiny工程/ridge/bootstrap/物理标签和独审不占GPU；真实模型重提特征和正常机制学习需要单卡。
完整剩余总GPU小时尚未冻结，NM1的4096更新/最多五臂/整阶段6GPUh仍仅建议，不能作为已批准总额。

继续当前chat监督同一parent/handle至权威终态；观察超时不是停止理由，不另开补齐任务。
只有512完整退出且原进程结束后，才从新auditor7c4affc clean代码对原run_git1584执行
1800秒独立CPU全对象审计并作冻结科学裁决；正式CLI必须提供外部launch/terminal双raw pins。
其前不读取中间科研值；工程通过不等于信号或方法有效。原f845资源失败及缺失触发sample限制保持，
见[旧运行及失败取证](DL_R5_LIVE_RESULTS.md)。正常压缩及有效S/P/W仍缺证据，完整目标active/M2不变。
