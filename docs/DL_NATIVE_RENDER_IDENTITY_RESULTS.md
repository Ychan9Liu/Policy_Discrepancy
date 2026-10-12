# DL-native-render-identity-r1：实际结果与资源

**终点同步核验（2026-10-12 12:48）**：研究分支已成功push，并以ls-remote核验到报告SHA `81fa38a216589ec39bbaef3f97b9f008c7da20b4`；GitHub/main仍 `595e982f1a3939775879728009e2105ee516e86d`，未合main。该报告版相对actual/tested `603e9132d58cc774865a5a68a512e4c3fe3b8c0c` 只改7份文档，源码仅sv3隔离checkout同步；后续终点归档提交仍仅文档，最终SHA见Git回复。12:48:52.572099北京时间fresh XML确认sv3物理4–7各0MiB/无compute与graphics，全部新owned PID已不存在；不是持续空闲保证。原native失败、第二信号/保护/性能尚未验证，完整目标active；继续当前chat，不自动创建或发送消息。下方较早push失败及资源快照均保留历史。

更新：2026-10-12。冻结规范见 [DL_NATIVE_RENDER_IDENTITY.md](DL_NATIVE_RENDER_IDENTITY.md)。**这个零动作工程对象验收完成；原 native assay 的失败没有转为成功，整个方法尚未验证成立。**

## 实际版本与验收

实际执行及 clean Linux 测试 SHA：`603e9132d58cc774865a5a68a512e4c3fe3b8c0c`。四组 CPU fixtures 共39项全部通过，66.292秒；外 CLI 66.519660秒/exit0，pip check0，包含真实 POSIX owned timeout/signals 和无关 sentinel。tiny模型、数值树和 fakephysics 是工程证据，不是方法效果。

主入口北京时间06:03:18.714218至06:04:04.481057，外45.766849秒/exit0，parent43.233232秒，两个串行 worker 分别19.190087和18.939493秒/exit0。parent、worker、外 CLI 完整终止且未超过180秒；`verify_completion`核源码/产物hash和退出链通过。root另核两worker各两个model、每model实际render索引 `[0,1,2]`，共4reset/12RGB/0suite transition/0actor/0Agent/checkpoint/旧episode读取；各worker两份graphics收据均绑定实际PID/物理GPU4，`compute_uuid=[]`、`graphics_verified=True`。没有独立重演renderer或认证完整GL输入。

执行仅在sv3隔离checkout `/data/Policy_Discrepancy/dl-worktrees/603e913-render-identity`，clean且tracked源码/config/lock及相关规范的checkout bytes等于Git blobs。服务器canonical在部署核查时仍clean `d9be5574878d2c3f117cc6667be844f9a527eb84`；本机canonical、sv1/sv2未改。相对原native运行SHA `fc4219f36449783f8751a9890a125cd218a2807f`，新增工程诊断、未采用的observation guard准备文件和报告；原Dreamer/embodied、19旧生产文件及native科学规范不变。

## 实际观察与解释边界

- 四个fresh model的初始snapshot SHA均为 `0779b6b8738845dfea3a39e6ac790e647757d35a5449e8fdbfdf17bca124d74c`。compiled MJB、695项已测model字段和derived transforms逐字节相同。
- 四个首帧的全部六组跨model比较，RGB hash均不同。同model的八组预定“首帧对后两帧”比较也都不同：3–33个像素、5–53个channel发生变化，最大channel绝对差为1。此范围仅对应这八组，不能外推旧96条轨迹或所有帧对。
- 原场景manifest实际不等，差异仅为 `scene/geomorder` 及四层 `ptr` 的同一数组别名。原分类 `measured_CPU_numeric_input_difference`及全部原hash保留，不改成通过或等价。
- 全部保存的RGB copy稳定，没有后续render改写旧live view；返回值确与当次Camera buffer共享内存。这个新对象没有观察到旧帧alias覆写，不能由此排除其他未测路径。

**geomorder的源码解释**：该安装MuJoCo3.15.0库SHA为 `d3a1476dd52e220031aca40b2b00f69d1febce2ee85c57d5015f68b0cc72afa9`。官方 `mjvisualize.h:306–309` 将geomorder描述为排序buffer；`dm_control/mujoco/wrapper/core.py:704,709–710` 分配至少1000个geometry容量，ptr返回self，因此重复字段不是五份独立状态。安装库静态证据显示 `mjv_makeScene` 分配geomorder但不清零；已核 `mjr_render` 透明排序路径先写入本次选中的geometry索引，再在实际计数范围内排序/读取。本hook是在update后、draw前，因此它捕获的是pre-draw scratch，不能将其直接当有效绘图排序或RGB根因。

root对已封口四个CPU NPZ先核SHA再用 `allow_pickle=False` 解码：ngeom23、capacity1000，geomorder前23项和尾部均有差异，前23项含大量不在0..22的值。不能仅去掉尾部就声称输入已认证。额外**事后解释性投影**只在另存manifest中排除五个scratch别名，六组跨model的其余已测scene字段相同；没有改原结果、标准或数组，没有新增模型/render/GPU调用。这仍不是全部有效GL/context/upload/shader/framebuffer/readback输入相等的证明，也没有定位像素变化来源。

不得据最大差1放宽原native RGB精确标准、round像素、盲加srand或只给其他组填入0的缓存首帧。该诊断表明无需Agent/CUDA训练也能出现像素不一致；不能写成图形噪声已经证实、原actor/rep结果有效或可立即重跑正式方法。

## 资源、产物与Git

| 项目 | 实际证据与范围 |
| --- | --- |
| GPU | sv3物理4，一张独占A100-SXM4-80GB；实际仅EGL graphics，无worker CUDA compute context；5–7未使用 |
| 时间 | 含fresh资源记录的外入口45.766849秒，单卡占用口径约0.012713GPU小时；180秒/0.05GPU小时是单次硬上限 |
| 显存 | 约2秒采样峰148MiB，仅本零动作对象；不代替size50m主流程6623MiB采样峰或最低显存保证 |
| 释放 | 06:05:51.747233北京时间完整XML：GPU4–7各0MiB、无compute/graphics进程 |
| 总预算 | 当前一张A100串行可做已有规模诊断，CPU解释与审计0GPU；整个剩余预验证总GPU小时未定 |

run：`/data/Policy_Discrepancy/runs/dl-native-render-identity-20261012-603e913/`，主 `primary-attempt01/`。完整命令/资源/测试/外退出在 `engineering/`，实际模型/图像/CPU字段在主目录。小型JSON/log镜像为本机ignored `analysis/outputs/dl-native-render-identity/603e913/`；无RGB、snapshot、MJB、NPZ或checkpoint入Git。

结果SHA256 `fae1a7abb04c7edd713ca4671b1479d7a31b5daf614fb00182025d2878720315`；外launcher SHA256 `7b885a309bb142feb6e024bd678e9f360b13e93191a7033aba93d9d27e35de7b`；Git bundle SHA256 `ebfcf5a15e22da44c94386e5393e9e6556a335d431459ebe8bcd9853b126d9fe`。root首次只读补充检查把 `runtime_graphics`误写为另一个键名而退出1；失败JSON保留，修正检查字段后 `root-verification02.json`通过，无环境或科学步骤重跑。geomorder解释记录在 `engineering/geomorder-readonly01.json`，本机事后投影另存 `analysis/outputs/dl-native-render-identity/engineering/scene-projection-readonly01.json`。

603e913研究分支push因connection reset失败，GitHub新SHA尚未确认，未合main。后续报告提交仅改变文档，不冒称为实际执行/测试版本；最终SHA及同步状态以终点Git核验为准。

## 下一步

首选继续当前已授权chat，先0GPU核实际renderer/GL的有效输入与可重复性条件，记录支持及反例，再审定必要工程修复。当前没有采用新的像素容差、renderer后端、缓存策略或native r2科学协议；不得用原失败产物通过native成功/独审前置。修复需要本机代码、确定提交、clean验收、新attempt，不能覆盖或补拼旧运行。

第二信号增量、真实nuisance/错误依赖、普通replay中的压缩损伤、选择性保护及连续actor性能仍待验证。全窗口rep停止是因果诊断，不是部署方法；observation guard仍未采用/执行，原M2及旧科学裁决保持，完整目标active。
