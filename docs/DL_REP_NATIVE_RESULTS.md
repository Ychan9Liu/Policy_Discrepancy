# DL-rep-native-r1：运行与GPU资源记录

**DL最新工程补记（2026-10-12 06:05）**：实际/tested SHA `603e9132d58cc774865a5a68a512e4c3fe3b8c0c` 的零动作render身份对象验收完成：Linux39全过/pip0，sv3物理4仅EGL、4reset/12RGB/0action/Agent，外45.766849秒/约.012713GPUh/采样峰148MiB。物理snapshot/MJB/已测model与derived相同但RGB不等；scene差仅pre-draw geomorder scratch别名，不能当根因或完整GL等价，原比较保留。06:05:51.747233完整XML核4–7各0MiB/no进程。原native fc4219f attempt01仍失败、无科学裁决/CPU独审，guard candidate未采用，方法未成立/M2不改/完整目标active；首选继续当前chat先0GPU解释renderer有效输入，不放宽或立即重跑。详见 [DL_NATIVE_RENDER_IDENTITY_RESULTS.md](DL_NATIVE_RENDER_IDENTITY_RESULTS.md)。603e913 push connection reset、未GitHub确认/未main，仅sv3隔离代码同步；后续报告版与实际测试版的差异和最终SHA按终点Git核验。下方带时间为历史。

## 2026-10-12 05:47：primary-attempt01失败，GPU已释放

**状态：已终止、主验收失败，无科学裁决；CPU独审未启动。** 实际执行/测试SHA为 `fc4219f36449783f8751a9890a125cd218a2807f`，原 [DL_REP_NATIVE_ASSAY.md](DL_REP_NATIVE_ASSAY.md) r1的科学设置和精确标准保持，不放宽比对、不拼接产物、不解读本attempt的returns。完整目标仍active，原M2不改。

主入口北京时间 `04:49:15.053030` 至 `05:31:59.490141`，外入口 `2564.437091` 秒/exit1，单卡外入口占用口径为 `.71234363645 GPU·小时`，不是GPU算子累计时间或完整科研总成本。A128学习和0/B/A各32条完整episode已经生成；主工程收据记录共96条episode、96000 transitions、96096 RGB，但这不是完整对象或独审验收通过。

`statistics` worker在 `1.615644589` 秒后exit1，拒绝点为实际版本 `scripts/dl_rep_native.py:467` 的跨组 `reset_receipts` 精确比较。已有小型失败metadata显示相同seed的初始物理snapshot hash相同而初帧RGB hash不同。没有发布 `result.json`，主 `complete_verified=False`，外CLI exit1；父收据elapsed为 `2562.8755269479007` 秒、timed_outFalse，失败不是超时。因此没有A−B统计裁决、压缩机制或模型性能结论。不得从已生成轨迹提取回报、改为非配对统计或忽略RGB条件继续发布成功。

**已核代码事实与未定原因**：初帧SHA在reset后、首次policy/step前立即计算，后续buffer alias不能改写该SHA字符串；native观察和轨迹首帧均copy。现snapshot只包括 `mjSTATE_INTEGRATION`、task、clock/reset及FromDM done，不含mjModel纹理/材质/相机/灯光、派生render data或GL/cache状态。已安装MuJoCo3.15.0的 `randomdot` 使用局部seed42，未见该函数调用全局rand/srand；不能据旧版本资料盲加 `ctypes.srand`，也不能写“纹理随机性/GPU噪声已是根因”。视觉model、derived camera/geom状态、renderer/cache和buffer来源仍待工程诊断。

| 本attempt资源事实 | 范围与限制 |
| --- | --- |
| 实际GPU | sv3物理4，一张独占A100；5–7没有本对象任务 |
| 主显存 | 约2秒采样峰6623MiB（约6.4678GiB）；不是瞬时峰或最低显存保证 |
| 主预算 | 原完整入口7200秒/单attempt≤2GPU·小时；实际失败成本另记，不能当总剩余预算 |
| CPU独审 | 原1200秒/0GPU阶段未启动；不把96条轨迹生成写成96000步独立复放完成 |
| 释放证据 | `05:36:17.595853` 北京时间fresh XML：物理4–7各0MiB，无compute/graphics进程；launcher2059813及parent2059834已不存在，GPU4释放 |
| 较新只读查询 | `05:47:14.807442` 北京时间memory/compute查询：4–7各0MiB、compute列表无4–7；这次不是完整XML/graphics复核，0–2未知作业未干扰 |

服务器失败证据为run的 `engineering/failure-statistics01.json`，同时保留 `engineering/primary01-terminal.json`、主 `supervisor_complete.json`、`statistics.log` 及所有原phase产物。run为 `/data/Policy_Discrepancy/runs/dl-rep-native-20261012-fc4219f/`，主为 `primary-attempt01/`；本机小型镜像为 `analysis/outputs/dl-rep-native/dl-rep-native-20261012-fc4219f/`。本次五报告归档已核镜像中外terminal、父退出、statistics traceback及05:36资源字段；失败metadata SHA256为 `6cfc2e3847348568bd3d5e92903ae7c934d1a405871f14eef655c36073e519f2`，外terminal为 `befd9e6971ac53719dcdfd6ab82583aef00906d3709b03b4ad123b5bd038369a`。这只是小型证据核对，不是完整产物/模型/optimizer独审；归档未读取episode pickle、图像或回报数组。

**Agent已采用的必要下一步，未执行**：[DL_NATIVE_RENDER_IDENTITY.md](DL_NATIVE_RENDER_IDENTITY.md) / `DL-native-render-identity-r1` 的零动作render输入身份诊断正在工程实施。单sv3物理4、完整入口180秒/≤.05GPU·小时；两个worker各构造两个fresh MuJoCo环境/model，共4 reset、12 RGB，0actor/0checkpoint/0Agent更新/0action/0transition。实际worker只核EGL graphics，历史compute/EGL probe不冒称新CUDA验证。诊断比较即时/保存buffer、compiled visual model、derived render data和同env重复render，只裁决输入/renderer层；未知字段明标，不宣称根因或方法有效。确定版本、CPU契约及fresh资源核查后才执行。

另一个 [DL_REP_OBSERVATION_GUARD.md](DL_REP_OBSERVATION_GUARD.md) 两路径终态对象仍是candidate，未采用、未执行；1800秒/.5GPU·小时是候选单次上限，不是新增总预算。其native成功封口及独审前置尚未满足。额外aux图与终态的等价也不等于整条历史优化轨迹等价；历史未记录的编译环境字段保持unknown。

**Git和同步边界**：当前文档编辑基准HEAD为 `422ae48e9f1c4728c3e5cfe57a48ce6e2e12b10d`，相对实际/tested fc4219f只含此前报告文档差异；本次五报告编辑未提交、未同步服务器，不冒称新代码已测试。422的第3次push也因443连接失败，未确认GitHub新SHA、未进入main。既有隔离checkout及canonical同步范围按下方历史记录保持；整个方法的总剩余GPU小时未定，当前CPU分析0GPU、已有规模诊断一张卡可串行。

## 历史：2026-10-12 04:57:02 状态快照

以下保留当时运行中及其下一步安排，已由上方失败状态取代，不构成当前继续执行或重跑指令。

状态：运行中，完整主/独审未验收，无科学裁决。规范见 [DL_REP_NATIVE_ASSAY.md](DL_REP_NATIVE_ASSAY.md)，完整目标继续active，原M2不改。

实际/测试代码SHA：fc4219f36449783f8751a9890a125cd218a2807f。clean Linux四组38项全过40.602秒，外CLI40.817279秒/exit0/owned终止，pip check0；含Windows跳过的真实Linux进程组/无关sentinel测试。tiny官方RSSM/Optimizer/Agent和fakephysics/hash/退出契约是工程验收，不替代实际size50m任务或方法效果。原19生产Git blobs与c030e26相同。

04:49:15.053030主入口启动，父PID2059834、外launcher2059813。旧source/B完整退出与独审链认证通过。A128学习owned143.612114秒/exit0，Optimizer int32步74493→74621，原cal/context/batch/keys匹配，保留其它损失、momentum、free nats1及系数dyn1/rep.1；Dt/Probe关闭。实际reset active2048/2048，whole active130129/131072、nonreset128081，reset/whole excess75089.42974853516/541603.4714355469。活跃KL不证明任务信息损伤或保护有效；旧B没有全窗口压力值，不伪造匹配。

04:57:02快照：eval0完成11/32个完整1000-step episode；B/A尚未评完、CPU独审未启动。真实A评价reload、全部96轨迹/96k transitions、A−B统计及主/审计完整退出仍待验。禁止局部裁决、缩短/更换样本、延长7200秒或拼接attempt。

## 资源口径

| 环节 | GPU与预算 |
| --- | --- |
| 主流程 | sv3物理4，一张独占A100-SXM4-80GB；7200秒硬限、单attempt≤2 GPU·小时，含认证/编译/A128/3×32×1000闭环/oracle/统计/写盘/退出，非实际ETA |
| 主显存 | 截至快照采样峰6623MiB（6.4678GiB）；约2秒采样，非瞬时峰或最低显存保证 |
| 零动作设备核查 | 外11.403533秒，采样峰61401MiB；0训练动作/0Agent更新，compute/EGL同物理4，退出后释放，另记成本；不把probe峰当模型训练需求 |
| 独立审计 | 1200秒完整CPU入口、0GPU；96000 stored-applied官方physics复放及独立fsum/配对bootstrap，非模型/renderer/optimizer重演 |
| 全部预验证 | 总GPU小时未定；普通replay、第二信号/nuisance/错误依赖、选择性gate和泛化仍缺证据，不能由本对象上限外推 |

资源实际XML快照：物理4为本项目PID2063211/C+G、2843MiB；5/6/7各1MiB但无进程，无本对象计划。快照不代表今后仍空闲。物理4 UUID为GPU-9c070d5a-ac68-4c26-9522-a790ad874b23；CUDA_VISIBLE_DEVICES同UUID、进程CUDA0、MUJOCO_EGL_DEVICE_ID=4；实际render后核计算/渲染映射。报告没有把1MiB底噪改为0。

## 产物、版本与下一步

服务器run：/data/Policy_Discrepancy/runs/dl-rep-native-20261012-fc4219f/，主primary-attempt01、待执行独审sibling independent-audit01。完整命令/config/source/environment/pins/roots在engineering/primary01-start.json及各phase完成收据；本机ignored小镜像analysis/outputs/dl-rep-native/dl-rep-native-20261012-fc4219f/。无checkpoint/RGB/轨迹pickle入Git。外launcher SHA256：64f7955c238ba45077c43d5e8618d77f8c260af215235a0ed0237e14026c6e8f。首次tiny fixture锁重入失败及修复、Windows证据见 [DL_VALIDATION_LOG.md](DL_VALIDATION_LOG.md)。

只向实际sv3隔离checkout /data/Policy_Discrepancy/dl-worktrees/fc4219f-rep-native同步代码；服务器canonical仍clean d9be5574878d2c3f117cc6667be844f9a527eb84，本机canonical/sv1/sv2未改。研究branch push两次分别connection reset/443连接失败，尚未确认GitHub新SHA，未合main。后续报告提交仅文档不同，不冒称实际测试/科学SHA。

下一首选继续当前完整授权chat，完成原attempt全部96轨迹、主exit0/完整封口，然后同代码SHA执行0GPU/1200秒CPU独审并核资源释放。失败保留、只修工程，不改科学设置/阈值/预算；成功仍按spec条件分支处理未决信号/gate，不能宣布整个目标完成或自动开始正式M2。无需另建chat或再审批已授权流程。
