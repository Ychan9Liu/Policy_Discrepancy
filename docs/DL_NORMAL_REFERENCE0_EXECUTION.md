# DL 正常共同历史 reference0 实际执行采用

## 2026-10-16当前状态：F1失败，F2已采用但未启动

F1实际producer仍为 `2a15c91aad16c0e3151fde3943e356dad1fcb69d`，在物理GPU4完成325/512后，
parent因资源guard主动停止：01:16:01保存的XML与pmon同时记录本worker542278为C、未知574111为C+G，
allowed仅542278；pmon还记录574111在GPU0为G。**不推断未知作业归属或根因，不干扰该进程。**
原worker返回−9，原parent与实际SSH均返回1，SSH完整耗时3553.406秒，小于原12600秒；这不是超时，
也不是NN、特征或科研阴性裁决。无worker_complete/generation，325partial不得复用、补齐或与新尝试拼接。
完整失败产物保留在原run；本机正确小镜像为
`analysis/outputs/dl-normal-reference0-2a15c91/failed-attempt01-metadata/`，
首次误用main.log的失败mirror另保留于`failed-attempt01/`，不能混作成功取证。

原F1 root SSH receipt 884B SHA
`9f27e9cc784bb52917b47902772393583849453b0d49876b16977b451c7b57fa`；
controller terminal 53235B SHA `b6c674463f626d22b7646f4454d203c8920348834ce6186f423bbbd89de9b263`；
supervisor SHA `da7e59a2d42bb4febef935add4b02afee97a38f693039c3c534bab397a1db97e`；
触发资源记录SHA `9f5f3f904689f1ffc17e9a6a996ccd29987e49eb1d7c89280d26dbe1efbabcbb`。
原parent539679/start410338164、worker542278/start410358034及owned组退场已由实际终态与root观察记录；
01:19:05–07的4–7空闲只是当时快照；root随后01:42:57–58再次核4–7空闲、原四PID gone和cccb/2a clean，
再启动独立C4；新F2仍须其自身启动前fresh核查。

root在用户sv3物理GPU4–7的完整预验证授权内采用**F2 fresh全量attempt02、独占物理GPU6**，
UUID `GPU-70db6c66-9917-e0e2-44db-0fc80fcf41b9`。最多同时两卡、每卡一个任务：C/probe5，F6。
producer代码与CODE保持原2a及`/data/Policy_Discrepancy/dl-normal-feature-run-code-20261015-2a15c91`，
仍为同512/32、128fit/384held、原source/labels/keys、逐帧oracle和整入口12600秒；没有改科学量或放宽资源guard。
原runner已接受物理4–7，无需修改producer。实际启动仍须fresh核GPU6的UUID、XML+pmon空闲与原失败进程退场。

- 新INPUTS：`/data/Policy_Discrepancy/runs/dl-normal-reference0-inputs-20261016-2a15c91/`。
- 新RUN：`/data/Policy_Discrepancy/runs/dl-normal-reference0-features-20261016-2a15c91/attempt02/`。
- 准备包：`analysis/outputs/dl-normal-reference0-gpu6-controller-r2/`；controller SHA
  `740f5cc8ef532b3d95ac495e9316b1f8902d71794c49d1cc6b3c5bb5a7bcd7d6`，plan SHA
  `f37daf3184925e77ffd9908b5a88c4fcfd2102eb0992d4773b27657bc238b874`。

新controller认证原F1失败whole/资源/日志rawpins、325partial边界和原PID/start/PGID/SID退场；
旧F及未知进程只查不发信号。全部原函数AST除接线main外保持，原3f清理helper原字节保留；
F2 controller/plan/observer仅本机准备并经独立只读peer，**未上传、未启动，尚无F2完整特征或读出结果**。
新reader r2的固定资源/路径重绑定及待验状态见[读出工程记录](DL_NORMAL_READOUT_RUN_ENGINEERING.md)；
旧dfd reader与已stage的e238读出外层禁止用于失败F1或新F2。

GPU5的cccb生成控制数组回归已有实际SSH0/11.406秒/300秒证据，但无NN、学习或校准资格结论；
C4已01:43:15在物理5启动、未完整验收，见[当前C记录](DL_NM1_CAL_RESULTS.md)。这些状态不改变F2或T的科研前提。

## 原F1采用与执行历史

以下保留原r1采用及F1当时的运行中观察；GPU4与旧attempt01路径不是当前F2安排。

2026-10-15，`DL-normal-reference0-execution-r1`。这是 Agent 在完整预验证授权内采用的
一次特征提取，不是用户逐项确认的新算法、正式训练、T4096 或旧6h建议的采用。
本轮先前完成真实标签及确定提交的工程测试；真实完整特征、吞吐和峰显存仍待执行验收。

## 研究用途及固定范围

采用 [DL_NORMAL_READOUT.md](DL_NORMAL_READOUT.md) 已冻结的任务量和特征定义，
为原 N1M 初态 reference0 取得共同正常历史上的 h、hq、hp；此入口不拟合读出。
原R5完整性及物理复放已通过，其信号主检验未获支持、sham concern 根因未知。
这些科学限制继续保留；这里仅复用已认证、已公开的处理前正常历史，不依赖R5科学阳性、
不将来源称作 fresh blind 或 actor关键性真值，也不按D/e/Y筛选样本。
本对象与C校准系数、C科学资格独立；C的完整退出和资源释放是串行运行条件，不是筛选reference0的科研门。

固定原512条、32family、每family16条，前8family/128条fit，后24/384条held。
每条从真正reset沿其固定RGB及raw[:t]，重建模型自己的h/z/carry；当前帧不伪造raw[t]。
每帧执行原public/native byte-exact oracle，保留两次原随机调用，概率查询不消耗随机流。
当前h4096为真实BF16、q/p概率32×32为包含官方unimix的F32，原dtype/bits、keys、完整只读模型身份均须认证。
输出h、概率、完整history及来源/标签闭包；无环境复放、渲染、backward、Adam、累计学习或native评价。

## 确定版本、资源及预算

实际/测试代码固定 `2a15c91aad16c0e3151fde3943e356dad1fcb69d`，
已在sv3 fresh clean Linux六模块36项全过/无skip，真实pending-auth alarm通过，
whole47.492秒、rootSSHexit0/48.078秒/300秒/0GPU；证据见
[DL_NORMAL_FEATURE_RUN_ENGINEERING.md](DL_NORMAL_FEATURE_RUN_ENGINEERING.md)。
实际code目录 `/data/Policy_Discrepancy/dl-normal-feature-run-code-20261015-2a15c91`，
采用记录的后续Git只改文档；不能把后续文档SHA冒作实际已测试代码SHA。

- 服务器sv3，独占物理GPU4，UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23`。
- **一次完整入口12600秒/3.5单卡小时上限**，包括所有初末CPU认证、初始化、全部512前向与oracle、
  保存、清理和完整退出；不是期望耗时或足够完成的实测保证。
- 初末source认证各自900秒上限包含在同一12600秒里，不另加预算。外部必须在Popen前安装同一硬时限，
  覆盖内部读取计划与设置timer前的bootstrap。任何成功须实际外层exit0、完整wall≤12600。
- 只有一个GPU worker；CPU parent与外部控制器隐藏CUDA、禁用MuJoCo GL，worker按plan显式设置该UUID。
  不创建EGL或renderer，`graphics_verified=False`。不自行换卡、共享、增加卡或自动扩预算。
- 原C实际完整退出/其owned groups gone、fresh XML+pmon确认指定卡无冲突后启动；
  C修复后仍在原物理5按同128批/3600秒完整新尝试，与本对象可独立并行，最多同时两卡、每卡一个作业。
  该设备安排在本对象尚未部署/启动时由Agent采用，用于避免阻塞C必要重跑；
  不把C初始化间隙或旧资源快照当GPU释放。未知作业不干扰。

3.5h是有限成本上限：参考已有R5真实2.58单卡小时，新的逐帧oracle工作量与吞吐尚不相同，
不据这个类比承诺ETA。超时或失败保留新attempt，不用partial补齐，不降低oracle精度或删除失败单位。
本次不采用任何后续学习步数、组数、native回报预算或方法性能判据。

## 输入及输出身份

原R5 producer Git `1584ef31c322fc3371ea9ce220c06fe7e654b23a`，run
`/data/Policy_Discrepancy/runs/dl-r5-live-r2-20261015-1584ef3/attempt01/`；
成功prior audit Git `277c18001d353d9101a3622959351d3994699adb`。
source index强制原producer launch/terminal及prior report/terminal/launch五份rawpins，完整原run inventory及源码身份初末一致。
原checkpoint SHA `c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24`，
config SHA `877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377`。
完整Agent.save的Python计数器保持原类型，fullstate/params/policycopy/optimizer/counters初末只读。

成功labels actual/tested47db，输入
`/data/Policy_Discrepancy/runs/dl-normal-labels-20261015-47db626/attempt01/labels.json`，
1539648B rawSHA `5db0529317c8c9c55674aee80df1ef0258c6bcf4e407ce1b4c74373ec0f46890`；
同目录terminal49321B SHA `cc0ec4728020290693f866b429de06a47eff3fcd917ba044cad310fee82c4361`。
plan还强制root真实外层退出receipt：
`/data/Policy_Discrepancy/runs/dl-normal-labels-inputs-20261015-47db626/outer-execution.json`，
588B SHA `0e443503674477bbc8a2331380498ad589fdf69e18fc0d5f206620ffe27a18ab`。
它记录真实SSHexit0/798.329秒/900秒/0GPU，不能用内部terminal自证未来退出。
标签是torso-local速度主量和upright辅助，source/frame/order/target逐项对齐，不能当作动作正确性真值。

新外部inputs目录 `/data/Policy_Discrepancy/runs/dl-normal-reference0-inputs-20261015-2a15c91/`；
fresh输出 `/data/Policy_Discrepancy/runs/dl-normal-reference0-features-20261015-2a15c91/attempt01/`。
命令、plan/controller/helper rawpins、实际启动时间/设备映射、父子原身份和完整退出收据均在外部inputs及run记录。
本采用记录不冒称这些实际执行证据已产生；启动后另行追加真实结果。

## 验收条件及解释限制

root先验证实际外层成功退出、whole wall、原父子及全部owned session退场，
再验512完整generation、source/labels/currentGit初末稳定、逐帧oracle、readonly和全部artifact图闭包。
完整图核验仍不是第二次独立NN重执行或RGB重渲染；不能写成方法有效、q信息增量或actor已利用。
任何资源冲突/错误键/carry断链/缺件/非有限/模型身份或逐位oracle失败均拒绝，保留原证据。
首选在完整特征验收后按已冻结数学程序拟合reference0读出，保留全部h/hq/hp与两标签绝对风险；
它回答固定读出族的可访问性，后续压缩和保护作用仍需独立学习、匹配对照及native行为证据。

## 2026-10-16 F1启动及当时运行中观察（历史，现已失败）

root与独立Agent已逐项核controller源码/真实2a接口/labels真实metadata与输入pins，无剩余静态阻断；
本机compile/12个纯schema负例，不冒称完整N1M或真实512执行已通过。
已核并实际部署外部launcher34350B rawSHA
`0a2e500cc5bc6b10647f093287e6f30d3b8662c2e3bf7767c718c9d7955ed7f6`，
execution-plan533B `4b1d2e6517d1b987e8a35dee247bc42a2464787e18e8d78a9f4fc9955efafcd1`，
generic helper34052B `3f0b8b025128d3280df2120f3987a6f7ad8806054fab5ec2e57b6decfeca28f1`。
helper只从已核rawbuffer加载通用函数、共享原12600 deadline，不调用C-specific main或资格函数。
其pidfd/fork→setsid清理已有实际Linux20秒fixture的13leaf/6继承组证据，范围不是任意daemon隔离保证。

00:16:32–33物理4–7均无可见GPU进程/0MiB，两个实际源码目录clean；
物理5已留给C新尝试（尚未建立CUDA context不表示可共享），本入口仅取物理4。
外部controller started_at为00:16:49.694137；实际feature parent539679/PGID=SID539679/start410338164，
outer539664；actual/tested仍2a15c91，本机原handle37919持续观察，整个12600秒不变。
launch48966B SHA `bff61c4f7fa1e2870fc222f6da6239e642126b6395593b63a6d3616ec0b51d7e`，
resource19588B `f9a3e2bc52ce38281cf3745e24f49a2277b5e519cae59fa0f1a337906d4045c1`；
完整metadata在本机`analysis/outputs/dl-prevalidation-launch-20261016/launch-metadata-20261015T161932/`。
00:18:19原parent在/proc身份匹配/存活，正CPU来源认证，无source_index、GPU worker或首unit，
未读取中间特征/科学数值。这不是完整特征/吞吐/显存/方法验收，GPU4仍保留至本对象权威退出。

实际外层observer在真正SSH完成后才写receipt，严格采用exactargv/实际2a/GPU4/returncode/whole wall，
强制成功为exit0、external_failure=null、elapsed≤12600、actual_outer_complete_within12600=true，
再绑定controller terminal的完整path/bytes/rawSHA。12630只用于失败观察，不能延长成功预算。
该receipt在当时尚未产生；其后实际结果为exit1，见本页顶部失败记录，不能作为成功读出前驱。

00:21:45只读观察：原parent539679/worker542278（PGID=SID542278/start410358034）均真实存活且身份匹配，
source_index及initial已存在，5个unit complete文件；无worker_complete/generation/终态。
这超出了单纯CPU mock接口，但未作完整单元/全图或科学验收，不从数量推断q信息或方法有效。
