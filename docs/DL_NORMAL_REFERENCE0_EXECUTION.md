# DL 正常共同历史 reference0 实际执行采用

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
