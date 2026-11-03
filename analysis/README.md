# 实验分析

## Pendulum seed2/3原参数复验已启动（2026-11-04）

用户已于11月3日明确批准四条复验并启动，11月4日又授权其余已分配空闲GPU并行。当前sv3 GPU1/4保留seed2/3 baseline，GPU5/6运行对应PPD-KL（dt），四条并行、GPU7保留；配置/实际SHA/工程与启动验收见`docs/PENDULUM_SEEDS23_PARALLEL.md`，原启动证据在`docs/PENDULUM_SEEDS23_RUNS.md`。没有新增alpha、损失、奖励或其他科学干预。四条都执行原训练SHA c5f86ca，控制器另记b7f42d4。

统计索引改为`/data/Policy_Discrepancy/runs/formal-m2-pendulum-seeds23-parallel-dt-20261104-sv3/actual-RUN_INDEX.json`，它包含原两baseline及新两Dt。原串行Dt槽位只有替代marker，没有训练数据，不纳入比较；原监督器对旧槽位的预期拒绝不代表新Dt失败。

正式统计命令为`python -m analysis.return_metrics --profile pendulum-seeds23 --index ACTUAL_RUN_INDEX.json --output NEW_DIR --require-complete`。此profile严格为pendulum、seed2/3、baseline/dt四条，沿用百万动作/21×10/原始return AUC及末段五点均值，输出逐seed指标和两条配对差值；缺seed/点不插值或重加权，不并入seed0/1、不把工程短程或评价episode当训练重复。完整批次指标需等四条结束验收。

`pendulum_seeds23_analysis.py collect/analyze`补充该四条的终点入口。collect只读sv3，要求新合并索引四条accepted与supervisor/completion一致，严格识别原两baseline和GPU5/6新Dt；不读取旧Dt替代槽位。镜像包含原始评价、metrics、全部成功更新收据、完整配置、c与终点审计、快照metadata/done及源hash，不下载checkpoint/replay、不写W&B。analyze调用原`return_metrics --profile pendulum-seeds23`，重新逐行核对收据和各seed c、与控制器指标交叉检查，生成逐seed学习曲线和门控图表。分布分位数仅为日志所存批次分位数的更新均值，不解释为全训练合并分位数。快照只复用元数据/完成标记及既有审计，不声称新解码参数或逐transition评价reward。

```sh
python -m analysis.pendulum_seeds23_analysis collect --output analysis/outputs/NEW_PENDULUM23_INPUT
python -m analysis.pendulum_seeds23_analysis analyze --input analysis/outputs/NEW_PENDULUM23_INPUT --output analysis/outputs/NEW_PENDULUM23_RESULT
python -m unittest tests.test_pendulum_seeds23_analysis tests.test_return_metrics tests.test_m2_analysis -v
```

2026-11-04准备时四条仍运行中；入口的范围拒绝、源hash和加权统计fixture验收不代表四条真实终点通过。命令/hash/测试和当前阶段记录在`analysis/outputs/pendulum-seeds23-analysis-preparation-20261104/`。正式采集、完整图表目视检查和校正结论待四条终点产物就绪；不移动活动训练/控制器checkout。

同模块`initial --output NEW_INITIAL_DIR`可在训练期间以CPU只读四条已保存的0步可信checkpoint和16条replay链各前128行（16次reset、2032动作），核对首次成功更新边界、逐参数数组hash/有限性及同seed输入差异。读取严格限定原两baseline/新两Dt路径和训练SHA，大文件留sv3，保存checkpoint文件hash、前缀数组hash、逐字段差异和实际分析版本。它核查实际初始条件，不宣称整个训练逐位确定，也不替代终点验收；执行结果在`analysis/outputs/pendulum-seeds23-initial-20261104/`。

## PPD-KL作用强度与无明显优势的解释（2026-11-03）

`docs/ANALYSIS_PPD_KL_STRENGTH.md`关联已有15任务return指标与门控统计，解释alpha、实际减弱、阶段/高分空间及单seed限制；不把无优势统一归为参数太小，不据已有曲线直接调参。核查脚本/输入hash/未舍入表在`analysis/outputs/ppd-kl-strength-20261103/`，无训练或协议变更。

## Pendulum失败链路与校正准备（2026-11-03）

`diagnose_pendulum_v1.py collect/analyze/prefix`只读两条正式seed0原始replay/0步checkpoint/JSONL，核对百万动作、episode奖励累计与首次进入奖励区间、动作与下一行奖励对齐、初始参数及首次更新前输入差异，输出统计与六面板图。完整来源、命令和范围见`docs/ANALYSIS_PENDULUM_CORRECTION.md`；产物为`analysis/outputs/pendulum-diagnosis-20261103/`。CPU读取，不执行agent/环境/GPU。4项手算新测试与既有指标统计共24项通过；该诊断交付时新seed方案待确认、尚不允许启动，后续授权与执行见上节，诊断产物和解释不变。

## 新增15任务分析（2026-11-02，用户排除acrobot）

`analyze_dmc15_v1.py collect --output NEW_DIR` 只读sv1/sv2/sv3正式索引，采集已逐条审计的非acrobot baseline/Dt原始评价/JSONL/配置和快照小证据，逐行复核收据及正式c；`analyze --output NEW_DIR` 从本机镜像复算15任务30条的AUC、末段均值和门控，导出CSV/JSON/PNG/SVG。使用普通Python（校验依赖assert，不用-O），绘图依赖同既有Matplotlib环境。完整命令、训练SHA及局限见 `docs/ANALYSIS_DMC15_V1.md`，实际产物 `analysis/outputs/dmc15-analysis-20261102-final/`。不修改原完整dmc16统计profile、不将15任务子集称全32条完成、不混算历史四任务或hopper补seed总体。

## hopper_hop 六画面任务示意GIF（2026-10-26）

`hopper_demo_gif.py collect/render`只读sv2既有seed0 baseline，取前六条保存链在百万动作前最后完整episode的固定2—8秒片段，生成三行两列、横纵间隙0、无文字/边框的6秒循环GIF。使用真实64×64观测最近邻放大，固定256色全局调色板；不执行模型/环境或GPU，不作方法对照。输出与命令、来源和量化限制见`docs/ANALYSIS_HOPPER_GIF.md`，GIF副本在`docs/proposal/hopper_hop_任务示意.gif`。

## 剩余后八任务与两机分片

本批后八任务用 `--profile dmc8-remaining`，配置和资源见 `docs/DMC8_REMAINING_RUNS.md`。sv1两个单卡计划、sv2六任务计划已启动八条baseline，八条Dt等待各自完整审计；GPU2编译失败的旧尝试保留，当前为retry03。23:54北京时间sv1 GPU7后进入的外来计算进程使独占检查未通过；23:58:59复核已退出，八卡当时只发现本项目C/G PID，短时冲突证据和影响未量化的限制保留。不能把任一快照当持续独占保证。单机/单卡audit生成逐任务指标且八任务总体保持null；全部八任务完整证据汇齐后，独立索引才可用 `--require-complete` 判定完整。沿用原始return、固定百万分母及指定五点，不自动与前八任务/原四任务/补seed汇总。前八批6条Dt缺项仍保留，不因新增后八任务而标完整。

## 新增前八任务单seed独立统计

sv1前八任务×baseline/Dt的16条授权运行见 `docs/DMC8_SERIAL_RUNS.md`，统计入口为 `python -m analysis.return_metrics --profile dmc8-first8 --index ACTUAL_INDEX --output NEW_OUTPUT --require-complete`。精确八任务/seed0/两组，沿用百万固定分母、21×10评价、指定末段五点；缺任务/点不重加权。不能将只完成八任务的批次用完整 `dmc16` profile宣称完整，也不自动并入原四任务或补seed总体。当前正式队列运行中，工程短程不能生成完整正式指标。

## M2 v1 三任务 PPD-KL 展示图（2026-10-21）

`plot_m2_v1_ppd_kl.py` 从已验收的 `analysis-final/return-audit/tables.json` 和 `gate_intervals.csv` 重绘，先检查原分析哈希、21×10评价、episode均值、主次指标差值及20个诊断区间。仅展示quadruped_run、quadruped_walk、reacher_hard；原dt组的图例名为PPD-KL。学习曲线只保留baseline/PPD-KL，另两图保留三个对照；门控只保留Mean Dt与Actual active weight。原四任务正式分析不变，不重算三任务总体，不平滑。

```sh
python -m analysis.plot_m2_v1_ppd_kl --source analysis/outputs/m2-v1-seed0-20261012/analysis-final --output analysis/outputs/NEW_PPD_KL_FIGURES
```

依赖同下述Matplotlib环境，输出必须新目录；保存三张PNG/SVG、逐曲线数据与输入/脚本/输出哈希。此次图与PowerShell复现命令见 `docs/FIGURES_M2_V1_PPD_KL.md`。

## Hopper seed1/2/3补测范围（2026-10-21）

用户已确认仅hopper的baseline/Dt六条，资源sv3 GPU1/3/4/5/6/7；本轮只准备，不启动。配置与范围见`docs/HOPPER_SEEDS123_PREPARATION.md`。新结果使用 `python -m analysis.return_metrics --profile hopper-seeds123 --index ACTUAL_RUN_INDEX.json --output NEW_DIR --require-complete`；固定新三seed、hopper单任务、两组和原预算/评价/指标。保留逐seed值及三条paired_seed_differences，缺seed不重加权；overall_metrics在此profile仅指hopper均值，不是四任务总体。默认CLI保持四任务seed0，原结果不变，不自动混入seed0或新增constant/shuffle。

此目录存放可复现的指标整理和绘图代码。输入来自 `docs/EXPERIMENTS.md` 索引的服务器运行目录；大型中间文件及生成图表放在 `analysis/outputs/`，不提交到 Git。

`walk_replay_media_v1.py collect/render`可只读提取sv2正式quadruped_walk的既有训练图像并编码并排视频。片段固定为第一条保存链在25万动作前最后一个完整episode，实际动作224001—239985；它不是25万步checkpoint评价录像。保留原帧/hash、完整20秒、8—18秒短片与来源标签；未执行环境或agent。严格25万步录像需用已保存checkpoint另行推理补录，GPU资源由用户分配。命令与证据边界见`docs/ANALYSIS_WALK_MEDIA_V1.md`。

同一脚本新增`collect --task dmc_quadruped_run --action-step 500000`，按相同规则提取run的480001—495985动作片段；`render`从输入识别任务/步数并核对原正式评价/源hash，输出完整20秒与固定8—18秒短片。原walk默认入口/旧输入保留兼容。训练片段return508.23/582.28与50万步正式均值442.03/725.63分别标注；不得把训练录像当固定checkpoint评价。见`docs/ANALYSIS_RUN_MEDIA_V1.md`。

`video_comparison_grid.py --video WALK_FULL_MP4 --output NEW_DIR`直接解码已有walk完整视频并核对原manifest哈希，旧默认生成含首尾的2×10 PNG/SVG。当前展示加`--columns 11 --hide-seed --no-footer --horizontal-gap 0`，取0、2、…、18、约20秒，保留标题和训练步数说明，仅删除seed0、页脚与横向间隙；最后实际19.96秒标`≈20 s`。每列两行来自同一视频帧，逐块核对成图像素与MP4裁剪，保存独立PNG和`sampling.json`；不插值或按动作结果挑帧。输出与命令见`docs/ANALYSIS_WALK_MEDIA_V1.md`的2026-10-22小节。

同一脚本也接受`run_before500k_training_full.mp4`，从已核对文件名/manifest自动使用quadruped_run标题与480k—496k训练动作说明，其余11帧版式/像素验证一致。run图与复现入口见`docs/ANALYSIS_RUN_MEDIA_V1.md`的11帧小节，旧walk默认入口保持。

## M2 v1 seed0 终点复核与门控分析

`collect_m2_v1.py` 通过只读 SSH 收集 sv2 正式16条的评价、JSONL、成功更新收据、配置、正式 c、04终点审计及快照小标记；不复制 checkpoint/replay，不写 W&B。小镜像包带原路径/SHA256，采集时只读交叉检查 W&B 精确摘要及资源。固定来源是 `formal-four-groups-pd1-20261010-final-attempt01`，不是启动交接版本。

Hopper低回报专项：`inspect_hopper_v1.py` 在sv2以CPU NumPy只读原0步可信checkpoint、replay的reward/action/边界数组及首次更新前图像，重建16条worker链与动作顺序，独立求和训练episode奖励；不会创建环境、调用agent、优化更新、使用GPU或写服务器。原大文件留服务器，保存逐NPY数组hash与初始294数组hash。`--frames`额外抽取已有图像，`--frames-only`可单独抽帧。

```sh
python -m analysis.inspect_hopper_v1 --frames --output analysis/outputs/NEW_HOPPER_INPUT
python -m analysis.diagnose_hopper_v1 --inspection analysis/outputs/NEW_HOPPER_INPUT/remote_inspection.json --mirror analysis/outputs/m2-v1-seed0-20261012/inputs --frames analysis/outputs/NEW_HOPPER_INPUT/replay_frames.npz --output analysis/outputs/NEW_HOPPER_ANALYSIS
```

诊断报告见`docs/ANALYSIS_HOPPER_M2_V1.md`。训练replay的raw reward求和是补充证据，不替代正式评价指标，也不声称已获得评价逐reward轨迹。**历史无版本日志**的`mode_vector_diff_frac`是因子mode不一致率，`mode_factor_diff_frac`是one-hot坐标不一致率，32类时前者为后者16倍；旧整向量比例缺失。2026-10-20修复后新增`mode_diagnostic_version=2`和`mode_coordinate_diff_frac`，factor/vector分别正确记录因子/整向量比例，不能跨版本直接拼同名曲线。兼容说明、CPU验证及限制见`docs/CODE_MODE_DIAGNOSTICS_REPAIR.md`。

独立兼容审计只读原16条已审查训练版本与镜像哈希，保留原字段并另列明确语义；旧整向量值保持null，不改旧表/日志或正式return。原分析脚本保留历史复算语义。

```sh
python -m analysis.mode_diagnostics --inputs analysis/outputs/m2-v1-seed0-20261012/inputs --output analysis/outputs/NEW_MODE_AUDIT
python -m unittest tests.test_mode_diagnostics -v
# JAX回归仅CPU；不创建环境或执行优化器更新
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 python -m unittest tests.test_mode_probe_regression tests.test_rep_probe tests.test_rep_overlay -v
```

`analyze_m2_v1.py` 校验镜像与04审计时输入哈希、逐条实际训练/审计SHA、冻结配置、随机流和同任务初始评价、百万动作/完整收据/21点×10episode、正式窗口S/N及先冻结后constant启动，再调用既有统计器 `--require-complete`，以第二种直接算式交叉核对指标，输出运行清单、c表、按阶段门控表及PNG/SVG图。

```sh
python -m analysis.collect_m2_v1 --output analysis/outputs/NEW_COLLECTION
python -m analysis.analyze_m2_v1 --inputs analysis/outputs/NEW_COLLECTION/inputs --output analysis/outputs/NEW_ANALYSIS
python -m unittest tests.test_return_metrics tests.test_m2_analysis -v
```

采集器仅依赖标准库和SSH，服务器使用现有dreamer环境。绘图需要 Matplotlib 3.10.8 / NumPy 2.4.2；本轮本机独立 `outputs/.../plot_deps` 安装，不修改服务器训练环境。输出必须新目录；失败尝试保留。诊断是每5万动作区间内的成功更新均值，以快照update ID得到实际更新数后加权；D分位数为逐批分位数均值，不能当作阶段合并分位数。仅logging收据持久化S/N，其余组三个matching字段为0是源码约定，不能据此解释为无活跃位置；各组作用取真实 `train/dt/` 日志。活跃区均值是每更新活跃均值的加权平均，不宣称已恢复逐位置梯度。详见 `docs/ANALYSIS_M2_V1.md`。

## 新增DMC16准备范围（2026-10-22，实测暂停）

`scripts.prepare_dmc16`仅解析其余16任务×logging baseline/Dt、seed0的32份配置，记录固定sv1物理GPU0–7的每卡4条排队计划，不创建环境、不使用GPU、不启动训练，也不监视空闲自动启动。用户已暂停本批实测；计划保留`authorized_to_train=false`、`launch_argv=null`，EGL实际映射未绑定。完整范围、条件与准备验收见`docs/DMC16_PREPARATION.md`。

未来实际产物使用独立`analysis.return_metrics --profile dmc16`，冻结百万动作/21×10评价/指定末段五点，仅seed0、这16任务、baseline/Dt；不改变默认原四任务四组profile，不自动合并旧四任务或hopper补seed。本批没有constant/shuffle，不能仅凭两组结果回答强M2选择性主张。成绩名`dmc_ball_in_cup_catch`显式映射为现有包装器的运行ID`dmc_cup_catch`。

## 原始 return 统计

实现为 `analysis/return_metrics.py`，只依赖 Python 标准库；协议依据为 `docs/EXPERIMENTS.md` C/C.1。生产 CLI 固定100万动作预算、21点、每点10完整 episode、指定末段五点、四任务、训练 seed0。它不接受新的预算、seed 或指标参数。`tests/run_return_metrics_fixtures.py` 的两 seed、小预算仅是手算统计 fixture。

先在忽略的输出目录准备一个 JSON run index，每个训练运行一条；`logging` 自动映射到 `baseline`。从不同服务器收集数据时只读镜像小日志，保留实际服务器目录与唯一 run ID，不合并两个相同 task/group/seed 的运行来选好结果。

```json
{
  "runs": [
    {
      "run_id": "sv1:UNIQUE_RUN",
      "directory": "/data/Policy_Discrepancy/runs/UNIQUE_RUN",
      "source_directory": "/data/Policy_Discrepancy/runs/UNIQUE_RUN"
    }
  ]
}
```

`directory` 是执行统计的机器能读取的目录，`source_directory` 是实际原始目录。完整16条 index 由04按真实运行索引构建。至少读取 `protocol_manifest.json` 和 `evaluations.jsonl`；可用 config、execution、final、audit 也记录哈希。新日志 schema v2 保留 `scores/mean`，增加 `returns_raw/mean_return_raw`、逐 episode 完整性、奖励来源、环境 seed/动作随机流、目标/实际快照动作步和唯一 `snapshot_id`。快照目录有 `evaluation_snapshot.json` 与 checkpoint 完成标记供轻量核对；统计器不修改输入。

旧日志的 `scores := episode_return_raw` 只针对已审查源提交自动映射（名单在代码中），不凭字段名推断归一化或完整性。旧聚合日志无法重新逐 transition 求 reward 和；完整 episode 依据为已审查的逐 episode `Driver(episodes=1)` 源路径，加上原始长度/seed/快照证据。该推断的限制随审计输出，不伪造不存在的 terminal 原始日志。

读取旧实际 checkpoint 时可能加载较大的可信项目 `agent.pkl`。已有04的小证据包可改用 index 的 `snapshot_audit_file`：该 JSON 顶层为 `runs`，每个 run 含 `directory/git_commit/artifact_sha256/snapshots`，其中 evaluations 文件哈希必须匹配；每个 snapshot 含 `action_step/counters.updates/params_sha256`。新 schema 还要在外部审计中保留完整 `snapshot_id`，可另给 `actual_action_step`。旧04 `resource-summary.json` 已符合旧日志格式；不同快照的冲突一律拒绝，不从文件名任选。可用 `source_archive` 记录原小证据包哈希。只对可信本项目 checkpoint 使用 pickle reader。

```sh
python -m unittest tests.test_return_metrics -v
python -m tests.run_return_metrics_fixtures --output analysis/outputs/NEW_HAND_FIXTURE
python -m analysis.return_metrics --index analysis/outputs/RUN_INDEX.json --output analysis/outputs/NEW_RETURN_AUDIT
```

输出目录必须新建。六张 CSV 为 `episodes`、`evaluation_points`、`seed_metrics`、`task_metrics`、`overall_metrics`、`dt_differences`；另有 `tables.json` 和 `audit.json`。CSV 中 null 为留空，JSON 中为 null，不能读作0。审计保存工具提交/工作区状态、固定规范、输入哈希/字段映射、重复重写计数、来源和精度。点均值核对容差事前固定 `atol=1e-9, rtol=1e-12`，仅用于 float64求和顺序误差；动作步、episode ID、快照与数量精确核对。

指标缺失/冲突时仍保存原始可用行，AUC 和 tail 独立标记 complete 与原因；不改变分母或替换点位。任务和总体需全部预定组成齐全才出数值；同一训练 seed 有多个运行时拒绝选取或平均。普通命令成功表示审计表已生成，不表示正式指标完整；`--require-complete` 在表生成后对不完整结果返回退出码2。工程短跑保留原始 return 点表，正式 AUC/tail/差值留空且标记不完整，不生成缩区间的伪正式指标。所有统计只支持当前单训练 seed 的描述性解读。
