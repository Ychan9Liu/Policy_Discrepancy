# 实验分析

此目录存放可复现的指标整理和绘图代码。输入来自 `docs/EXPERIMENTS.md` 索引的服务器运行目录；大型中间文件及生成图表放在 `analysis/outputs/`，不提交到 Git。

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
