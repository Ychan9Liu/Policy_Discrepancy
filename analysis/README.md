# 实验分析

此目录存放可复现的指标整理和绘图代码。输入来自 `docs/EXPERIMENTS.md` 索引的服务器运行目录；大型中间文件及生成图表放在 `analysis/outputs/`，不提交到 Git。

## DL 1M 探索结果复核

`dl_pilot_review.py` 对已完成DL-pilot-r3做CPU只读审阅：核验原始21点评价、全部更新收据、
21个实际模型快照与最终完整checkpoint的参数/计数、历史B/D来源及实际设备记录，
独立复算固定1M AUC/80–100万tail，并汇总已有batch级保护统计。
不训练、不重演环境、不改变参数或把历史单seed比较称为机制/跨seed证据。
checkpoint只在可信服务器本地读取，不下载或提交；输出目录必须新建。

```sh
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu python analysis/dl_pilot_review.py \
  --plan /data/Policy_Discrepancy/runs/dl-pilot-r3-plan-20261013-13ec72a \
  --output /data/Policy_Discrepancy/runs/NEW_DL_REVIEW
```

固定三段AUC和逐eval-seed终点差仅作事后描述。gate统计来自重复采样的replay batch，
不是独立状态覆盖；N批次上D-only潜在释放比不是实际Dt训练、正式κ或新增已批准对照。
复核生成 `review.json` 与小型 `points.json`，图表和运行记录见 `docs/DL_PILOT_R3_RESULTS.md`。

本机用Matplotlib从这两份只读JSON生成学习/保护曲线与固定AUC逐点贡献（不删除点）：

```sh
python analysis/dl_pilot_plot.py --review REVIEW/review.json --points REVIEW/points.json \
  --expected-review-sha REVIEW_SHA256 --expected-points-sha POINTS_SHA256 --output NEW_FIGURES
```

绘图在写盘前验证这两份固定字节、21点完整网格、原始episode均值和绝对统计；
相对AUC相同不能掩盖三条曲线共同污染。既有正确图保留，新图使用新目录。

`dl_signal_identifiability.py` 是 `DL-identifiability-r1` 的0GPU解释检查：
从已认证N批次的聚合量计算合法ρ可行域，并穷举有限四cue反例，区分信息可用性、
当前actor利用和保护收益。它不运行模型、环境、训练或replay，不提供新方法验收。

```sh
python analysis/dl_signal_identifiability.py --review REVIEW/review.json \
  --expected-review-sha REVIEW_SHA256 --expected-git FULL_CLEAN_COMMIT --output NEW_ANALYSIS.json
```

范围、未决问题及下一有限诊断候选见 `docs/DL_R4_IDENTIFIABILITY.md`；
同N批次的代数比例不能当实际Dt强度、正式κ或新训练表现。

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
