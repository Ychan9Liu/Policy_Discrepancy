# M2 v1 三任务 PPD-KL 展示图

日期：2026-10-21。05根据用户要求重绘，使用 `docs/ANALYSIS_M2_V1.md` 对应的原seed0复算产物；没有新增实验或修改训练源码、冻结协议、正式结果。

输出目录：`analysis/outputs/m2-v1-ppd-kl-figures-20261021/`。每张均提供220 dpi PNG和可编辑文字SVG：

| 图 | 文件 | 展示范围 |
| --- | --- | --- |
| learning_curves | `learning_curves.png` / `.svg` | 三任务一行；仅baseline、PPD-KL；21个未平滑评价均值 |
| descriptive differences | `descriptive_differences.png` / `.svg` | 主指标AUC/1M、次指标80万至100万五点评价均值；PPD-KL分别减baseline、constant、global shuffle |
| gate diagnostics | `gate_diagnostics.png` / `.svg` | 三任务行、Mean Dt与Actual active weight两列；四组曲线；Mean Dt保留原对数轴 |

三任务为quadruped_run、quadruped_walk、reacher_hard。仅在本组展示图省略hopper_hop，PPD-KL仅是原 `dt` 组的展示名称；Mean Dt是原诊断量名称，继续保留。原四任务16条比较及总体指标不变，没有另算三任务总体或从单seed推断显著性。

## 数据与验收

- 输入：`analysis/outputs/m2-v1-seed0-20261012/analysis-final/return-audit/tables.json` 与 `gate_intervals.csv`。读取前逐文件检查原 `output_hashes.json` 中的SHA256；不从W&B平滑图取数。
- 学习曲线126个点：三任务×两组×21点，每点10个episode；使用实际快照动作步和平均原始return，无平滑、插值或误差带。
- 差值图9个任务/对照组合×主次两指标：按原10 episode重新求点均值，再复算梯形AUC/1M和五点末段均值，核对原差值表；图中数字仅显示至小数点后一位。
- 门控图240个区间：三任务×四组×20个5万动作区间。Mean Dt使用 `D_mean`，Actual active weight使用 `active_weight_mean`，沿用每次成功更新内活跃位置均值的区间平均口径；不是恢复逐位置梯度或阶段合并分布。
- 三张图均注明seed0，已逐张目视检查标题、图例、任务范围和布局。验收范围为本地数据及绘图，不重新验收原16条训练，也不提供新科研结论。

`plot_data.json` 保存所绘数据；`manifest.json` 保存输入/脚本/输出SHA256、实际绘图Git SHA、Python/Matplotlib版本、筛选范围和核对项；最终版本及交接见同目录 `HANDOFF.md`。生成产物保留在忽略目录，不提交Git。

## 本机复现

在项目根目录的PowerShell运行，`--output` 必须使用尚不存在的新目录：

```powershell
$env:PYTHONPATH = 'D:/program/PD/dmr3/analysis/outputs/m2-v1-seed0-20261012/plot_deps'
& 'C:/Users/97370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m analysis.plot_m2_v1_ppd_kl --source analysis/outputs/m2-v1-seed0-20261012/analysis-final --output analysis/outputs/NEW_PPD_KL_FIGURES
```

本轮只在本机运行绘图、提交分析脚本与文档；无GitHub推送、服务器源码同步、训练或GPU作业，未核查服务器GPU现状。既有RESEARCH/STATUS/proposal未提交修改保留，STATUS仅提交本轮新增索引。

下一步可继续现有05核对投稿版式；完成标准为图标题、任务、图例、源数据哈希及PNG/SVG均与本说明一致，无需新建chat。科学协议决策继续由02与用户确认，本次不自动交接消息。
