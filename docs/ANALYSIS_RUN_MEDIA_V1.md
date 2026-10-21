# quadruped_run：50万步附近的已有训练录像

日期：2026-10-22。05沿用walk片段的选择规则，从sv2正式seed0运行的既有replay提取baseline/PPD-KL并排录像。本轮仅CPU读取产物、本机编码；未训练、推理、创建环境或使用GPU，未核GPU现状。PPD-KL是原dt组的展示名称。

## 交付与选取规则

输出：`analysis/outputs/run-500k-media-20261022/`。

- `final/run_before500k_training_8s_to18s.mp4`：固定截取episode第8—18秒，10秒并排短片。
- `final/run_before500k_training_full.mp4`：完整20秒并排录像，包括PPD-KL前段翻转过程。
- `final/contact_sheet.png`：2、5、8、11、14、17秒对照帧。
- `inputs/inspection.json`、`inputs/replay_frames.npz`：原始选定图像/reward、源路径/数组hash、链索引、正式评价及快照记录。
- `final/verification.json`、`final/manifest.json`、`final/decode_check.txt`：原评价/hash核对、源帧/reward验证、完整解码及输入/输出/版本记录。
- `legacy_compatibility.json`：原walk输入兼容检查；`COMMANDS.md`、`HANDOFF.md`、`DELIVERY.json`：复现命令、完整交接prompt与最终版本。

预先固定选取**第一条按首chunk创建时间排序的replay链，在50万全局训练动作前最后一个完整episode**，不依据回报或画面挑选。两组均选到链内第31个episode（ordinal=30）：实际全局动作**480001—495985**，1000动作、1001帧（含reset画面）。链排序不是独立保存的worker ID或环境初态证明。

每两帧取一帧、25 fps、1倍速，原任务控制间隔0.02秒、repeat1。64×64原始图像只作最近邻放大；未生成、插值或重新渲染环境画面。完整视频500帧/20秒、短片250帧/10秒，均已完整解码，源数组hash与去reset后的reward求和通过，接触图已目视核对。

## 训练片段与50万步正式评价分开解释

| 数据 | baseline | PPD-KL | PPD-KL减baseline |
| --- | ---: | ---: | ---: |
| 选定训练episode原始return | 508.234756 | 582.275476 | +74.040720 |
| 50万步10个完整评价episode平均原始return | 442.028357 | 725.627475 | +283.599118 |

该训练片段的return差异小于正式评价均值差异，不能用它替代正式评价。接触图可见PPD-KL在2—5秒有明显翻转，随后恢复四足姿态；固定8—18秒短片未覆盖这段，因此同时保留完整20秒供核对，不宣称PPD-KL在本episode全程无失稳。现有图像不含可独立核对的逐步physics/速度，不给出额外速度或步态统计。

50万步正式评价的baseline范围339.818125—514.157633，PPD-KL范围621.862861—789.242906；每组10个完整1000动作episode，原环境seed映射一致，逐episode原始return、均值、source hash、snapshot/update与原已验收表完全一致。这仍是单个训练seed的描述性证据，不能声称跨训练seed稳定性或显著性。

本片段是参数持续更新、使用train策略的历史训练行为，**不是50万步固定checkpoint评价录像**。两组原运行目录及W&B文件清单均未发现保存视频；原实际训练版本的评价回调没有持久化评价图像。正式结果和原始产物未改写。

## 快照与版本

实际训练SHA：`48d71f1541721208322442f5a1d5977124f3e888`。sv2两组原根目录：

```text
/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-seed0-20261009/dmc_quadruped_run/logging-attempt01
/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-dt-seed0-20261009/dmc_quadruped_run/dt-attempt01
```

`eval_snapshots/0500000/`下快照标签分别是`20261010T035021F099338`、`20261010T041839F411406`，动作500000、update124493，`agent.pkl`各513050093字节，完成标记存在。本轮仅确认元数据/存在性，未加载参数，不声称独立验证了checkpoint内容。

分析脚本`analysis/walk_replay_media_v1.py`新增受限的`--task dmc_quadruped_run --action-step 500000`入口，保留原walk/250000默认行为与历史输入兼容；内置原评价/hash交叉核对，不改训练源码/协议。实际采集、编码、真实源数据验证及旧walk兼容检查SHA：`aaff5e863d7a8941762c10a2ea19651f9e1d68fe`。最终交付仅新增文档/索引，完整最终SHA在HANDOFF/DELIVERY与回复中记录。只本机提交，无推送、服务器源码同步、服务器写入或W&B写入。

## 复现与下一步

在项目根目录PowerShell运行，输出目录必须尚不存在；复用本机已有NumPy/Pillow/imageio-ffmpeg依赖，不改服务器训练环境：

```powershell
$py = 'C:/Users/97370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -m analysis.walk_replay_media_v1 collect --task dmc_quadruped_run --action-step 500000 --output analysis/outputs/NEW_RUN_INPUTS
$env:PYTHONPATH = 'D:/program/PD/dmr3/analysis/outputs/m2-v1-seed0-20261012/plot_deps;D:/program/PD/dmr3/analysis/outputs/walk-250k-media-20261021/media_deps'
& $py -m analysis.walk_replay_media_v1 render --source analysis/outputs/NEW_RUN_INPUTS --output analysis/outputs/NEW_RUN_MEDIA
```

本次“同样截取”已完成。若需要严格50万步、相同评价环境seed的固定参数动作对照，首选由用户继续现有03合并窗口做checkpoint推理补录，保留全部10对完整episode，再按声明规则截取；只读准备可立即进行，GPU执行依赖用户按AGENTS分配服务器/物理GPU。不能把补录当原评价录像，或为了重现原return更改协议/动作/seed。验收需参数冻结、原seed/随机流、实际return与原值比较、完整视频与短片及来源hash；EGL像素/return偏差如实报告。完整可复制prompt见HANDOFF，不自动创建chat或发送消息。
