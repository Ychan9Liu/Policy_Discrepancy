# quadruped_walk：25万步差异与已有录像提取

日期：2026-10-21。05只读检查sv2既有seed0正式产物，并在本机编码录像；没有训练、模型推理、创建环境、GPU作业、源码同步或W&B写入，未核当前GPU占用。

## 可交付的片段

已经从原训练replay提取baseline与PPD-KL并排录像。PPD-KL是原dt组的展示名称。固定选取规则是**第一条按首chunk创建时间排序的replay链，在25万全局训练动作前最后一个完整episode**；规则在读取return和观看帧之前确定，没有按高低回报挑episode。

- 实际记录范围：两组均为全局训练动作224001—239985，链内第15个完整episode（ordinal=14），各1000动作、1001帧（含reset画面）。不能标成25万步固定参数的评价录像。
- 两组该训练episode的原始reward独立求和分别为baseline **500.288589**、PPD-KL **940.728044**。这是选定训练片段的return，不是下表的正式评价均值。
- 原图64×64，使用最近邻放大。原任务控制间隔0.02秒、repeat1；每两帧保留一帧、25 fps，按原速播放。没有生成、插值或重新渲染环境画面。
- 该片段PPD-KL保持行走姿态，baseline后半段失稳、翻倒。根据接触图选择8—18秒作为展示短片，同时交付完整20秒供核对。不能从一个片段推出所有episode或跨训练seed的行为。
- 训练时参数持续更新、使用train策略；两条链未保存可核对的逐步physics状态，不能声称画面来自完全相同的环境初态或冻结策略。

输出目录：`analysis/outputs/walk-250k-media-20261021/`。

| 文件 | 内容 |
| --- | --- |
| `final/walk_before250k_training_8s_to18s.mp4` | 10秒并排短片，包含baseline失稳过程 |
| `final/walk_before250k_training_full.mp4` | 完整20秒并排录像 |
| `final/contact_sheet.png` | 2、5、8、11、14、17秒的对照帧 |
| `inputs/inspection.json` | 运行/版本、25万步评价、快照元数据、源路径/数组hash、链索引与片段选择 |
| `inputs/replay_frames.npz` | 原始选定图像及reward；原大replay/checkpoint留服务器 |
| `final/manifest.json`、`verification.json` | 编码版本、输入/输出hash、时间/帧率、正式评价及源hash交叉检查 |
| `COMMANDS.md`、`HANDOFF.md` | 复现命令与完整交接prompt |

两段MP4已由ffmpeg完整解码核对，分别500帧/20秒和250帧/10秒；源帧hash及去reset后的reward求和通过，接触图已经目视核对。

### 2026-10-22：同一时刻的2×10对比图

用户指定上述完整MP4，按时间从前往后均匀抽取10帧；每列上排baseline、下排PPD-KL。图为`analysis/outputs/walk-250k-ten-frames-20261022/walk_comparison_2x10.png`，另有文字可编辑、内嵌原始裁剪PNG的同名SVG。源视频SHA256=`347acd3d0e08da88a37826f8dc50bd1c3315e5804f1a3e3762cfe9446f1280d6`，已与原视频manifest核对，不重新生成训练录像。

解码500帧、25 fps，以`round(i*499/9), i=0..9`选最近的已有帧，时间依次为0.00、2.20、4.44、6.64、8.88、11.08、13.32、15.52、17.76、19.96秒，覆盖视频首帧至最后一帧。每个时刻分别裁出视频中两个完整384×384画面，去除原视频重复文字；两行共20张，不插值、不改像素、不按动作结果挑帧。主图4148×1010像素，PNG标记300 dpi；放大画面仍保留原视频/64×64观测的分辨率限制。

脚本`analysis/video_comparison_grid.py`保存源视频/manifest/脚本/输出哈希、每列原视频帧号/精确时间/裁剪框、20张独立PNG和`sampling.json`。成图已目视核对，重新读取成图逐块验证其像素与对应MP4解码裁剪完全一致；源视频哈希保持。实际生成/检查SHA=`ffe82b95d18bdbf85758e2a0ecc4cdf53343053a`，后续交付仅文档索引，最终版本见产物HANDOFF/DELIVERY。继承本报告的训练replay证据范围，不新增固定checkpoint/方法有效性结论。本轮仅本机文件处理，无服务器访问、训练、推理或GPU，未核GPU现状；复现命令及继续现有05的版式核对prompt在产物`COMMANDS.md`/`HANDOFF.md`。

## 严格25万步的原始评价事实

| 组 | 10 episode平均原始return | 逐episode范围 |
| --- | ---: | ---: |
| baseline | 366.208158 | 101.759175—779.197173 |
| PPD-KL | 854.627989 | 729.836031—933.424423 |

差值为**+488.419831**。两组使用相同的10个正式评价环境seed，各episode均1000动作。但baseline有4个episode达到约725—779，另外6个约102—112；因此当前数据支持“这个训练阶段两组行为表现差异明显”，不支持“baseline每次都不会走”。单次评价的10个episode也不是10个独立训练seed。

本次重新读取两组原`evaluations.jsonl`、manifest、final_state的hash，与2026-10-12已验收镜像核对；25万步各10个原始return、均值、snapshot_id与原表一致。W&B仅只读列文件，没有使用平滑曲线或summary替代原始值。

## 没有保存原评价视频，checkpoint仍在

两组运行目录递归检查未找到MP4/GIF/WebM等视频；W&B可读文件清单也未出现视频。实际训练SHA的`protocol_v1.evaluate`回调只累计reward与episode信息，没有持久化评价图像或动作轨迹。

sv2原根目录：

```text
/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-seed0-20261009/dmc_quadruped_walk/logging-attempt01
/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-dt-seed0-20261009/dmc_quadruped_walk/dt-attempt01
```

各自`eval_snapshots/0250000/`保留完成标记和513050091字节`agent.pkl`，元数据均为动作250000、update61993、实际训练SHA `48d71f1541721208322442f5a1d5977124f3e888`；快照标签分别为`20261010T010545F612866`、`20261010T013410F294612`。本次检查存在性和标记，未加载或声称重新验证其参数内容。

严格对应25万步的首选下一任务是：由用户继续现有合并窗口03，准备并执行**只读checkpoint推理补录**，沿用原10个环境seed和eval随机流，保存全部配对episode后再截取示例。无需重新训练，不覆盖原产物，不把补录录像伪装成原始评价录像。GPU推理依赖用户按根`AGENTS.md`分配服务器/物理GPU；本提示不分配资源。不能假定EGL像素和原return逐位重现，若有偏差须保留并说明。完整prompt在产物`HANDOFF.md`，不自动创建chat或发送消息。

## 版本及复现

分析脚本：`analysis/walk_replay_media_v1.py`。正式编码和检查实际SHA为`200d7c74b5201a5921aaf228b661e827e00007a7`；最终交付提交只增加文档索引与交接记录，完整SHA写入`HANDOFF.md`/`DELIVERY.json`及最终回复，避免自引用。图表脚本先前已在`b5e536aa6e439d2ddc3a75d511e4e3c2c74c5af9`交付；本次在上述200d7c7再次生成三图并验证源数据范围，原图索引仍见`docs/FIGURES_M2_V1_PPD_KL.md`。

本机PowerShell复现，`--output`需用不存在的新目录：

```powershell
$py = 'C:/Users/97370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -m analysis.walk_replay_media_v1 collect --output analysis/outputs/NEW_WALK_INPUTS
$env:PYTHONPATH = 'D:/program/PD/dmr3/analysis/outputs/m2-v1-seed0-20261012/plot_deps;D:/program/PD/dmr3/analysis/outputs/walk-250k-media-20261021/media_deps'
& $py -m analysis.walk_replay_media_v1 render --source analysis/outputs/NEW_WALK_INPUTS --output analysis/outputs/NEW_WALK_MEDIA
```

采集仅使用SSH和服务器原NumPy/W&B读取；本机为Python3.12.14、Pillow12.3.0、NumPy2.4.2、imageio-ffmpeg0.6.0。ffmpeg可执行文件hash与版本存manifest，依赖仅装在忽略的本机输出目录，未改变服务器训练环境。本机提交，不推送或同步服务器；原RESEARCH/STATUS/proposal未提交候选修改保留。
