# 四任务训练录像对比GIF（2026-11-02）

用户指定cartpole_swingup 300k、finger_spin 1M、pendulum_swingup 1M、walker_run 1M。已从本轮正式baseline/Dt现有replay提取并排对比，左baseline、右PPD-KL，seed0。输出目录 `analysis/outputs/dmc-four-gifs-20261102/final/`，四组均提供GIF、MP4和首/中/末接触图。

## 取样与证据

沿用此前walk/run规则：第一条按首chunk文件名时间排序的保存链，取指定全局动作预算前最后一个完整episode；不看回报或画面选样。重建16条链和排除reset的全局训练动作计数，跨评价边界保持原Driver次序。链顺序不是独立持久化的worker ID证明。

这些是参数随训练变化的训练replay片段，不是300k/1M固定checkpoint的评价录像；实际动作范围直接写在图中。四组均保留完整1000动作episode，无裁掉起步或失败段。逐任务原始控制间隔根据安装dm_control suite源码/XML核对：cartpole .01秒，finger/pendulum .02秒，walker .025秒；因此完整时长分别10/20/20/25秒。

|文件前缀（均有.gif/.mp4）|两组实际训练动作范围|时长|baseline片段return|PPD-KL片段return|对应正式评价均值B/PPD-KL|
|---|---|---:|---:|---:|---|
|cartpole_swingup_300k|272001—287985|10s|697.46|843.31|497.33 / 842.36|
|finger_spin_1000k|976001—991985|20s|494.00|902.00|498.70 / 926.80|
|pendulum_swingup_1000k|976001—991985|20s|0.00|699.00|90.50 / 813.40|
|walker_run_1000k|976001—991985|25s|344.10|805.33|353.20 / 812.95|

训练片段的return仅描述该片段，不替代正式10 episode均值。finger_spin选定训练episode也出现902的高return，为终点高回报提供一条训练侧佐证，不证明此前整个阶段持续高回报或跨seed可复现。

## 验证与文件

- 输入源为 `analysis/outputs/dmc15-analysis-20261102-final/run_manifest.json` 的八条已分析运行；sv1的cartpole baseline和finger两组、sv2的pendulum/walker baseline、sv3的cartpole/pendulum/walker Dt，按实际目录只读CPU采集。原manifest、评价、final_state文件hash与前轮分析一致；完整训练SHA/路径在每条inspection.json中。
- `inputs/<task>-<group>/`保存实际远程采集源码、SSH日志、压缩包、inspection.json和frames.npz；含源replay chunk数组hash、原始1001帧、reward、所选episode及目标步数评价。大checkpoint/replay留服务器。
- 原64×64图像最近邻放大；输出640×414，左右画面各320×320，20fps原速。取最近已有帧，不插值；.02秒任务采样与理想播放时刻最多偏差.01秒。固定256色GIF调色板有颜色量化损失，MP4为有损H.264。
- 八条源帧hash及非reset reward求和通过；GIF完整逐帧解码与调色板帧像素一致、总时长准确；四个MP4完整解码通过。首/中/末接触图已目视检查标签、范围与左右对应。manifest.json记录实际版本、输入hash、取样索引、GIF/MP4输出hash及检查结果。
- 实际采集、编码、验证SHA：`575e9916077e4e9c8b731453d3d284d489599acf`；后续提交仅文档，最终完整SHA见同目录上级DELIVERY.json。本轮只本机提交、只读服务器文件；无推送、服务器源码同步、训练、推理、环境创建或GPU作业，未核GPU现状。原草稿与原实验产物保留。

## 复现

```powershell
$py='C:/Users/97370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -m analysis.dmc_comparison_gifs collect --source analysis/outputs/dmc15-analysis-20261102-final --output analysis/outputs/NEW_INPUTS
$env:PYTHONPATH='D:/program/PD/dmr3/analysis/outputs/m2-v1-seed0-20261012/plot_deps;D:/program/PD/dmr3/analysis/outputs/walk-250k-media-20261021/media_deps'
& $py -m analysis.dmc_comparison_gifs render --source analysis/outputs/NEW_INPUTS --output analysis/outputs/NEW_MEDIA
```

所有输出目录必须新建。复用已核对的walk_replay_media_v1链重建代码；原脚本未修改。未重新执行原训练或改变科研协议。

## 后续与可复制prompt

当前四组GIF全部完成。若要改版式或截短，继续现有05，标准为仍保留准确来源/动作范围与左右对应；若要求严格指定checkpoint的评价视频，则由03准备推理补录，GPU执行需用户分配。

```text
项目D:\program\PD\dmr3，继续05。交接最终版本见analysis/outputs/dmc-four-gifs-20261102/DELIVERY.json；实际采集/渲染/验证SHA 575e9916077e4e9c8b731453d3d284d489599acf。
必读AGENTS.md、docs/ANALYSIS_DMC_FOUR_GIFS.md、docs/ANALYSIS_DMC15_V1.md和产物final/manifest.json。
四任务训练replay GIF已完成，左baseline右PPD-KL，实际272001—287985或976001—991985动作，不是固定checkpoint评价。后续仅按用户反馈调整版式/时段，未决为新的展示偏好；不调参、不改协议、不推理/训练或覆盖原产物。输出analysis/outputs新目录，验收源hash、时序/速度、左右标签和完整解码，完成后报告最终/实测SHA与限制。不自动建chat或发送消息。
```
