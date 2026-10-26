# hopper_hop 六画面任务示意 GIF

日期：2026-10-26。用户确认需要六段hopper_hop动作示意，不作baseline/PPD-KL方法比较；要求三行两列、画面之间无缝隙。已完成纯画面、无文字/边框的循环GIF。

交付文件：`analysis/outputs/hopper-demo-gif-20261026/final/hopper_hop_任务示意_3x2.gif`。同字节副本放在既有任务示意图目录`docs/proposal/hopper_hop_任务示意.gif`，没有修改该目录其他文件或报告。512×768像素，每格256×256，150帧、25 fps、6秒、1倍速、无限循环；横向/纵向间隙和外边框均为0。

## 素材与选择

- 使用sv2原seed0正式logging baseline的六段不同训练replay；同一次训练的前六条保存链按行优先排列，属于任务动作示意，不是六个独立训练seed或方法对照。
- 固定选取按首chunk创建时间排序的前六条链，在100万训练动作前各自最后一个完整episode（ordinal=61），不按reward或画面选样。六个完整episode实际约976001—991990动作；每格统一截episode的2—8秒，每隔一个原帧取样，帧索引100、102、…、398，原任务控制间隔0.02秒，因此25 fps为原速。
- 原始图像64×64，最近邻放大；不创建环境、不加载模型、不重新渲染，不插值或生成动作。GIF使用固定256色全局调色板、无抖动，量化造成颜色精度损失，不能声称GIF逐RGB像素等于原始观测。三个时刻的无损参考PNG保留在final目录。
- 原服务器根：`/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-seed0-20261009/dmc_hopper_hop/logging-attempt01`。实际训练SHA=`48d71f1541721208322442f5a1d5977124f3e888`。大replay/checkpoint留服务器，仅收集六段必要帧、reward及小证据。
- 不将示意GIF作为固定checkpoint评价录像、正式return指标或跨seed结论。纯示意图不意味着所示训练策略每个episode都成功。

## 复现与验收

脚本`analysis/hopper_demo_gif.py`提供`collect`和`render`。实际采集、编码与检查SHA=`fa5b54a7bf678717bc090cf8121da43700034998`；后续交付仅文档。最终SHA、代码差异和完整可复制交接prompt在产物HANDOFF/DELIVERY。输出目录必须新建，不覆盖原始输入或历史产物。

```powershell
$py = 'C:/Users/97370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -X utf8 -m analysis.hopper_demo_gif collect --output analysis/outputs/NEW_HOPPER_GIF_INPUTS
$env:PYTHONPATH = 'D:/program/PD/dmr3/analysis/outputs/m2-v1-seed0-20261012/plot_deps'
& $py -X utf8 -m analysis.hopper_demo_gif render --source analysis/outputs/NEW_HOPPER_GIF_INPUTS --output analysis/outputs/NEW_HOPPER_GIF
```

`inputs/inspection.json`记录run/源文件与数组hash、六段片段选择、源NPZ和帧号；输入manifest/evaluations/final_state哈希与原hopper专项已验收镜像一致。重建16条replay链的完整转移/动作/reset数一致，六段源图像hash和完整episode reward求和通过。

`final/manifest.json`记录版本、来源、拼接/帧率/量化与输出hash；全部150帧完整解码，解码像素等于预期量化后的拼图；量化前每格像素等于对应源帧，帧时长/循环设置与三行两列无间隙尺寸通过。接触图已目视核对；图像中保留真实倒伏等行为，不修改原画面。只验证任务示意素材及编码，不重新验收正式训练或宣称方法有效。

本轮只读访问sv2历史产物及sv3六条补seed产物的简要目录/终点元数据；用户确认示意范围后实际GIF仅使用上述sv2六段，不将sv3指标或审计状态并入本任务。没有训练/推理/环境/GPU作业、服务器写入/源码同步或W&B写入，未核当前GPU占用。本机提交分析脚本与报告/索引，不推送；既有RESEARCH/STATUS/proposal和根TeX未提交文件保留，新增GIF副本作为本机素材不加入Git。

交付可直接用于任务介绍。若需裁剪时长、尺寸或版式，首选继续现有05，无需新建chat或GPU；完成标准为六段真实来源不变、三行两列无缝隙、时长/循环/像素处理与用户要求一致，并保留新的hash/复现记录。不自动创建chat或发送消息。
