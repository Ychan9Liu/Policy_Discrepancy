# 03 · sv1存储清理与Denoising候选（2026-11-03发起，11-04完成）

**后续更新（2026-11-04）**：用户随后明确要求“本机归档不要留”，两份本机完整归档以及未完成传输/测试包均已删除。原归档hash和验收记录仅作历史证据，不能再从这些本机路径恢复。当前可用状态、授权与删除核验见docs/LOCAL_ARCHIVE_REMOVAL_20261104.md。下文保留当时操作记录。

**后续更新（11月4日01:11）**：用户随后明确授权仅Denoising训练/评价replay直接删除，62目录26143个NPZ/295.429GB已清理、未归档，模型/预算快照/指标与配置审计保留；01:15 /data余340.990GB。当前事实与损失边界见docs/DENOISING_REPLAY_CLEANUP_20261104.md；下方保留本次前一阶段的执行与建议原貌，不继续作为replay仍在的依据。

用户授权执行前一份sv1清理方案；Denoising本轮仅核查可清理项。缓存于11月3日23:04清理，模型于11月4日00:13归档核验后移除，00:15:50完成只读复核，均为北京时间。无训练、推理或GPU作业，未核GPU占用。

## 已完成的清理

| 项目 | 实际处理 | 容量（十进制GB） |
| --- | --- | ---: |
| cache/pip、pip-cache下载缓存 | 删除778个常规文件，保留空目录 | 3.326599168 |
| 20条已验收工程integration的agent.pkl | 80个模型完整归档后移除sv1原件 | 41.029238784 |
| 合计移除的常规分配块 | 详细路径和收据可核对 | 44.355837952 |
| /data可用空间 | 1.236GB → 45.583GB，最终98% | — |
| /data/Policy_Discrepancy占用 | 370.156GB → 325.807GB，含新增归档收据 | — |

缓存范围仅/data/Policy_Discrepancy/cache/pip与pip-cache。Conda缓存、envs/dreamer及继承的base-py311保留；envs分配容量仍6,494,744,576字节。共享硬链接不能用目录du直接推断回收量。

工程模型范围严格为以下三个根的integration：engineering-dmc8-serial-20261022（16条）、engineering-dmc8-remaining-20261024-sv1-gpu2（2条）、engineering-dmc8-remaining-20261024-sv1-gpu7（2条）。只移除ckpt/eval_snapshots中的agent.pkl；replay、协议状态、所有原有配置/JSON/JSONL/日志/收据/审计/profile保留。其他初期预检查、失败、旧size50m报告及固定输入/渲染诊断目录保留。20条均为4098动作/517更新的工程fixture，不是正式百万步结果。

## 归档与验收证据

- 本机归档：D:/program/PD/dmr3/analysis/outputs/sv1-storage-cleanup-20261103/engineering-models.tar.zst，18,828,210,895字节。80模型的原始内容共41,029,094,544字节。
- 整包SHA256：a57e14473e791dea87fbc6b41fd3821f9ef3484aec7eb5d6cac92165c8c8ecec。
- 完整读回80个tar模型成员，每个长度/SHA256与源一致；再次计算落盘整包SHA256，与接收流一致。删除前重新读取服务器80个源文件比对内容hash、长度、mtime/inode及单硬链接，刷新项目进程和严格路径范围，全部通过后才逐文件移除。未反序列化模型或执行agent。
- 删除前后复核1190项保留文件/链接：常规文件使用完整SHA256，链接检查目标。19587个正式文件的路径/模式/长度/mtime/inode元数据集合相同；没有重新计算所有正式大文件的内容hash，不把元数据检查写成全体内容独立复算。
- 00:15复核：80个目标均不存在，20条目录均有engineering_model_archive.json；服务器最终收据success=true，源码checkout仍6e468ad4a8252f21401eab6a9761626b33659f0e且干净。
- 每条工程目录的归档标记保存本机归档路径、整包/文件hash与恢复限制。服务器小收据在/data/Policy_Discrepancy/runs/storage-cleanup-20261103/，有archive-manifest.json、cache-cleanup-summary.json、deletion-receipt.json。

这些工程目录的模型已移到本机；需要恢复归档后才能重做原快照/完成复开验收。没有放宽reader或伪装原目录仍在线完整，也不声称中断训练可以无缝恢复。正式模型/replay/评价快照及科学协议、alpha/c、梯度、seed、预算均未改动。

## 本机归档保留与限制

归档目录含RETENTION.md与.KEEP_ARCHIVE，不能按可再生analysis/outputs缓存整体删除。使用隔离的archive_deps/zstandard==0.25.0及max_window_size=2147483648读取长窗口zstd，再用tarfile流式读取；Windows自带tar默认解码内存上限不足，不据此认定归档损坏。完整80模型和整包均已实际读回验证。迁移本机归档前先校验新副本，再更新服务器及各运行目录引用。

首次gzip传输在未验证、未移除源模型时主动停止，随后改用服务器现有zstd；没有改变数据验收标准。本机解码器仅装入本轮忽略目录，未安装或升级服务器实验依赖。

本机未完成传输engineering-models.tar.gz.partial（10,938,744,832字节）的Remove-Item清理被自动审批拒绝，返回blocked by policy、未提供更细原因。命令未执行，文件保留；未换用其他删除API绕过拒绝。它不是有效归档，已验证tar.zst与sv1清理不受影响，具体见local-partial-policy-rejection.json。该临时文件的本机清理尚未完成。

## Denoising可清理候选：本轮未删除

/data/denoising约470.545GB。首选是精确重复GIF去重归档：21671份GIF共25.153GB，同大小候选17088份实际做SHA256，确认8528组8552份冗余副本，全部为独立单硬链接，可去重6.744915968GB。需保留每组原件和原路径映射，避免破坏本地W&B引用；线上完整上传尚未核实。这个6.74GB包含于25.15GB，不相加。

第二项是sv2_formal_runtime_transport_20261012的环境/视频迁移包约5.621GB；需核接收端验收和可恢复备份后移除sv1运输副本，保留打包/传输收据。源receipt不能代替接收成功证明。代码字节码/测试缓存约0.0304GB，打包pip缓存约0.002GB，收益较小。

全树replay约295.429GB、按名称分类的ckpt/budget等约116.731GB先保留。其中sparse_h4主体replay约289.005GB、task_nuisance_xyz_v5的budget_snapshots约47.771GB，属于研究数据；本轮未核该项目科研保留协议和逐条终点，不自行选择保留预算点或方法组。W&B全目录约28.851GB，也不能整体当缓存。Denoising原件均未删除或迁移。

详细路径、精确重复组、硬链接与证据限制见analysis/outputs/sv1-storage-cleanup-20261103/DENOISING_CANDIDATES.md及denoising-*.json。

## 版本、输出与下一步

本轮开始本机HEAD bf996bdf4cf4d733cd3e5b12b781f05c7b6e4554；维护核查基于sv1 checkout 6e468ad4a8252f21401eab6a9761626b33659f0e。原20条工程实测/产物SHA为18b85c4d85ff456e6822626f17cb14cc0f44d2b8和6e468ad4a8252f21401eab6a9761626b33659f0e，保留原验收范围。本次无训练代码/实验变更或新的数值/完整agent测试SHA，不把文档HEAD当旧模型训练版。维护脚本、SHA、命令和实际收据保存在本机忽略目录；最后提交仅本报告与STATUS索引，最终完整SHA及同期差异见DELIVERY.json。未GitHub推送或同步服务器源码。

sv1存储清理与Denoising只读盘点已完成；本机未完成gzip临时文件仍受自动审批拦截。剩余45.58GB仍低于原单条60GB启动容量下限，更不代表满足并行任务总容量要求；空间快照也不是启动授权。本轮没有核GPU或启动新实验。

首选下一项继续当前03，确认Denoising 6.74GB精确重复GIF的保留原件/路径映射方案后再执行获批范围；当前可解读/只读核对清单，真正删除/链接替换或迁移仍需用户明确授权。新科研选择交02和用户。本机临时文件保留可由用户另行处理，勿误删有效tar.zst。可复制prompt见本机HANDOFF-current03.txt。不自动建chat或发送消息。
