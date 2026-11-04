# 03 · 按用户要求移除本机归档（2026-11-04）

用户在上一轮sv1清理完成后明确要求“本机归档不要留”。本次按该最新授权删除两轮清理留下的本机完整归档、未完成gzip传输文件及微小压缩测试包，保留清单、日志、配置与审计证据。此前RETENTION.md/.KEEP_ARCHIVE中的payload保留要求在这个明确范围内被用户最新指令替代；未把payload移至其他目录或服务器继续保存。

## 已删除文件

| 本机路径（相对analysis/outputs/） | 类型 | 字节 |
| --- | --- | ---: |
| sv1-storage-cleanup-20261103/engineering-models.tar.zst | 80个旧工程模型的完整归档 | 18828210895 |
| sv1-cleanup-execute-20261104/models-and-transport.tar.zst | 90个旧工程模型及两运输包的完整归档 | 20671207007 |
| sv1-storage-cleanup-20261103/engineering-models.tar.gz.partial | 先前停止的未完成传输 | 10938744832 |
| sv1-storage-cleanup-20261103/zstd-long31-probe.tar.zst | 微小压缩测试包 | 113 |
| 合计 | 4个具体文件 | **50438162847（50.438162847GB）** |

以上均位于D:/program/PD/dmr3/analysis/outputs/下两个已明确的清理目录。操作使用同一PowerShell流程的Remove-Item -LiteralPath逐文件删除，未递归删除目录；逐个核绝对路径、精确父目录、workspace内全部祖先无reparse、文件长度/创建及修改时间，并以独占读取句柄核内容。两份完整归档当前SHA256仍与先前读回验收收据一致，另记录partial和probe的实际hash。

执行时段：2026-11-04T11:34:28.8352777+08:00 至 2026-11-04T11:35:09.9801381+08:00，北京时间。收据success=true、errors为空，四个目标均不存在。删除日志4条index/路径/长度/hash与清单逐项一致且无重复。删除阶段两个原清理目录内144个非目标文件的长度/mtime/完整SHA256全部保持；之后仅按下文更新指导文字与可用状态，新旧文本hash另行留档。

D盘可用空间由483.437662GB到533.875782GB；50.438GB是删除文件的逻辑容量，磁盘变化还包含元数据及其他工作写入。本轮没有核服务器磁盘/GPU现状、启动实验或修改训练代码/协议。

## 恢复限制与索引更新

这两份完整归档原来保存170个旧工程agent.pkl文件和两份运输包；相应sv1原件此前已移除。现在不能再通过本次归档恢复它们。没有核查其他独立副本，不声称全局再无副本；保留的指标、配置、hash和清单不能重建模型payload，也不能作为无缝恢复证明。未重新建立备份、重跑实验或替用户改变科研协议。

本机两个原产物目录新增ARCHIVE_REMOVED_20261104.json，RETENTION.md、旧REPORT/HANDOFF和.KEEP_ARCHIVE加明确失效说明；旧完整验证/删除JSON、日志及原始sha记录保持为历史事实。原指导文字另存original-guidance-before-update.json。此前对gzip.partial删除的自动审批拒绝记录也保留；本轮有新的明确授权，使用相同原生PowerShell删除方式成功执行，没有改用其他删除API绕过拒绝。

sv1只更新25个既有备份索引（20个旧工程run标记、3个后续工程根标记、运输包标记及最近清理completion）并在两处旧审计根新增失效通知。每个原索引的所有既有字段保留，附加local_archive_availability.available=false；原始字节/hash另存服务器审计。原失败收据和历史验证事实不改写。本轮服务器payload删除数为0，正式模型/replay、Denoising现存模型/预算快照、指标和活动任务均未处理。

## 验收记录和版本

本机证据：analysis/outputs/local-archive-removal-20261104/，含authorization.json、实际PowerShell脚本、local-manifest.json、local-deletion-receipt.json、4行deleted.jsonl、指导文字前后hash、服务器更新清单/收据、RESULTS.json与DELIVERY.json。服务器审计：/data/Policy_Discrepancy/runs/local-archive-removal-20261104/。

- 清单文件SHA256：d1970aea46c56fd857bbfa6187bbb9fe8f156b1c93d5106cee134cf00466faee。
- 删除日志SHA256：32f8241feee2cd4a5f183ad46841d678f60518bb2d034ca63f3d0ed1842e5a99。
- 实际PowerShell程序SHA256：77b2a44c0dd36b8778ede0334e10ea686b76e6f0bc49adcc2f7fef2ad8256056。
- 服务器索引更新程序SHA256：2a714c5fb41e3f4e8af3696b29f7edcded4a48a37e6057152dffc1bf62dcd7ec。

首次预检因PowerShell Measure-Object未读取OrderedDictionary的bytes值而在总量保护条件处停止，文件未删；改用精确Int64求和并保持50438162847字节的原定条件。另一个只读祖先探查使用了不兼容的Split-Path参数，已中止，改为有深度上限的明确父路径检查；原预检脚本与错误说明保留。最终删除和保留核验均通过，不把前期失败隐藏或用放宽目标条件代替修复。

起点HEAD为880045733b6a9658e1bab6dee3bccf32697360ee；文件核验/文档生成时HEAD为bce76441e9206846ce7b1a9449276641f972d9ca。本次无训练代码或实验变更，也无算法/完整agent测试SHA；实际验收为以上文件和索引检查。最终Git提交只包含当前清理报告、两份旧报告的后续说明和STATUS索引；完整最终SHA及差异见DELIVERY.json。只在sv1更新审计元数据，没有服务器源码同步、依赖变更、GitHub推送或sv2/sv3操作；并行窗口草稿与提交保留。

本次授权的本机归档移除与可用状态更新已完成。今后应先核查是否存在其他独立副本，再判断旧工程快照能否复核；不能继续引用已失效的本机tar路径，也不自动重建归档或开启新实验。
