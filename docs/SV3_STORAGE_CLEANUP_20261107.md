# 03 · sv3四项存储清理与校准文件去重（2026-11-07）

用户在本轮可清理项报告后要求“全部清理”。执行范围限定为报告中的四项：重复incoming、PD pip-cache、旧Denoising派生分支replay，以及保留两处原路径/内容的校准数组存储去重。四项已完成并独立复核，没有扩大到正式PD数据或其他失败运行；不留payload归档。

## 实际结果

| 操作 | 文件/对数 | 移除或共享原分配块GB |
| --- | ---: | ---: |
| 删除旧Denoising训练/评价replay | 8960 | 75.855196 |
| 删除重复incoming | 1280 | 11.424129 |
| 删除pip-cache内安装缓存 | 81 | 0.103137 |
| 校准数组写时复制去重，保留两份原文件 | 128对 | 19.095093 |
| **合计** | **10321文件删除、128对去重** | **106.477556** |

操作窗口（北京时间）：2026-11-07T20:34:58.241006+08:00至2026-11-07T20:35:39.631460+08:00；独立后查：2026-11-07T20:38:18.299983+08:00。
删除前/data可用340.685197GB，独立后查可用447.157506GB，净增106.472309GB。
GB为十进制。文件分配块/去重前目标独占extent与整盘净变化是不同口径；其他作业、审计文件及文件系统元数据会影响df差值。共享块回收通过FIEMAP和df核对。

## 准确范围与保留

- `/data/Policy_Discrepancy/runs/dmc_policykl_half_a20_20261010/logs/`下8个replay/eval_replay目录，仅清单中的8960个NPZ。来源provenance为sv1旧Denoising分支，4条旧metrics终点step=1,000,000；不把这些日志按当前v1重新验收。四个agent.pkl、全部原metrics/scores/config/日志/来源与源码、旧runtime保持。轨迹未创建本次备份，不能从模型/指标重建或声称无缝恢复。
- `/data/Policy_Discrepancy/runs/dl-nm1-replay-stage-20261015-e70419b/incoming/`中的1280个NPZ删除。同根replay中的完整同字节副本及stage/input/worker收据保持；实际stage认证在删除后及独立后查均通过。
- `/data/Policy_Discrepancy/pip-cache/`仅删除81个缓存常规文件，空目录可留。Python环境、base-py311及服务器源码checkout没有修改。
- `dl-nm1-cal-20261015-47db626/attempt02/batch_*/evidence.npz`与`dl-nm1-cal-20261015-abefbb3/attempt03/batch_*/evidence.npz`的128对同字节数据按下节方式共享块，两份失败目录、文件、路径和记录都保留。
- 正式PD模型/评价快照/replay、其他DL失败/研究产物、其他用户目录均未列入删除或去重范围。所有原运行记录未覆写；8个退役replay目录及incoming新增共9个STORAGE_CLEANUP_20261107.json，指向独立审计。

## 校准去重语义与验收

sv3 `/data`为XFS、reflink=1。采用内核FIDEDUPERANGE就地共享相同数据块，两个原路径继续是各自独立inode、nlink=1的常规文件；没有改成符号链接或硬链接，没有删除任一失败尝试的文件。

- 准备时128对完整SHA256逐一相同，stat前后稳定；原目标extent全部未带SHARED标记。保留原始extent映射，操作记录逐段保存请求长度、实际去重长度和状态。
- 独立小文件fixture先验证：非块对齐EOF同内容去重通过，不同内容被拒绝；原inode/长度/权限/所有者/mtime/link count保持。去重后写入其中一份，另一份hash不变，证明该文件系统写时复制隔离；fixture文件随后移除，只保留收据。
- 原数组以最多16MiB段执行内核相同内容检查；每段status=0且有正的实际去重长度，覆盖全部19094635776逻辑字节。未放宽错误或内容一致要求。
- 独立后查重新完整计算256个原文件SHA256，全部与准备清单一致；原路径、inode、大小、mode、uid/gid、mtime和nlink=1保持。两侧FIEMAP逐逻辑区间映射到相同物理块且带SHARED标记，原目标独占extent共19095093248字节已转为共享。
- 文件的物理extent及ctime允许因存储操作改变；它们不是原失败计算结果。没有重算模型、梯度、随机流或科学指标。普通du会分别计算两份文件的逻辑分配块，可能仍显示原大小；实际块共享证据为FIEMAP与df，而非把两个目录相加。

## 删除与保留检查

准备清单锁定10321个具体路径、种类、uid、单硬链接、inode/dev、mtime、大小和分配块；incoming完整hash、旧replay/cache只保存元数据。执行前及逐文件再次检查，严格限定允许目录，逐项unlink并落审计日志，未递归删除整个项目。

本机路径fixture接受4项、拒绝8项（正式数据、模型、环境、越界层级、错误所有者/硬链接等）。它只验证选择器，不能替代实际清理。真实操作收据success=true、errors为空；独立后查核删除清单全部缺失、目录无残留NPZ、cache没有常规文件、9标记正确，日志连续且没有重复。

保留基线共78461个常规文件和91个链接，独立后查元数据一致；其中41981个保留常规文件重算完整hash，加上256个校准原路径，共42237个路径有完整内容hash保持证据。包含旧Denoising保留模型、源码/runtime/原日志，stage保留replay以及两次失败校准非数组记录。其余runs仅元数据核对，含20880个formal下pkl/npz文件；不写成正式数据全量hash复算。

准备/执行/后查检查可读/proc命令、cwd和FD，目标引用及未分类同UID可见性缺口均要求为空。仅已确认身份和父进程的SSH/PAM会话分类；其他UID句柄仍有权限限制，不声称root级全机占用证明。未停止任何已有作业，没有GPU资源查询或GPU执行。删除前后原stage读取器认证均通过。

## 证据、命令与版本

服务器审计根：`/data/Policy_Discrepancy/runs/sv3-storage-cleanup-20261107/`。本机：`analysis/outputs/sv3-storage-cleanup-20261107/`。保存authorization、实际源码、fixture、完整manifest、operations.jsonl、活动检查、execute-receipt、postcheck、RESULTS、镜像hash与DELIVERY；元数据不是模型/轨迹归档。

| 依据 | SHA256/Git SHA |
| --- | --- |
| 实际维护源码字节 | `1183bceaceca1e76e8129130c65712d27f8230f631ea070d27e404dfdd29b9fa` |
| 规范manifest | `5def17fb4ad977055566951ec11878d0a85211c3c7f537dccfe2aec49a74f46c` |
| 原操作日志 | `7a3b496e3206d9fe752a94866fec39c3b9dc11a5c5d696236a28cf47a4387bf3` |
| 操作开始时本机HEAD | `f7e56a4ad2a8e8e53496f8744bb1d614d2241674` |
| 本机实际结果核验HEAD | `f7e56a4ad2a8e8e53496f8744bb1d614d2241674` |

维护脚本位于ignored输出目录，实际执行版本由字节SHA256标识；上述Git SHA记录现场仓库版本。四阶段实际命令在仓库根、现有本机Python执行：

```text
python analysis/outputs/sv3-storage-cleanup-20261107/build.py
python analysis/outputs/sv3-storage-cleanup-20261107/check_selectors.py
python analysis/outputs/sv3-storage-cleanup-20261107/remote_call.py fixture fixture
python analysis/outputs/sv3-storage-cleanup-20261107/remote_call.py prepare prepare
python analysis/outputs/sv3-storage-cleanup-20261107/fetch_audit.py prepared
python analysis/outputs/sv3-storage-cleanup-20261107/remote_call.py execute execute
python analysis/outputs/sv3-storage-cleanup-20261107/remote_call.py verify verify
python analysis/outputs/sv3-storage-cleanup-20261107/fetch_audit.py final
python analysis/outputs/sv3-storage-cleanup-20261107/validate_results.py
```

这些是实际执行记录；fixture/prepare/execute含排他保护，不用于重新清理。validate_results可重读现有本机镜像。此前只读review中的首次hash脚本语法错误及修正版结果仍保留；本次执行没有靠改验收条件处理异常。

本轮仅向sv3执行授权维护并保存小审计记录，未接触sv1/sv2，未同步服务器repo或GitHub。Git仅提交本文和STATUS本轮维护条目，原共享工作区草稿保留；最终完整SHA、与核验版本的差异见本机DELIVERY。没有训练/算法/科学协议变更或新的完整agent测试SHA，存储fixture不构成方法效果结论。

四项清理均完成，无剩余授权项。首选继续当前合并03处理既有运行监测或后续存储需求；这次释放空间不自动扩展其他目录清理/新训练授权，不创建chat或发送消息。
