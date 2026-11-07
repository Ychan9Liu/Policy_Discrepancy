# 03 · sv2 Denoising replay清理（2026-11-07）

用户在sv2 replay盘点后明确要求“可以，删除denosing的replay”。本次已完成该范围的直接删除与独立复核，沿用“不留本机归档”；保留模型、指标、配置和原审计记录。此次是sv2的独立副本，与11月4日sv1 `/data/denoising` 的清理分开记账。

## 范围与结果

唯一数据根为 `/data/Policy_Discrepancy/denoising_sv2_formal_20261012`。只删除事先列明的8条运行、16个 `log/replay` / `log/eval_replay` 目录中的常规NPZ，未递归删除项目或目录。完整路径清单见审计manifest。

| 项目 | 核验结果 |
| --- | --- |
| 删除窗口（北京时间） | 2026-11-07 19:41:27—19:42:34 |
| 独立后查 | 2026-11-07 19:42:49 |
| 删除文件 | 16,720个NPZ，无重复 |
| 文件分配块 | 297,201,504,256字节，297.201504256GB |
| 文件逻辑长度 | 297,167,223,392字节 |
| 删除前 `/data` 可用 | 116,701,437,952字节，116.701GB |
| 独立后查 `/data` 可用 | 413,899,563,008字节，413.900GB |
| Denoising保留文件 | 118个常规文件，全部完整SHA256及元数据前后相同 |
| PD保留replay检查 | 69,120个NPZ、513,341,927,424字节分配块，元数据前后相同 |
| 剩余目标NPZ / 删除说明 | 0 / 16个 |

GB均为十进制。空间净增受审计文件、目录块及其他作业同时写入影响，不要求与删除文件分配块逐字节相等。此次没有核查GPU资源，也没有启动、停止或排队训练。

八条来源为cheetah_run seed0的XYZ_H4baseline、XYZ_main_gray8、XY_H4；walker_walk seed0的同三组及seed1的XYZ_H4baseline、XYZ_main_gray8。均位于上述Denoising根，命名和完整路径以授权清单为准，不扩展到Policy_Discrepancy正式数据。

## 实际检查与边界

- 准备时重新盘点，不沿用当日00:25的容量作为当前值。每个目标核规范绝对路径、允许目录、类型、所有者、单硬链接、dev/inode、mtime、长度和分配块；16目录各抽一个NPZ只读ZIP成员表，包含action/reward/is_first/is_last。
- 本机选择器fixture验证1项允许和6项拒绝，包括非NPZ、嵌套路径、父目录文件、PD路径、硬链接及其他所有者。真实删除另作服务器规范路径/实时元数据检查，fixture不能代替实际后查。
- 删除前及每1000文件检查可读 `/proc` 的命令、cwd和文件句柄。23次准备/执行/后查均无目标占用或未分类的同UID不可见进程，没有保护条件失败或继续删除尝试。只对可确认身份及父进程的SSH/PAM会话作系统分类；其他UID句柄可能不可读，不声称拥有root级完整可见性。原始postcheck中的 `other_UID_visibility_limits` 计的是全部partial_visibility条目，含已分类同UID会话；不能直接解释为其他UID进程精确数量。
- 逐项再次核文件身份后unlink，逐条记录删除日志，周期性flush/fsync。执行收据success=true且errors为空；独立后查再次验证所有路径不存在、16目录没有NPZ、16个 `REPLAY_REMOVED_20261107.json` 指向正确清单。
- Denoising的8个agent.pkl、原日志、config、scores、metrics及检查点辅助状态等118个保留文件（逻辑长度3,710,731,919字节）全部完整内容hash通过；已有PD四组/扩展seed0/seed1/DtL的54个replay目录、69,120文件仅作元数据保持检查，不写成内容hash复算。
- 本机从服务器取回29个小审计文件（清单可较大，但不含轨迹数组），验证镜像hash、清单规范hash及16,720行日志与原文件逐行一致、无重复。未下载或另存replay payload。

被删replay仅有文件元数据和样本ZIP结构记录，没有全量内容hash或内容备份。若无其他独立副本，原始交互轨迹不能从保留模型/指标恢复；检查点中留存的辅助状态也不能使空replay目录恢复可用。不能将这次清理后续跑称为无缝恢复。原始指标文件保留，本次没有重新计算指标或检验方法效果。

## 证据、版本与复核命令

服务器审计根：`/data/Policy_Discrepancy/runs/denoising-replay-cleanup-20261107-sv2/`。

本机：`analysis/outputs/sv2-denoising-replay-cleanup-20261107/`。包含授权、实际执行源码、prepare/execute/verify请求和收据、清单、逐文件删除日志、过程占用检查、镜像hash、RESULTS.json与DELIVERY.json。不提交大产物，也没有模型/轨迹归档。

| 版本依据 | SHA256 / Git SHA |
| --- | --- |
| 实际远程维护源码字节SHA256 | `149e08e2efc4829b1318c39860801d14bb38e27f8ce2301180d7a2424ce903bd` |
| 规范manifest SHA256 | `7656193cf70d57396ec158008b248ad2d0e9a3f351315a4af1de29950e8fed23` |
| 逐文件删除日志SHA256 | `5ab36816f07010f43ec4e4664e5510e87953fe3057040ecdcc4b0e0d9c4beeb5` |
| 操作/本机文件验收时Git HEAD | `83bb51cb74b81f46349d780a75d498081a17bdf2` |

实际操作由上述完整源码字节hash标识，不把工作区Git SHA当成未入库维护脚本的代码版本。服务器未修改项目源码，未同步checkout、依赖或科学配置；无算法/完整agent测试、GPU执行和新实验SHA。本轮Git仅提交本文及STATUS维护条目，最终完整SHA及相对验收版本的差异留在DELIVERY.json；共享工作区其他草稿保留，未GitHub推送。sv1/sv3未连接或同步。

以下命令在仓库根运行过，本机Python为 `analysis/outputs/dmc16-preparation-20261022/venv/Scripts/python.exe`：

```text
python analysis/outputs/sv2-denoising-replay-cleanup-20261107/check_selector.py
python analysis/outputs/sv2-denoising-replay-cleanup-20261107/remote_call.py prepare prepare
python analysis/outputs/sv2-denoising-replay-cleanup-20261107/fetch_audit.py prepared
python analysis/outputs/sv2-denoising-replay-cleanup-20261107/remote_call.py execute execute
python analysis/outputs/sv2-denoising-replay-cleanup-20261107/remote_call.py verify verify
python analysis/outputs/sv2-denoising-replay-cleanup-20261107/fetch_audit.py final
python analysis/outputs/sv2-denoising-replay-cleanup-20261107/validate_results.py
```

这些是已执行记录；prepare/execute含排他保护，不用于再次删除。`validate_results.py` 可对现有本机审计镜像复核。

本次清理已结束，无待删余项。首选由当前合并03继续核查待运行批次的实时容量与队列条件；413.90GB仍低于先前八卡批次480GB初始门槛，本次不修改该门槛、不启动训练。其他类别清理或新的科学验证按既有授权范围处理，不自动创建chat或发送消息。
