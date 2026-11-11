# sv2 已完成运行 replay 清理（2026-11-11）

在sv2磁盘核查后，用户要求“已完成任务，replay全部删除”。此次仅处理sv2已确认完成的运行，沿用直接删除、不留本机或服务器payload归档的偏好；模型、评价快照、指标、配置与原审计记录保留。93条运行的81,248个replay NPZ全部删除并独立复核，原分配块共663,608,717,312字节（663.608717312GB）。未扩展到sv1/sv3，也未删除中断或完成证据不足的运行。

## 结果与当前容量

| 项目 | 结果 |
| --- | --- |
| 删除开始（北京时间） | 2026-11-11T17:56:53.567738+08:00 |
| 删除结束 | 2026-11-11T17:59:59.614061+08:00 |
| 独立文件后查 | 2026-11-11T18:00:14.102424+08:00 |
| 完成运行 | 93条：63条百万动作运行，30条工程短跑 |
| 删除文件 / 逻辑长度 | 81,248个NPZ / 663,442,471,766字节 |
| 删除前 `/data` 可用 | 143,620,337,664字节 |
| 独立后查 `/data` 可用 | 807,211,388,928字节 |
| 最新容量采样 | 2026-11-11T18:00:21.784598+08:00 |
| 最新 `/data` 总量 / 已用 / 可用 | 2197.950 / 1390.738 / 807.211GB |
| 最新 `/data` 使用率 | 64% |
| 已清理目录 / 不可用标记 | 93 / 93 |

GB为十进制。文件原分配块与磁盘可用空间净增分别记录，审计文件、目录块及其他作业的同期写入会造成差异。精确df/du和inode输出见本机 `final-disk.json`。Denoising的sv2 replay已在11月7日清理，此次未重复计账。

## 完成判据与清理清单

只选择自身 `final_state.json` 与 `protocol_manifest.json` 可核对的运行：实际动作数达到自身预算、更新数大于零、评价动作网格覆盖起点和预算终点、评价记录数与终态相符，每点episode数量符合自身配置；存在episode完成字段时同时核其为true。原终态、配置、评价文件SHA256留存。百万动作运行63条包括旧正式PD运行、DL pilot和八条dynamic baseline；30条工程运行预算分别为12动作×12条、2048动作×4条、4098动作×14条，不称为正式实验验收。

该核查用于判断存储清理资格。部分旧索引仍写着待整组审计，本次没有篡改历史索引或补发科学验收；也没有重算结果、启动训练、评估算法有效性。

以下仅列出本次实际删除的运行数量，完整逐路径清单见manifest；不代表同名批次下所有运行均被清理。

| runs下批次目录 | 运行数 | NPZ数 | 原分配块GB |
| --- | ---: | ---: | ---: |
| `engineering-return-cuda-20261009-27261bf` | 1 | 16 | 0.018448 |
| `formal-m2-v1-pd1-seed0-20261009` | 4 | 5120 | 42.013819 |
| `formal-m2-v1-pd1-dt-seed0-20261009` | 4 | 5120 | 41.733366 |
| `formal-m2-v1-pd1-shuffle-seed0-20261010` | 4 | 5120 | 41.369305 |
| `formal-m2-v1-pd1-constant-seed0-20261010` | 4 | 5120 | 41.576038 |
| `dl-pilot-r3-20261013-13ec72a` | 1 | 1280 | 11.424129 |
| `engineering-dmc8-remaining-20261024-sv2` | 3 | 24 | 0.000098 |
| `formal-m2-dmc16-remaining8-serial-20261024-sv2` | 7 | 8960 | 66.082058 |
| `engineering-missing-dt-20261030-sv2` | 3 | 24 | 0.000098 |
| `engineering-missing-dt-20261030-partial1357-sv2` | 3 | 24 | 0.000098 |
| `formal-m2-dmc16-missing-dt-20261030-partial1357-sv2` | 4 | 5120 | 33.717932 |
| `formal-m2-dmc16-missing-dt-20261030-remaining3-sv2` | 1 | 1280 | 8.123552 |
| `engineering-vision20-seed1-20261103-sv2` | 3 | 24 | 0.000098 |
| `formal-m2-vision20-seed1-20261103-sv2` | 14 | 17920 | 120.100676 |
| `engineering-vision20-seed1-remaining-20261105-sv2` | 4 | 128 | 0.154952 |
| `formal-m2-vision20-seed1-remaining-20261105-sv2` | 7 | 8960 | 64.912830 |
| `engineering-vision20-walker-dt-20261105-sv2` | 1 | 32 | 0.043917 |
| `formal-m2-vision20-walker-dt-20261105-sv2` | 1 | 1280 | 10.622841 |
| `engineering-vision20-hopper-stand-dt-20261105-sv2` | 1 | 32 | 0.036868 |
| `formal-m2-vision20-hopper-stand-dt-20261105-sv2` | 1 | 1280 | 9.288532 |
| `engineering-dtl-three-20261105-f1f33bc` | 3 | 48 | 0.066392 |
| `dtl-three-clean-seed0-20261105-f1f33bc` | 3 | 3840 | 33.800978 |
| `engineering-dcs-dynamic-preparation-20261106` | 8 | 256 | 0.536572 |
| `dcs-dynamic-vision20-seed0-20261106-5245096` | 8 | 10240 | 137.985118 |

## 保留项、验证与限制

- 精确删除前核对规范绝对路径、允许目录、NPZ类型、当前所有者、单硬链接、dev/inode、mtime、大小和分配块；拒绝符号链接、越界、嵌套目录和变更文件。93目录各抽一个NPZ只读ZIP成员表，核action/reward/is_first/is_last，无payload下载。选择器7项fixture通过，真实操作另作实时核验，fixture不是服务器后查的替代。
- 准备、执行、每1000文件及独立后查共88次可读进程命令/cwd/FD检查，没有目标占用。暂态同UID不可见进程的实际退出复查次数为0；只有原PID确已消失并再次扫描通过才继续。其他UID的隐藏句柄不可完全检查，不声称root级完整可见性。未停止或干扰其他作业。
- 删除采用逐文件unlink及同序日志、周期flush/fsync，未递归删除项目或目录。执行收据success=true、errors为空；独立后查确认81,248路径全部不存在、日志同序且无重复、93目录无NPZ并有 `REPLAY_REMOVED_20261111.json`。
- 9,022个保留常规文件元数据保持，其中7,001个日志/配置/指标等完整SHA256复核通过；1,805个PKL模型/评价快照只核元数据，未全量重读其内容。248个符号链接目标、inode、mtime保持。未把模型元数据核对写成完整内容hash验证。
- 12个未纳入目录的7,940个NPZ共105,842,798,592字节分配块均元数据保持：八条磁盘保护中断的dynamic Dt，另有一个旧中断工程运行及三个缺少完成证据的smoke运行。这些不属于此次“已完成任务”删除范围；不能声称sv2全部replay为零。
- 原模型、评价快照、指标、配置、检查点及审计仍在。被删轨迹没有备份，模型和日志不能重建原轨迹；DL pilot原N运行replay也已退役，后续依赖该原路径的读取不再可用。任何空replay目录均不能凭保留检查点称为无缝恢复，旧运行收据不证明payload仍存在。

## 证据与版本

服务器独立审计目录：`/data/Policy_Discrepancy/runs/sv2-completed-replay-cleanup-20261111/`。

本机审计镜像：`analysis/outputs/sv2-completed-replay-cleanup-20261111/`，包含inventory、授权、实际维护源码、准备清单、逐文件删除日志、占用检查、独立后查、镜像hash、RESULTS.json、final-disk.json和DELIVERY.json。仅保存小审计数据；无replay/model归档，且未提交大产物。

| 版本依据 | 完整值 |
| --- | --- |
| 实际远程维护源码SHA256 | `570c13797207c49fb3e9c0ecaa620263cc1ebec71661fde298d772f87065504f` |
| manifest规范SHA256 | `749ff62bed04ce9cc9ad5ddf1f175738eb94219b1cdc41a54c662be191206a9b` |
| 删除日志SHA256 | `dbb0fa840256311e0e59d9383c8ee719f54e168ecfbd828abfc7df9463110f79` |
| 文件验收时本机Git HEAD | `0c0555450dbc2c93182cf3613a84cebaa328f048` |

维护源码由以上字节hash标识，不把未入库脚本虚称为Git代码版本。本轮无项目代码、科学协议、依赖、算法/完整agent测试或实验变更，无GPU核查/运行。只在sv2执行维护并写入独立审计目录和93个删除标记；未同步服务器repo checkout，也未推送GitHub。最终本机Git仅提交本文和STATUS新增维护记录；最终完整SHA、相对文件验收HEAD的差异与共享工作区保留检查见DELIVERY.json。其他未提交草稿未纳入本次提交。

以下命令均已执行。Python为本机 `analysis/outputs/dmc16-preparation-20261022/venv/Scripts/python.exe`；prepare/execute具有排他保护，不应重新执行删除。

```text
python analysis/outputs/sv2-completed-replay-cleanup-20261111/test_selector.py
python analysis/outputs/sv2-completed-replay-cleanup-20261111/remote_call.py prepare prepare
python analysis/outputs/sv2-completed-replay-cleanup-20261111/fetch_audit.py prepared
python analysis/outputs/sv2-completed-replay-cleanup-20261111/remote_call.py execute execute
python analysis/outputs/sv2-completed-replay-cleanup-20261111/remote_call.py verify verify
python analysis/outputs/sv2-completed-replay-cleanup-20261111/fetch_audit.py final
python analysis/outputs/sv2-completed-replay-cleanup-20261111/validate_results.py
python analysis/outputs/sv2-completed-replay-cleanup-20261111/final_disk.py
```

本次已确认完成运行的replay清理已结束，无待执行删除。首选继续当前磁盘管理chat按需要只读复查容量；剩余中断replay是否退役由用户另行明确，不能因为本次空间释放就自动删除或启动实验。实验启动/恢复问题由用户在现有工程实验窗口处理，不自动创建chat或发送消息。

可复制的后续核查prompt：

```text
继续当前磁盘管理chat。项目D:\program\PD\dmr3；先核对当前Git和共享工作区，不回退或丢弃修改。必读AGENTS.md、docs/STATUS.md、docs/SV2_COMPLETED_REPLAY_CLEANUP_20261111.md及analysis/outputs/sv2-completed-replay-cleanup-20261111/DELIVERY.json（记录最终交付SHA）；文件验收HEAD为0c0555450dbc2c93182cf3613a84cebaa328f048。
sv2已确认完成的93条运行replay共81,248文件/663.608717312GB已删除，无payload归档；模型、评价快照、日志、配置保留，12个中断或完成证据不足的replay目录仍保留。仅只读核查最新磁盘容量及剩余占用，不自动扩大删除、停止作业、启动训练或改变科学协议。需要退役剩余数据时先列具体目录、完成/占用证据和影响，等用户明确范围。
核查输出放新的analysis/outputs目录；保留服务器/data/Policy_Discrepancy/runs/sv2-completed-replay-cleanup-20261111原证据。完成时报告检查时间、容量、已验证范围、限制、实际版本及下一步，不将旧快照当实时状态，不自动建chat或发送消息。
```
