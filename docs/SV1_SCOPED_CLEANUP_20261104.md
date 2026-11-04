# 03 · sv1第二轮定范围清理（2026-11-04）

**后续更新（2026-11-04）**：用户随后明确要求“本机归档不要留”，两份本机完整归档以及未完成传输/测试包均已删除。原归档hash和验收记录仅作历史证据，不能再从这些本机路径恢复。当前可用状态、授权与删除核验见docs/LOCAL_ARCHIVE_REMOVAL_20261104.md。下文保留当时操作记录。

用户明确要求“优选候选、旧PD工程模型都删了”。承接01:37候选清单，已完成sv1 Denoising精确重复GIF、小缓存、两份运输包及三个旧PD工程根内90个模型的清理。运输包和模型先完整归档到本机并验证，再移除服务器原件；重复GIF通过硬链接合并存储，所有原路径仍可读。未扩展到Denoising模型/预算快照、PD正式数据或当前活动工程目录。

## 实际处理

| 类别 | 数量与处理 | 移除的常规文件分配块（十进制GB） |
| --- | --- | ---: |
| 旧PD工程模型 | 90个agent.pkl，完整归档后移除原件 | 36.931829760 |
| Denoising运输包 | 2份，完整归档后移除原件 | 5.620981760 |
| 精确重复GIF | 8528组中的8552份冗余存储替换为同组保留内容的硬链接 | 6.744915968 |
| 可再生缓存 | 1393个.pyc和19个下载缓存文件直接移除 | 0.032419840 |
| 合计 | 1504次unlink、8552次精确重复文件替换 | **49.330147328** |

旧模型的三个范围分别为/data/Policy_Discrepancy/runs/下的engineering-size50m-20261009-1393548（42个）、engineering-size50m-repair-20261009-98225e1（22个）、engineering-dmc8-remaining-20261024-sv1（26个）。只处理agent.pkl，原日志/指标/配置、replay、profile、故障诊断和其他证据保留。

运输包为/data/denoising/sv2_formal_runtime_transport_20261012/下的dreamer_dmc_runtime_v3.tar.gz和video_assets_v1.tar。两个完整payload的SHA256同时与原打包收据一致。本次通过独立本机备份保留可恢复来源，没有宣称验收了sv2部署状态。

Denoising缓存按预先核准的1412个具体文件处理，未递归删除项目或环境目录。base-py311、envs/dreamer、正式模型/评价快照/PD replay、Denoising预算快照及模型均不在操作清单。sv2/sv3未操作；没有启动训练、推理、GPU作业、核GPU占用或停止已有任务。

## 归档与恢复

本机归档：D:\program\PD\dmr3\analysis\outputs\sv1-cleanup-execute-20261104\models-and-transport.tar.zst。

- 压缩包长度：20671207007字节（20.671207GB）；92个payload原始长度合计42552539382字节。
- 整包SHA256：0736de5485babe290cfd22b4b7a7a9af5298be3910597d4e1f2c166ff8c2f481。
- 实际完整读回90个模型和2个运输包，每个成员的长度与SHA256均等于源流；另重读落盘整包，SHA256与接收流一致。未反序列化模型、加载环境或执行agent。
- 使用服务器既有zstd -3 --long=31 -T2；本机复用前次隔离zstandard==0.25.0，解码max_window_size=2147483648。未安装或升级服务器依赖。
- 源工程三个根与运输目录均留有ARCHIVED标记，引用本机归档、完整清单/哈希及服务器收据。需要原快照/复开验收时先按archive-manifest.json恢复对应原路径并复核SHA256；不据此声称环境/replay可无缝恢复。

本目录含RETENTION.md与.KEEP_ARCHIVE，必须保留已验证tar.zst及其清单、脚本和收据，不能作为可再生analysis/outputs缓存删除。此前另一批80模型的sv1-storage-cleanup-20261103/engineering-models.tar.zst仍须独立保留，本归档不替代它。前次本机未完成gzip临时文件未重试删除，也不在本轮范围。

## GIF语义与保留验收

- 去重前重新读取17080个候选GIF，逐一验证完整SHA256；同组均独立单硬链接，文件系统、权限、UID/GID相同，执行前又核xattrs相同。
- 保留每组一份完整内容，使用同目录临时硬链接和原子替换处理8552个重复副本，保留全部原文件路径。执行及独立后查确认8528组内容hash正确、17080条原路径可读、各组inode和硬链接数符合清单。其余GIF按保护基线保留。
- 同组历史GIF现在共用inode、mtime和可写内容，应作为不可变媒体使用；原每份文件的mtime_ns、inode、权限与路径映射在manifest中，恢复独立副本时可据此还原。W&B本地媒体引用路径仍存在；本次未核线上上传完整性，也未清空.wandb日志。
- 操作前后27687个保留常规文件元数据、其中12214个文件完整SHA256，以及1412个符号链接检查全部通过。大部分保留Denoising大二进制使用元数据保护，不能把本检查写成所有大模型内容独立复算。旧PD三个根的非模型常规文件全部做完整hash。
- 原deletion-receipt.json保留success=false：10056项操作、保留检查和GIF内容检查均已完成，最后的进程快照因PID1636570句柄暂不可读而失败。02:27:50复核该PID已不存在，无法追溯它当时的身份；没有改写失败收据或重跑任何删除。
- 随后独立只读postcheck.json为success=true、errors为空：沿用原进程检查条件，当前可读目标使用者和未分类同UID不可见进程均为空；重新核所有目标缺失、8528组内容/17080路径、保留文件及日志，全部通过。其他UID可见性限制仍保留，不能回推未知PID当时从未使用目标。
- 操作日志10056条由本机逐条核index/路径/类型/长度/分配块/hash/原与新inode、无重复；本机与服务器日志SHA256一致：c662ecf09d0389b402148e5ec68ed7f8846ab2919c261ba2d2829cc776b287cc。

## 时间与容量

- 清理开始：2026-11-04T02:25:22.628774+08:00；完成：2026-11-04T02:26:50.702359+08:00，均为北京时间。
- 操作前/data可用277.402665GB；操作后可用326.729470GB。
- 独立后查时间2026-11-04T02:29:22.086090+08:00，/data可用326.723957GB；Denoising目录162.713756GB，Policy_Discrepancy目录352.498356GB。

49.330GB来自精确目标原分配块，不把同期其他作业持续写入下的磁盘净增量当作删除量。空间只是当时快照，不是新实验启动授权或未来容量预约。

进程检查覆盖可读cmdline/cwd/FD，未见目标使用者。同UID工作进程要求可见；已识别的sshd/PAM系统会话按PID/argvhash记录，新增sshd会话还核其root所属sshd父进程。其他UID的部分/proc信息不可读，这一限制完整留档，不能声称获得root级全主机句柄证明。最初两次权限假设预检停止，均发生在任何目标数据修改之前；实际分类证据和失败输出保留。未提供或索取密码，未停止未知作业。

第一次发起执行时遇到Windows CreateProcess命令行长度限制（WinError206），SSH尚未启动、目标数据未修改。将传输编码改为标准库zlib压缩后执行，实际程序组合SHA256保持相同，未改变清单或任何数据验收条件；记录见execution-launch-error.json。

## 版本、证据与后续

任务起点HEAD为73f0f88b1a9a9f698446d7675dc57a4725e76b64，生成本报告时HEAD为1f1e40ceff75624faf1c9d1172f1cb549d1c7884。期间共享工作区有其他窗口的pendulum分析代码/测试和运行文档提交，均保留；本次维护没有执行这些分析测试，也不为其验收背书。本次没有训练代码/科学协议/模型算法变更，无新的数值或完整agent测试SHA；实际验收为上面的文件内容、保留基线、操作日志和容量核查。最终文档完整SHA及同期差异见本机DELIVERY.json。

实际执行维护脚本的组合SHA256为5432378c1d2d31df726b5d6330f5616bb1b4781b52409da02964faa472349e86；冻结清单SHA256为d39b92d00c2d7593b1961f3893f5579ce501c98ed4ebc6faa714354987711991。本机保留实际成功准备、归档、数据操作和独立后查的source与hash，以及全部请求、预检/启动失败输出、归档与操作证据。服务器canonical仓库后查为6e468ad4a8252f21401eab6a9761626b33659f0e，status=''；本轮未同步或修改服务器源码、推送GitHub、移动活动checkout。

服务器审计根：/data/Policy_Discrepancy/runs/scoped-storage-cleanup-20261104/，含authorization.json、manifest.json、archive-manifest.json、archive-verified.json、operations.jsonl、deletion-receipt.json、activity-checks.json、postcheck.json及进度。各源根留有归档/去重标记。

本机完整产物：analysis/outputs/sv1-cleanup-execute-20261104/；本报告、STATUS索引入Git，其余大产物不提交，保留所有并行窗口草稿/已有提交。实际通过项为本次清理、归档与保留核验，不是新真实任务或方法有效性验收。

本轮授权清理已完成。首选继续当前03，只读核对现有任务队列的容量增长与终点产物；不因空间增加另启实验或扩展删除。未来归档迁移须先验证目的副本并更新源标记。无需新建chat，不自动发送消息；可复制后续prompt在本机HANDOFF-current03.txt。
