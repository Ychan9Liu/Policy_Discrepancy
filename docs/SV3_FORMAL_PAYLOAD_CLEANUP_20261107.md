# 03 · sv3旧正式运行 replay 与历史评价模型清理（2026-11-07）

用户在16条旧正式运行的内容盘点后要求“清理这两项”：正式训练replay与历史评价模型。已完成精确清单删除及独立后查。保留16份最终ckpt模型、原始指标/配置/来源/审计等全部非目标文件；无payload归档。此授权不扩展到正在运行的DCS批次或其他服务器。

## 交付与磁盘结果

| 删除内容 | 文件数 | 原分配块GB |
| --- | ---: | ---: |
| 16条旧正式运行的训练replay NPZ | 20480 | 142.936666112 |
| 0至100万、每5万步的历史评价agent.pkl | 336 | 172.328509440 |
| 合计 | 20816 | 315.265175552 |

删除时间（北京时间）：2026-11-07T21:54:25.533084+08:00至2026-11-07T21:57:14.914465+08:00。独立后查：2026-11-07T21:57:41.081368+08:00。
/data可用由删除前407.664894GB变为后查719.476101GB，净增311.811207GB。GB为十进制。315.265176GB为已删除文件的原分配块总数；整盘净变化还受同时运行的DCS、其他作业及文件系统/审计元数据影响，不能将二者混称。

本次另应用户状态询问只读sv1：2026-11-07 21:47:55北京时间，/data使用率98%、可用47.372546GB；21:58:08复查可用47.372349GB，仍为98%。PD目录du为601.310941GB，其中runs为594.606318GB，两次相同。sv1没有执行清理、启动/停止作业或GPU核查。这是带时间的磁盘快照，不代表当前GPU空闲或实验状态。

## 精确范围

每条运行只删除即时子目录replay中的1280个NPZ和eval_snapshots中21个指定网格步的agent.pkl。不递归删除运行目录，也不按formal前缀批量匹配。完整原路径如下：

- `/data/Policy_Discrepancy/runs/formal-m2-dmc16-missing-dt-20261030-remaining3-sv3/dmc_cartpole_balance/dt-attempt02`
- `/data/Policy_Discrepancy/runs/formal-m2-dmc16-missing-dt-20261030-remaining3-sv3/dmc_cartpole_swingup/dt-attempt02`
- `/data/Policy_Discrepancy/runs/formal-m2-dmc16-missing-dt-20261030-sv3/dmc_pendulum_swingup/dt-attempt02`
- `/data/Policy_Discrepancy/runs/formal-m2-dmc16-missing-dt-20261030-sv3/dmc_reacher_easy/dt-attempt02`
- `/data/Policy_Discrepancy/runs/formal-m2-dmc16-missing-dt-20261030-sv3/dmc_walker_run/dt-attempt02`
- `/data/Policy_Discrepancy/runs/formal-m2-dmc16-missing-dt-20261030-sv3/dmc_walker_walk/dt-attempt02`
- `/data/Policy_Discrepancy/runs/formal-m2-hopper-seeds123-baseline-dt-20261021/seed1/dt-attempt01`
- `/data/Policy_Discrepancy/runs/formal-m2-hopper-seeds123-baseline-dt-20261021/seed1/logging-attempt01`
- `/data/Policy_Discrepancy/runs/formal-m2-hopper-seeds123-baseline-dt-20261021/seed2/dt-attempt01`
- `/data/Policy_Discrepancy/runs/formal-m2-hopper-seeds123-baseline-dt-20261021/seed2/logging-attempt01`
- `/data/Policy_Discrepancy/runs/formal-m2-hopper-seeds123-baseline-dt-20261021/seed3/dt-attempt01`
- `/data/Policy_Discrepancy/runs/formal-m2-hopper-seeds123-baseline-dt-20261021/seed3/logging-attempt01`
- `/data/Policy_Discrepancy/runs/formal-m2-pendulum-seeds23-20261103-sv3/seed2/logging-attempt01`
- `/data/Policy_Discrepancy/runs/formal-m2-pendulum-seeds23-20261103-sv3/seed3/logging-attempt01`
- `/data/Policy_Discrepancy/runs/formal-m2-pendulum-seeds23-parallel-dt-20261104-sv3/seed2/dt-attempt01`
- `/data/Policy_Discrepancy/runs/formal-m2-pendulum-seeds23-parallel-dt-20261104-sv3/seed3/dt-attempt01`

删除前各final_state仍记录100万训练动作、21点评价，与20:42原盘点对象完全一致。此次没有重新计算return、重验原210个episode或重新判定科研结果。没有新增/重跑实验，也没有算法、协议、环境、GPU配置或随机流变更。

## 验收与保留证据

- 20816个目标与既有盘点清单逐路径匹配；删除前及逐项删除时核对规范绝对路径、范围、普通文件、uid、nlink=1、inode/dev、mode、mtime、大小和分配块。逐文件unlink后落日志，未使用广泛递归删除。
- 本机选择器fixture共11项通过（允许3、拒绝8），拒绝最终ckpt、错误快照步、嵌套replay、越界、新DCS路径、其他uid及硬链接。fixture只验证选择器，真实删除由服务器清单和独立后查验收。
- 16份最终ckpt模型与20:42完整SHA256基线一致。删除前、独立后查均对这16条运行的全部1449个保留常规文件完整计算SHA256并核元数据；全部一致，另64个原链接路径/目标/inode/mtime一致。保留文件总数包含16份最终模型，其余1433份为原日志、指标、配置、收据与辅助状态。
- 删除收据success=true、errors为空。独立后查20816条路径全部缺失、目标glob为空；20816条逐项日志与manifest严格同序、唯一、无重复；16个新不可用标记齐全。执行和后查的journal SHA256一致。
- 执行前、每1000文件及结束/后查检查cmdline/cwd/可见FD，没有目标引用。若同UID暂时不可读，只接受原PID已实际退出且新一轮全扫描通过；不跳过仍存活的未知进程。其他UID不可见FD限制仍存在，不能宣称root级全机证明。新DCS运行明确不在删除或静态保护清单内，未干预其进程；不以正在增长的全runs目录作不变断言。
- 删除目标主要以原元数据锁定，没有对全部315GB内容重新做hash；原盘点对16个最终模型及16个最后评价模型做过完整hash。保留1449文件则有本次前后完整内容hash。区分两类证据，不夸大目标内容检验范围。

## 恢复与分析限制

历史评价模型包含0步及100万步快照，均已退役。此前16对最终ckpt与最后评价模型的整文件hash为0/16相同，不能把最终ckpt当成所删评价快照的逐位副本。原评价结果/快照元数据及收据保持，但对应agent.pkl已不可用；本次在各运行根新增FORMAL_REPLAY_AND_EVAL_MODELS_REMOVED_20261107.json，明确replay/历史评价模型不可用并指向清理清单。

没有归档训练轨迹或模型payload；不能据保留日志重建轨迹、恢复历史固定快照评价或声称无缝续跑。保留的16份最终ckpt可供其本身适用的后续用途，但本轮未重新加载/运行它们。旧失败/DL校准材料、已共享校准块、旧Denoising保留模型/runtime、运行环境及base-py311均不在本次范围。

## 版本、复核命令与产物

- 文件实测/本机核验时仓库HEAD：`65d5db8984845a692c870591a3725109537843aa`。无新训练代码测试SHA；实际维护程序存独立audit目录，准确源码SHA256：`2cccba443aa9628d5b814a23b9b56b74f743b93bd93c3e87ad83d1a9f6cf8f29`。最终提交只记录本报告和STATUS，完整最终SHA/测试版本差异在同目录DELIVERY.json，不把文档提交称为训练实测。
- manifest canonical SHA256：`b3d6aa91ab42bca452fec87ab10fa09479b2af3e2a629f2f0388c954fe2f5c43`。
- 删除journal SHA256：`f26b1ed665ff6c305a0cfc19919904e5b56ce769ffa0f452b9e2ec0b56cfd513`。
- 本机：`analysis/outputs/sv3-formal-payload-cleanup-20261107/`；服务器：`/data/Policy_Discrepancy/runs/sv3-formal-payload-cleanup-20261107/`。包含authorization、actual_remote_source.py、manifest、活动检查、deleted.jsonl、execute-receipt、postcheck、RESULTS和小审计文件镜像/hash。只镜像审计元数据，没有模型/NPZ归档。
- 实际顺序：本机python执行test_selector.py；remote_call.py prepare prepare；fetch_audit.py prepared核对远端清单；remote_call.py execute execute；remote_call.py verify verify；fetch_audit.py final；validate_results.py。每次SSH调用使用同一源码hash与清单hash。execute为已完成的破坏性动作，不能作为复算命令重跑；只读复核收据应使用validate_results.py。
- 同步范围：仅sv3目标文件清理、新不可用说明及审计材料；sv1只读磁盘检查；本机文档提交。没有服务器源码checkout移动/Git同步、GitHub推送、sv2连接或任何训练启动。共享工作区已有AGENTS/CHATS/RESEARCH/STATUS及论文草稿保留。

## 后续

本次两项授权全部完成，无需跨03/04交接。首选继续现有03只读检查sv1容量及需要保留的产物清单；没有将本次sv3删除授权扩大为sv1正式数据删除。正在运行的DCS监测仍按其原授权、索引及保护门槛执行，本轮容量检查不新增训练授权。新的科学复验由02与用户确认；本次文件维护不构成方法有效性结论，不自动创建chat或发送消息。

可复制给现有03的下一步prompt：

> 项目D:\program\PD\dmr3。以本报告提交（完整SHA见analysis/outputs/sv3-formal-payload-cleanup-20261107/DELIVERY.json）为交接版本，先核实际HEAD和工作区并保留已有修改。必读AGENTS.md、docs/STATUS.md、本报告及docs/DCS_DYNAMIC_SV3_SIX_20261107.md。sv3指定16条旧正式运行的replay和全部336历史评价agent.pkl已删除；16最终ckpt与日志/指标/配置/审计保持，无payload归档，不可声称历史快照可补评或无缝恢复。当前任务先只读刷新sv1磁盘和PD产物分类，按保留证据能力列出具体候选、占用与依赖；不删除其他正式数据、不启动/停止作业、不改变协议或门槛。新的实际产物放analysis/outputs/独立目录；需变更保留能力时由用户明确范围。验收须有带时间磁盘数据、来源清单及不可恢复限制，报告完整SHA和同步范围。继续现有03，不自动创建chat或发送消息。
