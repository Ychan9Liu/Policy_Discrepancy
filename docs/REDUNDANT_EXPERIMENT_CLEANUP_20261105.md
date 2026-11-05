# 03 · 已完成工程短跑冗余清理（2026-11-05）

用户要求“根据前面几轮实验运行的情况，清理废弃的实验冗余磁盘占用”，并延续此前“本机归档不要留”的要求。本轮检查sv1/sv2/sv3，按已通过验收的具体工程运行清单直接移除模型和回放payload，不生成、转存或保留payload归档。105条均为4098训练动作步、517次成功更新和3点评价的工程短跑；没有把正式实验或科学效果不佳的运行判为废弃。

## 结果

| 服务器 | 工程运行目录 | agent.pkl数 | replay NPZ数 | 移除分配块GB | 后查/data可用GB | 后查时间（服务器北京时间） |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sv1 | 43 | 108 | 1376 | 56.994820 | 63.903711 | 2026-11-05T18:10:26.511705+08:00 |
| sv2 | 46 | 184 | 1472 | 96.097247 | 265.015599 | 2026-11-05T18:20:17.854056+08:00 |
| sv3 | 16 | 64 | 512 | 33.380499 | 498.411971 | 2026-11-05T18:07:57.420167+08:00 |

合计删除3716个明确文件：356个工程agent.pkl和3360个工程replay NPZ，逻辑容量186464922048字节，原分配块容量**186472566784字节（186.472567十进制GB）**。sv1/sv3的execute01一次完成，sv2三次在进程可见性保护处停止后，核完整已删前缀并仅继续剩余部分；原失败收据保留。最终三台独立只读postcheck全部通过。容量以目标原分配块计，不用其他作业仍写入时的磁盘净增量代替；可用空间只是表内时间快照，不是新增启动授权。

## 具体范围

以下根均位于 `/data/Policy_Discrepancy/runs/`；仅清理验收收据逐条列出的成功运行，未递归删除根目录。

| 服务器 | 工程根 | 成功运行目录数 | 移除GB |
| --- | --- | ---: | ---: |
| sv1 | engineering-dmc8-serial-20261022 | 16 | 0.548094 |
| sv1 | engineering-vision20-seed1-20261104-sv1-retry02 | 24 | 50.172170 |
| sv1 | engineering-vision20-seed1-reallocated-20261104-sv1 | 3 | 6.274556 |
| sv2 | engineering-dmc8-remaining-20261024-sv2 | 12 | 25.090101 |
| sv2 | engineering-missing-dt-20261030-partial1357-sv2 | 4 | 8.343306 |
| sv2 | engineering-missing-dt-20261030-remaining3-sv2 | 1 | 2.084811 |
| sv2 | engineering-missing-dt-20261030-sv2 | 7 | 14.597874 |
| sv2 | engineering-return-cuda-20261009-27261bf | 4 | 8.374440 |
| sv2 | engineering-size50m-20261009-1393548 | 4 | 8.401793 |
| sv2 | engineering-vision20-seed1-20261103-sv2 | 14 | 29.204922 |
| sv3 | engineering-missing-dt-20261030-remaining3-sv3 | 2 | 4.169724 |
| sv3 | engineering-missing-dt-20261030-sv3 | 4 | 8.361079 |
| sv3 | engineering-pendulum-seeds23-20261103-sv3 | 4 | 8.340038 |
| sv3 | engineering-pendulum-seeds23-parallel-20261104-sv3 | 2 | 4.170047 |
| sv3 | engineering-size50m-20261009-1393548 | 4 | 8.339612 |

每条运行的配置、protocol_manifest、final_state与验收收据先核SHA256，要求engineering_fixture=true、action_budget=4098、终点4098/517/3；integration收据还要求passed=true、engineering_only=true、退出码0。旧size50m/report-return运行使用各自passed的audit.json。清理路径进一步限定为 `ckpt/<id>/agent.pkl`、`eval_snapshots/<step>/<id>/agent.pkl` 和该运行 `replay/*.npz`，并逐个核绝对真实路径、无符号链接跳转、独立单硬链接、UID、inode、长度、mtime、分配块与完整内容SHA256。

保留所有未列入成功工程运行的失败/中断/冻结输入证据、CPU一致性fixture、profile、源码和Git包、日志、指标、配置、成功更新收据与原验收报告。sv1先前已删模型的16条工程运行本次仅清其剩余replay；没有改写此前归档移除事实。sv2 11月5日新工程/DTL目录及关联活动工作本轮排除；所有formal根的模型/评价快照/replay均不在删除选择器中。DL/旧探索运行、Denoising剩余模型与预算快照、环境/base-py311和服务器源码也不在本次清单。

## 核验与恢复边界

- 删除前完整计算3716个目标文件SHA256并冻结清单；删除前再次核验原验收证据与文件身份，逐项unlink，按index写审计日志，定期重新检查占用；没有按名称通配批量递归删除。
- 删除前、删除后和独立后查，7903个工程保留文件完整SHA256/元数据、36个链接均通过。在准备时final_state记录已达百万动作的91个正式运行目录，118755个pkl/npz文件元数据保持；这部分没有重读全部大payload做完整hash，也不替代正式终点科学审计。活动正式运行仅作范围排除，不能将其正常新增数据误判成清理改动。
- 服务器与本机镜像逐文件hash相同；本机逐条复算3716条删除日志，路径、类型、长度、分配块、目标hash和序号与清单完全一致、无重复；原路径缺失及105个失效标记均经服务器独立复核。
- 进程检查覆盖可读cmdline/cwd/FD。目标目录引用者及未分类的同UID可见性缺口必须为空；仅对已识别并核父进程的系统SSH/PAM会话记录例外。其他UID的/proc部分句柄不可读，不能宣称root级全主机占用证明。没有停止或挤占既有作业，也未核GPU占用或启动GPU工作。
- 每条清理后的运行新增 `ENGINEERING_PAYLOADS_REMOVED_20261105.json`，available=false、payload_archived=false，并指向完整清单。原latest、checkpoint元数据和历史验收不改写；模型和轨迹已不可从这些原文件恢复，hash/指标不是备份。需要重新打开旧工程checkpoint或重放原轨迹时，必须先核其他独立副本；不能继续宣称现存模型可复开或无缝恢复。

## 命令、异常和版本

本机使用现有 `analysis/outputs/dmc16-preparation-20261022/venv/Scripts/python.exe`。实际脚本、请求、stdout/stderr均在本机证据根，维护程序SHA256：`37a7865f253ab6a32f90c3ca824c3816b0ad7f14d9171a06ba918f258c83cb6b`。脚本通过SSH在各服务器临时执行，只写清理审计及明确payload目标；没有安装依赖或手改服务器项目源码。

sv2分段继续/核验程序SHA256：`e3c6f1a2af93c06cef28ea9309c3671a4ba94e3d9ebb405c6d50ab1e62bb10d1`，与实际请求、服务器源码及三份分段收据一致；本机另存的CRLF文本副本hash不同，不能替代实际执行源码hash。程序直接复用原维护程序的路径、内容和占用检查函数，未修改检查条件。初次删除100文件后遇PID3407152、continuation01再删900文件后遇PID3409809、continuation02再删500文件后遇PID3410528，均为同UID的nvidia-smi comm且cmdline为空、fd暂不可读。原错误分别留在deletion-receipt、continuation01/02-receipt；continuation03完成余156文件。每段新文件日志保持连续index 0…1655，只接尚存尾段、不重删或覆盖旧日志；三段继续前均先重新核原验收、全体保留文件、剩余文件身份和完整占用检查。首次问题PID在单独precheck时已退出；当前可见性通过不能追溯证明这些进程故障时的隐藏FD从未引用目标。最终continued-postcheck逐项核四段内容/收据hash、全部原路径缺失与46处失效标记，success=true。

实际命令依次为：

```text
python inventory_local.py
python collect_evidence.py
python make_plan.py
python check_scope.py
python remote_call.py <sv1|sv2|sv3> prepare prepare02
python fetch_audit.py <host> prepared
python remote_call.py <host> execute execute01
python remote_call.py <host> verify verify01
python fetch_audit.py <host> completed
python continue_call.py inspect_partial partial-precheck01
python continue_call.py continue continuation01
python continue_call.py continue continuation02
python continue_call.py continue continuation03
python continue_call.py verify_continued continued-verify01
python fetch_audit.py sv2 continued
python validate_results.py
```

路径边界检查接受全部3716个实际清单路径，并拒绝8个越界/错误类别fixture。它只证明选择器边界，实际删除验收以服务器内容、文件身份、占用、保留基线与后查证据为准，不当作训练或方法有效性验收。

初次sv1 prepare01在验收证据读取处遇到系统python3缺少hashlib.file_digest；发生在建立审计目录和删除前。改为兼容的标准SHA256分块读取，原脚本/失败日志保留，目标和验收条件未放宽。sv1若干SSH请求在认证前连接关闭，失败日志保留，后续原入口请求成功；没有因连接失败重做已完成删除。上列通用verify/completed命令适用于sv1/sv3，sv2使用分段继续及continued核验/镜像命令。

- sv1：清单 `c42805de286afbab1d0d18b205b0356c7574925f86cf2141d3b415b2db1d063a`；删除日志SHA256 `9e97c09a7a712d17fb7aa1a92ca17bc742ce2887214325556fcebefa626094ab`。
- sv2：清单 `12920d1224947fce32896f8e7dfb9ded485764accbaf29eb6d92c4592113bc31`；各段原始日志按序拼接的SHA256 `cb9065b66eeac803d22f2d4a190e9123edaaeb523082acd73b8603d800920d8d`。
- sv3：清单 `86e6b039c3ae26b20cf7ef021c3698321ab26f9acb2cee7b16116df9839ab187`；删除日志SHA256 `5f6d0ea569a1c62f0d92d1ffa6dfe7b811eaa50a335f91a048c996c12341637e`。

起点HEAD `479c25b71d2383208e9f7c13f1de7942b6d6755c`；本机结果核验HEAD `8fd7bed1a0ab238cddf3f3937e70673eeb4bb01d`。这期间其他窗口提交和未提交文稿保留。本次没有训练代码、算法、科学协议或实验运行变更，没有新的agent/GPU测试SHA；验收对应上面的实际维护脚本hash。最终文档提交及其与核验版本的差异写在本机DELIVERY.json，只提交本报告和STATUS的本次索引，不收纳其他窗口草稿。

本机证据根：`analysis/outputs/redundant-experiment-cleanup-20261105/`。三台服务器各自审计根：`/data/Policy_Discrepancy/runs/redundant-experiment-cleanup-20261105/`。其中保存授权/完整文件清单、删除前hash、维护源码、进程检查、删除日志、收据和独立postcheck；大模型及replay没有下载到本机。同步范围为三台服务器的清理及审计记录、随后本机文档Git提交；没有GitHub推送或服务器repo/checkout同步。

本次确认可废弃的工程冗余已清理。进一步减少正式模型/评价快照/replay或旧科研分支的数据，需要先明确科研保留和复核需求，不能仅按时间、低return或目录名认定废弃；本轮不替用户作此决定。
