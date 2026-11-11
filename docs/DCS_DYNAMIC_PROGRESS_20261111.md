# Vision20 dynamic 40条进度核查（2026-11-11）

北京时间2026-11-11 17:42:37分别只读核查sv2、sv3。范围为原Vision20 dynamic、seed0、size50m、alpha20、baseline/原Dt共40条；原百万动作/repeat1、21点各10完整episode、背景声明和科学协议保持。本次未启动、恢复、停止或删除任何实验。

| 状态 | 条数 | 范围 |
| --- | ---: | --- |
| 原监督器已验收完成，本次终点计数复核通过 | 20 | sv2八条baseline；sv3六任务的baseline/Dt十二条 |
| 中断，尚未重跑 | 8 | sv2八条Dt，仍为11月8日磁盘保护中断的attempt01 |
| 未启动，队列监督器已退出 | 12 | sv2剩余六任务各baseline/Dt |
| 本批运行中 | 0 | 原训练和监督器PID均不在；可读进程未见本批训练 |

按单条运行完成20/40（50%）；按完整baseline/Dt任务对，完成6/20（30%）。不把中断进度折算为完成，也不把原索引的陈旧running当作正在训练。

## 任务范围

- **两组均完成的六任务（sv3）**：cartpole_swingup、cartpole_swingup_sparse、cheetah_run、finger_spin、finger_turn_easy、finger_turn_hard。最后一条finger_turn_easy Dt于11月9日00:01:05结束，00:01:07整批supervisor-final为passed=true、engineering_only=false，十二条均exit0/accepted。
- **仅baseline完成、Dt中断的八任务（sv2）**：hopper_hop、quadruped_run、quadruped_walk、reacher_hard、acrobot_swingup、cup_catch、cartpole_balance、cartpole_balance_sparse。Dt最后成功更新起始动作仍为77.26万–80.22万，均无final_state，没有发现新attempt目录。详情见[11月8日中断核查](DCS_DYNAMIC_INTERRUPTION_20261108.md)。
- **两组均未启动的六任务（sv2）**：hopper_stand、pendulum_swingup、reacher_easy、walker_run、walker_stand、walker_walk。原6 queued/6 backlog仅为历史队列状态；监督器已退出，输出目录不存在，不会自动接续。

## 验收依据与资源

合并新S28与T12索引得到40个唯一run_id。20个accepted运行均具备100万训练动作、249493成功更新的终点状态，评价动作网格准确为0至100万每5万一次，每点10条完整episode，合计420点评价和4200条完整episode；原索引exit_code均0并有audit。本次没有重跑完整更新收据/模型/奖励重放/W&B终点审计，保留原监督器accepted语义，不把工程结束等同于方法有效。

sv2原索引与11月8日中断时的完整SHA256相同，仍错误保留八个running；supervisor-error的hash也相同，原因仍是5GB磁盘保护。八个训练PID和监督器均不存在，未发现重跑或新的DCS运行根。派生observed-RUN_INDEX与原始索引分开存储，没有改写服务器旧记录。

17:42资源快照：sv2 /data可用143.657GB，nvidia-smi枚举GPU0–7均0MiB/0%；sv3 /data可用158.568GB，nvidia-smi已恢复可查询，与14:44的驱动不可用快照不同。当前sv3仅枚举7卡（0–6），旧UUID `GPU-4725722b-bd64-ec77-2fcd-9692cf3ee36e`未在输出出现，后续若干索引已改变，不能沿用旧物理编号/EGL绑定。当前索引3–6约48,500MiB（47.4GiB）、利用率100%，归属未判明，未干扰。这里只记录查询结果，不诊断或维护驱动，也不把当前可查询当作全部GPU可启动。

本批原训练进程均已结束；当前未知作业及新的GPU编号不属于重新分配授权。磁盘可用量较11月8日/11月11日早些时候改变，未核定谁清理或扩容。现有容量不应据单条60GB门槛直接启动八卡并发；此前容量预留不足与异常退出状态缺陷尚未修复。

## 版本、证据与下一步

本轮开始HEAD `ec62ec30e62422d20dfa1273d4e1bf45326b50f0`。相对11月8日核查e85c098，只增加sv3状态及磁盘chat范围文档，未改算法/调度实现。实际训练固定`5245096ee52de5ba1ff6bcafb96d19298b74dcad`、原监督器/此前CPU实测`0b222ea905ad0bcfe45ff6315ba7655435d25de4`。本次只读服务器及本机证据核对，无新算法测试、GPU测试、训练或服务器写入；只提交本机进度报告和STATUS，现有未提交工作保留。

证据在`analysis/outputs/dcs-progress-20261111/`：双端snapshot、`collect_readonly.py`、`reconcile.py`、`observed-RUN_INDEX.json`、`MANIFEST.json`和`DELIVERY.json`。纯CPU只读核对通过，区分原始状态与当前实际状态；服务器原根仍为sv2 `runs/dcs-dynamic-vision20-seed0-20261107-sv2-remaining28/`、sv3 `runs/dcs-dynamic-vision20-seed0-20261107-sv3-six/`，均位于`/data/Policy_Discrepancy/`。sv3较早的完整小日志复核见[SV3_STATUS_20261111.md](SV3_STATUS_20261111.md)，该报告的14:44资源快照只作历史。

首选继续当前合并03窗口，处理并发容量预留、异常退出状态收尾及新尝试准备，之后重新核验已授权资源与容量，才能接续剩余20条训练（8条重跑、12条新跑）。已确认协议和原授权保持；不能擅自删除产物、共享或换卡、复用失败目录，也不能称环境/replay未完整恢复的续跑为无缝恢复。磁盘清理按用户已指定的专用窗口范围处理；具体删除范围仍须明确，不能由本次进度核查扩展。完整六对日志可交现有05先做子集分析，但不能报告为全20任务比较。不开新chat、不自动发消息。
