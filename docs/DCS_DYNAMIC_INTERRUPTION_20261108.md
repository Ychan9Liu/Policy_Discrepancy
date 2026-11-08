# Vision20 dynamic：sv2八条Dt磁盘保护中断核查

2026-11-08只读核查，以下均为北京时间（UTC+8）。用户问“八点多到九点启动的一批实验，中断了，是哪些、为什么”。对应今天08:42–09:03在sv2 GPU0–7接续启动的八条原Dt/PPD-KL，均为dynamic、seed0、size50m、alpha20、百万动作/21×10评价；各自baseline已正常完成并被原监督器审计为accepted。

## 受影响运行

| sv2物理GPU | 任务（均Dt） | 11月8日进程启动 | 最后成功更新的起始动作步 | 已完成评价点 |
| --- | --- | --- | ---: | ---: |
| 0 | hopper_hop | 08:42:12 | 802172 | 17/21 |
| 1 | quadruped_run | 08:44:41 | 796768 | 16/21 |
| 2 | quadruped_walk | 09:03:20 | 772616 | 16/21 |
| 3 | reacher_hard | 08:45:13 | 800000 | 16/21 |
| 4 | acrobot_swingup | 08:43:49 | 800000 | 16/21 |
| 5 | cup_catch | 08:44:08 | 800000 | 16/21 |
| 6 | cartpole_balance | 08:43:50 | 800216 | 17/21 |
| 7 | cartpole_balance_sparse | 08:42:14 | 800424 | 17/21 |

动作列取成功更新收据的start_action，不伪装为精确的环境停止计数。八条均无final_state.json，不满足正式完整指标条件。W&B只读回查均为crashed；W&B创建时间晚于进程启动约一分钟。保留原目录、ID、快照和失败证据，不重写原日志。

## 原因与证据

1. 18:47:45监督器记录/data可用5,826,568,192字节，18:48:00降至4,654,387,200字节（十进制4.654GB），跨过现有5,000,000,000字节紧急保护线。
2. 18:48:00.281946的supervisor-error.json明确记录`RuntimeError('Disk below 5GB; stopping only this campaign workers')`；manager.log回溯指向固定控制器的第280行。其finally调用stop_owned，向自身创建的训练进程组发SIGTERM，超时才SIGKILL。各训练日志最后写入在18:48:01–18:48:12，22:42复核八个训练PID和监督PID265311均不存在。没有持久化各Dt的实际退出码，不猜测是否某条使用了超时SIGKILL。
3. 第一条Dt接续时（08:42:07附近）/data仅189.407GB，最后一条接续时（09:03:16附近）仅176.286GB。现有后继检查只要求每次启动时全盘余量至少60GB，没有按同时在跑的八条预留完整剩余产物空间。这个容量门槛不足以保证八条并发完成，是本次容量安排的不足；5GB保护触发本身符合既有设置，不能算正常训练完成。
4. 22:44附近只读du显示该原正式根约408.156GB；八条中断Dt的replay、评价快照、最终训练checkpoint及plugins目录合计约179.040GB。资源序列与实际产物证明持续写盘带来容量压力；未逐文件归因整个共享文件系统的所有写入，不能把所有额外占用都归因于本项目。

证据根S：`/data/Policy_Discrepancy/runs/dcs-dynamic-vision20-seed0-20261107-sv2-remaining28/`，主证据为`resources.jsonl`、`supervisor-error.json`、`manager.log`、`actual-RUN_INDEX.json`。八条产物仍在原根`/data/Policy_Discrepancy/runs/dcs-dynamic-vision20-seed0-20261106-5245096/<task>/dt-attempt01/`。

## 状态记录缺陷与当前范围

原S索引停止在18:48，仍把八条Dt标为running，这是陈旧状态，不能当作仍在训练。控制器低空间分支只给接管的旧worker改写failed_storage；此次八条Dt是接管后新创建的worker，停止发生于finally，该路径没有再写结束状态/退出码。故磁盘保护真实生效，但索引收尾存在缺陷；本轮仅定位和记录，未修复代码或改写服务器索引。

22:42:23现场核查：sv2 /data仅4.339GB，八条Dt及监督器已退出；GPU0/1/3/5/7无计算占用，GPU2/4/6有另一个Python环境的未知作业，未干扰。sv3 /data约402.197GB，六条Dt PID均在，最近成功更新约90.24万–91.25万动作，六条baseline均accepted。14个baseline的原accepted状态、100万动作/249493更新/21×10计数得到核对；这不是重新执行全套终点审计。

合并原40槽的实际观测为：**14条baseline已验收、8条sv2 Dt中断、6条sv3 Dt运行、12条sv2任务尚未启动且因监督器退出阻塞**。后12条是hopper_stand、pendulum_swingup、reacher_easy、walker_run、walker_stand、walker_walk的baseline/Dt；原索引里的6 queued/6 backlog不会自行接续。

## 版本、输出及后续

实际训练版本保持`5245096ee52de5ba1ff6bcafb96d19298b74dcad`，监督器为`0b222ea905ad0bcfe45ff6315ba7655435d25de4`。本次开始HEAD `b2f63b30cc41fc0ff6b9c053154a7ca2e513d86f`，较上次交付65d5db8仅增加sv3旧正式payload清理报告/STATUS；本次核查没有发现运行源码变化，保留现有未提交工作。本轮无训练/算法/配置更改、无重启或删除、无服务器源码写入、无GPU测试；只新增本机核查产物、报告和STATUS。未重跑算法测试，不把只读核对写成新工程验收。

本机小证据在`analysis/outputs/dcs-interruption-audit-20261108/`：双端summary、sv2-details、四份原始监督器证据、W&B只读快照、`diagnose.py`、`observed-RUN_INDEX.json`和`MANIFEST.json`。四份原始证据与服务器SHA256一致；纯CPU核对40唯一ID、14个baseline终点计数、8条无进程/无终点且W&B crashed、6条实际存活、12条未启动和5GB阈值跨越，均通过。派生索引与原始索引分开，不覆盖原日志。

首选继续当前合并03窗口，先明确可释放的数据范围或扩容，完成容量预留与异常退出状态的工程修复/测试，再在原授权资源中重新核占用并准备独立attempt02。不得用本次核查授权删除其他产物、共享GPU或改科学协议。现有runner在`embodied/run/protocol_v1.py:414–418`明确拒绝未完成训练的checkpoint续跑，因为环境及replay RNG未完整恢复；若保持当前已验收语义，这八条需新尝试从头训练，不能把从80万附近接上称为同一条无缝续跑。

本次是故障事实核查，无方法效果结论；sv3六条持续运行和其余待执行槽位保持原授权。后续运行/修复继续当前03，科学协议变化交02与用户，不自动新建chat或发消息。
