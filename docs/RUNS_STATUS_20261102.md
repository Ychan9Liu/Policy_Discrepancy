# 03合并窗口 · 当前完成情况（2026-11-02）

北京时间12:07索引/资源核查、12:07—08线上读回。**新增16任务×baseline/Dt共32条：31条训练结束，1条运行，未启动/排队/失败均0。全32条最终汇总尚未完成。** 本轮只读核查和文档记录，未新启动、停止、迁移或修改训练，不新增测试/科学配置。

## 本次核实的进展

昨晚23:34启动的六条Dt均accepted、退出0，各百万训练动作、249493次成功更新、21点严格0..百万每5万、每点10完整episode，已有逐条run_audit和分片completion，表hash一致；线上六条finished、完整配置一致、原始return精确JSON/hash与本机源表读回一致：

| 服务器/原物理GPU | task（省略dmc_） | 完成时间（北京时间） |
| --- | --- | --- |
| sv2/1 | cup_catch | 11月2日10:34 |
| sv2/3 | cartpole_balance_sparse | 11月2日10:34 |
| sv2/5 | cartpole_swingup_sparse | 11月2日10:34 |
| sv2/7 | hopper_stand | 11月2日10:34 |
| sv3/1 | cartpole_balance | 11月2日10:43 |
| sv3/3 | cartpole_swingup | 11月2日10:43 |

sv2四条表SHA256 `8a9fb5024c696371d99c07e901529c95406d15f196f08a2f768e675cc29e8f99`；sv3两条表 `deed92d0feb5905d2e17591141ba2148d643da3261c9ef206c25b5bb70abe02f`。分片shard_complete=true/cohort_complete=false，不能替代32条全体审计。此前已accepted后补sv3四条本轮交叉核final/评价/尾收据/completion与表hash一致，原17条所属批次汇总未因本次六条完成自动收尾。

**唯一剩余：acrobot_swingup / Dt / seed0，sv2物理GPU1。** cup终点审计成功后，队列按授权自动释放，11月2日10:34:50实际启动agent1791869。12:07:41核查34893次连续成功更新、最近更新开始动作141600（约14.2%预算），invalid累计0；0/5万/10万三点评价各10完整episode。manifest完整配置/SHA一致，CUDA UUID/EGL1/进程内CUDA0同卡，17个C/G PID均属此agent/子进程，未发现外来共享；W&B running、完整配置一致。最近更新开始动作不是精确实时总动作；本轮未评价收益或预测精确结束时间。

queue-progress旧字段training_started=false是释放前/ready阶段快照，不能继续当实时状态。queue-released及正式目录/manifest/实时收据证明已启动，不再报告“acrobot仍排队”。真实前驱完成→审计→串行释放已发生；本次也没有重复运行队列或重跑已结束任务。

## 资源、范围和证据

12:07 sv2 GPU1仍有本项目acrobot；GPU3/5/7已由本批释放，当前无C/G PID，sv2 GPU6未使用。sv3新增GPU1/3及此前4—7均由本批释放、当时无C/G PID；GPU0/2仍为外来作业，不干扰。/data余sv2 677,158,989,824字节、sv3 117,977,235,456字节。资源快照不代表未来预约；sv1本次未检查，不把历史空闲快照当现状。

原21条已结束计数继承11月1日各索引核查，本次重点核新六条、原补四条和acrobot，未对旧21条再次全量重算；31条训练结束不等于31条全部完成所属批次汇总。hopper补seed1/2/3六条保持独立既有accepted记录，若仅盘点新增38槽位为37结束/1运行，不自动混合科研指标；历史原四任务四组16条结果不变。

冻结clean视觉/size50m/seed0/alpha20/free_nats1/原KL系数及梯度开关/百万动作/21×10/JSONL+在线PD_1继续有效。c来源和raw-return/固定百万分母/指定末段五点/缺项不重加权按EXPERIMENTS C/C.1，不调参或扩大任务/seed/组。实验完成数量和工程通过不证明方法有效或强M2因果，跨机逐位/真实环境/旧GPU7并发等限制保留。

新七条运行及CPU114/CUDA工程实测SHA `c7689d50f4caed0fef89912c65280b8d2558b126`；队列/17项实测 `2cccd5e006a639f10699907ad3f5b2f49c2c0b56`；早先补四条运行 `d8997f64a301bfdcfe08758faeae8957a7395766`。本轮相对起点`6b7214f2ea8a1d446b347ffd3f61364191139f7b`仅状态文档变化，无代码/测试/实验变更，无服务器Git同步或GitHub推送，用户既有草稿保留。最终完整SHA见本机DELIVERY及回复。

源正式根仍为`/data/Policy_Discrepancy/runs/formal-m2-dmc16-missing-dt-20261030-{partial1357-sv2,remaining3-sv3,remaining3-sv2,sv3}/`。新六条线上/表校验及acrobot审计保存在相应工程根`startup-20261101/status-audit-20261102.json`；本机状态与线上审计小证据/DELIVERY/prompt：`analysis/outputs/run-status-20261102/`。完整命令、设备、配置、旧验收和恢复限制见`RUNS_STATUS_20261101.md`及`MISSING_DT_REALLOCATION_20261030.md`。

首选继续现有03监测acrobot并完成其百万动作/21×10/收据/快照/raw-return/W&B终点，再统一旧31条与最后一条的全32条来源和最终统计/线上收尾。可立即监测/CPU只读核查，完整表依赖最后训练及全体审计；无需新建chat、不自动发消息，不以交接替代任务。新科学复验仍由02与用户决定，完成审计后由用户交现有05分析。
