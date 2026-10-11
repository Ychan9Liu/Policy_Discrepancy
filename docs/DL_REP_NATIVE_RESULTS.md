# DL-rep-native-r1：运行与GPU资源记录

## 2026-10-12 04:57:02 状态快照

状态：运行中，完整主/独审未验收，无科学裁决。规范见 [DL_REP_NATIVE_ASSAY.md](DL_REP_NATIVE_ASSAY.md)，完整目标继续active，原M2不改。

实际/测试代码SHA：fc4219f36449783f8751a9890a125cd218a2807f。clean Linux四组38项全过40.602秒，外CLI40.817279秒/exit0/owned终止，pip check0；含Windows跳过的真实Linux进程组/无关sentinel测试。tiny官方RSSM/Optimizer/Agent和fakephysics/hash/退出契约是工程验收，不替代实际size50m任务或方法效果。原19生产Git blobs与c030e26相同。

04:49:15.053030主入口启动，父PID2059834、外launcher2059813。旧source/B完整退出与独审链认证通过。A128学习owned143.612114秒/exit0，Optimizer int32步74493→74621，原cal/context/batch/keys匹配，保留其它损失、momentum、free nats1及系数dyn1/rep.1；Dt/Probe关闭。实际reset active2048/2048，whole active130129/131072、nonreset128081，reset/whole excess75089.42974853516/541603.4714355469。活跃KL不证明任务信息损伤或保护有效；旧B没有全窗口压力值，不伪造匹配。

04:57:02快照：eval0完成11/32个完整1000-step episode；B/A尚未评完、CPU独审未启动。真实A评价reload、全部96轨迹/96k transitions、A−B统计及主/审计完整退出仍待验。禁止局部裁决、缩短/更换样本、延长7200秒或拼接attempt。

## 资源口径

| 环节 | GPU与预算 |
| --- | --- |
| 主流程 | sv3物理4，一张独占A100-SXM4-80GB；7200秒硬限、单attempt≤2 GPU·小时，含认证/编译/A128/3×32×1000闭环/oracle/统计/写盘/退出，非实际ETA |
| 主显存 | 截至快照采样峰6623MiB（6.4678GiB）；约2秒采样，非瞬时峰或最低显存保证 |
| 零动作设备核查 | 外11.403533秒，采样峰61401MiB；0训练动作/0Agent更新，compute/EGL同物理4，退出后释放，另记成本；不把probe峰当模型训练需求 |
| 独立审计 | 1200秒完整CPU入口、0GPU；96000 stored-applied官方physics复放及独立fsum/配对bootstrap，非模型/renderer/optimizer重演 |
| 全部预验证 | 总GPU小时未定；普通replay、第二信号/nuisance/错误依赖、选择性gate和泛化仍缺证据，不能由本对象上限外推 |

资源实际XML快照：物理4为本项目PID2063211/C+G、2843MiB；5/6/7各1MiB但无进程，无本对象计划。快照不代表今后仍空闲。物理4 UUID为GPU-9c070d5a-ac68-4c26-9522-a790ad874b23；CUDA_VISIBLE_DEVICES同UUID、进程CUDA0、MUJOCO_EGL_DEVICE_ID=4；实际render后核计算/渲染映射。报告没有把1MiB底噪改为0。

## 产物、版本与下一步

服务器run：/data/Policy_Discrepancy/runs/dl-rep-native-20261012-fc4219f/，主primary-attempt01、待执行独审sibling independent-audit01。完整命令/config/source/environment/pins/roots在engineering/primary01-start.json及各phase完成收据；本机ignored小镜像analysis/outputs/dl-rep-native/dl-rep-native-20261012-fc4219f/。无checkpoint/RGB/轨迹pickle入Git。外launcher SHA256：64f7955c238ba45077c43d5e8618d77f8c260af215235a0ed0237e14026c6e8f。首次tiny fixture锁重入失败及修复、Windows证据见 [DL_VALIDATION_LOG.md](DL_VALIDATION_LOG.md)。

只向实际sv3隔离checkout /data/Policy_Discrepancy/dl-worktrees/fc4219f-rep-native同步代码；服务器canonical仍clean d9be5574878d2c3f117cc6667be844f9a527eb84，本机canonical/sv1/sv2未改。研究branch push两次分别connection reset/443连接失败，尚未确认GitHub新SHA，未合main。后续报告提交仅文档不同，不冒称实际测试/科学SHA。

下一首选继续当前完整授权chat，完成原attempt全部96轨迹、主exit0/完整封口，然后同代码SHA执行0GPU/1200秒CPU独审并核资源释放。失败保留、只修工程，不改科学设置/阈值/预算；成功仍按spec条件分支处理未决信号/gate，不能宣布整个目标完成或自动开始正式M2。无需另建chat或再审批已授权流程。
