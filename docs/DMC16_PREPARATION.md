# 03合并窗口 · 其余16个DMC视觉任务准备

日期：2026-10-22，Asia/Shanghai。**准备完成；用户已暂停本批实测，32条训练均未启动。** 用户先要求其余16任务的baseline与Dt在sv1开测，并确认物理GPU0–7、全部沿用原v1；发现八卡均被占用后，用户明确“先暂停实测，有任务再跑，做好准备工作”。当前暂停优先，不设置空闲自动启动、不继续真实环境/GPU测试、不停止既有作业。恢复本批实测须用户明确恢复，并重新核查资源与工程条件。

## 范围、配置与排队

范围为仓库 `scores/dmc_vision-dreamerv3.json.gz` 中20个任务，排除原4任务hopper_hop、quadruped_run、quadruped_walk、reacher_hard。新范围 `M2-v1-dmc16-seed0-baseline-dt`、预设 `m2_dmc16`、入口 `protocol_dmc16` 只接受新增16任务、训练seed0、logging/Dt。原 `m2_v1/protocol_v1` 的四任务限制与hopper补seed范围保持，原16条结果和现存hopper六条训练不由本次准备重开、替换或停止。

下表省略共同前缀 `dmc_`，数字为每张物理卡未来队列内顺序；每卡最多一条，总并发上限8。排队表是静态计划，**不是已运行的监督器或GPU预约**。

| sv1物理卡 | 组 | 第1条 | 第2条 | 第3条 | 第4条 |
| ---: | --- | --- | --- | --- | --- |
| 0 | logging baseline | acrobot_swingup | cartpole_swingup | finger_turn_easy | reacher_easy |
| 1 | Dt | acrobot_swingup | cartpole_swingup | finger_turn_easy | reacher_easy |
| 2 | logging baseline | cup_catch | cartpole_swingup_sparse | finger_turn_hard | walker_run |
| 3 | Dt | cup_catch | cartpole_swingup_sparse | finger_turn_hard | walker_run |
| 4 | logging baseline | cartpole_balance | cheetah_run | hopper_stand | walker_stand |
| 5 | Dt | cartpole_balance | cheetah_run | hopper_stand | walker_stand |
| 6 | logging baseline | cartpole_balance_sparse | finger_spin | pendulum_swingup | walker_walk |
| 7 | Dt | cartpole_balance_sparse | finger_spin | pendulum_swingup | walker_walk |

接球任务的成绩标识为 **`dmc_ball_in_cup_catch`**；当前 `embodied/envs/dmc.py` 按首个下划线拆domain，运行标识须用 **`dmc_cup_catch`**，包装器再映射为 `suite.load('ball_in_cup','catch',...)`。准备产物保留runtime→score显式映射，两者是同一任务，不增加第17个任务，也不修改环境/奖励。不能把成绩标识直接当作当前CLI任务ID。

共同配置：clean纯视觉64×64 RGB、size50m、seed0、alpha20、free_nats1、dyn/rep=1/.1、ac_grads=false、reward_grad/repval_grad/repval_loss=true；batch16×64、replay_context1、16训练env、train_ratio256、repeat1、CUDA/bfloat16、prealloc=False。其余模型/优化器/replay/观测处理完整继承已声明v1；每对配置只差mode与独立logdir，不增debug、干扰或奖励修改。

每条100万训练动作含预填充、排除reset/评价；0..100万每5万评价，共21点，每点10完整episode，固定快照/采样动作/隔离评价RNG。总计划32条、3200万训练动作、672点评价、6720完整评价episode，不据此预估吞吐或完成时间。action维数、任务原始horizon与实际渲染尚未实测，不把所有任务假定为已通过同一接口或相同episode长度。

baseline实际rep权重1，候选w同alpha20；每任务自身seed0完整logging baseline按开始动作 `[100000,300000)`、raw rep KL>1有效位置S/N计算冻结c，仅保留供审计；本批无constant/shuffle，Dt不等待c。正式c目前全部null，fixture的c=.5不是实验参数。无活跃位置/非法值按原规则停止报告，不自动填1或调alpha。

每任务原始return AUC/固定百万分母、80/85/90/95/100万五点末段和Dt−baseline保持。新增独立 `--profile dmc16` 为16任务完整组成的等权描述性汇总，保留逐任务；缺项不重加权。默认原四任务四组和hopper补seed profile保持，不自动把历史四任务或不同seed合并成20任务总体。本批缺少constant/shuffle对照，不能据两组差异独立归因于M2逐时刻选择价值，也不作跨训练seed稳定性结论。

## 资源核查与当前暂停

唯一现场资源快照：2026-10-22 **14:52:17北京时间**（06:52:17 UTC），sv1 SSH配置实际落到host `lyg2153`。八张A100-SXM4-80GB均有Ray计算进程，PID按物理卡0..7依次为3239378、3239384、3240840、3240837、3239525、3239549、3240885、3240886；归属均未识别，记未知作业，不干扰。显存依次629/629/2965/2965/2993/2849/9897/9895 MiB，低占用也不等于空闲。/data当时约1.1T可用，不是后续容量保证。

当时sv1仓库HEAD `313532571f0a0e39759b9229264361c0fd9bd48f`、工作区干净。这只是只读快照，不是代码已同步或环境已验收的证明。用户暂停后没有再执行SSH命令、服务器写入/安装、环境创建、模型推理、GPU探针或训练。本批占用GPU数0、没有本批需要释放的卡；没有因本次准备停止既有sv1或sv3作业。

## 实现与CPU验收

代码及实际测试提交 **`e83a012a8dde255b8cb9df0631243eddab2454db`**。后续交付仅报告/索引文档，最终完整SHA见本机DELIVERY及最终回复。代码本机提交，未推送GitHub、未同步任一服务器；可验证bundle仅在本机outputs留存，待恢复时同步确定版本。

- `dreamerv3/protocol_scope.py` 明确新16任务范围；`configs.yaml/main.py/protocol_v1.py` 增加独立入口及任务允许列表，不放宽旧范围。agent、RSSM、rep probe、JAX agent、DMC包装器文件与接手 `133888f4116db1898784757269fce7ae0b629c67` 一致；runner除任务允许列表外的run AST以及所有计步/评价/随机流函数和ProtocolState AST保持一致。
- `scripts/prepare_dmc16.py` 仅CPU准备，无launch子命令、无Popen或空闲监视器。检查运行源文件已入Git，允许保留不参与运行的本机文档草稿；未来服务器仍须干净确定提交。生成32份JSON+YAML完整配置、16份随机流实际seed/key映射、任务清单/别名、八卡四条队列、配置/产物哈希及待执行命令。所有运行 `authorized_to_train=false`、`launch_argv=null`，EGL为未绑定，指标及c为null。
- `freeze_c.py` 接受新范围但保持S/N、半开窗口、去重与错误处理；新增范围拒绝其他seed、任务、mode、工程fixture。Windows测试发现旧write_text的CRLF转换使文件字节与LF payload校验串不一致；修为写出固定LF字节，保留严格校验、不归一化读取旧坏文件。Linux原输出不变，原v1及hopper产物语义与旧函数fixture逐字段一致（仅冻结时间不同）。首个失败日志保留，不放宽容差。
- `analysis/return_metrics.py` 增加独立dmc16范围，不改奖励、积分/末段公式或原容差 `atol=1e-9,rtol=1e-12`；动作步/ID/数量/快照为精确检查。新任务/seed/mode混入旧profile不能成为完整结果。
- 本机 **33项CPU测试通过，10.598秒，无跳过**：14项新准备/入口/ledger/c/统计fixture，原15项return测试与4项hopper范围测试。新fixture覆盖32生产配置/配对差异、20−4及接球映射、独立范围拒绝、实际runner前置校验、固定队列和暂停状态、源代码污染拒绝、16×21×10 RNG映射、c边界/文件哈希/幂等/旧范围兼容、6720合成episode/672点/32指标行的完整CLI与缺点拒绝；手算总体7.5/9.5、Dt差2仅是合成数据。没有执行真实环境或训练。
- 已有真实原四任务16运行的小镜像CPU只读复算，`--require-complete` 返回0，3360 episode；`tables.json` 与先前验收版本逐字节相同，SHA256 `3d02f5cca66644877e55e45a79fd160949ff9b8d72c45920bf09e1235d75e9ef`。这验证旧统计兼容，不是新16任务实测。

本机隔离venv仅安装配置解析依赖：Python3.12、NumPy1.26.4、elements3.22.2、ruamel.yaml0.19.1，完整实际版本在local-environment.json；未修改系统Python或服务器环境。没有JAX/ninjax完整agent环境；测试中的生产入口校验、ProtocolState和freeze通过AST提取实际函数执行，DMC dispatch用假suite，不调用真实环境。**没有新增完整agent、CPU/float32训练、CUDA/bfloat16、梯度数值或真实任务可运行性验收。** 原phase1/2算法测试保留未修改，不把静态计算路径一致性当作重新通过全部算法测试。

## 产物与复现

本机根 `analysis/outputs/dmc16-preparation-20261022/`，Git忽略：32配置、16随机映射和queue/index在 `plan-final/`（最终交付SHA绑定）；代码测试提交的独立快照在 `plan-e83a012/`。另含resources-observed.json、CPU stdout、原v1复算/相同哈希、开发失败记录、环境、validation/DELIVERY、bundle哈希和完整HANDOFF。既有用户RESEARCH/STATUS、proposal和根目录TeX相关文件未混入提交。没有新32条结果表，计划不得伪称actual-RUN_INDEX。

```powershell
# 本机仅CPU准备；输出目录必须新建。
.\analysis\outputs\dmc16-preparation-20261022\venv\Scripts\python.exe -m unittest -v tests.test_dmc16_preparation tests.test_return_metrics tests.test_hopper_repeat_scope
.\analysis\outputs\dmc16-preparation-20261022\venv\Scripts\python.exe -m scripts.prepare_dmc16 --output analysis/outputs/NEW_DMC16_PLAN --campaign formal-m2-dmc16-baseline-dt-20261022 --resources analysis/outputs/dmc16-preparation-20261022/resources-observed.json
```

未来实验根的计划值为 `/data/Policy_Discrepancy/runs/formal-m2-dmc16-baseline-dt-20261022/`，下接每个运行task的 `{logging,dt}-attempt01/`，本轮未在服务器创建。恢复时如根已存在，保留旧尝试并使用新运行标识，不覆盖、不将环境/replay不完整的恢复称为无缝续跑。

## 恢复后的必要条件及下一步

首选继续本03/04合并窗口，**等待用户恢复本批实测**。允许继续只读准备审查；不能因GPU后来空闲自动探针/训练，也不把原启动指令覆盖后来的暂停。

恢复后先核实际HEAD/差异，更新sv1入口/host/UUID及全部类型C/G占用、磁盘/RAM和依赖；八卡仍被占用则等待用户调整，不换卡或共享。同步本机确定提交后核干净源码；在分配卡上验证实际CUDA/EGL落卡，检查新16任务动作空间/seed/64×64纯视觉/相机/horizon及真实size50m链路，补受影响完整agent/梯度回归和W&B账号/PD_1在线写读；这些尚未通过。完整命令从最终计划派生，实际记录须保存当时SHA/config/seed/环境/设备映射。

随后由本窗口完成启动绑定与受限队列监督，逐卡空闲核查后按批准的32条计划从头训练、保存失败尝试/退出码/更新收据/快照/匹配和诊断；不改alpha、预算、模型或奖励来掩盖故障。最终逐条验收百万动作、21×10评价、无非法值/冲突收据、自身baseline的S/N/c、完整原始return及W&B精确发布，并报告实际GPU释放。全部审计后才交05分析；缺少constant/shuffle和单seed边界保留，新的科学协议仍由02与用户决定。不自动创建chat或发送消息。
