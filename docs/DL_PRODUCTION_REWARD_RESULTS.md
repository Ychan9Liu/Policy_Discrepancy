# DL 生产奖励均值诊断结果

版本 `DL-production-reward-r1` / 工程r1，2026-10-12。固定B对象及独立CPU审计已完整验收；**科学分类为 MC-inconclusive，停止此B，不代表方法有效或整体路线阴性**。规范见 [DL_PRODUCTION_REWARD.md](DL_PRODUCTION_REWARD.md)。这是用户完整预验证授权内的Agent诊断设计与实际有限证据，非用户逐项确认的算法。

## 实际对象与版本

- 执行及clean Linux测试完整SHA：`d685a2d3fc8b8783f6c2e99e9d57ed8cb6702512`。本提交新增五B脚本、一个独立auditor、六测试和工程文档；未修改Dreamer原算法或正式M2配置。后续汇总文档提交不作为实际执行SHA，差异与同步按Git终点收据。
- sv3/`lyg0326`隔离执行目录 `/data/Policy_Discrepancy/dl-worktrees/d685a2d`；canonical `/data/Policy_Discrepancy/repo`保持clean `d9be5574878d2c3f117cc6667be844f9a527eb84`。本机canonical main为 `595e982f1a3939775879728009e2105ee516e86d`，原dirty RESEARCH/STATUS保留，不合并/覆盖；sv1/sv2未同步。
- 正确生产f1来源 `f1ae4d6513e81ed680992b64ba8ebade7d256d52`；cal episode `[9,14,15,18,20,24,30,31]`、位置 `[1,300,600,900]`，同实际h/四已存sampled q/p接口、七首normalized控制`.25`＋19步normalized zero，H20。六源JSON/八数值白名单NPZ原SHA、真实F64累计成本及source/carry/完整20步proof认证复用；不解records、不读blind、不新执行physics。
- checkpoint/config/params原pin与完整counters按规范；size50m、vision64/camera2/repeat1/propriofalse、native BF16、原RSSM/unimix.01/native symexp TwoHot，DL/rep off。旧MC代码仅复用factory/checkpoint loader，metadata分别记录B外层版本与旧loader归属，没有复用legacy科学输入。
- 4块×32 future paths、chunk8、每条20步；实际512chunks、4,587,520 latent/reward步。官方只读private-root imagine；future seed20261024/统计20261025；固定四初始sample不重抽，branch/control共noise，生产policy RNG/counters不推进。
- Python3.11.16、JAX/jaxlib0.4.33、Ninjax3.6.3、NumPy1.26.4、ml_dtypes0.5.4、Mujoco3.15.0、dm-control1.0.48。环境完整配置、设备及命令在plan/运行收据。

## 资源、完整退出与工程证据

只用**sv3物理GPU4一张独占A100-SXM4-80GB**。00:45:15.649498北京时间fresh检查4–7各0MiB，无compute/pmon作业；5–7未安排本B。启动record再次查实际idle/UUID，CUDA_VISIBLE_DEVICES为GPU4 UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23`，进程内CUDA0；实际初始化与终点CUDA身份/独占核通过。EGL环境physical4及旧zero-action同卡proof经过hash/绑定复核；本B无renderer，不称新EGL绘图验收。

运行00:45:47.826572–00:46:33.669653北京时间，父PID1976247、worker1976253：父完整入口**45.802874845秒**，外部CLI终点45.843115024秒，exit0/timed_out false/worker terminal/ownedgroup cleanup/fullchain verified。固定1800秒硬限，含来源/加载/编译/预测/统计/写盘。采样显存峰值**2505MiB（约2.45GiB）**；约`.0127 GPU小时`按单卡完整入口折算，不是GPU kernel有效计算时间或整体预验证预算。0新physics/render/actor/optimizer。

Windows release54项：50pass/4Linux-only skip，10.495秒。相同确定SHA的Linux**54项全过，无skip，19.993秒**；pip check与独审小selftest通过，工程入口合计21.489秒/0GPU。覆盖实际tiny官方phase/unimix抽样/私有seed/attempted-write拒绝/初始sample固定/q=p/common keys/prefix、BF16精确回转、F64BC偏差及零反例/LOEO/分层权重/CI/停止、源白名单/对象字段、Linux入口解析/metadata-helper/owned后代/无关sentinel/验证与末次写盘超时。不把fixture写成真实任务或方法验收。

实际参数/optimizer/norm/counters整个checkpoint state前后hash均为 `e666c93929194015303e7138fa0c92414c96482dc2a0dd1de450efec8d532cb4`；checkpoint/config输入与24相关源码Git blob及源文件终点一致。模型实际read keys仅dyn/rew。原near-uniform极大symexp bins tiny浮点失败证据保留，既定容差未放宽；实际首chunk8960点全过，F32native对独立F64对称均值最大差`1.0017648099935883e−7`，门为`1e−6+1e−5*abs(reference)`。

独立CPU审计：进程1979800，完整auditor2.568799903秒、父2.807263974秒、外部terminal2.807582287秒，120秒全入口、exit0/fullwrite/ownedterminal/0GPU。全部path的F64统计用独立residual off-diagonal U-stat复算：84,487numeric、4,767exact字段、212零侧条件，最大差**1.6653345369377348e−16**；选择/CI零侧/分类、327,680 root keys及327,680 sample keys字节完全匹配，首chunk与完整episode前缀相同。key枚举复用JAX CPU PRNG primitives，不称独立Threefry/Ninjax算法；首chunklogit oracle不称全部模型独立重演。

通用父不授予实际B-worker来源：其actual reads/GPU保持unknown、real B artifact chain=false；审计自身正文及父verification明确八源/八新预测数值读取、实际GPU0。继承的B planned-GPU1字段不是审计资源分配，不能据此统计CPU审计用了GPU。验收同时依据CLI exit0/完整120秒/已核文件链，不能只读`status=passed`。

## 科学结果与停止

主量是逐初始sample的控制中心化signed BC-MSE，扣sample variance/N，不截负；再平均四sample/state/episode。选择效用单独计算，global max−value≤1e−10取最低control ID。下面均为固定N128，八已见calibration episode的2000 whole-episode bootstrap名义区间，非blind确认或正式显著性结论。

| 主量（越小越好） | 数值 |
| --- | ---: |
| q BC-MSE | .057380510939101834 |
| p BC-MSE | .05809063287541952 |
| LOEO常数MSE | .03806762950408622 |

| 必要误差改善 | point | 名义CI95 | epsilon | 固定epsilon门 |
| --- | ---: | --- | ---: | ---: |
| p−q | .0007101219363176845 | [−.0033021810180138213, .005420749282689339] | .001558142009261399 | .0004 |
| constant−q | −.01931288143501561 | [−.03759327825852403, −.003038931068385313] | .000966220659000893 | .0004 |

q相对p的描述性误差改善很小且区间跨零，q比常数误差高；两个有限MC可靠性指标均超原绝对门。不能绕过`any unstable→MC-inconclusive`的优先级，将常数方向的负名义区间改判成`not_supported`。没有必要双阳性支持，主分类保持**MC-inconclusive**。

| 真实控制选择效用 | 数值 |
| --- | ---: |
| q | 5.555312310815943 |
| p | 5.563286766340461 |
| LOEO常数 | 5.7131551921454875 |

| 必要选择改善 | point | 名义CI95 | epsilon_U | 固定epsilon_U门 |
| --- | ---: | --- | ---: | ---: |
| q−p | −.007974455524517544 | [−.0724214570331801, .04126858074881843] | .012434810575042591 | .002 |
| q−constant | −.157842881329545 | [−.30241354860997366, −.03325481990869958] | .012133891435632815 | .002 |

选择分类也为**MC-inconclusive**；描述性负方向保留，不能以MSE或负区间替代预定选择判据。32/64/128、两half、四block、全部choice一致比例和真实switch代价/极值/MC correction均保存在result，未筛选状态或clip尾值。有限epsilon不是严格积分误差界。

已按规范停止B：不增L/block/seed/H/控制/episode/checkpoint，不重算挑选子集，不进入blind/gate/训练。工程完整但科学不确定是正常完成状态，**不需要为寻求阳性重跑**。如果发现新的实现缺陷应另记证据/工程影响，不能改原门掩盖失败。

## 证据位置与同步

服务器基目录 `/data/Policy_Discrepancy/runs/dl-production-reward-20261012-d685a2d/`：

- `cpu/{tests.log,pip-check.log,audit-selftest.log,summary.json}`。
- `resource-startup.json`、`gpu-preflight01.json`、`gpu-postflight01.json`；`launch-attempt01.{log,json}`与`launch-independent-audit01.json`。
- `cp300000-attempt01/`：budget→plan→result→production_reward_complete→supervisor_complete；八episode NPZ、first_chunk_oracle NPZ/check JSON及production_reward.log。
- `cp300000-attempt01/independent-audit01/{audit.json,budget.json,production_reward.log,supervisor_complete.json}`。

本机ignored镜像：`analysis/outputs/dl-production-reward/dl-production-reward-20261012-d685a2d/`。31文件逐一与server hash/size相等，清单SHA `94c63a491ad79d0fca85dff26aba4dc073f1986ee0a61494dc8a3142b3c4046c`；archive SHA `aaf5521d714e1c1cd2e5f7119e169e4a1cda282c8430de3cabc4620d8c1c1225`。独审正文SHA `14271fd4fe999f6fcb17f6824bcefdb27a49e02829a9eafad6aa6c33730c092d`，主result SHA `c9162129f6251e9881d379bef21cbf0da16b0b42dc8199cf56a94b70b9d917dd`。所有原始产物不入Git。

Operational helper位于bootstrap目录，CPU helper SHA `69b5b9ffa9e5ca86bafcc195bc5217577335f652545178a2ac81774670b0bbd4`；GPU terminal observer `fccd97ef2127cd9b82b48c3b30764effd253ec7639b1e293b315cffb7dfb1305`；CPU audit launcher `f30d430d8a866e0d042c030ee74274eb3e8c624fb29d91fd3191bc35730bf112`。完整实际命令保存在上述launch及plan，源码/auditor固定在d685提交。

00:51:19.592302北京时间fresh终点检查：4–7各0MiB、无compute/pmon作业；本B GPU4已释放，5–7没有本B作业。这是带时间快照，不保证以后持续空闲。研究分支d685已push并远端SHA核验；未main，后续文档提交同步状态以最终Git收据。

## 证据限制与下一任务

B只描述已见cal32/四固定初始接口、当前RSSM＋head、七控制和固定zero尾的H20条件均值/选择；H20超过当前imag_length15。不能否定H1 head整体技能、raw latent KL、latent可学信息或未来actor学习路径，也不能将工程/数值一致升级为当前RGB新增信息、normal active rep损伤、保护或性能证据。A的有限拟合readout探索性增量及原H20/H100/legacy裁决保留，不互相改判。

**首选继续当前完整授权chat，先做0GPU协议分析**：独立识别“当前观测相对共同历史新增、后验可学习且可能受active rep压缩损害”的决策信息。完成标准是明确共同历史及合法信息干预、观测可获取性、独立外部动作成本、normal rep因果作用与充分/必要证据、唯一有限对象及停止规则；不得按D/K/head结果挑有利状态。可见cue/paired-history、第二信号对D的独立增量、正常压缩损伤及保护、强度/conditional-shuffle/reward-only/OOD、后续学习性能仍未完成。候选需另审定，不把B不确定改成新gate/训练许可，不自动读blind。

资源建议仍是已有规模模型诊断**1独占A100串行即可、2卡仅用于相互独立环节并行**；sv3 GPU4–7现有分配足够、无需全占。本对象实测约.013GPUh/上限.5GPUh；完整剩余GPU小时与新训练显存未定，不能沿用旧20.4GPUh条件外推或本项2.45GiB作为整体预算/保证。完整方法目标仍未完成。
