# DL-r4 当前动作与记忆效用诊断

版本 **DL-r4-utility-r1**，2026-10-15。状态：**Agent 在完整预验证授权内采用，工程实施中，真实对象未执行**。这不是用户逐项确认的算法，也不是正式训练协议。用户授权的计算范围为 sv3 物理 GPU4–7；本对象计划仅用 GPU4 独占一张 A100，5–7 不预占。原 M2、旧阴性/不确定/失败记录均保留。

## 主张与识别边界

前置证据为 [DL_R4_IDENTIFIABILITY.md](DL_R4_IDENTIFIABILITY.md) 的有限反例及 [DL_PILOT_R3_ANALYSIS.md](DL_PILOT_R3_ANALYSIS.md) 的实际 N1M 分析。它们已说明 H1 支持可能漏掉延迟依赖、现有 N 的释放很弱；未证明真实任务中的信号辨别、压缩损伤或保护收益。

本对象的工作假设是：高 D 的普通历史位置中，当前观测对当前 actor 的有限闭环作用可能不同；H1 e 或 raw latent K 可能在 D 之外提供预测增量。反例包括：三臂均无可辨差异；q 使 actor 更差；仅动作改变有作用；H1 对延迟作用无增量；e 只代理时间；MC 或渲染误差与效应同量级。失败不等于信息不存在或应被压缩。

三臂首下标表示**首动作**来源，次下标表示**本帧潜在记忆**来源：

| 臂 | 首动作 | 当前 latent | outgoing prevact |
| --- | --- | --- | --- |
| qq | q sample actor | q sample | 自己实际 q raw |
| pq | p sample actor | q sample | 自己实际 p raw |
| pp | p sample actor | p sample | 自己实际 p raw |

共同 h；q/p 使用真实 categorical/actor API 与相同的实际 latent/actor subkeys。qq 与 pq 的当前记忆须逐字节相同，pq 与 pp 的首 raw/applied 动作及首真实奖励须逐字节相同。之后三臂都正常接收**自己的未来真实 RGB**、使用 posterior 和自己的上一 raw action，不持续屏蔽观测，也不喂另一臂的未来帧。

对固定 H=100，定义 G_am=sum(real suite F64 reward)/100，真正终止后吸收零；不以实际存活长度替代分母。主量 U_obs=G_qq−G_pp，辅助 U_act=G_qq−G_pq、U_memory=G_pq−G_pp。三臂足以得到这些量，不能识别 action×memory 交互；第四臂 qp 仅在后续明确要估计交互时有必要。本对象不是信息最优读出，也不测 rep 更新后的保护收益。

## 输入、固定采集与隔离

- 实际模型：已验收的 quadruped_walk / N / training seed0 / size50m / vision64 / camera2 / repeat1，原 BF16 参数与 policy；不更新模型、optimizer、normalizer 或公共随机计数。
- 源完整 checkpoint：sv2 `/data/Policy_Discrepancy/runs/dl-pilot-r3-20261013-13ec72a/dmc_quadruped_walk/N-seed0-attempt01/ckpt/20261014T005815F054475/agent.pkl`，文件 SHA256 `c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24`，params `cffc13c0dcdfbd0c8b0e5c503c9d6251f4b1550c08b85a8742a638655d66d372`，294 leaves。输入完整状态与标准 load 后的 save 做 typed-byte digest 核验；此 cp 的 batches=updates=249493，不能泛化为任意 cp 的 counters 都无改变。
- 源配置：`/data/Policy_Discrepancy/runs/dl-pilot-r3-plan-20261013-13ec72a/config.yaml`，SHA256 `877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377`。只改变执行平台/输出路径/不预编译等记录明确的诊断设置，不改模型、梯度、训练科学设置。冻结模型，无新增训练。
- 32 个新正常 episode，env/eval seed=4096+episode_id，id0–31；calibration=id0–7，held=id8–31。沿实际公开 eval policy 采集各1000步，公共 eval seed/counter 在每次采集后恢复。新 seed namespace 不将已开旧留出重新命名为 blind。
- 全部普通候选位置固定 t=64,128,…,896 共14个，既排除 reset0，也保证100步窗口不越过原1000步 clock。每位置记录 D/K/e 及 mode/sample 区别；所有候选都保留，不由 U、e、K 或结果挑选。
- 每 episode D 降序/position 升序取 top2，再并入固定 t512；重合不补选，最多3状态。**主训练和主评估仅 D-top2**，固定512只作辅助，用同一封口预测器，不重拟合。top2是episode内相对高D，不声称总体绝对high-D或普通replay总体。
- 每状态16个独立私有 root，分支共用 root/subkeys，命名不依赖臂/运行顺序。source 与 branch 命名域分离。有限16-root均值不是状态期望真值；不能为阳性增 root、episode、窗口或换 cp。
- 三臂按预定6种排列循环执行，顺序不由结果改变；每状态 pair0 额外完整 qq 重复作零效应控制。共享真实 source 当前 RGB；同一 live 环境对象恢复完整 physics/task/RNG/clock/wrapper 状态并核视觉 model 身份。未来实际 render；**不声称共同 policy key 能保证严格配对像素**。旧±1 RGB问题及原精确比对失败不翻案。
- H1 e 严格沿 dt_latch 的 q/p modes、source recorded raw action 和下一真实 reward；不按各臂不同 action 的 reward 重标 e。e与U的条件对象不同，失配不自动判 head 错误。

## 统计、判据和停止

cal8 完成并验收后，先冻结输入 SHA 和预测器，才采集/开启 held24。主拟合按 episode 等权、episode 内 top2 各1/2。固定 ridge：mean weighted squared loss + .1×||beta||²，截距不惩罚，标准化仅用 cal；不搜 λ。

source 的全部14候选及每selected状态branch pair0的初始qq均实际调用公开policy验raw/完整carry；不是其它15roots或全部未来步的公共policy独立复演。后续每步用各自cached-own-RGB再算私有原policy，和physics复放分开。

固定比较器 D+t、D+K+t、D+K+e+t、e+t，其中 t=position/1000 是共同时间控制。主比较是 held top2 的 D+K+e+t 相对 D+K+t 的逐 episode 等权 MSE 改善；D+K 相对 D 等联报。bootstrap10000/seed20261028 按完整 episode 采样，不能把位置/MC当独立 episode；名义95%区间只描述这一固定探索程序，不含重新拟合、模型或训练 seed 不确定性。连续 signed U 为主，不以逐状态均值符号当真标签。

t仅线性协变量，不排除e代理非线性时间、轨迹质量或阶段；增量通过也只支持这个固定线性比较程序，不称已排除时间混杂。N1M是高回报晚期单模型，选择它针对旧欠训练actor限制，但任何结果都不能归因r3早期慢学或训练中保护收益。

实用尺度 δ=.01（100步平均 suite reward 的1%）事前固定，仅作本诊断的描述尺度。逐state t(df15)名义区间据此分 exploratory_useful / harmful / practically_small / unresolved，不做多重检验后的真值宣称；前8/后8 roots 一致性与 MC SE 联报。

有限测量资格：所有 selected 状态 pair0 的 |G_qq−G_qq_null|≤.0025=δ/4，所有工程身份/physics/control/cached-policy审计通过。单重复不构成全MC渲染误差界。任一失败即 measurement-inconclusive，保留差异，不放宽、不换顺序重跑求通过。

有限覆盖前置：held top2 的主 U_obs 至少4个不同 episode 含 exploratory_useful，且至少4个不同 episode 含 exploratory_harmful 或 exploratory_practically_small；这不是检验功效证明。只有测量资格、覆盖及主增量区间下界>0同时满足，才称本有限对象 exploratory supporting。否则按实际为不确定/本固定程序未支持；始终联报连续结果，不把跨零写成等效或无信息。任何结果都不自动进入保护训练。

## 工程验收与资源

必须先通过：真实 tiny BF16 public policy 的 qq raw/after carry byteexact；真实 subkeys/distribution复算；三臂 action/memory/prevraw反例；未来posterior/自身RGB；错误namespace/SHA/时间索引与state写入拒绝；完整冻结；physics F64 reward/control/fullstate 重放与 cached-policy 重放分开；父进程全入口期限及自己创建的process group清理。不以fixture通过冒充N1M真实对象通过。

最大主分支动作 32×3×16×3×100=460800，额外null≤9600，source32000；真实physics和cached-policy复放另列，不能隐去。建议占卡硬上限：source1800秒、qualification1200秒、labels18000秒，总21000秒≈5.8333 GPU小时（探针、失败成本另记）。这是固定对象的停止预算，不是ETA或完整科研总预算；较前一资源答复的约4小时建议增加，原因是本次已明确MC16及H100。已实测单项吞吐不能保证本新入口按时完成。

每次真实启动前重新确认 sv3 physical4 的 XML+pmon 空闲及 CUDA/EGL UUID 映射；冲突就保留证据等待，不自行换卡或干预未知作业。实际执行使用 clean Git SHA 的 sv3 隔离 checkout。checkpoint、images、rollouts 不入 Git，结果存 `/data/Policy_Discrepancy/runs/dl-r4-utility-<date>-<sha>/`，失败不覆盖。当前没有真实科学数据或GPU执行结果。

## 后续必要证据

即便本对象支持 e/K 增量，仍需正常 active rep 对任务信息/actor 的损伤和选择性保护、强度匹配、conditional-shuffle、reward-only 等对照，以及新训练 seed/任务性能。当前actor负效用与信息可用性必须分开；若本对象不支持H1，依据已固定反例考虑延迟/动作后果信号的另立规范，不能仅改阈值让 gate 触发。完整目标保持 active，不能以本规范、fixture或一次诊断结案。

## 工程证据（执行前）

Windows六组首轮50项49通过/1 Linux进程组skip，23.726秒；独审新增静态模型身份反例后，最终独审10项单独8.288秒全过（此前全组含9项独审）。真实tiny BF16/实际CPU物理与stub边界如上，不是N1M实际验收。四模块Agent交叉审查已修复F32/F64、观测标量容器、trusted-pickle hash图、bundle输入来源和阶段期限；尚需确定提交Linux全组验收。

2026-10-15 14:58:09北京时间，sv3输入目录 `/data/Policy_Discrepancy/runs/dl-r4-utility-inputs-20261015-0655/` 的checkpoint513050093字节、config4927字节均与上述原SHA匹配，`transfer_verified.json`保存收据；仅复制/hash，未解码或模型执行。15:01:16 XML+pmon确认sv3 GPU4–7各0MiB/无作业，0–2未知作业未干预；这只是准备快照，启动前须重新核查。canonical checkout仍d9be5574878d2c3f117cc6667be844f9a527eb84，没有服务器源码修改。

确定提交 `d61d8ae45b380dd4e54907e5669ce3cb731749f3` 在 sv3 隔离clean checkout的51项Linux检查全部通过32.629秒，pip check0，完整检查34.724秒/0GPU。真实tiny及CPU fixture范围不变。超时反例的两个管道ResourceWarning通过显式close修复，不改变科学语义；修复版确定SHA及相关复验另记录。d61d8ae已推研究分支，main仍595e982f1a3939775879728009e2105ee516e86d；未开始真实GPU诊断。
