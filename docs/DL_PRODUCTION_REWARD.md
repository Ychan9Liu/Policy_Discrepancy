# DL 生产奖励均值诊断

版本 **DL-production-reward-r1**，2026-10-11。这是用户完整预验证授权内、Agent 科学与源码交叉审查采用的有限诊断协议，**不是用户逐项确认的算法**。2026-10-12 工程r1实施中，最终确定SHA/Linux验收及真实执行状态以验证日志为准；规范采用时尚未实现/执行。取代 RESEARCH/DL-24 的 B-r0 建议；旧 A、H20/H100、legacy 原结果与停止门保留，M2 正式协议不变。

## 问题、必要性与边界

检验：在同一实际历史 h 和已存 sampled q/p 接口下，当前生产 RSSM＋reward head 的有限 H20 条件均值预测，是否具有相对 p 及整 episode 留出常数的后验特异改善？此前 legacy 动作路径不能回答这一生产对象；A 的拟合 readout 不证明现成奖励模型有技能。

这是现成 reward-support 候选的必要诊断，不是 raw latent KL、可学习信息或后续 actor 学习路线的普遍前置。H20 超过当前 imagination length15；阴性只限制指定联合开放环均值估计器，不能扩大到 H1 head 无技能、后验无信息或整体保护路线无效。阳性也不能证明当前 RGB 超越完整历史、正常 rep 删除信息、独立信号可辨关键性、gate 有效或训练性能。

## 唯一对象与来源

- 正确生产动作来源 f1ae4d6513e81ed680992b64ba8ebade7d256d52 的 DL-action-information-r1：episode `[9,14,15,18,20,24,30,31]`、每 episode 位置 `[1,300,600,900]`、四已存 paired q/p sample。来源目录 `/data/Policy_Discrepancy/runs/dl-action-information-20261011-f1ae4d6/cp300000-attempt01/`。六发布 JSON 的固定 SHA 沿用 DL-nested-probe-r1 的 SOURCE_PINS，但不得 import 该 CPU worker（它清空 CUDA）。八 NPZ 的 SHA 从固定 result 取得，读前及结束前复核。
- 六字段惰性白名单：episode int32 scalar、positions int32[4]、h F64[4,4096]、latent F64[4,4,2,32,32]、reward_f64 F64[4,15,20]、first_commands F32[4,15,12]。`allow_pickle=False`；不访问/解码 records，不读 blind 数组。验证有限性、shape、精确 onehot、F64→native BF16→F64 字节相等。
- 只用前七首控制：zero、all+、all−、lift-LR+、lift-LR−、extend-diag+、extend-diag−，幅度 .25，逐字节核已存 raw normalized F32 向量。后19步 normalized zero；zero 的物理 lift 是 .05，不是零力矩。真累计成本严格 F64 `sum(reward_f64[:, :7, :], -1)`，与原发布 rewards 字节相同。
- 完整 physics/carry/key、实际 h/sample z 与无早 terminal 依据是 f1 固定实现、完整退出及独立来源审计的**认证复用**，不是 B 新物理验证。源原完整20步执行/flags 条件不能成立时为 semantic/source failure，不借真未来 flags 修改模型预测。
- checkpoint `/data/Policy_Discrepancy/runs/dl-prevalidation-bootstrap-20261010-ae8372e/sources/cp300000-agent.pkl` SHA `2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9`；同目录 config.yaml SHA `11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445`；参数 digest `14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2`，updates/batches74493、actions18769、eval_actions1001、eval_seed3404107006。可信 pickle 前核字节。按生产 spaces 无环境构造 Agent，size50m/vision64/camera2/repeat1/propriofalse/BF16/unimix.01/symexp_twohot，DL/rep off；参数、optimizer/norm、counter、policy RNG 保持不变。

## 模型时序、随机流与保存范围

每个已存 `(h,z)` 是当前状态；**不 observe、不再次当前 core、不重采样初始 z、不取 mode**。首 raw normalized action 推进一步 core→prior→实际 `_dist().sample`，取得下一 feature 后调用原 `feat2tensor/rew.pred()`；重复20次。不能输入 applied 物理控制或加入起点的过去 reward。

TwoHot 使用 physical symexp bins，原 `.pred()` 是对称求和的概率均值；各步 native F32 点预测按 F64 求和，再积分未来 latent。不是把 H20 标签喂一个 TwoHot，不是累计 return 分布/联合概率或 proper score。本协议不新增 CE 主量。

未来固定四 block×32 paths=128，真实 chunk8，每次一个 state、四初始 sample、两 branch、七 controls。root从 `PRNGKey(20261024)` 开始，先 fold tag `0x50525231`，再依次 fold episode、position、initial-sample0–3、block0–3、block内path0–31、future step1–20，全部uint32。path是块内绝对索引，不是chunk内索引或跨block的0–127。branch/control 不进入 key，同 q/p 及七控制共享噪声；各自边际正确，但不能保证方差降低。初始四 sample 只是固定接口重复，非完整 posterior/prior 边际积分；不调用生产 Agent `_seeds` 或改变 counters。各块 key 不随 chunk 划分改变。

采用官方 `RSSM.imagine(single=True)`，对每条轨迹/每步用显式私有 root 的只读 `nj.pure` 调用再 vmap。只读必须 `create=False, track=True` 并拒绝任何 modified/created 尝试，不能只凭最终数值相同判只读。该私有 context 的实际 `nj.seed()` 只消费本步 root，现有 Ninjax3.6.3 首次 sample key 为 `split(root,16)[1]`；保存 root 与实际 sample key，并核消费恰好一次。不声称官方 imagine 不调用 nj.seed；禁止它推进生产 policy/training 随机流。固定同一编译函数对所有 chunks 返回 native pred/logits/bins，仅首 chunk 写 logits，避免审计开关产生不同估计图。

保存全部 native per-path/per-step reward F32、精确 keys、bins、源 h/z/控制 typed hashes 和轴标签；raw reward 约18.35 MB。另保存预定首 episode9/position1 的首 chunk 原始 reward logits/bins，用对称公式核点预测。全部数值/统计可独立复算，首 chunk oracle 不宣称全部模型独立复现。实际批量 BF16 差异必须说明，不把 CPU tiny 分块一致写成真实50m字节不变。

## 主量：修正有限 MC 的平方误差偏差

对每条 future path 的 H20 累计预测先做七控制中心化，得 `X[l,a]`；真累计成本同样中心化为 `Y[a]`。对每个固定初始 z、branch：

```text
BC_MSE(N) = mean_a[(mean_l X[l,a] - Y[a])² - var_l(X[l,a], ddof=1)/N]
```

主量使用 N128，**不截断负值**。BC-MSE只估计该条件均值的平方误差，不评价完整奖励分布。不同 q/p future variance 会使 plugin MSE 含不同 Var/N；公共噪声不能消除此偏差。F64工程实现先形成 centered residual `E_l=X_l−Y`，再用 `mean(E)^2−var(E,ddof=1)/N`，平移不改科学公式，避免重复均值运算使数学零产生微小残差；不做阈值归零。独立审计用等价 off-diagonal residual U-stat 复算。原 plugin MSE、MC correction、未中心化 level MSE 单列描述。

每初始 z 先计算误差，再四 sample 平均→每 episode 四 state 平均→八 episode 等权，不先平均四初始 sample 后平方。两个必要有符号 gap 为 `BC_MSE_p−BC_MSE_q`、`MSE_constant−BC_MSE_q`。常数是 LOEO 其他七 episode 真 centered 成本向量的均值，无 MC correction，不看留出 episode 标签。whole-episode bootstrap2000、统计 seed20261025、linear percentile95；八 calibration 已观察、仅名义探索性区间，不作独立确认或正式显著性结论。

路径固定按 block0→1→2→3、每块 path0→31 排序：first32为block0，first64为block0/1，last64为block2/3，pooled128为全部四块；不择子集或顺序。32/64/128、first64/last64、四 block32均各用自身 N 和同公式；联报 path extrema、各 MC correction、block/half 数值，不 clip 极值、不保证未见 rare latent 尾部。每主 gap 的有限可靠性指标：

```text
epsilon = max(abs(g32-g64), abs(g64-g128),
              abs(gfirst64-glast64), 3.182*std(four block gaps, ddof=1)/2)
```

数值分辨率预先定为源 δ²/100=.0004 reward²；它是 Agent 事前标尺，不是理论最优值或严格 MC 误差上界。必要阳性还要求 point>0、epsilon≤.2×point、lower95−epsilon>0。any epsilon>.0004 → `MC-inconclusive`；稳定且任必要 upper95+epsilon≤0 → `finite_native_mean_not_supported`；稳定但未双正 → `finite_native_mean_inconclusive`；仅双正 → `finite_native_mean_exploratory_supported`。全部状态完成后停止此 B，不扩路径/seed/H/控制/位置/checkpoint。

## 外部选择独立联报

每个初始 z 用128-path 累计均值的最高控制，再以原真实成本评 utility/regret，最后平均四 sample/state/episode。沿原选择规则，在 `max(value)−value≤1e−10` 的全局集合取最小 control ID；科学epsilon/epsilon_U和独审容差均不改变此 tie 规则。常数选择上述 LOEO 向量最高控制，不拟合新 readout。

分别报告 q−p、q−constant 的 episode效用及同2000 bootstrap 区间。若声称有限选择也获支持，两个 epsilon_U 用相同四项结构，绝对门 δ/100=.002 reward，并要求20%正 margin及 lower95−epsilon_U>0；half/block choice 一致比例和真实 cost switch 代价全部联报，不删不稳点。预测主量双正而选择未双正，仅支持有限均值预测，不转成关键性或保护许可；选择双正仍仅 calibration 探索。

## 工程验收与资源

**sv3/lyg0326 物理 GPU4，一张独占 A100；1800秒完整入口硬限，最多0.5 GPU小时。** 总4,587,520 latent/reward步骤；0新 physics/render/optimizer。该上限不是实测耗时/显存保证，也不是完整预验证总预算。GPU4冲突即停止等待，遵守原资源规则。CUDA UUID/进程内0/既有EGL证明及本轮无render写入运行记录；启动前 fresh占用与映射，加载后核实际 CUDA context；当前未执行或核查实时占用。

使用独立 stdlib B supervisor，入口计时含校验/模型加载/编译/预测/统计/成功收据写盘；只清本次 owned group，失败目录不覆盖，partial 不给科研裁决。不得直接复用 nested supervisor 的 CPU环境/版本/验链。预算先写 planned reads，完整B验链后才写 actual8；链为 budget→plan→result→production_reward_complete→supervisor_complete。确定提交、clean工作区、相关源码Git blob及源输入前后hash一致。

必要工程证据：真实 tiny CPU 的官方 imagine 单步时序/实际unimix抽样/original reward pred、q=p公共key、future key/动作只影响相应prefix、已有初始sample未变、BF16无损回转、Ninjax无create/modify、私有seed每步一次且生产随机流与全部状态/counter不变；来源/轴/对象字段拒绝守卫；U-stat偏差修正、LOEO标签隔离、等权/身份重复拒绝、bootstrap/停止反例；Linux owned后代/无关sentinel/末次写盘超时。工程通过不代表真实50m或方法有效。

固定episode/state/initial身份与统计权重不因重复数组行变成更多样本；入口拒绝额外/重复身份。future MC的 BC correction 只对预定独立路径适用：root须按上述域精确重获，拒绝复制future path后增加N；control/branch共用root是唯一预定重复，不当作额外独立路径。

独立审计固定全部 source/hash/完整覆盖、全path F64累计/center/BC-MSE/U-stat、常数/选择/regret、episode统计和有限MC状态；首chunk TwoHot对称点预测单列。实际数值 oracle 容差在工程验收前另固定并注明浮点范围，不借调容差取得科学阳性。

**工程事前容差（2026-10-12、真实B前固定）**：首chunk native F32对称点预测与独立CPU F64对称公式要求 `abs(error)≤1e−6+1e−5×abs(reference)`；纯F64统计独审 absolute1e−10，选择/身份/key bytes/CI零侧/分类 exact。这些工程门不改变科学epsilon或tie。tiny近均匀极大symexp bins可因softmax F32舍入相消超门，此反例须保留，well-conditioned tiny通过不保证真实head稳定；真实超门保留失败，不能据结果调容差。

**工程r1交付与独审范围**：入口 `scripts/dl_production_reward_supervisor.py` → `scripts/dl_production_reward.py`，数值来源、模型、统计分别为同名前缀的 `source/model/stats.py`。`analysis/dl_production_reward_audit.py` 独立实现CPU F64 residual off-diagonal U-stat、完整报表、LOEO/选择/CI/有限MC分类以及私有key枚举，不导入三个生产模块；JAX CPU PRNG primitive复用须声明，不称独立PRNG算法。独审自身120秒完整入口/0GPU；独立parent监督及源/B输入前后hash，成功必须在父完整退出后复算，不给partial科研裁决。相关24源码/测试/规范文件在执行前后核Git blob与工作区字节；依赖JAX/jaxlib0.4.33、Ninjax3.6.3明确记录。生产model metadata说明只复用旧pinned factory/checkpoint loader，不复用legacy预测对象。首chunk原始NPZ与oracle检查JSON先保存再作固定门判断，以保留可能真实浮点失败，失败不写科学result/complete。

## 保留反例与后续依赖

q 信息可能可学但当前 dynamics/head 忽略；H20开放环和固定zero尾可能离开policy支持；两分支都差却q稍好；当前RGB只是历史重编码；四初始sample漏掉稀有信息；有限MC误差或BF16批量浮点差异；预测改善而动作排序不改善。当前标尺/八episode bootstrap 均不能替代独立真实性能证据。

本协议任何结果均不直接进入 blind/gate/训练。完整目标仍缺：配对历史的可见关键cue、第二信号对D的独立辨别增量、normal active rep损伤/保护、强度及conditional shuffle/reward-only/OOD替代解释、后续学习和性能。下一首选继续本聊天完成实现及确定SHA工程验收，再一次有限执行与独审；全部 missing 条件不得以 B 工程通过替代。
