# DL 有限嵌套接口探针

科学版本 `DL-nested-probe-r1`，工程 r1，2026-10-11。此规范由 Agent 在用户完整前验证授权内交叉审查后冻结，非用户逐项认可的算法。实际执行前须确定提交、通过 clean Linux 工程检查；结果另记。原 H20 λ1 接口零增量及 H100 负平均、原停止门、M2 协议保持。

## 问题与解释范围

固定合成 assay 已给出稀疏已知线性信息的漏检反例，不能再把原探针阴性当成 latent 信息不存在。新对象问：下述**单一、预先限定的共同选参程序**，能否从既有 calibration 的实际 q 接口获得相对 p 与常数控制的选择效用增量？这是已看过 cal8 后的探索，不是 blind 确认，不检测所有线性族或完整历史之外的新 RGB 信息。

共同 λ 的理由是不用 q−p 最大化或两侧单独调参制造比较优势。它不能保证相同有效自由度（两侧特征谱不同），也会漏检：强 λ 时 q/p 效用均 .75、弱 λ 时 q=.9/p=.5，pooled 选则仍会选强 λ。因此新阴性只停止本程序，不否定整个固定网格、表示中的信息或后续学习路径。

## 冻结输入与边界

- 只用 `DL-action-information-r1` 实际提交 `f1ae4d6513e81ed680992b64ba8ebade7d256d52` 已完成的八 NPZ，源目录 `/data/Policy_Discrepancy/runs/dl-action-information-20261011-f1ae4d6/cp300000-attempt01/`。episode `[9,14,15,18,20,24,30,31]`、位置 `[1,300,600,900]`、四实际 paired sample、h4096、stoch32×classes32 不变。
- 先核发布的 plan/result/complete/budget/supervisor 和独立数值审计收据的固定 SHA256，再核八 NPZ 原始字节；`allow_pickle=False` 只读取 `episode/positions/h/latent/reward_f64` 白名单，不读取 `records` 或其它数组。既有 source BF16→F64、carry/key/动作生产路径及物理恢复是**复用已验来源证据**，本次不重新运行模型/物理验证。
- 只用前七预定控制的真实 H20 reward，严格按原 `np.sum(reward_f64[:, :7], axis=-1, dtype=float64)` 求和，逐字节对原结果成本数组；不用 actor 八项、D/K/reward-head、额外标签、概率均值或新样本。
- 全部八 episode 保留，零 blind 数组、模型/render/physics/optimizer/GPU。不修改旧产物、不重判旧门、不采用 gate 或方法训练。新目录发布完整退出与 hash 链，部分点不能给结论。

## 单一测量程序

1. 固定 mean-objective λ 网格 `(1., .1, .01, .001)`；这是三数量级有限敏感性检查，未证明最优或检测完备。8 个 outer LOEO，每个 outer 的七个训练 episode 再做 7 个 inner LOEO（训练六 episode）。任何 heldout episode 的特征、标签都不参加该 fit 的均值/RMS/截距或选 λ。
2. 各 fit 分开处理 h/z：训练均值居中、训练 total-RMS 缩放，严格零 RMS 则整组含 heldout 置零，不加 eps。七控制 reward 先减每行控制均值，再减训练列均值；列均值作不正则化截距。每 state 的四 sampled z 各占一行，共同 h 复制；inner 96 行、outer112 行，样本不作为独立统计单位。
3. F64 kernel ridge：`K=X Xᵀ/N`，无截断 eigensystem，`(K+λI)`；每训练分区/接口缓存预处理与谱供四 inner λ，outer 只拟合被 inner 选中的 λ。保留谱/残差与均值/RMS证据，数值验收固定绝对 `1e-10`，不能按结果加大容差。组缩放给 `trace(K)<=2`，λ最小 .001 时条件数上界2001只是数值可行性依据。
4. inner 每 λ、q/p 分别记录真选择 reward 和预测 MSE。先逐 sample argmax（`1e-10` 展示/预测 tie 取最小编号），再 MC、state、episode、q/p 等权平均。共同 λ 使用 **global tie 集合**：所有 λ 最大 pooled utility 的 `1e-10` 邻域；在该集合内取最小 pooled MSE 的 `1e-10` 邻域；最后取最大/最强 λ。不能用迭代近似比较、外层标签、q−p 最大值、强制非塌缩或阳性偏好选参。
5. outer 主量为真选择效用、regret 和 q−p/q−constant；MSE独立联报。常数控制只由该 outer 的训练成本选取。联报全部 inner candidate、tie 集合、selected λ、outer预测/choice/效用及预处理元数据；预计128 eigensystem、464 candidate evaluation，不计算 outer 全 λ 的事后最佳结果。
6. q−p/q−constant 使用原 whole-episode bootstrap：2000次、seed20261021、linear percentile95%，不把 CI 裁成0。这是固定交叉拟合分数的名义 calibration 描述；训练折重叠，未包括重拟合不确定性或独立确认。两 lower>0才有本程序探索性正支持；任 upper<=0为本程序不支持；其余不确定。沿用旧外部 δ=.2 控制选择依据，仅作已验前置，不重新选 δ 或标签；连续效应大小与 δ尺度联报。

精确复制所有 MC 行不改变 mean objective、训练统计或预测（除浮点误差）；增加独立 sample 可能改变 MC 近似，不能混称复制。测试须验证复制不变、outer heldout 特征/标签不影响 inner 选参、q/p 对换共同 λ 不变、whole-episode 隔离、严格零组、λ1 与原 ridge 数学等价、global tie 顺序无关及输入轴/有限性拒绝。工程 fixture 不等于真实方法验收。

## 执行与停止

仅 sv3/lyg0326 Linux CPU，NumPy1.26.4、BLAS/OMP 单线程；全父入口120秒，覆盖载入/import/Git核查/拟合/写盘/验链，清理仅自身新建 process group。该预算尚未实测；超时保留失败，停止，不扩大时间、网格、seed、维数或样本。实施/小 fixture/clean server 检查后完整对象只执行一次，独立数值审计另有120秒CPU对象，不读取 blind 或按结果改程序。

无论结果何种，均不自动进入 gate/压缩/训练。尚缺正确生产动作路径上的 reward-head 预测技能（旧 legacy 诊断不能裁决该对象），以及配对历史/可见当前 cue 的因果辨别、正常活跃 rep 的信息损伤、保护效果与替代解释排除。reward-head 技能是对应信号候选的必要验证，不是所有可学习信息路线的必要条件；frozen actor 未用信息也不否定后续 actor 学习。上述新分支须各立完整规范，不能把建议写成已批准协议。
