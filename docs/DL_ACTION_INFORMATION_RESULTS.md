# DL 有限第一动作与接口信息：结果

2026-10-11。科学协议为 [DL_ACTION_INFORMATION.md](DL_ACTION_INFORMATION.md) 的 `DL-action-information-r1`；执行源码、服务器CPU测试和真实对象提交均为 **`f1ae4d6513e81ed680992b64ba8ebade7d256d52`**，runner/physics/stats工程初版r1。本文仅记录这个有限校准对象，不改变原M2或H100反馈裁决，不是独立blind确认或方法有效性结论。

**冻结裁决：`do_not_advance_gate_or_training`。** 有限外部控制依据成立；两个接口及constant选择完全相同，接口选择收益增量不支持，actor首动作收益不确定。独立数值／源特征审计已通过，没有改变原门或裁决。

## 执行与输入

| 项目 | 实际值与证据范围 |
| --- | --- |
| 服务器／资源 | sv3 / lyg0326，CPU单线程；GPU0张，无JAX、模型、render或新actor RNG调用 |
| 来源 | 已验production feedback engineering-r2 `0d49c589b525ba0c26f9f85bf52c23573dc19507`；仅八calibration IDs `[9,14,15,18,20,24,30,31]`、位置 `[1,300,600,900]` |
| 控制／未来 | 七固定raw F32首控制，另八已存actual q/p首raw；共同19步normalized-zero，H20/.4秒；实际生产Clip/Unify/Normalize后的F64 applied独存 |
| 特征／统计 | actual同h BF16 `[4096]`、四paired sampled onehot z `[32,32]`，逐字节源绑定后F64转换；完整episode LOEO、train-only group RMS、mean-objective ridge λ1、2000 whole-episode bootstrap/seed20261021/linear分位数 |
| 外部门 | δ=.2，closed近优集合；敏感状态覆盖≥4ep且存在不同ep不相交近优集合；没有事后改margin/tie/库/窗口 |
| 时间 | 20:46:59–20:47:33北京时间；父监督33.9108223063秒，worker收据33.7754023112秒；180秒总硬限未超时 |
| 退出／完整性 | exit0、worker1890410已terminal、仅owned group清理；8episode/32state/480record，无partial或failed-replay文件 |
| 实际步数 | 主物理9600，stored-applied复放9600；均为完整20步window，无terminal/padding；restore/integration最大误差0，F64/F32 reward/discount bytes及完整task/RNG/clock/reset/done通过 |
| 工程测试 | sv3 clean工作区29项全部通过3.062秒，包括真实Linux进程组失败/超时清理及不影响无关进程；pip check通过 |
| 边界 | source文件及五published JSON/hash保持；不读24blind，不修改Dreamer参数/optimizer/counters，不采用gate或训练；实际FromDM wrapper未在本CPU对象执行，复放不是新增独立physics或模型复现 |

服务器目录：

`/data/Policy_Discrepancy/runs/dl-action-information-20261011-f1ae4d6/cp300000-attempt01/`

确定源码工作区：`/data/Policy_Discrepancy/dl-worktrees/f1ae4d6/`。原服务器 `/data/Policy_Discrepancy/repo` 保持干净 `d9be5574878d2c3f117cc6667be844f9a527eb84`；未同步sv1/sv2。源码通过本机Git bundle同步；本次首次GitHub push连接失败，最终推送状态见最新Git收据／交接，不据bundle推定GitHub已更新。

## 结果与解释

外部选择门成立：16/32状态有reward range≥.2，覆盖episodes `[9,14,18,24]`；存在18对不同episode不相交的δ-optimal集合。这支持这个七首脉冲/zero续行对象有实际状态选择要求。它不是当前RGB在完整历史之外提供新信息、actor能利用或任务全局关键性的证据。

q与p两接口的128个heldout sample选择，及八fold constant，全部为索引2／`all-`。两个selected-reward增量每episode都严格0，bootstrap上下界均0：

| 量 | mean | 名义95% whole-episode区间 | 冻结解释 |
| --- | ---: | --- | --- |
| q readout − p readout selected reward | 0 | [0,0] | 接口选择增量不支持 |
| q readout − train-fold constant selected reward | 0 | [0,0] | 不优于固定动作参照 |
| 实际actor q首动作 − p首动作 signed H20 reward | .0150941934522 | [−.0258798524745,.0544297596488] | `actor_first_action_exploratory_inconclusive` |

q/p平均selected reward同为5.7131551921，有限库平均regret同为.0835643374。q/p centered reward预测MSE分别.0367093270/.0367150627；这一很小的误差差不改变实际选择或事前门，不能另换指标宣布接口增量成立。全部原成本向量、近优集合、预测、sample选择、fold预处理收据、per-state/RNG/episode signed差及散布保存在result。

**工作假设，未获归因证据**：有限控制库和共同续行、constant动作优势、固定正则和线性族、只有32状态/四sample、历史已足够或当前观测未被模型有效表示，都可能限制检验。选择相同不等于所有latent信息相同、prior已完全充分、RGB无价值或保护后验无效。不能按此次结果调λ/H/库/样本/checkpoint后将新结果算原r1通过。

**固定输出的只读解释，非新拟合／裁决**：constant位于26/32状态的closed Aδ，另外6状态在库内regret>.2；最大regret .5081129010。q/p特征项均使OOF MSE低于截距单独预测的.0380676295，但均未越过截距的控制排序；最低最终constant领先margin分别.0086299563/.0081144224，大于原tie。独审的16fold/interface kernel `XᵀX/n` trace约2，最大eig/(eig+λ)为.20238–.23771。这是固定λ/RMS的实际收缩几何，不能证明收缩是失败主因；特征贡献非零也不证明新增RGB可用信息。未扫描λ或改变metric。原result代数QA收据 `fixed-readout-qa01.json` SHA256 `91de5cf37adff62a547f0e9ab1d82acb37d2a3ff2becbd7230ee0836ae059ace`，本机脚本／bootstrap同字节 `fixed-readout-qa.py` SHA256 `0b3094e439c0d3b3e28a30b0f70b2dba1a319f3e6b079569a55353dff94e3aad`；零新fit/physics/model/GPU。

**Agent授权内决定**：完成一次冻结对象后停止，维持不推进gate或训练、不启用blind；保留H100负平均与本H20actor不确定的对象差异。整体“独立信号辨别→active压缩保护→性能”的科研目标未完成。下一问题须先论证新信息相对完整历史的独立控制依据及合法可见cue；仅用D/K、偶然正号或低reward挑点不能满足。新因果对象、正常shared-loss压缩与后续actor学习均未由本对象授权启动或冻结。

两代理独立源码／科学审查一致：此刻不直接运行新physical challenge。**未冻结的唯一条件候选**为从可信 `p−1` complete snapshot、原raw/applied完成同一真实transition，施加预定有界torso world-y torque，检验camera2侧视pitch线索；在p与实际pre-policy carry/prevraw对齐。纯velocity修改可能在单帧不可见，teleport姿态可能改contacts，p再额外step却用p旧carry会错位，trackcom与近正立reward对称也可能使符号不可用。须先论证64RGB可辨、合法/support/OOD及实际不同动作要求，幅度／控制／门／预算未定；不增力或延窗求阳性。即时actor收益不是所有后续学习路径的普遍必要门，但正常loss压缩仍依赖独立可学任务信息和active rep作用证据。

## 独立数值审计与限制

独审脚本 SHA256 `32c610aacaca27a77708aa48b411acf9752580b044b42d8877095c9e578b1579`，服务器八项合成／重新hash篡改自检通过4.005秒，实际审计12.720443秒／120秒Linux总alarm，exit0／passed。先核公布feedback五JSON及全部cal8 NPZ字节，才解码256份calibration源record，逐first BF16 h／sampled onehot z／actor raw转换与新数组精确绑定。核新8artifact/schema/480record opaque envelope hash、完整退出／预算／版本／源链、production mapping／terminal masks／累计cost，并独立kernel eigendecomposition复算16fold读出与所有selected reward/regret／whole-episode CI／判据。最大算术差 `7.105427357601002e−15`，q/p prediction最大差均 `4.996003610813204e−16`；动作选择、分类及CI零侧精确一致，不用算术容差放宽科学门。

独审结果 `independent-numeric-audit-v1.json` SHA256 **`cbb99e43ad72a6960d86e5044455a18fecc38f068172defe60deb0d82f8678e5`**。它不import任何生产runner/stats/physics，不执行JAX／模型／render／新physics，不解新action-info完整record pickle；state真值、NJ子key语义和源模型正确性依赖此前已验来源／复放。是独立数值及保存源特征一致性复核，不是独立科学数据、完整新模型／physics复现。

## 原始证据及重演

| 文件 | SHA256 |
| --- | --- |
| plan.json | `474bad6adf6558b7a47c40711546dc4cf2adc870684b8cff2766a802605f44d6` |
| result.json | `06df6efa5f9ce84b428921b59fa508df8badd1dd54e30335bf2d3a273e585fad` |
| action_information_complete.json | `28481a749476be0b3e8c05f28e719cb22e4cfffb638e56c04d8f86128026d37e` |
| budget.json | `ddc8f43a3f4b57c11858be61f4c31315484c8c42b1e887f48391e71d077e2a68` |
| supervisor_complete.json | `a920a0191429c7ffd05af5a897a30839da47d046d9126a1002a4997928dc1ea3` |
| independent-numeric-audit-v1.json | `cbb99e43ad72a6960d86e5044455a18fecc38f068172defe60deb0d82f8678e5` |

完整8artifact hash、480主/复放收据及32初proof byte来源在result，实际六source文件hash在plan。小JSON/日志镜像在本机ignored `analysis/outputs/dl-action-information/f1ae4d6/`，不提交科学大产物或trusted pickle。

本机准备的完整launcher位于bootstrap `action-information-f1ae4d6/check-and-run.sh`，SHA256 `8c4f5871aa94b31079770583252cca322ef4f2d0d5b6da9b3c719f0bfa47412e`。CPU测试命令为：

```sh
/data/Policy_Discrepancy/envs/dreamer/bin/python -m unittest discover -s tests -p 'test_dl_action_information_*.py' -v
```

真实入口（目录必须不存在；不应重复运行同attempt）：

```sh
cd /data/Policy_Discrepancy/dl-worktrees/f1ae4d6
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /data/Policy_Discrepancy/envs/dreamer/bin/python scripts/dl_action_information.py \
 --dataset /data/Policy_Discrepancy/runs/dl-feedback-20261011-0d49c58/cp300000-attempt02/feedback \
 --output /data/Policy_Discrepancy/runs/dl-action-information-20261011-f1ae4d6/cp300000-attempt01
```

独审完整入口（output须为新路径）：

```sh
/data/Policy_Discrepancy/envs/dreamer/bin/python \
 /data/Policy_Discrepancy/runs/dl-prevalidation-bootstrap-20261010-ae8372e/action-information-f1ae4d6/independent_action_information_audit.py \
 --runroot /data/Policy_Discrepancy/runs/dl-action-information-20261011-f1ae4d6/cp300000-attempt01 \
 --dataset /data/Policy_Discrepancy/runs/dl-feedback-20261011-0d49c58/cp300000-attempt02/feedback \
 --expected-git-sha f1ae4d6513e81ed680992b64ba8ebade7d256d52 \
 --output /data/Policy_Discrepancy/runs/dl-action-information-20261011-f1ae4d6/cp300000-attempt01/independent-numeric-audit-v1.json
```

资源结论：这一项实测GPU小时为0。当前最低一张独占A100可串行完成已有模型诊断，两张可并行真正独立的环节，现有四张分配足够；未形成剩余全部前验证或正式训练的可信总GPU小时。过去source/feedback约2.5GiB、完整单步工程约6.5GiB是不同A100单项，不是训练显存或更小显存卡可用性验证。本轮未核当前GPU占用，不引用旧快照作当前状态。
