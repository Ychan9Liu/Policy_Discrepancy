# DL 有限路径边际预测诊断

版本 `DL-predictive-MC-r1`，2026-10-10。**这是用户总体授权范围内由 Agent 决定的必要排除诊断，未采用为训练方法，也不是用户逐项确认的新算法。** 原 mode 诊断结果保持：300k 固定控制 H10 未支持 clean 相对 prior、clean 相对 gray-body 的必要预测方向。不能据此断言奖励头完全没有技能；生产 RSSM 在初始及未来 latent 上采样，而旧诊断逐步取 mode，因此仍需一次有限、预先固定的边际诊断。参见 [DL_PREDICTIVE_DIAGNOSTIC.md](DL_PREDICTIVE_DIAGNOSTIC.md) 和 [DL_REVISIONS.md](DL_REVISIONS.md)。

## 已决定的诊断对象

只用原 300k baseline checkpoint：文件 SHA256 `2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9`，参数 digest `14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2`，updates74493，原完整配置 SHA256 `11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445`。真实 size50m、vision64、camera2、repeat1、propriofalse、RSSM unimix=.01，模型参数/状态保持冻结。

原 H100 collection SHA `5765c893a419a59db46e5a3e3197af20b9059757`、seed20261012、32episode 中原八个 calibration episode，每个沿用16个预定 H100 位置，合128位置。只读这些 calibration NPZ；其余24个 episode 只核对完整 plan/complete 元数据，不打开 blind 数组、不选择替补位置。

每位置仍是六条原 H10 控制：记录 raw float32 Gaussian 动作序列、零动作、每维+.25、每维−.25、正负交替pattern、反pattern。记录动作不得裁剪或改写。真实 reward/controls 复用已完整物理复原并哈希绑定的 `be9e32aa018d8b61c0b2931fe44f72a6f480d4b5` mode 诊断，不新建环境、不执行物理步、不 render。固定五控制为主，记录控制单列；H1/H3仅描述，H10是唯一主窗。

mode result SHA256 `cc4ccf9e4842cd6fe2e0e94a85f7be2fceecb50996aa68d7fda134f87c778d2a`、plan SHA256 `4ac004571cd60211400a28d30c965556f2e127b1826e38b29db2fca6bce6b843`、独立审计 SHA256 `4bd7e91af939d12834f989999dcccc3080577563eee1f28c50b11f77282a99d5` 固定。执行前核对 source plan/complete、八个 source 与 modal artifact 的字节 hash、episode/split/seeds、预定位置、动作序列、记录 reward float32编码、repeat0、checkpoint/config bytes；全部成立且 Git clean 后才构造 Agent 并只读加载可信项目 checkpoint。真实五控制 H10累计跨度≥.1的原68位置/8episode须复核，不能换控制取得覆盖。

每个位置额外用不改动的 `modal.Predictor.predict`、同before carry/images/controls/history seed重演mode路径，保存logits、逐位置字节hash一致性及最大数值差。该审计共128次mode前向，包含在3600秒内、不新增physics；它不是额外主科学门、不设放宽容差。若mode重演不逐位一致，须保留浮点/历史重建混淆，不得把MC差异全部归因于去mode。

## 概率对象与随机流

重建历史保持原 seed20261012 和每 episode/frame 的原命名 seed。当前四图像共用干预前 carry、前一实际动作及同一个 h；四个 q 为 clean/backgroundA/backgroundB/gray-body，p只计算一次，共五分支。gray-body 保留轮廓、姿态、地面和影子，它不是完整运动学信息剥夺。

本诊断条件在这一固定、原样抽样重建的历史carry上，只积分当前及未来latent，未积分所有过去latent。保存初始h/qlogits/plogits的float32数值和实际compute dtype；BF16数值转float32无信息损失，但不冒称原生dtype字节dump，便于独立CPU读取。

初始 z0 从实际 q/p 分布抽样，每一步按同一预定控制推进 core，再从该步实际 prior 抽样 z，最后输出 TwoHot reward 概率。**所有初始及未来 latent 都采样，不残留未来 mode。** 复用 RSSM 的实际 `_dist`/OneHot.sample 接口，unimix恰好一次，不重新混匀。不同 sample/control/step 使用独立命名 key，同一 sample/control/step/factor 的五分支共享噪声；耦合保持各边际正确，但不保证降低任意非线性模型的方差。

MC seed20261017，四个独立 block，每 block32路径，主估计预先固定 pooled128。block/sample/control/step/episode/position 均进入私有 key 名称，sample分块不改变命名 key；不同CUDA/BF16 GEMM批量仍可能改变浮点前向，CPU tiny 分块一致不能证明真实50m逐位一致。历史/policy RNG不受推进；路径函数不调用 `nj.seed()`。真实本轮sample chunk固定8，每次一个位置，控制固定六条；不并行展开全部128路径，不择chunk寻找有利结果。32/64路径只作数值检查，不能按方向选择 L、key、prefix 或控制。

对每位置/分支/控制/步先计算 `mean_l softmax(logits_l)`，再使用原 bins 和准确 TwoHot target 求 CE。稳定计算使用 log-softmax 与 logsumexp。不能平均 logits 后 softmax、不能平均各路径 CE、不能平均正部；H10平均边际 CE不是完整 reward 序列 joint likelihood，也不是策略价值。所有原始 chunk logits、精确 keys、bins、各block概率sum及logsum保存到新 ignored 产物目录，供独立 CPU 重建；不把这些数组提交 Git。

## 主要统计、可靠性与停止

必要量仍为固定五控制、H10平均的两个有符号差：`CE_prior−CE_clean` 与 `CE_gray−CE_clean`。episode为统计单位，bootstrap2000、statistical seed20261016；主估计只用 pooled128。四 block 分别计算其 aggregate gap，并计算 first32、first64、pooled128、last64 的 aggregate gap。

六控制逐项报告同一H10差值及episode区间，均为描述，不选最优控制；记录控制与固定五控制合并主结果同时保留。

对每个必要量冻结有限可靠性指标：

```text
epsilon = max(
  abs(gap32−gap64),
  abs(gap64−gap128),
  abs(first64−last64),
  3.182 * std(four block gaps, ddof=1) / 2
)
```

它是有限数值可靠性指标，**不是严格积分误差上界或修正后的正式显著性检验**。两个epsilon均≤.001 nats才称 MC稳定；每个必要方向还须 point>0、epsilon≤point的20%、lower95−epsilon>0。控制敏感覆盖也须满足原校准诊断条件。原独立68个控制敏感位置单列描述，不代替全128位置主结果，不通过挑子集改变裁决；逐位置 half差异同样报告，不能以aggregate稳定宣称新gate排序可靠。

- 预算内 MC不稳定：`MC-inconclusive`，停止本次分支，不继续提高 L 或追加 key。
- MC稳定而两个必要方向未同时支持：停止当前 checkpoint/有限控制上的多步 reward支持候选；不扩大为“奖励头在所有对象上无技能”。
- MC稳定、必要方向及原控制敏感覆盖成立：只允许论证一个新候选，并冻结独立新数据和必要对照；不能直接训练或宣布正式准备就绪。

不改变 D/v/gate、alpha/rho、H、controls、数据选择和科学阈值，不拟合信号，不更新 optimizer。所有输出保持 `training_authorized=false`、`proceed_short_training=false`、`eligible_method_training=false`。

## 工程、预算与证据边界

最大工作量 `128 positions ×5 branches ×6 controls ×10 steps ×128 paths =4,915,200` latent/reward步骤，加原历史前向；零新增物理步和render。资源限定 sv3物理GPU4，单卡真实size50m，启动前 fresh占用和CUDA映射遵守现有guard。公开CLI用父进程监督单worker，**3600秒硬上限包含源校验/模型加载/编译/评分**，超时终止本worker并保留部分原始证据、新attempt目录和失败原因，不加预算或换卡。这是上限，不是实测ETA。峰值/实际耗时由根Agent实际监督记录。

模型直接按冻结生产DMC wrapper的spaces构造：image uint8[64,64,3]、reward float32、三bool flags、normalized action float32[12]范围[-1,1]；Agent配置传入与生产make_agent一致。不能调用旧 `d.load_agent`（它会为取spaces构造环境），也不调用make_env。完整配置/hash及参数digest复核确保此处只重建同一冻结模型；输入raw Gaussian动作仍保留，不因声明的action空间范围而裁剪。

脚本 `scripts/dl_predictive_mc.py`，测试 `tests/test_dl_predictive_mc.py`。执行输入为 `--checkpoint --config --dataset --modal-scores --modal-audit --output --resource-record`；`--sample-chunk`默认8、真实入口只接受8；底层CPU fixture可以检验其他分块的key契约。没有可调整的 L/H/seed/gate 参数。真实worker拒绝CPU；CPU只验证底层小真实模型和统计/来源契约。原始float32 logits未压缩约5.0GB，另有概率sum约0.3GB和logsum约0.3GB；实际NPZ压缩量和写盘时间须记录，不能从其数量推断硬预算内一定完成。

CPU验收需涵盖实际tiny Agent：q=p同噪声精确一致、实际unimix分布频率/不重复混匀、未来各步采样、私有seed重现与分块不变、无Ninjax seed/状态写入、未来控制不影响过去prefix，以及概率混合与平均CE的反例、稳定/不稳/缺控制覆盖停止规则、source/hash/axis拒绝。CPU fixture不能写成真实size50m验收，更不能写成方法有效。

本诊断即使阳性，仍无法排除共同模型错误、有限控制偏差、gray-body泄漏和校准数据复用；缺少独立控制价值辨别、真实压缩保护、有效S/P/W及条件训练证据。若该有限排除分支结束仍不支持reward代理，合法关键状态挑战或正常未来观测的closed-loop是新的研究对象，须另立适用范围/独立数据/预算和停止规则，不能无限延长窗口寻找通过。
