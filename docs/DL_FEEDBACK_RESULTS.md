# DL 正常反馈有限对象：结果与证据

2026-10-11；科学协议 `DL-feedback-r1`，实际工程 `DL-feedback-engineering-r2`。**完整运行、独立产物统计和独立官方CPU物理审计均通过；冻结平均控制机会裁决为 `exploratory_average_opportunity_not_supported`。停止本有限对象，不追加样本、RNG、H或checkpoint，不进入gate或训练。** 这不证明所有条件子集、训练中持续KL压缩或后验保护路线无效；完整方法目标仍未完成。

协议见 [DL_FEEDBACK_DIAGNOSTIC.md](DL_FEEDBACK_DIAGNOSTIC.md)，来源见 [DL_PRODUCTION_SOURCE_RESULTS.md](DL_PRODUCTION_SOURCE_RESULTS.md)，失败历史与版本同步见 [DL_VALIDATION_LOG.md](DL_VALIDATION_LOG.md)。科学选择是用户完整前验证授权内的Agent决定，不是用户逐项确认或正式M2新协议。

## 1. 实际版本、输入和运行

实际代码与clean server CPU测试SHA均为 `0d49c589b525ba0c26f9f85bf52c23573dc19507`。只同步sv3独立clean worktree `/data/Policy_Discrepancy/dl-worktrees/0d49c58`；canonical repo保持干净 `d9be5574878d2c3f117cc6667be844f9a527eb84`，未触及sv1/sv2或正式训练。代码只修source单帧0D数组容器并增加反例，科学r1及所有精确判据不变。此前bd0af3a工程r1 attempt01失败记录保留。

| 字段 | 实际记录 |
| --- | --- |
| 对象 | `dmc_quadruped_walk`，baseline训练seed0、300000训练动作阶段，size50m/64×64 RGB/propriofalse/camera2/repeat1 |
| checkpoint SHA256 | `2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9` |
| 参数digest | `14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2` |
| config SHA256 | `11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445` |
| 来源 | f2d24da1831da2ffca22d313e7f534c96fc0d3eb production source attempt02；collection seed20261020，calibration IDs `[9,14,15,18,20,24,30,31]`，24blind数组未读 |
| 状态与随机流 | positions `[1,300,600,900]`，32状态×4paired RNG；private20261018/stat20261019；q/p共用实际NJ latent/actor消费位置 |
| 干预 | 首步q当前RGB posterior与同h真实prior p；此后双方接收各自正常RGB和posterior，不永久遮盲 |
| 真实奖励与统计 | suite实际F64 signed H100sum，F32联存；八episode等权，2000整episode bootstrap，linear名义95%区间 |
| 资源 | sv3/lyg0326，物理GPU4 A100 SXM4 80GB，UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23`，进程内cuda:0，EGL物理4 |
| 时间与退出 | launcher 18:21:44–18:31:01北京时间；父监督537.614524秒/exit0/非超时/自身group清理，worker1835714 terminal；结果写盘前elapsed534.930052秒 |
| 硬预算／显存 | 父监督3600秒；1秒GPU采样521点峰2587MiB，非连续峰值上界；父监督约0.14934 GPU小时，不含资源探针及后续CPU审计 |
| 训练状态 | optimizer更新0，参数/optimizer/norm及原counter冻结；无alpha/c/shuffle新训练设置、无gate变化 |

原counter前后均为updates74493/batches74493/actions18769/eval_actions1001/eval_seed3404107006；actions是policy calls，不误称训练动作预算。只推进私有命名诊断RNG。七依赖实际为Python3.11.16、JAX/jaxlib0.4.33、NumPy1.26.4、dm-control1.0.48、MuJoCo3.15.0、elements3.22.2、ninjax3.6.3，完整配置和版本以plan/resource为准。

完整产物根：

```text
/data/Policy_Discrepancy/runs/dl-feedback-20261011-0d49c58/cp300000-attempt02/
  feedback/{budget,plan,result,feedback_complete,supervisor_complete}.json
  feedback/episode_{009,014,015,018,020,024,030,031}.npz
  feedback/independent-feedback-audit-r2.json
  feedback/independent-feedback-physics-audit-v1.json
  {resource,resource-device-probe}.json
  gpu-samples.csv / inventory-start.xml / pmon-start.txt
  independent-artifact-audit.log / independent-physics-audit-v1.log
```

完整数组/pickle只在server，未提交Git。本机小收据镜像 `analysis/outputs/dl-feedback/0d49c58/`，包括clean CPU、审计器自检及attempt02；attempt01镜像在bd0af3a子目录。

## 2. 工程与独立验收范围

- 新SHA clean server CPU28项通过：statistics9、actual tiny BF16 model2、真实quad/wrapper physics7、synthetic cached-policy replay4、supervisor6。Linux真实owned-group timeout及失败leader descendants清理均执行通过；pip check、source七版本/全部32artifact hash/仅cal8 decode通过。模型和renderer stub的fixture不称实际闭环验收。
- 本次32个实际size50m源RGB/pre-carry/source-eval-key上，公共生产policy与适配器q的raw/after-carry逐字节相等。256真实反馈branch、25600动作全部完整；同量stored-applied physics及own-cached-RGB policy条件复放通过，实际F64/F32奖励、raw/after-carry/proof逐字节检查，完整state/task/RNG/counter/wrapper误差最大0。192branch普通H100窗、64branch在H100末端正常time-limit；无提前终止、无吸收padding、无跨reset。
- 32个初帧只新render一次，fresh/source像素最大差为0或1（2帧0、30帧1），初state误差0。未反复render选图，没有完整renderer/闭环逐位复现声明；source/fresh差异与全部真实反馈RGB留产物。
- 独立NumPy/stdlib artifact/statistics审计passed：八artifact bytes、全部25600 planned keys、源pins/位置/carry/eval-key、数值dtype/masks/奖励编码/动作mapping/退出完整性一致；独立全部signed prefix、episode统计/2000bootstrap/分类最大算术差 `5.684341886080802e-14`。算术允许1e−10，CI零侧与分类严格一致，不用容差改变裁决。它不解码record pickle、不执行model/physics/renderer，NJ child key及物理clock只核收据。
- 额外独立CPU官方physics审计passed：先核完整authority/hash再只解码cal8 trusted records，256branch/25600步直接suite执行stored-applied；F64 reward/discount/F32编码逐字节一致，integration/restore误差0，task/RNG/counter/reset/flags及预期FromDM done一致。审计本体39.280067秒、CLI39.317794秒，120秒POSIX硬限，GPU使用0。脚本不import生产helper/JAX/model/render，也不执行实际FromDM wrapper；是同官方后端及既有独审作者的独立实现，不是第二物理模拟器或新source-policy生成。

独立artifact auditor SHA256 `b3a18990f3e9c29606e60216ea06fa5d022169ea1c8cae3f3bc39f39c47c6cbc`，server五基础/三完整synthetic/24重新hash篡改自检通过。physics auditor `54e5b5a24fc9971ef8f454979a1ee9a9bc47c536f52bcac42b4c50efd96ef1ab`，server12项selftest通过1.127189秒，含F64单ULP/F32相同、非有限、末time-limit/早timeout和吸收反例；ml_dtypes只注册可信BF16 pickle。

完整launcher及两项独审命令保存在bootstrap根 `/data/Policy_Discrepancy/runs/dl-prevalidation-bootstrap-20261010-ae8372e/`；本机ignored同名脚本及收据可复核。实际GPU入口为 `bash <bootstrap>/launch-feedback-0d49c58.sh`，launcher SHA256 `730e4d36b8413c8162acb53915d385390bcd4b7b676ad639cab558c8708b292c`。两项CPU审计的完整参数为：

```bash
B=/data/Policy_Discrepancy/runs/dl-prevalidation-bootstrap-20261010-ae8372e
R=/data/Policy_Discrepancy/runs/dl-feedback-20261011-0d49c58/cp300000-attempt02
P=/data/Policy_Discrepancy/envs/dreamer/bin/python
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu MUJOCO_GL=disable timeout 120 "$P" "$B/independent-feedback-audit-r2.py" --dataset "$R/feedback" --expected-git-sha 0d49c589b525ba0c26f9f85bf52c23573dc19507 --output "$R/feedback/independent-feedback-audit-r2.json" > "$R/independent-artifact-audit.log" 2>&1
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu MUJOCO_GL=disable "$P" "$B/independent-feedback-physics-audit-v1.py" --dataset "$R/feedback" --expected-feedback-sha 0d49c589b525ba0c26f9f85bf52c23573dc19507 --output "$R/feedback/independent-feedback-physics-audit-v1.json" --deadline-seconds 120 > "$R/independent-physics-audit-v1.log" 2>&1
```

以上记录实际命令；后续重验使用新的输出路径和日志，保留这两份已完成审计。

| 权威文件 | SHA256 |
| --- | --- |
| budget | `91e38a756d131ae279a8b8da0c6372422fafcd1ee49b1c76a88fc677fa81f4af` |
| plan | `f046120aa9a54fe52abdd6a9c86adaa679fc976239aff27f9304f90ca74e74f0` |
| result | `4592e363a54385a36e849f937fae259aba06890d0f5e11a86e44393135915309` |
| complete | `a4bfda1872553fb82706f90e79870d69cf69202ab1bd6885c4a3745c5917657c` |
| independent artifact/statistics | `86aad8aea9f90f5b054879ea5d6b4da6d69aac0500b133f366a03e823b9d384f` |
| independent CPU physics | `75d2917da0d0c0b7111370ebb0e3a2ab096ffba6d3983fbb53e1000037f1975d` |

## 3. 冻结裁决及分散性

主量是H100累计 `q−p` reward，未取正部或按存活长度归一化。

| 摘要 | 实际结果 |
| --- | --- |
| 八episode等权primary | −1.0046095186171677 |
| 名义95% episode-bootstrap | [−2.255450222882697, −0.1739823539403551] |
| 原冻结裁决 | `exploratory_average_opportunity_not_supported`（upper95≤0） |
| position1/300/600/900总体mean | −0.523096 / −2.658098 / −0.613528 / −0.223715 |
| RNG0/1/2/3总体mean | −0.157714 / −1.037728 / −2.272178 / −0.550818 |
| first-two / last-two RNG mean | −0.597721 / −1.411498 |
| 四leave-one-RNG mean | −1.286908 / −0.993570 / −0.582087 / −1.155873 |

八episode均值依次为 −0.251784、−1.029683、−0.114727、−4.925094、−0.059703、−1.533615、−0.118647、−0.003623。少数episode影响大，四RNG在单state的散布也大；完整u_pair/std/range和H1/10/50 signed prefix在result及独立审计中联报，不能把四路径均值当精确state正负标签。不得删除主导episode/位置/RNG或用描述prefix改变主裁决。

区间仅针对固定calibration、四预定随机流及此checkpoint的探索对象，八episode bootstrap只有名义口径；不声称罕见子集覆盖、完整policy噪声积分或跨训练seed泛化。当前结果支持停止有限平均对象，**不解释成“所有后验信息有害”或“保护压缩必然无效”**。未来正常RGB补偿、prior已足够、actor未有效利用、prior/actor表征支持差异等仍是待证解释。

## 4. 授权内下一决定与尚未采用建议

**Agent决定**：停止该平均对象；不启用24blind、不采用新gate、不进入短训练。先用现有calibration核camera/body/contact/历史可见性与任务语义，再定义独立外部关键对象。关键性不能来自D/K/e或四RNG中的偶然正号，必须有有限真实动作排序/代价、信息可获取性与现有actor利用的证据。图像查看仅是可见性初查，不替代三个前置。自然状态没有合适对象时才论证合法单transition有界物理挑战，并明确OOD/适用范围；其类型/幅度/控制集合/标签/预算当前未冻结，不启动。

**已完成的CPU图像初查**：独立导出器 `DL-calibration-image-export-r1`（脚本SHA256 `5a115a6624b8b4f561f125a2edd30a2dfc0a74be4d636ac50f4c819e928796a9`，server八项synthetic自检通过）核完整authority及8artifact hash，只读取四个numeric字段source_image/fresh_image/positions/episode；不解析pickle、U或模型数组，不重render。105PNG（64native/32pair/8episode/1overview）和frames.json位于attempt02的 `calibration-image-qa01/`，本机同名镜像。frames收据SHA256 `5b1a31e39e43c6c7007ce405f7009f3e9c776147649b5fca8865acddb9224329`，总览PNG `700048620689f42900c15a830d843e160fa8fdac26065b0a6cfb589fa752b222`。全部32pairs按固定顺序查看，nearest四倍只供视觉QA；原生64RGB保留。

**目测观察及未确认推测**：图中本体、足部及地面轮廓可见，source/fresh整体场景一致；部分图呈足部朝上或侧倾外观。后者只是目测，尚未用物理torso orientation/contact核验，不作为“已跌倒”标签或actor失效结论。单帧也不证明速度、接触力、当前动作排序或历史之外的新增关键性。下一可立即做的CPU任务语义核查可用官方torso_upright/torso_velocity等连续量，不以U/D/K/e选点；动作排序、信息可获取及actor利用仍需独立因果证据。

**Agent建议、未采用**：`DL-signal-increment-r0` 可比较同自由度非线性D-only、D+rawK、D+H1 reward支持的连续utility选择排序，并联报相对增量及selected绝对收益。每episode离线top2/4不是部署gate，也不匹配活跃压缩强度；拟议ridge/10800秒/24blind方案均未冻结。负平均不逻辑排除条件子集，但只有可能性不足以消耗留出或把原门翻案。

**理论事实及工作假设**：真正latent边际策略KL受data-processing界约束，当前mode条件Normal D不受单向rawK约束；K补充分布变化但非统计独立或任务价值证据。模式近tie可K趋0而D不趋0；背景新奇/弱prior可同时提高D/K但真实收益非正。D+K有额外辨别信息、保护有效active rep、排除强度/对应关系/reward-only和OOD、最终提高性能，均仍需证据。

完整方法下一阶段仍需：独立目标/反例覆盖、第二信号相对D增量、active free-nats压缩的学习介导证据、有效S/P/W与actor/OOD对照、未见数据和条件训练。当前工程和有限负结果不能替代这些环节。首选继续当前完整授权聊天，无需自动创建或发送其他chat。
