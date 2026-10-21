# 03 · Hopper seed1/2/3 baseline 与 Dt 补测准备

日期：2026-10-21，Asia/Shanghai。**状态：CPU准备验收完成，6条训练均未启动。** 用户本轮确认任务/seed/组别和资源，并要求先准备；尚无启动训练的指令。

## 用户与03的本轮配置声明

| 项目 | 确定设置 |
| --- | --- |
| 任务及组别 | 仅 `dmc_hopper_hop`，训练seed1、2、3；每seed logging baseline与Dt，共6条；不跑constant/shuffle |
| 范围标识 | `M2-v1-hopper-seeds123-baseline-dt`；新的受限复验范围，保留原M2-v1四任务seed0协议及16条结果 |
| 模型/观测 | 沿用原正式配置：size50m，clean、64×64 RGB、image=True/proprio=False，无debug覆盖或干扰 |
| 方法/梯度 | alpha20，free_nats1，dyn/rep=1/.1，ac_grads=False、reward_grad=True、repval_grad=True、repval_loss=True；原actor/RSSM/优化器/gate不改 |
| 训练链路 | batch16×64、replay_context1、16训练env、train_ratio256、CUDA/bfloat16；每条从头创建模型/optimizer/replay/env/counters，不分叉seed0或baseline参数 |
| 预算 | 每条100万执行训练动作，含预填充、排除reset/评价；repeat1，6条共600万训练动作 |
| 评价 | 每条0至100万、间隔5万，共21不可变快照；每点10完整episode，采样动作、独立评价流；共1260评价episode，按原1000动作horizon预计126万评价动作，另计不进入训练预算 |
| c | 用户明确每seed从自身完整logging baseline的更新开始动作 `[100000,300000)`、raw rep KL>1有效位置累计S/N；沿用原去重/非法停止规则，冻结并记录来源。本批c仅供审计，不启动constant，Dt不依赖c、不等待其产生；不复用seed0的c |
| 执行设置 | 延用已声明prealloc=False、JSONL+W&B project `PD_1`、无scope；每条唯一run ID与全新目录，不覆盖原产物 |

资源按用户指定sv3物理GPU集合1/3/4/5/6/7，由03给出一对一安排：

| 训练seed | baseline物理GPU | Dt物理GPU |
| ---: | ---: | ---: |
| 1 | 1 | 3 |
| 2 | 4 | 5 |
| 3 | 6 | 7 |

每卡同一时刻一条作业，不共享、换卡或增卡。该分配是本批范围，不沿用旧sv2四任务GPU0–7监督器。原16条正式任务训练SHA `48d71f1541721208322442f5a1d5977124f3e888`、已完成分析和hopper失败记录均保留，不能按新增seed结果择优删除旧结果。

## 新增工程支持及原流程回归

本轮初始HEAD `b3a4dd72ecd5c852be941de92ad66de59a6029ec`；原RESEARCH/STATUS候选归档及docs/proposal保留，不混入提交。

**实现/实际测试SHA：`0b764bd91e0bc145f8b0c4234c9fefc719ad96eb`。** 最终交付SHA见本轮DELIVERY/HANDOFF/回复；测试后仅文档，最终配置索引重新由干净最终提交生成并固定execution SHA。

- `dreamerv3/protocol_scope.py`、`m2_hopper_seeds123` preset及 `protocol_hopper_seeds123` 分派：只允许hopper、seed1/2/3、logging/Dt。原 `m2_v1`/`protocol_v1` 仍只允许seed0；新范围拒绝seed0/4、其它任务、constant/shuffle。所有动作计数、预算、评价、free-nats、loss/梯度和RNG派生代码不变，manifest明确记录新范围。
- `dreamerv3.freeze_c` 对新范围核对config/manifest seed一致、完整窗口/网格/阈值并写真实root_seed；原seed0冻结产物与修改前受信任实现比较，除生成时间外精确相同。新c仍须完整baseline及产物审计，无活跃/非法值停止，不填默认c。
- `analysis.return_metrics --profile hopper-seeds123` 固定仅hopper、seed1/2/3、baseline/Dt，沿用原始return AUC/固定100万分母与80/85/90/95/100万尾部五点；先各seed算指标，再任务内三seed算术平均，并输出三条 `paired_seed_differences`。缺seed/点/episode不完整，不补值或重加权。默认CLI仍为原四任务seed0。
- 新结果的 `overall_metrics` 仅是hopper单任务三seed汇总，**不是四任务均值**；主表不混入seed0。历史seed0可在报告中单独列示并披露诊断版本差异，不自动改成四seed主分析。比较只有Dt−baseline，不能因此宣称已优于constant/shuffle或裁决M2选择机制。
- `scripts.prepare_hopper_seeds123` 仅CPU解析/核对/写计划，无启动分支。六份生产CLI配置已解析并YAML保存，同seed仅mode/logdir不同，跨seed仅另差seed。输出包含48个训练env seed和630个评价env seed及首个评价key、replay映射、独立目录/参数哈希/审阅命令。各组同seed共用映射，不保证EGL图像或真实轨迹逐位确定。

本机27项统计/范围测试与sv3 **59项CPU测试**通过，无失败/跳过；包括678个env seed与实际runner映射核对、c原始收据半开窗口/真实seed/不可变与错误身份拒绝、完整6条手算指标、配对差值 `[1,2,-3]` 及不完整/默认profile/覆盖拒绝，还有既有梯度隔离、mode修复、shuffle跨CPU shard、计数/恢复边界回归。手算fixture没有真实百万步或正式c。

以新统计代码默认profile只读复算原16条：`--require-complete`退出0，3360 episode/336点评价，tables.json与原05复算逐字节相同，SHA256 `3d02f5cca66644877e55e45a79fd160949ff9b8d72c45920bf09e1235d75e9ef`。

以上是CPU/配置/产物验收；没有新agent优化更新、真实DMC rollout、GPU探针或训练。新范围的真实size50m CUDA启动/长期运行仍由04在用户授权后执行，不把fixture写成真实补测结果。此前mode语义版本2继续使用；原日志缺失的整向量值不补写。gate与输入微扰的因果问题仍未解决。

## sv3准备与资源证据

现场只读快照为北京时间 **2026-10-21 14:12:42**：sv3/lyg0326，指定GPU1/3/4/5/6/7均0MiB且XML全部进程类型及pmon未见作业；GPU0/2已有未知归属作业，不干扰。对其cwd读取受系统权限限制，未推测归属，未尝试提权。该快照不是预约或启动时状态；结束资源快照及时间另见DELIVERY/最终回复，04启动前必须重查。

| 物理GPU | CUDA UUID |
| ---: | --- |
| 1 | GPU-5b1ebede-99ec-d080-29b0-3c04011dc663 |
| 3 | GPU-4725722b-bd64-ec77-2fcd-9692cf3ee36e |
| 4 | GPU-9c070d5a-ac68-4c26-9522-a790ad874b23 |
| 5 | GPU-79704918-e096-92b4-252c-e3244497a8b3 |
| 6 | GPU-70db6c66-9917-e0e2-44db-0fc80fcf41b9 |
| 7 | GPU-c9a4f076-5f8a-8136-9535-7e74e14e895f |

每条计算只用对应UUID屏蔽后logical CUDA0。**EGL实际设备映射尚未验证**，计划中的`egl_index/MUJOCO_EGL_DEVICE_ID`与`launch_argv`均null，审阅argv不是启动授权。04获得启动指令后先验证计算/渲染PID→同一物理UUID，不直接假设EGL编号等于物理编号；冲突时等待用户调整，不自行换资源。

快照/data可用757432659968 bytes、RAM可用909657952256 bytes。六条按旧规划每条60GB盘/RAM预留约360GB，快照余量足够；这是容量规划，六并行真实CPU/IO争用和时间/内存峰值未新实测，启动前仍须检查。不得删除已有replay/checkpoint或base-py311腾空间。

sv3原checkout干净 `d9be5574878d2c3f117cc6667be844f9a527eb84`，仅本机提交、经Git bundle哈希校验/verify/快进同步至实际测试版本；最终文档同样快进，不在服务器直接编辑项目源码，不同步sv1/sv2或推送GitHub。bundle包含sv3落后期间的既有提交历史，不冒充全是本轮改动。

环境Python3.11.16、JAX/JAXlib0.4.33、NumPy1.26.4、dm-control1.0.48、MuJoCo3.15.0；CPU测试显式CUDA_VISIBLE_DEVICES=''、JAX_PLATFORMS=cpu及两虚拟CPU。sv3原缺W&B，已用现有 `env/requirements-wandb-pd1.lock` 与安装前实际包版本作constraints，dry-run确认只新增10包再安装；W&B0.30.0，全部已有包版本未改，pip check通过。未更改/删除继承环境。W&B账号在线认证/日志上传尚未做，不创建测试run；04启动前核查，不能泄露凭据或以停用已声明logger绕过。

## 产物、命令与交04

本机新证据：`analysis/outputs/hopper-seeds123-preparation-20261021/`，含资源XML/pmon、resources、代码bundle、CPU日志、原16条回归、依赖前后列表、最终六份JSON/YAML与计划/随机映射、COMMANDS/validation/DELIVERY/HANDOFF。

sv3证据：`/data/Policy_Discrepancy/runs/hopper-seeds123-preparation-20261021-cpu/`。CPU初版配置包为repo `analysis/outputs/hopper-seeds123-preparation-20261021-0b764bd/`；**承接只使用最终 `analysis/outputs/hopper-seeds123-preparation-20261021-final/`**，code_sha等于交付最终SHA，初版保留。计划不是真实结果，新实验根仍未创建：

`/data/Policy_Discrepancy/runs/formal-m2-hopper-seeds123-baseline-dt-20261021/seed{1,2,3}/{logging,dt}-attempt01`

```sh
# CPU准备，不会启动；新目录命名，不覆盖初版/最终包
cd /data/Policy_Discrepancy/repo
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu /data/Policy_Discrepancy/envs/dreamer/bin/python -m scripts.prepare_hopper_seeds123 --resources /data/Policy_Discrepancy/runs/hopper-seeds123-preparation-20261021-cpu/resources.json --campaign formal-m2-hopper-seeds123-baseline-dt-20261021 --output analysis/outputs/NEW_HOPPER_PLAN
# 训练结束后的CPU统计，用真实实际运行index，不用计划冒充结果
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu /data/Policy_Discrepancy/envs/dreamer/bin/python -m analysis.return_metrics --profile hopper-seeds123 --index /path/to/actual-RUN_INDEX.json --output /path/to/NEW_RETURN_AUDIT --require-complete
```

04不得套用硬编码sv2/四任务/seed0的 `run_formal_baselines.py` 或 `audit_formal_pd1.py`。按六份审阅argv及实际验证的设备环境监督执行；必要监督工程代码在本机提交/测试后同步同一版本，不改服务器源码。正式启动、每条实际SHA/config/设备映射/PID/状态/资源/退出必须记录；baseline完整审计后各自freeze，Dt无需等c。收尾同时核对连续成功update ID、开始步窗口S/N、invalid0、final/checkpoint/快照、21×10完整评价与真实c来源，再调用新统计profile；W&B不是权威源值的替代。

首选下一步：用户明确启动后，继续现有04使用本次资源与配置。可立即做CPU/只读部署检查；训练及GPU探针仍等待启动授权。完成标准为6条完整预算/评价/收据/快照与三份各自baseline c审计、逐seed/三seed均值/配对差值与失败记录、真实设备/版本/资源释放记录；缺失按协议不完整，不补点/重加权/择优重跑。没有无缝恢复环境/replay能力，不完整尝试保留后新ID，不覆盖。训练波动可检验重复性，但本批缺constant/shuffle重复且无配对输入干预，不能作为M2选择性或hopper唯一根因结论。交接prompt见本轮HANDOFF及最终回复，不自动创建chat/发消息。
