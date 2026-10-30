# 03合并窗口 · 补齐十一条未启动Dt（2026-10-30）

## 授权、范围与固定配置

用户先授权sv2运行剩余任务并排除GPU6，随后明确“这台服务器上跑七条，sv3 gpu 4-7 跑另外4条”。本批仅补DMC16范围中从未启动的11条Dt，不重跑已有16条logging baseline或5条Dt，不追加训练seed、constant、shuffle。旧blocked索引、已完成产物和历史失败证据原样保留；新目录各为attempt02，从头训练，不从旧checkpoint恢复。

| 服务器 | 物理GPU | task（均带dmc_前缀） | 组/训练seed |
| --- | ---: | --- | --- |
| sv2 | 0 | acrobot_swingup | Dt / 0 |
| sv2 | 1 | cup_catch | Dt / 0 |
| sv2 | 2 | cartpole_balance | Dt / 0 |
| sv2 | 3 | cartpole_balance_sparse | Dt / 0 |
| sv2 | 4 | cartpole_swingup | Dt / 0 |
| sv2 | 5 | cartpole_swingup_sparse | Dt / 0 |
| sv2 | 7 | hopper_stand | Dt / 0 |
| sv3 | 4 | pendulum_swingup | Dt / 0 |
| sv3 | 5 | reacher_easy | Dt / 0 |
| sv3 | 6 | walker_run | Dt / 0 |
| sv3 | 7 | walker_walk | Dt / 0 |

每卡一条、11条并行。**sv2 GPU6不使用；sv3 GPU6属于用户指定4—7范围。** 不替换资源、不停止外来作业。sv1空间不足且未分配，本批不向sv1部署或启动。

沿用`M2-v1-dmc16-seed0-baseline-dt` / `m2_dmc16`：clean，仅64×64 RGB视觉，size50m，seed0，alpha20，free_nats1，dyn/rep外层系数1/.1，ac_grads=false，reward_grad=true，repval_grad=true，repval_loss=true。batch16×64、replay_context1、16训练env、train_ratio256、repeat1、CUDA/bfloat16、prealloc=False；正式report关闭。每条100万训练动作含预填充，排除reset/评价；0..100万每5万评价，共21点，每点固定快照/采样动作/独立评价随机流、恰好10个完整episode。JSONL加在线W&B `ychan9liu-tsinghua-university/PD_1`，独立ID、resume=never。

各任务自身已完成seed0 logging baseline继续作为c来源，窗口为成功更新开始动作`[100000,300000)`，有效loss位置raw rep KL>1，累计候选权重`1/(1+20D)`的S/N；logging实际权重1。此11条Dt不使用c，不在线匹配c替代训练权重。完整原baseline审计/c来源复制在新plan的`source_baseline`中，不修改旧baseline。原始return、实际x梯形积分/固定百万分母和指定80/85/90/95/100万五点末段遵循EXPERIMENTS C/C.1；缺项不重加权。

## 实现、提交和同步

代码/验收目标版本 **`d8997f64a301bfdcfe08758faeae8957a7395766`**。仅新增：

- `scripts/run_missing_dt.py`：严格限制11个缺失槽位及7+4分配；原blocked必须无PID/开始时间，原logging必须已有百万动作/249493更新/21点评价审计和合法c。解析后新Dt完整配置与原Dt只差logdir、与原baseline只差mode/logdir；禁止自行改模型、seed、alpha、预算或离线W&B。
- `tests/test_missing_dt.py`：资源/槽位/配置篡改拒绝、非法或未完成来源拒绝、ID冲突、上下文隔离、输出不可覆盖等7项测试。

没有修改agent、RSSM、rep probe、梯度、优化器、奖励、动作计数、评价或随机流实现。复用现有strict GPU门禁、CUDA/EGL实测探针、logger、完整agent和终点审计。父进程恢复此前已验证的非交互启动PATH，清除LD_LIBRARY_PATH/LD_PRELOAD/CUDA_HOME/CUDA_PATH/PYTHONPATH注入；子进程另清理CPU/MPS flags并显式绑定CUDA UUID和实测EGL。无依赖安装或全局transfer guard关闭。

通过Git bundle核哈希、verify/fetch、ff-only同步到两台实际执行服务器，源码均干净且固定d8997f6；未GitHub推送、未同步sv1。原sv2为`dbd28b6703d51d61e5d3485dbfea6c413075ea85`，sv3为`743089f40662b031a5cf2b6bf988decb52ba0b28`。

| bundle | SHA256 |
| --- | --- |
| code-sv2.bundle | `04324d3aa981f29ab92379e736b63ed4c3b9b1376fc50ab2fe4808ddc3c69f8a` |
| code-sv3.bundle | `24f93f226cf0074a4a5592d80313058ff962e1553e695e82007b9eaa690239e8` |

环境均为`/data/Policy_Discrepancy/envs/dreamer`：Python3.11.16、JAX/jaxlib0.4.33、numpy1.26.4、dm-control1.0.48、mujoco3.15.0、elements3.22.2、ninjax3.6.3、wandb0.30.0，pip check退出0。环境/锁文件校验和实际设备UUID/EGL/进程内CUDA0映射见工程根记录。保留继承的base-py311。

## 验收及实际启动记录

本机53项测试通过（21.815秒）；两服务器各107项CPU回归通过（sv2 53.112秒、sv3 54.068秒），均对应d8997f6。CPU三条完整float32 agent对比report关闭/打开及评价episode数2/1：训练参数digest、计数、batch key、更新收据、匹配累计和共同评价episode结果精确一致。容差未放宽，离散字段/配置/哈希/动作边界使用精确检查；保留已有算法、梯度、global shuffle跨shard、协议与return回归测试。

上述精确比较为**各服务器内**三变体：sv2参数digest为`41a913d9d50f02df34e387ba514354e51ca924aeefed36a643aebcd2fabb6be4`，sv3为`3c8d1e42637d602fab435dcdc445e1b509df5408035964e4c4a5b58a94c9a1aa`，跨机未逐位相同，内部原因未定位。没有把各机隔离通过写成跨机逐位重现。整理证据时首轮辅助检查误要求两机均等于sv3 digest而失败；保留该辅助记录，修正为协议要求的同机原始参数/收据精确比对，不修改算法测试、训练代码、容差或已生成的结果。

完整命令在两台工程根`check-commands.json`。本机可重放子集：

```powershell
.\analysis\outputs\dmc16-preparation-20261022\venv\Scripts\python.exe -m unittest -v tests.test_missing_dt tests.test_dmc8_remaining tests.test_dmc8_supervisor tests.test_dmc16_preparation tests.test_return_metrics
```

14:04北京时间启动前初核：sv2指定七卡、sv3指定四卡均0MiB且无C/G PID；sv2 GPU6和sv3 GPU0/2已有非本项目作业，未使用或干扰。两台/data可用约821GB/722GB，分别高于7×60GB/4×60GB容量门槛。真实CUDA工程验收使用分配卡，正式启动前还须重新检查工程进程已退出且卡空闲。旧13:40状态报告为此前快照，不代表本批启动后状态。

11条真实size50m CUDA/bfloat16短程全部通过：每条4098动作、517更新、3点各1完整episode、2次报告，合计5687更新、33点评价、22次报告；退出0，固定快照/收据/终点/只读报告与评价指纹检查通过。训练日志中活跃比例非零，Dt实际平均权重0.9992692709—0.9999476372，活跃位置实际权重也非1；这是存在gate作用的工程证据，不是收益。Dt不累计baseline匹配S/N，短程match_count=0属预期，不代表无活跃位置。

**14:20资源冲突**：sv2工程进程退出后复核，GPU2/4新出现非本批python3.10计算进程（GPU2 PID552701、GPU4 PID553506，解释器位于root的jaxgcrl_atari环境；详细cwd读取无权限，不猜测项目归属）。七卡整体启动前门禁拒绝，正式根尚未创建；GPU6原有外来作业亦未使用。原始证据保存在`prelaunch-resource-conflict.json`，已请用户协调释放或重新分配，不共享/停对方/自行换卡。

**14:21:10 sv3正式启动**：14:21:06—10复核4—7卡均无C/G PID，CPU监督器PID618156，在d8997f6代码启动四条Dt独立session。agent PID分别为GPU4 pendulum_swingup=618571、GPU5 reacher_easy=618580、GPU6 walker_run=618594、GPU7 walker_walk=618917。实际argv/env白名单在`formal-supervisor-command.json`。

**14:25:12 sv3启动审计通过**：四条零点评价恰好10完整episode，固定快照动作0/update0；成功更新数分别437/316/331/329，最近更新开始动作3776/3292/3352/3344，收据从0连续、invalid_count=0。四份manifest完整配置/运行SHA精确匹配，CUDA可见UUID、进程内cuda:0和实测EGL4/5/6/7一致；68个计算/图形PID均为对应agent/子进程，四卡未发现外来C/G作业。W&B四条running、完整配置一致，独立startup JSON/hash线上读回精确一致。启动审计不是正式AUC或百万动作终点验收。首轮辅助审计在reacher尚预填充、收据未创建时抛出FileNotFoundError，之后等待首批更新完成才验收，没有放宽检查或修改训练。

**14:25:36 sv2资源再核查**：GPU0又新增python计算/java图形进程（PID558403/558530），GPU2/4外来作业仍在；GPU1/3/5/7当时无C/G PID，正式根仍不存在、7条均未启动。占用变化已报告并更新资源询问，原七条计划/绑定仍保留，未自动改为部分启动或等待空闲后台启动。此时/data可用805,602,594,816字节；sv3启动审计时706,274,996,224字节。

**14:29:36交付资源/进度复核**：sv2冲突仍为GPU0/2/4，正式根不存在、七条未启动，/data余805,570,580,480字节。sv3四条进程及监督仍存活，GPU4/5/6/7仅本批C/G，未释放；更新分别2153/2030/2045/2035，最近更新开始动作10640/10148/10208/10168，invalid累计0，已完成评价仍仅0点、各10完整episode，/data余705,764,364,288字节。最近更新开始动作不是精确实时总动作；此次快照不保证之后始终独占。sv2本项目工程GPU进程已全部退出，不能将外来占用卡称作空闲。当前已实际完成4条正式启动、7条待用户资源处理，未取消原授权或用交接替代可执行工作。

## 输出、监督和复现边界

服务器工程根：

- sv2：`/data/Policy_Discrepancy/runs/engineering-missing-dt-20261030-sv2/`
- sv3：`/data/Policy_Discrepancy/runs/engineering-missing-dt-20261030-sv3/`

正式根：

- sv2：`/data/Policy_Discrepancy/runs/formal-m2-dmc16-missing-dt-20261030-sv2/`
- sv3：`/data/Policy_Discrepancy/runs/formal-m2-dmc16-missing-dt-20261030-sv3/`

每task独立`dt-attempt02/`，W&B ID为`formal-m2-dmc16-missing-dt-20261030-<task去dmc_>-seed0-dt-02`。工程根保存未绑定/已绑定plan、环境及Git同步收据、设备proof、CPU/真实CUDA验收、完整命令和启动审计；正式根保存`actual-RUN_INDEX.json`、resources.jsonl、stdout、manifest、RNG映射、逐更新收据、评价/快照/最终状态和审计。不得对现有目录再次执行run；环境/replay不能完整续接，故控制器明确拒绝恢复/覆盖。

受限执行链（下列为复现说明，现有运行不要重复启动）：

```sh
# 同SHA、同host、真实CUDA验收/online logger/设备证据齐全后绑定。
/data/Policy_Discrepancy/envs/dreamer/bin/python -m scripts.run_missing_dt bind --plan ENGINEERING_ROOT/SERVER-plan.json --evidence ENGINEERING_ROOT --output ENGINEERING_ROOT/bound-plan.json
# 由干净CPU父进程以独立session启动；完整实际argv/env白名单见formal-supervisor-command.json。
/data/Policy_Discrepancy/envs/dreamer/bin/python -u -m scripts.run_missing_dt run --plan ENGINEERING_ROOT/bound-plan.json
```

监督器逐卡重新核C/G占用，独立session启动，更新索引/退出码。每条退出0后全量审计百万动作、249493更新、21×10、固定快照/manifest/收据合法性；全部本机Dt通过后生成本机partial `dmc16`表和精确W&B摘要。**`shard_complete=true`不等于全16任务×两组完成**：本批7/4条均只有Dt，overall指标必须不完整、`cohort_complete=false`；全32条汇总须后续将旧21条与新11条按实际索引/快照证据统一审计，不把缺失组重加权。

本机小证据/plan/测试/同步bundle/最终交付与继续prompt在`analysis/outputs/missing-dt-start-20261030/`（Git忽略）。不提交大产物、checkpoint、replay或凭据；原日志全部保留。

14:27小证据镜像逐文件SHA256校验通过，sv2 145文件压缩包`77077dcec3ed404d0aa53101044107c34976d6a50cbc32aaf28aed290b0b67e9`、sv3 144文件`b7e1f69837f1d6019ff3c437286eefa702c1098281caf8323ea77195be6b0410`；仅配置、日志、工程收据/隔离指纹和启动证据，不含checkpoint/replay。正式运行中的JSONL副本为完整行前缀快照，不当完整终点数据；14:29资源快照另存并同步。最终交付晚于实测代码时只含工程报告/STATUS/索引文档，不移动活动服务器checkout；最终完整SHA和对象同步收据见DELIVERY。

## 未完成、限制与下一步

正式百万动作和21×10终点结果须运行结束才验收；工程4098动作/3×1只验证执行/计数/只读报告和评价隔离，不能冒充正式指标。相同seed配置不保证真实环境轨迹或逐位计算重现，既有hopper gate/输入微扰因果问题未解决。此处两组单seed结果不能独立证明强M2逐时刻选择价值或跨seed稳定性。既有sv1 GPU7短时外来并发等旧运行限制仍保留，未因本次新部署解除。

首选继续**当前03合并窗口**监测同批11条、处理不改协议的工程故障、保存失败并以新尝试重开，完成逐条终点/本机shard及全32条来源一致性与原始return/W&B收尾。可立即监测与只读审计，终点汇总依赖完整训练和产物；未授权新任务/seed/组、换卡或删数据。全部审计后由用户交现有05分析；新科学复验由02与用户确定。不自动创建chat或发送消息，可复制prompt见本机`HANDOFF-current03.txt`。
