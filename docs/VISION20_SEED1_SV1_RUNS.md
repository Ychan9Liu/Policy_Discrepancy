# 03合并窗口 · Vision20 seed1 sv1续启（2026-11-04）

接续`VISION20_SEED1_RUNS.md`。用户确认sv1已可用，现场核查GPU2—7无C/G进程、/data约341GB，低于原六卡360GB初始预留。助手报告缺口后，用户明确回复**“直接启动，我会继续清理”**。本轮在原六卡/原科学配置内继续验收和启动24条，不重复要求启动确认、不擅自改卡，不删除或迁移任何产物。

## 明确容量例外

本次只对sv1初始并行预留执行显式例外，不将不足360GB写成原门槛通过。新增CLI开关`--accept-initial-capacity-deficit`默认false、仅接受sv1；开关开启仍要求至少60GB剩余，每次后续新训练继续执行原60GB底线；不足时阻断并保留证据，不自动删除、缩预算或修改replay。

正式根`capacity-at-start.json`记录free_bytes、normal_reservation_bytes、reservation_satisfied、explicit_sv1_capacity_exception、per_run_floor_bytes和用户授权文字。例外不说明341GB足以最终留存全部24条；用户持续清理/扩容是外部执行条件，运行期间继续核查。不会把临时容纳六条与全批累计产物容量混为一谈。

## 原配置和固定队列

Vision20 seed1共40记录，复用hopper seed1旧一对、19任务新跑38条。本次只sv1的12任务×baseline/PPD-KL（logging/dt）24条；sv2七任务14条继续原运行，不重开/替换。每卡严格baseline→完整终点审计→Dt→下一任务baseline→审计→Dt，前驱失败/审计失败/卡占用/容量不足阻断后继，无外来空闲自动重试或共享。

| sv1物理GPU | 第一任务 | 第二任务 |
| --- | --- | --- |
| 2 | quadruped_run | hopper_stand |
| 3 | quadruped_walk | pendulum_swingup |
| 4 | reacher_hard | reacher_easy |
| 5 | acrobot_swingup | walker_run |
| 6 | cup_catch | walker_stand |
| 7 | cartpole_balance | walker_walk |

task均有`dmc_`前缀。继承已确认`m2_vision20_seed1`：clean、仅64×64 RGB视觉，size50m、训练seed1、alpha20、free_nats1、dyn/rep=1/.1、ac_grads=false、reward_grad/repval_grad/repval_loss=true；batch16×64、replay_context1、16env、train_ratio256、repeat1、CUDA/bfloat16、prealloc=False、正式report关闭、JSONL+W&B PD_1（ychan9liu-tsinghua-university）、独立run ID/resume=never。百万训练动作含prefill不含reset/评价，0..百万每5万21点，各10完整episode/固定快照/采样动作/独立评价RNG。

每任务自身seed1完整logging baseline按成功更新开始动作[100000,300000)、raw KL>1有效loss位置累计候选1/(1+20D)的S/N提供c审计；logging训练实际权重1、Dt不用c；正常replay重采样保留、仅同update ID重写去重，无活跃/非法停止，不填c1或调alpha。raw-return、实际x积分/固定百万分母、指定80/85/90/95/100万五点、缺项不重加权按EXPERIMENTS C/C.1。科学协议、原seed0/补seed/候选方法均不改变。

## 本机实现和实际运行版本

sv1代码/验收目标SHA **`bef29480df12d07659c640b22218804dd1d69e63`**。基于当前本机HEAD，增加上述显式容量开关；工程验收支持多任务同卡计划分轮：第一轮六任务baseline/PPD-KL各一组、第二轮六任务各一组，保证每张卡同时只有一条工程训练。每轮调用原严格审计，原完整plan先验证，再仅允许其精确任务切片，最后合并24条验收、拒绝缺项/重复。

本机最终9项`tests.test_vision20_seed1`通过（15.669秒）；先前容量单测8项通过。模拟验证默认360GB不足拒绝、用户开关开启但低于单条60GB仍拒绝、显式341GB例外记录false/true及底线保持、24条队列/前驱审计、两轮每卡无重叠及篡改拒绝。fixture不是真实CUDA或正式结果。

24份完整config重新解析与原7769938计划精确相同，只更新执行SHA；58个模型/训练/环境Python blob与sv2实际77699380851701f7a03a9d13985c5cc0de5d0702逐一相同。显式排除的入口/范围文件为configs.yaml/main.py/freeze_c.py/protocol_scope.py，其间其他窗口已增加独立pendulum seed2/3范围，本次vision20配置/路由保持原行为；原agent/RSSM/rep probe、梯度/优化器、协议runner RNG不改。sv2继续原7769938工作树，不切换其活动代码。

代码提交`1a16345628442a7464b7547fa0c56e34970589fc`同时收进当时共享index内其他窗口已暂存的Denoising存储维护文档/索引，保留历史不reset；后续bef2948仅包含本轮控制器/测试。最终交付较实际运行如有后续变化需逐项记录，不将其他窗口代码/测试称为本轮验收。

Git bundle SHA256 `21af52971adb5393cc2f72b842f5eb17dfac758397c152e122c49586d3df0fcd`已verify/fetch，同步仅sv1，在`/data/Policy_Discrepancy/checkouts/vision20-seed1-bef29480df12`创建独立干净detached工作树；canonical repo保持原`6e468ad4a8252f21401eab6a9761626b33659f0e`、旧产物不动。首次用无命名正ref创建bundle被Git拒绝为空包，未部署错误包；随后从HEAD打包Git对象并选定bef2948运行，验证记录保留。无sv2/sv3源码同步/GitHub推送，无服务器项目源码手改或依赖安装。

## 资源与启动记录

01:16北京时间：sv1 GPU2—7空闲、UUID与原计划相同，/data余340,989,685,760字节；GPU0/1为未知项目Ray作业，未干扰。sv2七baseline运行、七Dt queued、/data约577GB，GPU6未使用。本次不挪sv2工作；sv3独立pendulum seed2/3范围与本批seed1不同，不重复其任务/seed。

01:23 sv1同提交CPU/完整agent/设备/logger/24条真实size50m CUDA验收流程已启动，检查进程1560557。独立CPU启动pipeline1561504等待本项目checks-acceptance成功后绑定；再刷新指定六卡无C/G/UUID和实际空间，以显式容量例外开关启动正式监督器，不等待外来作业空闲重试。工程失败则pipeline停止留证据，不会把工程进程、plan或Popen当正式启动成功。

01:48:16，retry02合并24条真实size50m CUDA/bfloat16工程验收全部通过：每条4098动作/517成功更新/3点各1完整episode/2报告，合计12408更新、72点评价、48次报告，持久化收据/完整快照/报告和评价只读指纹通过。logging实际权重1、各任务匹配窗口活跃位置非零，Dt实际权重/活跃性单独留诊断，不将工程值当正式c/收益。工程参数不成为正式协议，无降低数值/归属/完整性检查容差或全局关闭transfer guard。

**01:48:39 sv1实际正式启动**：CPU监督器1609040，GPU2—7首批logging agent1609139/1609161/1609452/1609475/1610035/1610430（依上表第一任务顺序）。其余18条queued；不是24条同时运行。完整启动前六卡fresh require_free/UUID与argv/父环境在retry02 formal-supervisor-command.json，正式root已创建且含capacity-at-start.json，显式传入用户容量例外，实际代码bef2948。实际初始空闲289,725,108,224字节，低于360,000,000,000预留，reservation_satisfied=false；60,000,000,000单条底线保留。

**01:56:52启动审计通过**：六条各0点恰好10完整episode、固定0动作/0更新快照，成功更新按GPU顺序1937/1889/1924/1841/1881/1928，ID连续且invalid=0。CUDA UUID/EGL、进程树及全部C/G同卡独占、完整config/seed/SHA、W&B running和精确startup JSON/hash逐条读回一致。六条均继承本批代理变量，训练环境无CPU强制或交互CUDA路径污染。/data余283,305,242,624字节。此时GPU2—7仍被本项目占用、未释放；GPU0/1未知项目作业未干扰。证据retry02/formal-startup-audit.json。

01:57:25只读复核sv2七baseline存活、更新65971—66320、开始动作265912—267308，末条invalid均0；七Dt仍queued，GPU0—5/7为本项目、GPU6无计算进程且未使用，/data余569,931,264,000字节。全40记录为2复用验收完成/13运行/25排队，无本轮失败或存储阻塞状态；不能把尚未触发容量底线写成全部24条存储有保证。

### 初次网络失败与同SHA重试

bef2948在sv1的123项CPU回归通过（94.315秒）、三变体完整float32 agent精确隔离和六张分配卡CUDA/EGL同卡探针通过。随后online logger返回403 Forbidden，checks和启动pipeline均停止，未创建正式训练根；初次工程根/失败stdout/traceback完整保留，没有CUDA训练或正式run被该失败覆盖。

定位：sv1当前SSH账号codex-runner有已配置netrc，但会话无HTTP/HTTPS代理，直连api.wandb.ai返回403；sv2已有127.0.0.1:10808代理变量，其同端点返回正常404/405。sv1现有本地10808代理也可用，使用后只读viewer和PD_1项目权限检查通过。未新建代理服务、复制/打印凭据、修改netrc或换W&B账号；首次辅助viewer读回误把SDK User当dict报AttributeError，已改为直接验证viewer存在及项目访问，不据该辅助解析错误断言凭据无效。

retry02仅给本批父进程及子进程补齐现有HTTP_PROXY/HTTPS_PROXY/ALL_PROXY及小写变量为http://127.0.0.1:10808，不改模型/优化器/随机流/依赖/源码。新工程根`engineering-vision20-seed1-20261104-sv1-retry02/`，使用同SHA/同机已通过的CPU/agent/六卡proof及37份小文件，逐文件记录来源/hash；重试online logger使用新run ID后缀sv1-retry02，保留原失败身份。01:30 retry检查进程1570968、启动pipeline1571090开始执行，正式根仍使用原独立sv1计划、不从旧失败检查点恢复。后续训练父环境亦显式继承该网络配置，原GPU/容量门禁继续执行。

## 产物与后续

正式根`/data/Policy_Discrepancy/runs/formal-m2-vision20-seed1-20261103-sv1/`（目录日期为准备日，实际11月4日启动）；原失败工程根`/data/Policy_Discrepancy/runs/engineering-vision20-seed1-20261104-sv1/`保留，实际通过根为`/data/Policy_Discrepancy/runs/engineering-vision20-seed1-20261104-sv1-retry02/`，wave01/wave02各12条，合并integration/acceptance.json严格对应24个task/mode。Dt工程实际平均权重范围0.9989511489868164—0.9999504376901314，logging实际全1，窗口活跃计数289960—358400；这些短程诊断不是正式c或方法收益证据。完整命令、配置、环境、CUDA UUID/EGL/实际C-G映射、运行/退出/容量门禁/收据/快照/匹配c保留服务器独立目录；env仍`/data/Policy_Discrepancy/envs/dreamer`，不能删除base-py311。

本机证据/来源比对/计划/测试/同步bundle及最终DELIVERY/继续prompt在`analysis/outputs/vision20-seed1-sv1-20261104/`。原始38条计划与hopper复用证据见11月3日目录。不提交checkpoint/replay、密钥或大产物，用户及其他chat草稿保留。

462份小文件已镜像并逐项验SHA256，运输包738254字节、SHA256 `8b3e8221e6301e0d08f782fd6660e4dc1253b9f5c8a6a95c73bf7ed210fff544`。运行中JSONL仅截取完整行前缀，不含replay/ckpt/eval_snapshots/wandb二进制；这不是可恢复训练备份。镜像清单、核验结果在mirror-manifest.json和mirror-verification.json，sv2快照单独为sv2-final.json。

本机验收命令为`analysis/outputs/dmc16-preparation-20261022/venv/Scripts/python.exe -m unittest -v tests.test_vision20_seed1`。服务器用`/data/Policy_Discrepancy/envs/dreamer/bin/python -m unittest -v`运行原工程check-commands.json列出的17个模块（123项）；完整float32三变体/精确比较命令在原工程check-commands.json，源helper在本机checks.py。实际CUDA命令为`python -u -m scripts.run_vision20_seed1 integration --plan /data/Policy_Discrepancy/runs/engineering-vision20-seed1-20261104-sv1-retry02/sv1-plan-bef2948.json --evidence /data/Policy_Discrepancy/runs/engineering-vision20-seed1-20261104-sv1-retry02`，cwd为固定bef2948 checkout，使用同根check-commands.json记录的清洁启动环境和本批代理。不是数值近似比较替代：原隔离检查要求训练参数/随机键/成功收据精确一致，未事后放宽容差。正式命令和完整父环境在formal-supervisor-command.json；已启动队列禁止再次执行run命令。

实际运行后本机HEAD另有d18d759/73f0f88文档及34c3526的独立pendulum终点分析代码/fixture提交，均未进入本批活动checkout；最终本轮文档提交不宣称验收这些其他窗口变更。最终完整SHA及逐文件差异见本机DELIVERY.json和final-vs-tested-files.txt。本批不改原训练目标/RNG/冻结配置，也未做新科学比较结论。

首选继续当前03监测已启动六卡24条和sv2七卡14条原队列；容量低于单条底线或外来冲突停止后继并报告，不自动删数据。全部38条新运行及hopper两复用源齐全后，验收每条百万动作/249493合法更新/21×10快照/正式c/raw-return/线上finished与精确JSON/hash，并统一全40来源和缺项标记；每host partial不称全20完成。新科学协议仍交02与用户，完成全批审计后由用户交现有05，不自动建chat/发消息。
