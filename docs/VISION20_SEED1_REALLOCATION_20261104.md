# Vision20 seed1：sv1六条阻塞槽位改配（2026-11-04）

用户先提出GPU1/2，随后明确更正GPU0/1，并追加GPU4各两条；最终仅sv1物理GPU0、1、4，各串行两次训练，共六次，不增加任务/seed/实验组。14:28现场GPU0/1/4无C/G，GPU2/6为非本项目进程；原GPU3/5/7的Dt继续，不共享/停止外来作业、不自动换卡。

| 物理GPU | 第一条 | 第二条 |
| --- | --- | --- |
| 0 | quadruped_run Dt | reacher_hard Dt |
| 1 | cup_catch Dt | reacher_easy logging baseline |
| 4 | hopper_stand logging baseline | hopper_stand Dt |

task均有dmc_前缀。将hopper baseline/Dt放同卡以保留完整审计屏障，GPU1第二条相应为原阻塞reacher_easy baseline。本批不同时启动同卡两条。原reacher_easy Dt、walker_stand两组仍未获本次调度、不自动追加；后继必须当前驱完整审计通过，失败/资源冲突/容量不足即阻断，不轮询外来作业等空闲重试。

## 版本与科学配置

控制器提交`d6b9604ddf860d451d74d3cf6513fb66973a9094`，新增scripts/run_vision20_reallocated.py及tests/test_vision20_reallocated.py。实际训练始终固定既有已验收`bef29480df12d07659c640b22218804dd1d69e63`，checkout `/data/Policy_Discrepancy/checkouts/vision20-seed1-bef29480df12`；新控制器checkout `/data/Policy_Discrepancy/checkouts/vision20-reallocated-d6b9604ddf86`。启动前分别验证两者完整SHA/干净状态，控制器后续提交不进入活动训练源码。

原m2_vision20_seed1/protocol_vision20_seed1、scope M2-v1-vision20-seed1-baseline-dt、profile vision20-seed1不变；clean/仅视觉64RGB/size50m/seed1/alpha20/free nats1/dyn-rep1-.1/ac_grads=false/reward_grad和repval_grad/repval_loss=true、batch16×64/replay_context1/16env/train_ratio256/repeat1/CUDA-bfloat16/preallocFalse、正式report关闭，JSONL+线上ychan9liu-tsinghua-university/PD_1，独立ID/resume=never。

每条百万动作含prefill排除reset/评价；0..百万每5万21点，各10完整episode、固定快照/采样动作/独立评价随机流。每任务自身seed1 logging原[100000,300000)有效loss位置S/N审计c；非法或无活跃停止、不调alpha/填c1。Dt不读取c；raw return与固定百万分母AUC/指定末五点按EXPERIMENTS C/C.1，不因单seed/两组比较宣称方法选择价值。本次解析六份完整config，仅logdir改变，CUDA/EGL及W&B新身份在环境层改变；预算/损失/梯度/RNG不改。

## 防重复与容量

只接受原actual-RUN_INDEX中state=blocked、无PID、旧目录不存在的六个固定task/mode；任何已开始、旧目录/产物出现、源身份改变即拒绝。旧监督器不重试blocked槽位，本次不修改其持续写入的索引。原正式根reallocation-claims/逐slot排他创建源→替代run映射；新目录reallocated01、run ID附reallocated01，不覆盖原阻塞记录，不把新运行称恢复。

补启动根 `/data/Policy_Discrepancy/runs/formal-m2-vision20-seed1-reallocated-20261104-sv1/`。未来全40汇总以六份claim替换对应旧blocked身份，各task/mode只接纳一条；原索引与新索引不能直接拼接成重复槽位。原38新运行数量不变。

本次三卡要求初始180GB、每条后继60GB；14:28余203.624GB，没有使用初始容量例外、删数据或缩预算。用户继续清理仍是全批留存条件；三卡启动足够不等于全部后续产物有保证。无完整环境/replay状态恢复承诺。

## 验收与执行记录

14:34:16在sv1启动独立验收pipeline2166225。先跑新增3项控制器测试及保留123项CPU回归，再用训练SHA在0/1/4各做真实CUDA/EGL同卡探针；首条各一条实际size50m CUDA/bfloat16短程，工程4098动作/517成功更新/3点各1完整episode/2次报告，复用原严格收据/快照及只读隔离审计。短程不是正式指标。复用此前同训练SHA、同机24条完整任务验收及完整float32隔离；本次新增的是调度和设备证据，不改变训练代码。

新增测试检验同卡不重叠/前驱审计失败阻断、原槽已启动/目录存在拒绝、固定六槽完整config及SHA/新身份。首轮fixture使用默认prealloc和未JSON序列化tuple造成预期配置检查拒绝，已修正fixture为实际正式配置/持久化格式，未放宽生产验证。

代码bundle SHA256 `20651b219af8d4fce0278387fce49bacce7837a70ddc502f1b1523edb06507ff`已verify/fetch，仅sv1增加独立控制器工作树；canonical仍6e468ad4a8252f21401eab6a9761626b33659f0e，原活动训练工作树和sv2/sv3不动，无GitHub推送或服务器源码手改。实际网络继承现有127.0.0.1:10808代理，不改凭据/依赖。

工程根 `/data/Policy_Discrepancy/runs/engineering-vision20-seed1-reallocated-20261104-sv1/` 保存commands、完整计划/源映射、proofs、CPU/CUDA收据及pipeline/failure/正式监督器启动记录。全部验收成功后pipeline重核三卡再启动已授权六槽位；启动本身不视为完成，随后核实三个首条的0点10episode、连续更新、训练SHA/config/实际C-G归属与W&B读回。

14:36已完成126项服务器CPU回归（93.900秒）、三卡真实CUDA/EGL同卡探针；本机14项相关测试通过（14.254秒）。Git逐路径核对dreamerv3/、embodied/及复用run_hopper_seeds123/run_dmc8_serial/run_missing_dt文件在训练bef2948和控制器d6b9604之间没有差异；source-identity.json保存核验范围。三项真实CUDA短程随后执行，不能把CPU fixture当真实GPU验收。

本机产物 `analysis/outputs/vision20-reallocated-20261104/` 不提交大产物/凭据；报告后续段记录真实通过/失败、正式PID、实际资源时间和最终版本差异。首选继续当前03完成此六槽及原队列、每条终点/去重替代与全40指标收尾；不自动创建chat/向他人发消息。

## 实际启动

14:43:41三项真实size50m CUDA/bfloat16工程短程严格审计通过，每项4098动作/517更新/3×1完整episode/2报告，总1551更新/9点评价/6报告；hopper logging匹配位置324879。Dt工程match_count=0是仅logging提供c匹配的语义，不据此判断gate活跃/无效；原24项诊断/算法验收保留。既有代理下W&B viewer/PD_1访问也通过。

14:43:54—58三条正式训练实际启动：监督器2178932，GPU0 quadruped_run Dt PID2179037、GPU1 cup_catch Dt PID2179057、GPU4 hopper_stand logging PID2179355；另三条queued，正式训练SHA仍bef2948，控制器d6b9604。三卡fresh require_free及原槽位未启动复核通过，六个claim已排他写入；prelaunch记录要求初始180GB/后继60GB，14:44余194243751936字节，无容量例外。启动审计随后记录，不把Popen当验收。

14:46:46启动审计通过：三条各0动作/0更新固定快照、恰好10完整episode，连续成功更新82/119/151，invalid均0；实际训练manifest SHA及完整config精确一致，三卡全部C/G属于各自agent进程树，CUDA/EGL同卡独占，线上三个run均running、config及精确startup JSON/SHA256写入读回相同。子进程无CPU/XLA或交互CUDA路径污染，继承现有网络代理。证据formal-startup-audit.json。

14:47:26现场：原GPU3/5/7三个agent PID2020017/2013401/2013440仍存活，本项目GPU0/1/3/4/5/7均有运行，GPU2/6非本项目作业未触碰，没有释放本批指定卡。/data余191034204160字节。按六份claim排除旧源blocked行后，sv1恰好24唯一task/mode：6条训练/单条审计完成但待组级/W&B收尾、6运行、9排队、3阻塞；新六槽为3运行/3排队。剩余三个原槽为reacher_easy Dt和walker_stand baseline/Dt，未自动启动。

92份小证据已镜像，包105928字节，SHA256 `6102e29686d41e3d8e19991ef6b6a96896ebcfe6f7ed85a7474064a1a9f58570`；只包含小JSON/完整行前缀JSONL/config/stdout，排除checkpoint/replay/eval_snapshots/wandb二进制，不是可恢复训练备份。manifest及本机mirror-verification.json逐文件验证，原始大产物留服务器；本机DELIVERY.json记录最终SHA、实际CPU控制器/训练CUDA SHA、Git同步和最终差异。最终本轮仅文档变更，不移动d6控制器/bef训练活动checkout，不声称已完成百万步或方法有效。
