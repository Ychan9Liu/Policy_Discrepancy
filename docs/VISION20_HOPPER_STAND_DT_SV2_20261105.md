# Vision20 seed1 hopper_stand Dt 转至 sv2 GPU3

用户于2026年11月5日指定sv2物理GPU3重新运行唯一阻塞的 `hopper_stand Dt`。15:48:13核查时GPU3无计算或图形进程，`/data` 可用204302639104字节。本项目另四条Dt在GPU0、1、2、4运行。此次使用单卡60GB启动门槛。

sv1原尝试在GPU占用检查处阻塞，队列无PID，目标目录尚未建立，原监督器已结束。本次从第0步训练，保留原始身份到 `reallocated01` 再到sv2 `dt-moved01` 的两级来源。15:51:31以排他 `mkdir` 预留sv1旧目录并写入 `MOVED_TO_SV2.json`。原正式根的 `reallocation-claims/dmc_hopper_stand-dt-reallocated01-to-sv2.json` 记录第二级替代。全40汇总时沿两条映射排除前两个槽位。源行快照SHA256为 `8d9e4b785021f01709e85d58000aef4857a3074be621531272d54f0e120a7462`。

正式配置为clean、64×64 RGB视觉、size50m、训练seed1、alpha20、free nats 1、dyn/rep 1/0.1、`ac_grads=false`、`reward_grad=true`、`repval_grad=true`、`repval_loss=true`。batch 16×64、replay context 1、16环境、train ratio 256、repeat 1、CUDA bfloat16、prealloc false。100万动作计入prefill并排除reset和评价。评价在0至100万每5万一步，固定参数快照采样10个完整episode。运行使用JSONL及线上W&B `ychan9liu-tsinghua-university/PD_1`，独立run ID和 `resume=never`。逐字段校验只允许训练配置的logdir发生变化，服务器和GPU记录写入执行元数据。

对应sv1 seed1 baseline已通过249493次更新及21点评价的单条审计。其 `c_frozen.json` 的SHA256 `86b053a6ed289e2ba258cc4940288350c9d5c4057c6201a70145cab5e7370b86` 与审计记录一致。[100000,300000) 内有效位置27332254，匹配权重和21693075.982681274，非法计数0，冻结c为0.7936804620168273。新Dt启动计划的 `baseline_c_source` 为 `None`，c仅随来源证据归档。原始return统计沿用 `EXPERIMENTS.md` C/C.1。

控制器使用 `a81b797348e03a3c1767afa50b523472995ceb85`，在原单条迁移控制器增加绑定hopper_stand、blocked来源和GPU3的配置。Walker入口仍使用原默认配置。新增回归检查两级来源、GPU拒绝、配置差异和旧报错剥离。本机7项相关测试通过。训练checkout固定 `bef29480df12d07659c640b22218804dd1d69e63`。代码bundle SHA256 `90aa0dea266ff7391b2d50592461f68bb0c1ccd4492f317f36250787d860015c` 已校验并同步至sv2独立控制器checkout。sv1保存来源记录，sv2活动训练及canonical checkout保留原版本。

15:52:26启动验收进程3320285，完整命令见 `pipeline-process.json`。启动顺序为CPU回归、GPU3 CUDA/EGL探针、真实size50m短程、PD_1访问、GPU及磁盘复核和正式启动。短程预算4098动作，按原审计检查517次成功更新、3点各1个完整episode和2次报告隔离。正式启动审计另查0点评价、实际落卡、连续更新和W&B配置读回。

来源记录在sv1 `/data/Policy_Discrepancy/runs/migration-vision20-hopper-stand-dt-20261105-sv1/`，sv2工程根为 `/data/Policy_Discrepancy/runs/engineering-vision20-hopper-stand-dt-20261105-sv2/`，正式根为 `/data/Policy_Discrepancy/runs/formal-m2-vision20-hopper-stand-dt-20261105-sv2/`。本机镜像保存在 `analysis/outputs/vision20-hopper-stand-dt-sv2-20261105/`。

## 启动与验收记录

服务器133项回归耗时91.543秒并通过。用POSIX路径在本机重算已归档的walker_run计划，逐字段与旧计划一致。训练及复用审计源码Git diff为空。环境快照记录Python 3.11.16、JAX和jaxlib 0.4.33、dm-control 1.0.48、MuJoCo 3.15.0、W&B 0.30.0，`pip check` 通过。GPU3 UUID为 `GPU-92630efa-be4e-7074-d29a-ccff0ad7f801`，进程内 `cuda:0`，EGL设备3。真实CUDA短程验收通过，证据为 `checks-acceptance.json`。

15:57:45.231正式启动，监督器PID 3326261，训练PID 3326285。16:00:24启动审计核对零动作及零更新的快照，零点评价含10个完整episode，前102次成功更新连续且非法计数为0。全部GPU3计算及图形进程属于训练进程树。manifest与实际配置逐字段一致，PD_1在线配置及启动摘要SHA256读回通过。证据见 `formal-startup-audit.json`，正式启动命令见 `formal-supervisor-command.json`。

16:01复核时新运行已到3280动作，sv2 GPU0至4上的五条Dt均存活，最近更新非法计数为0。sv1全部训练进程已退出，旧hopper Dt目录只有迁移标记。sv2可用200062750720字节，sv1可用6940573696字节。汇总两台的有效槽位共38条新运行，任务与实验组组合唯一，再计入两条hopper_hop复用记录后，全40为35条训练结束及5条运行。完成的33条新运行已通过单条审计，整批指标和W&B终点收尾待五条运行结束。

已镜像47份证据文件，逐项SHA256通过。归档大小45492字节，SHA256 `3db3f5ac06f54e21134a6572d931e194ba0aa63d20c1406716da4bb560f86d98`。运行中的JSONL按完整行前缀归档，模型、回放及评价快照留在服务器。最终版本与文件清单见本机 `DELIVERY.json`。
