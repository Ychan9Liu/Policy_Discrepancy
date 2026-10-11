# DL 有限嵌套接口结果

科学 `DL-nested-probe-r1`／工程 r1，实际执行与 clean Linux 测试提交 **`315346b2d0df7e3b9d81f3ca7cd46944d5ac8dd0`**；冻结规范以该提交 [DL_NESTED_PROBE.md](DL_NESTED_PROBE.md) Git blob为准。Agent总体授权内选择，非用户逐项认可的方法。

## 有限结果与解释

原cal8/32位置、四paired sampled z、h4096/32×32、七首控制和normalized-zero续行H20真成本不变。mean λ四值、whole-episode outer8/inner7、共同inner utility/global tie/MSE/最强λ规则一次完整执行，outer只评价被选规则。episode `[9,14,15,18,20,24,30,31]` 对应λ `[.001,.1,.1,.1,.01,.001,.01,.1]`；112inner分区/16outer fit、128 eigensystem、464次预测完整。

| 主量 | 均值 | 名义95% whole-episode区间 |
| --- | ---: | --- |
| q−p 真选择reward | .003927799889331143 | [.0005159088678856373, .008195409614653176] |
| q−constant 真选择reward | .0156441998693859 | [.006870717118694178, .02843293686018337] |

q/p/constant选择reward分别5.728799392014874／5.7248715921255435／5.7131551921454875；q/p centered per-control MSE分别.03592851635620402／.0352057100285985，q的MSE更高。只能说本程序选择效用有探索性正增量，不能写成q整体预测更准。两平均增量为旧δ=.2的约1.96%／7.82%；δ用于原外部控制分离，不是本次效应成功阈值，不能将小效应当性能承诺。

预定状态 `selected_procedure_exploratory_supported`。八episode已观察过、训练折重叠，固定cross-fitted分数bootstrap未包含重拟合不确定性，不作blind独立确认或正式显著性结论。q优势可能来自历史重编码或不同有效自由度；未证明当前RGB超越完整历史的因果增量、正常KL损伤、保护或训练效果。原λ1 H20零增量、H100负平均及停止门不变。

## 执行与来源

- sv3/lyg0326 Linux CPU，Python3.11.16／NumPy1.26.4／线程1。GPU0；零新模型/render/physics/optimizer、未读blind。八cal NPZ只读数值白名单、records未解码，原成本F64 bytes相同；6发布JSON/8NPZ前后核hash。物理/carry/key是认证复用旧证明，不是重新运行。
- 本机39fixture：35pass/4POSIXskip2.157秒；clean上述SHA的sv3 **39全部pass**，process4.770104秒，含入口parse alarm、验链/最终写盘硬限、自身后代清理与无关sentinel。工程fixture不是方法有效证据。
- 2026-10-11 **23:14:15.920838–23:14:18.140913北京时间**；一次完整父终点2.216830279秒／120秒，worker1939562 terminal/exit0/非超时/owned-group清理/完整链。parent-entry收据2.155755494秒。未核当前GPU占用，不声明历史卡持续空闲或本轮释放卡。
- 只部署sv3隔离 `/data/Policy_Discrepancy/dl-worktrees/315346b`，canonical服务器干净d9be5574878d2c3f117cc6667be844f9a527eb84、本机main595e982f1a3939775879728009e2105ee516e86d原dirty两docs保留，无sv1/sv2同步。bundle `40e484f781c8acf83d2e4c64d0e5968a0e95c915f91d97ab359c658d3db18eaf`；最终文档与GitHub同步另记。

服务产物根 `/data/Policy_Discrepancy/runs/dl-nested-probe-20261011-315346b/`（`cpu/`、`attempt01/`、外部launcher终点）；本机ignored镜像 `analysis/outputs/dl-nested-probe/315346b2d0df7e3b9d81f3ca7cd46944d5ac8dd0/`。完整命令、源码/环境/源pins、预处理、所有inner候选与outer预测在plan/result/CPU summary，14小文件镜像manifest可核。原始产物不入Git。

14文件本机／服务器逐字节匹配；mirror manifest SHA256 `a17cb62f1695d293a7bc4c88d7e5c6a2cdb8cfa88d7fe7e93041a78605d73bfb`。实测提交后的最终文档差异仅本结果报告、规范状态段及RESEARCH/STATUS/EXPERIMENTS/验证日志；代码和参数未改，不据文档提交冒称重复验过模型。

| 证据 | SHA256 |
| --- | --- |
| frozen315规范 | `0e251dd4b72da2d9ebf127d47bb70f306c1fd0154b70d4a003c3eeacab67036b` |
| plan.json | `32043263e0101c39e1a4eec5c2204164cb18422c6d7c7d2ff1b7401f129fcb8b` |
| result.json | `7a80036dc29b9d59d2c17de5b1f30733313f6562bca62e607c958ccf96cf9085` |
| nested_probe_complete.json | `d4df654e0903e4a1bdecf7fdc0b2ae8b4dbb174fb0df744d83063955ef71e0bf` |
| supervisor_complete.json | `69d60236c23b2a8aaf48a55112f0c3c9fc88650eb65ae2e76fcbbe7121e7acf5` |
| CPU summary | `60f27c0faaa594d4e2ee2886a3e12b3e152567ef67fcb22ee60cd08c5276b7de` |
| outer launcher | `fe3c4ffcf0caf2d3d612287e9d6e7df108f1c3a1c24ee16addb4705d0c237db6` |

## 独立审计与下一步

独审ignored源码 `analysis/outputs/dl-nested-probe/independent_nested_audit.py` SHA `eb2272431472fbcf96d5e2be8721a6bf0654f9d6daf8280f054a28fec4e3136c`。只stdlib+NumPy、不import生产stats/loader；八小synthetic selftest在本机与sv3通过（sv3 .172662秒）。root在真实执行前加4 Git blob/working bytes/plan pin前后核验；实际最终315 metadata一致，没有放宽双字符串兼容或改变科学判据。

独立 **direct solve** 重获448inner/16selected outer、train mean字节/RMS/截距、选择/真效用/regret/MSE、常数、共同λ/tie与2000episode bootstrap；**140557比较通过，max abs1.0658141036401503e−14**，choice/λ/digest/classification/CI零侧exact。原true cost、6来源JSON/8NPZ、新五JSON链及4源码Git字节前后匹配。eigh特有重建证明仅核原固定容差收据；不重跑模型、环境或renderer。

正文1.956172780秒，父终点2.025088133秒／120秒，worker1941441 terminal/exit0/only-owned-group清理。`attempt01/independent-audit01/audit.json` SHA `c934bc2cc2c367b4e163802e45c782a53c9fca9be1c1b6a81281d7b1ff28f373`；父收据 `841c336525d2a229af5275585820312cddac31d3c413d925a1ecb75b7e94feb5`，launcher `335c34d41c2a7bbdb3ad377fcaea6f6dfc92d7eae8bb768e88cec3a9d489b352`。父通用process API的`generic_fixture_without_scientific_verification=true`仅表示未运行nested-worker验链，actual reads保持unknown；实际八episode读取及独立数值结论在auditor正文，不将正文误当纯fixture。

**停止此有限对象**：不追加λ/seed/样本、outer winner搜索、blind、gate或训练。A验收时B-r0仍为未采用建议；后续交叉审查已采用 [DL_PRODUCTION_REWARD.md](DL_PRODUCTION_REWARD.md) r1 科学协议，工程及真实执行尚未开始。首选继续当前完整授权聊天实现并验收B，不因A阳性推进训练；配对历史/可见cue、独立第二信号、active rep损伤/保护与性能仍缺证据。当前0GPU、已有规模模型诊断1A100串行/2可独立并行，sv3 4–7足够不用全占；历史2.5–6.5GiB采样峰不保证新训练显存，总剩余GPU小时未定。本B的GPU4/1800秒是单个有限对象上限，不是整体预算。
