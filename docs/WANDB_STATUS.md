# 04 · 三台服务器W&B登录核查

检查时间：2026-10-09 21:58:45–21:58:46（Asia/Shanghai，请求开始时间；UTC13:58:45–13:58:46）。范围为当前SSH用户及项目 `/data/Policy_Discrepancy/envs/dreamer`，不推断其他用户/环境的登录情况。

| 服务器 / hostname | SSH用户 | 当前凭据与在线验证 | W&B账号 | dreamer SDK / 当前PATH CLI | 检查时repo SHA |
| --- | --- | --- | --- | --- | --- |
| sv1 / lyg2153 | codex-runner | netrc有凭据；只读viewer请求HTTP200、身份有效 | ychan9liu | 未安装 / 不可用 | 27261bf6df875939195879ab23e1dc161507fe5c |
| sv2 / lyg0360 | lyc | netrc有凭据；只读viewer请求HTTP200、身份有效 | ychan9liu | 未安装 / 不可用 | feb00f75eea9adc39d3e2121e2d6d29e34eea29d |
| sv3 / lyg0326 | lyc | netrc有凭据；只读viewer请求HTTP200、身份有效 | ychan9liu | 未安装 / 不可用 | d9be5574878d2c3f117cc6667be844f9a527eb84 |

三台非交互SSH环境均未设置WANDB_API_KEY/WANDB_MODE，凭据来源是当前用户netrc对应api.wandb.ai条目；没有发现用户或repo的wandb/settings，使用默认官方API。凭据仅在检查进程内读取，用HTTPS、禁止重定向的只读GraphQL viewer请求验证；未打印、保存或提交密钥，也没有wandb.login/init、实验run创建或凭据更新。

证据：本机 `analysis/outputs/wandb-login-check-20261009/status.json`，保留时间、主机、用户、SDK/CLI、凭据存在性、认证状态、HTTP状态及账号；`check_wandb.py`为本机检查脚本，经SSH stdin执行，不落入服务器仓库，不导入agent或使用GPU。SDK安装检测用importlib，不触发W&B初始化。

本次仅只读核查和文档记录：没有代码、依赖、登录状态、服务器工作区或实验变更，不涉及训练验收。在线认证通过不代表已验证具体entity/project写权限、run上传、网络长期稳定性或其他Python环境；未安装SDK的dreamer环境不能直接使用项目的W&B logger。

如用户要启用W&B，首选在现有04按用户与03共同声明的日志配置处理SDK依赖及日志验证；当前正式baseline的prealloc/logger声明仍待明确，不因本次检查自动改为W&B或启动训练。无需另开chat，不自动发送消息。后续产物用analysis/outputs新目录，大产物留服务器runs；依赖变更需记录锁文件/版本及pip check，不删除base-py311。

可复制交接prompt：

```text
继续现有04，项目根D:\program\PD\dmr3，从完成回复最终SHA接手，核对差异不回退。
必读AGENTS.md、docs/CHATS.md、docs/STATUS.md、docs/WANDB_STATUS.md及docs/FORMAL_BASELINES.md。
三台当前SSH用户均有netrc凭据，2026-10-09 21:58在线viewer认证账号ychan9liu；
项目dreamer环境均无wandb SDK，当前PATH无CLI，未验证entity/project写权限或run上传。
本次没有安装、登录修改或训练；凭据不得输出/提交，不能把本次核查当启用W&B授权。
后续若用户要求启用W&B，先依据用户/03已声明日志配置，在实际执行环境处理依赖并验证，
不改模型、梯度、随机流或科学协议；不删除base-py311，不在服务器直接改源码。
正式baseline启动授权与sv2 GPU0–3分配已给出，执行设置声明仍待明确；不重复请求启动确认。
完整命令/版本/证据/依赖记录入文档，outputs用新目录，不自动创建chat或发送消息。
完成按AGENTS交付完整最终/实测SHA、差异/同步范围、证据与限制、下一步prompt。
```
