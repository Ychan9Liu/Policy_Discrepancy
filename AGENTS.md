# Policy_Discrepancy 工作约定

- 本仓库 `D:\program\PD\dmr3` 是项目根目录。`D:\program\PD` 只是父目录，不在外层仓库提交项目文件。
- DreamerV3 原始代码位于 `dreamerv3/` 和 `embodied/`。研究问题与未定假设写在 `docs/RESEARCH.md`；实验协议和结果索引写在 `docs/EXPERIMENTS.md`。按任务需要读取相应文档。
- 将“想法、待验证假设、已决定的设计、实验支持的结论”明确区分。未获用户确认的科研选择不要写成既定事实。
- 代码在本机仓库修改并通过 Git 同步；服务器 `/data/Policy_Discrepancy/repo` 用于运行同一提交。启动实验前记录 Git 提交、实际配置、seed、服务器、GPU 和环境版本。
- 实验产物存放在服务器 `/data/Policy_Discrepancy/runs` 的独立目录；不要把检查点、回放数据、密钥或登录信息提交到 Git。
- 三台服务器的 Python 环境位于 `/data/Policy_Discrepancy/envs/dreamer`，依赖快照见 `env/requirements-dreamer.lock`。该环境继承 `base-py311`，不能单独删除后者。
- 项目聊天的分工与交接见 `docs/CHATS.md`；聊天记录不是配置和实验结论的唯一来源。
