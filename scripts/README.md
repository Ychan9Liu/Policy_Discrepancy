# 运行脚本

此目录预留环境核对、实验启动和结果汇总脚本。脚本实现前，实验依照 `docs/EXPERIMENTS.md` 记录提交、实际配置、seed、服务器和 GPU。服务器无调度器，正式运行方式待确定后固化到脚本。

`run_size50m_integration.py` 固定隔离的4098动作工程配置，覆盖四个真实DMC任务的size50m四模式、工程c冻结、完成重开及可选补验；每次启动检查GPU占用，拒绝覆盖既有目录。它不启动正式百万步预算。`analyze_size50m_integration.py` 只读审计收据/快照/配置，汇总资源。使用范围、已发现故障和不能宣称通过的事项见 `docs/SERVER_INTEGRATION_SIZE50M.md`；这些脚本不是正式实验调度器。
