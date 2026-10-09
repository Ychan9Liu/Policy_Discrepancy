# 运行脚本

此目录预留环境核对、实验启动和结果汇总脚本。脚本实现前，实验依照 `docs/EXPERIMENTS.md` 记录提交、实际配置、seed、服务器和 GPU。服务器无调度器，正式运行方式待确定后固化到脚本。

`run_size50m_integration.py` 固定隔离的4098动作工程配置，覆盖四个真实DMC任务的size50m四模式、工程c冻结、完成重开及可选补验；每次启动检查GPU占用，拒绝覆盖既有目录。它不启动正式百万步预算。`analyze_size50m_integration.py` 只读审计收据/快照/配置，汇总资源。使用范围、已发现故障和不能宣称通过的事项见 `docs/SERVER_INTEGRATION_SIZE50M.md`；这些脚本不是正式实验调度器。

04 原始return及修复后真实CUDA验收见 `docs/SERVER_RETURN_CUDA_AUDIT.md`。`run_size50m_integration.py` 的supervisor显式设置并记录 `CUDA_VISIBLE_DEVICES` 与 `MUJOCO_EGL_DEVICE_ID`，依然要求每次启动GPU占用为0；EGL枚举与实际UUID映射需在使用的服务器/卡现场确认。该脚本固定缩小工程预算，不是正式启动入口。
