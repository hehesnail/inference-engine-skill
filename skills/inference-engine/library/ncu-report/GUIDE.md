# NCU report 技术资料与 helpers

来源：MIT HAN Lab [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill/tree/c7980a43cb88a8734d113b22b57ff66746cf5719)，MIT [LICENSE](LICENSE)。保留完整 reference、helpers 与硬件专题；原入口存为 [UPSTREAM.reference.md](UPSTREAM.reference.md)，作为该快照的历史参考。

当前任务先用本包 [NCU 指引](../../references/nsight-compute.md)。上游主要针对 NCU 2026.3 和 Blackwell；其中 `--set full`、clock/cache 配置、SM100/103、固定百分比和采集流程按需选择，不成为所有设备的默认要求。原文的硬件优化建议仍需要目标代码和测量支持。

| 任务 | 读取 |
|---|---|
| 真实 workload 与 harness | [harness](reference/02-harness-guide.md)、[模板说明](helpers/README.md) |
| 采集模式、CLI 与报告格式 | [collection](reference/03-collection.md) |
| Python 报告 API | [API](reference/04-python-api.md) |
| occupancy、tail、stall、tensor core、memory | [analysis dimensions](reference/05-analysis-dimensions.md) |
| 信号到候选优化 | [diagnosis](reference/06-diagnosis-playbook.md) |
| metric 精确名称 | [metrics](reference/08-metric-names.md) |
| 空报告、source mapping、PM 与版本问题 | [common issues](reference/09-common-issues.md) |
| Blackwell 专题 | [programming](blackwell-cuda-programming.md) |

helpers 的离线报告解析不需要 GPU，但需要对应 Nsight Compute 的 `ncu_report`；当前维护环境缺该依赖，尚未进行真实报告验证。源码/语法检查不代表运行通过。模板中的待填写段落是用户编写 harness 的位置，不是已实现的通用 kernel。

注意报告缺失/无匹配 action 时部分 helpers 会 skip 后正常退出；逐项核对输入 tag/action 和实际产物，缺指标保留 unknown。上游文中的固定阈值只能作案例线索，理论 speedup 不能代替真实对照。
