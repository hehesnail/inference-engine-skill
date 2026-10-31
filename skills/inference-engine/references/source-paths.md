# 正文中的引擎源码与生成文档路径

以下链接来自固定快照，目标是实际引擎 checkout、构建生成的页面或历史文档，不是本地 skill 的缺失依赖。日常专项说明和工具已经本地化；修改实现时仍应检查用户实际版本的源码。

| 来源 | 原链接 | 定位方式 |
|---|---|---|
| `sglang/.agents/skills/mechanical-refactor-verify/SKILL.md` | `scripts/tests/` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/.agents/skills/model-recipes/SKILL.md` | `../../../recipes/models` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/.agents/skills/pr-checklist/SKILL.md` | `../../../tests/AGENTS.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/AGENTS.md` | `tests/AGENTS.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/README.md` | `../../DCO` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/README.md` | `../../docker/Dockerfile` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/README.md` | `../../mkdocs.yaml` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/README.md` | `../community/reviewers.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/README.md` | `../getting_started/installation/README.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/README.md` | `../governance/committers.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/README.md` | `./jit_kernel_warmup.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/ci/failures.md` | `../../../.buildkite/scripts/ci-clean-log.sh` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/ci/failures.md` | `../../../.buildkite/scripts/ci-fetch-log.sh` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/ci/failures.md` | `../../../.buildkite/scripts/rerun-test.sh` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/incremental_build.md` | `../getting_started/installation/gpu.md#troubleshooting` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/config/compilation.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/model_executor/layers/mamba/mamba_mixer.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/model_executor/layers/mamba/mamba_mixer2.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/model_executor/layers/mamba/short_conv.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/model_executor/models` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/model_executor/models/config.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/model_executor/models/jamba.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/model_executor/models/lfm2.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/model_executor/models/llama.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/model_executor/models/mamba.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/model_executor/models/mamba2.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/model_executor/models/nemotron_h.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/model_executor/models/opt.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/v1/attention/backends/linear_attn.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/v1/attention/backends/registry.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `../../../vllm/v1/attention/backends/short_conv_attn.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/basic.md` | `registration.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/tests.md` | `../../../tests/models` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/tests.md` | `../../../tests/models/multimodal/processing` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/tests.md` | `../../../tests/models/multimodal/processing/test_common.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/tests.md` | `../../../tests/models/multimodal/processing/test_phi3v.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/tests.md` | `../../../tests/models/registry.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/tests.md` | `../../../tests/models/utils.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/tests.md` | `../../models/generative_models.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/contributing/model/tests.md` | `../../models/pooling_models/README.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/design/nixl_kv_cache_lease.md` | `../features/nixl_connector_usage.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/features/disagg_prefill.md` | `../../examples/disaggregated/example_connector/run.sh` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/features/disagg_prefill.md` | `../../examples/disaggregated/flexkv_connector/prefix_caching_flexkv.py` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/features/disagg_prefill.md` | `../../examples/disaggregated/lmcache/README.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/features/disagg_prefill.md` | `../../examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/features/disagg_prefill.md` | `../../examples/disaggregated/mooncake_connector/run_mooncake_connector.sh` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/features/disagg_prefill.md` | `../../tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/features/disagg_prefill.md` | `../serving/online_serving/token_in_token_out.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/features/disagg_prefill.md` | `kv_offloading_usage.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/features/disagg_prefill.md` | `mooncake_connector_usage.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/features/disagg_prefill.md` | `mooncake_ec_connector_usage.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/features/disagg_prefill.md` | `moriio_connector_usage.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/features/disagg_prefill.md` | `nixl_connector_compatibility.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/features/disagg_prefill.md` | `nixl_connector_usage.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/usage/security.md` | `../design/endpoint_plugins.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/usage/security.md` | `../design/prefix_caching.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |
| `vllm/docs/usage/security.md` | `../features/lora.md` | 在目标 checkout 检索；CLI 生成页使用该环境的帮助。 |