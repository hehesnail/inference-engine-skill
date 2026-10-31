<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# PyTorch Profiler API Reference

Check the installed version and event object. PyTorch 2.0.1 exposes CUDA-named fields; newer versions use device-named fields and may keep deprecated CUDA aliases. Do not infer that every `cuda_time_total` access or sort key fails on all 2.0+ versions.

| Field on current aggregated events | Meaning |
|---|---|
| `device_time` | Mean device duration per invocation: `device_time_total / count` |
| `device_time_total` | Accumulated device duration, microseconds |
| `self_device_time_total` | Accumulated duration excluding child attribution |
| `cpu_time` / `cpu_time_total` | Mean per invocation / accumulated CPU duration |

Choose the family once and keep units and count:

```python
for event in prof.key_averages():
    family = "device" if hasattr(event, "device_time_total") else "cuda"
    total_us = getattr(event, f"{family}_time_total")
    self_us = getattr(event, f"self_{family}_time_total")
    mean_us = total_us / event.count if event.count else 0.0
    print(event.key, event.count, total_us, self_us, mean_us)
```

For `table(sort_by=...)`, use a field supported by that version. Current `_build_table` translates CUDA/XPU sort names to device names; old examples are not uniformly errors. Summing attributed operator durations can double-count nesting or overlap; it is not request latency or the union of GPU-busy intervals.

Source: [PyTorch 2.0.1 implementation](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/autograd/profiler_util.py), [current FunctionEventAvg contract](https://docs.pytorch.org/docs/main/generated/torch.autograd.profiler_util.FunctionEventAvg.html), [current implementation](https://github.com/pytorch/pytorch/blob/main/torch/autograd/profiler_util.py). NVTX domain strings and push/pop names follow the [official annotation API](https://nvidia.github.io/NVTX/python/reference.html).
