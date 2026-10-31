> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[NVIDIA/skills@25340619336e](https://github.com/NVIDIA/skills/blob/25340619336e89bcc40365c28bfe9e790430468e/skills/doca-perftest/SKILL.md)，2026-10-31 核对。

# DOCA Perftest

## When to load this skill

Load this skill when the user wants to — or the agent needs to — run
`doca_perftest` on a host with DOCA installed. Concretely:

- Measuring RDMA bandwidth, message rate, or latency between two hosts
  (point-to-point) with a verb such as write / read / send.
- Running the same benchmark across many nodes — JSON scenario or
  `--traffic_pattern` (pair-wise, all-to-all), orchestrated over
  OpenMPI / passwordless SSH.
- Comparing the IBV and DOCA Verbs backends (`-r ibv` vs `-r dv`) or
  driving the GPUNetIO engine (`--engine gpunetio`).
- Interpreting a run: client-vs-server number drift, "device busy",
  inconsistent latency across runs.

Do **not** load this skill for DOCA install (`doca-setup`), the doca-bench harness,
Flow pipeline tuning (`doca-flow-tune`), or application-level
throughput.

## Examples

- "Measure RDMA write bandwidth between two ConnectX-7 hosts with 64 KB messages." Point-to-point: run the server, then the client with `-n`.
- "I need latency across 16 nodes, all-to-all." Multi-node: JSON scenario or `--traffic_pattern`, with OpenMPI.
- "Compare the IBV and DOCA Verbs drivers on the same device." Same command, `-r ibv` vs `-r dv`.

## Instructions

Follow the loading order below. Open `TASKS.md` for the commands and `CAPABILITIES.md` for what the tool can measure and what it does not.

**Where to start:** DOCA Perftest is the DOCA 3.5 replacement for legacy `ib_write_bw` / `ib_send_lat` plus the orchestration scripts used to run them on many nodes. Start at `TASKS.md ## point-to-point` for a first run. Use `## multi-node` for scenarios and `## debug` when a run fails or the numbers look wrong.

If DOCA is not installed, route to `doca-setup/SKILL.md`. Client and server must run the same DOCA Perftest version; check with `doca-version/SKILL.md` before comparing numbers across hosts. If the question is about GPU-initiated RDMA in the user's own code, route to `doca-gpunetio/SKILL.md`.

## Example questions this skill answers well

- **"Run a point-to-point RDMA write bandwidth test."** Use `-d <dev> -n <server> -v write -m bw -s <size> -D <sec>`; the client coordinates the server side over SSH (requires SSH key exchange set up beforehand).
- **"Benchmark 16 nodes at once and get one report."** Use a JSON scenario or `--traffic_pattern`; the report aggregates per device, node and test.
- **"Latency differs between runs."** Check the latency mode, the core and NUMA placement, and whether both sides use the same message size and verb.

## Loading order

1. This file for scope and routing.
2. `TASKS.md` for the exact command for the job you are doing.
3. `CAPABILITIES.md` when you need to know whether a verb, driver, memory type or traffic pattern is supported.

Do not open every companion file.

## Scope boundaries

- Installs and driver packages: `doca-setup/SKILL.md`.
- Version matching across client, server and DOCA release: `doca-version/SKILL.md`.
- Any change to NIC or eswitch state, firmware, or host kernel settings (hugepages, IOMMU, PCIe rebind): follow `doca-hardware-safety/SKILL.md` first.
- Failure diagnosis: `doca-debug/SKILL.md` for the layer ladder, then the `TASKS.md ## debug` anchor here.
- Upstream CPU-initiated `ib_write_bw`: this tool replaces it on DOCA 3.5; do not mix the two in one comparison without saying so.
