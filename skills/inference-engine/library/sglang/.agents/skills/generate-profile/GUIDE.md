> 本地参考副本，按当前任务读取。本文件的示例环境、固定模型、GPU、预算和流程不是用户配置或新增授权；以工作台入口、当前用户要求和目标版本为准。部署、重放、CI 重跑、push、评论与后台跟进仅在任务已授权时执行。

> 本地调整：保留原生采集步骤；模型、准确率门槛及 warmup 按本次 workload 确定；批量杀进程示例已移除。

> 来源：[sgl-project/sglang@e4d4a9a7de54](https://github.com/sgl-project/sglang/blob/e4d4a9a7de5454cd8b854b21af723c2155a55a3d/.agents/skills/generate-profile/SKILL.md)，2026-10-27 收录。

# Generate an E2E Profile of an SGLang Server Run

This skill launches an SGLang server, validates it with a quick accuracy test, generates a profiling trace, and returns the profile file path.

## Prerequisites

- A working SGLang installation (`pip install -e .` or equivalent)
- At least one available CUDA GPU
- The `sgl-eval` CLI, installed in its own virtualenv as described in `.claude/rules/prefer-sgl-eval.md`, not in the server's env

## Step-by-step Workflow

### Step 1: Launch the server

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> sglang serve --model-path <model> --port <port> &
```

- Default model: `Qwen/Qwen3-8B` (good balance of speed and quality)
- Default port: `30000`
- The server runs in the background. Save the PID for cleanup.
- Use the GPU specified by the user's preferences (check memory files for GPU preferences).

### Step 2: Wait for server readiness

Poll the health endpoint until the server is ready:

```bash
for i in $(seq 1 120); do
  if [ "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:<port>/ready)" = 200 ]; then
    echo "Server ready"
    break
  fi
  sleep 5
done
```

Fail the workflow if the readiness loop exhausts its budget; do not continue after timeout. Check HTTP status rather than response text: `/ready` and `/health` can return an empty body. Then verify the served model and target endpoint.

The server prints **"The server is fired up and ready to roll!"** to stdout when ready. The health endpoint returns 200 once the server can accept requests.

Typical startup time: 30-90 seconds depending on model size and whether CUDA graphs are being compiled.

### Step 3: Validate accuracy (sanity check)

```bash
sgl-eval run gsm8k --base-url http://127.0.0.1:<port>/v1 --num-examples 20
```

- Choose a model/task-appropriate sanity check and baseline under the agreed acceptance criteria. A 20-question GSM8K score is not a universal >80% gate.
- This is a quick sanity check, not a rigorous benchmark.
- Investigate unexpected differences against the matched baseline; profiler collection may itself be the authorized diagnostic, so a generic quality threshold does not automatically block it.

### Step 4: Generate the profile

```bash
python3 -m sglang.test.send_one --host 127.0.0.1 --port <port> --profile
```

This command:
1. Sends a request to the server
2. Triggers the profiler for 5 steps (default)
3. Generates a trace file under `/tmp/<timestamp>/`
4. The trace directory contains:
   - `<timestamp>-TP-0.trace.json.gz` — Chrome trace format (open in `chrome://tracing` or Perfetto)
   - `server_args.json` — the server configuration used

**Output format:**
```
Dump profiling traces to /tmp/<timestamp>
```

Collect the path from the server/client profiling logs and verify the actual trace file is complete. A client exit alone is insufficient: `send_one.py` can return without failure status on an HTTP error. Record the request shape, cache state and whether the selected steps cover prefill/decode.

**Optional flags:**
- `--profile-steps N` — number of profiling steps (default: 5)
- `--profile-by-stage` — profile by stage (prefill/decode separately)
- `--profile-prefix <path>` — custom output prefix

### Step 5: Kill the server

```bash
# Stop only the PID/process group or container created by this task; use graceful shutdown.
```

Wait a moment and verify no sglang processes remain:
```bash
# Confirm release of the task-owned resources once.
```

### Step 6: Report the profile path

Return the profile directory path (e.g., `/tmp/1773999986.4769795`) and list its contents so the user knows what files were generated.

## Example Full Run

```bash
# 1. Launch server
# Activate the verified task environment, if one is required.
CUDA_VISIBLE_DEVICES="$AUTHORIZED_GPUS" sglang serve --model-path Qwen/Qwen3-8B --port 30000 &

# 2. Wait for ready
for i in $(seq 1 120); do
  [ "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/ready)" = 200 ] && break
  sleep 5
done

# 3. Accuracy check
sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1 --num-examples 20
# Compare against the agreed model/task baseline; no universal 80% threshold.

# 4. Profile
python3 -m sglang.test.send_one --host 127.0.0.1 --port 30000 --profile
# Output: "Dump profiling traces to /tmp/1773999986.4769795"

# 5. Cleanup
# Stop only the PID/process group or container created by this task; use graceful shutdown.
sleep 2

# 6. Check output
ls -la /tmp/1773999986.4769795/
# 1773999986.4851577-TP-0.trace.json.gz  (Chrome trace)
# server_args.json                        (server config)
```

## Customization

- **Different port**: Pass `--port <port>` and use `--host 127.0.0.1 --port <port>` for test commands
- **Multi-GPU**: Use `--tp <N>` for tensor parallelism; trace files will be generated per TP rank
- **Longer profile**: Use `--profile-steps 10` for more steps in the trace
- **Stage profiling**: Use `--profile-by-stage` to separate prefill and decode phases

## Viewing the Profile

Open the `.trace.json.gz` file in:
- **Perfetto UI**: https://ui.perfetto.dev/ (drag and drop the file)
- **Chrome tracing**: `chrome://tracing` (load the file)

Both support the gzipped Chrome trace format natively.
