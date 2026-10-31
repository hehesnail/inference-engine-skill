> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[NVIDIA/TensorRT-LLM@f405e10b730e](https://github.com/NVIDIA/TensorRT-LLM/blob/f405e10b730ea953961ecf5719242c581527587a/.claude/skills/trtllm-visualgen-bench/SKILL.md)，2026-10-31 核对。

# /trtllm-visualgen-bench

Both halves of a VisualGen benchmark: the server it runs against, and the client that measures it.

```
0. prepare the environment
1. pick the workload, copied into a fresh run directory
2. bring up its server (skip if given one)
3. run
4. read the result
5. shut the server down (only if step 2 started it)
```

## Arguments

```
/trtllm-visualgen-bench [--workload <path>] [--server_config <path>] [--model_path <checkpoint>]
```

Each settles a decision this flow otherwise puts to the user, so a run given all three asks
nothing further about those inputs. Comparability also requires the same checkpoint revision, installed source/container, hardware, traffic, warmup/cache state and measurement boundaries. Parse them from
`$ARGUMENTS`; anything else there is the request in prose.

| flag | settles | left to ask without it |
|---|---|---|
| `--workload` | the request — backend, prompt, shape, references, `extra_params` | bench step 1, *Pick the workload*: which one, since a model name names several |
| `--server_config` | the whole `--visual_gen_args` YAML, served byte for byte | serve step B, *Generate the config*: the parallel sizes and the attention backend |
| `--model_path` | the checkpoint `trtllm-serve` is given | serve step A, *Resolve the model*: `--find` resolves it, and asks when more than one row matches |

**`--server_config` is served as given — nothing is appended to it**, including the
`compilation_config` serve step B would have derived. Most shipped configs carry no warmup shapes, and
without explicit warmup, compilation can enter the first measured request for a new shape. The benchmark also sends an initial test request by default outside the measurement; record that pre-test and warmup policy rather than labeling every first measured request cold.
`serve_wait.py` says so, as a warning rather than a failure, because a config that asked for no
warmup got none. Read that line before trusting the numbers.

A `model:` key inside it does not set the checkpoint. `VisualGen.__init__` overwrites that field
with the CLI positional, so the YAML's value never reaches the engine and nothing raises: a
config naming one checkpoint, served against another, measures the other one. `--model_path` is
what decides it.

## 0. Prepare the environment

Use the target, version-recorded container where `import tensorrt_llm` works. Check ffmpeg availability on the server for MP4; prepare missing dependencies only within the authorized environment. Do not assume every image lacks ffmpeg or run a system install as part of read-only analysis: `--format mp4` fails without it, and the server's `auto` default quietly writes AVI instead — which also drops the audio track of an `enable_audio` workload.

Serving needs more — the checkpoint store, and whatever the model demands; [`serve/README.md`](serve/README.md) lists those.

## 1. Pick the workload

**`--workload`, or the user names one. Nothing here picks one for them.** A model name is not enough: most of these checkpoints have several files and each is a different measurement, so a bare model name gets the list back.

> `nvidia/Cosmos3-Super` has `-t2v`, `-i2v` and `-t2av`. Which one?

**A `--workload` outside `workloads/` is run where it sits.** Its references resolve against its
own directory, so copying it into the run directory without the media beside it fails at load.
Keep the exact source and dependency hashes, or snapshot the workload and its referenced files preserving relative layout. A path alone is mutable and does not preserve the measured workload.

`workloads/` sits next to this file. One directory per run, holding everything the run produced and the workload it ran:

Create a unique run directory. Snapshot the selected workload plus referenced media, prompt and action files while preserving their relative directory layout. Set `RUN_WORKLOAD` to that snapshot, and record hashes of each dependency. If absolute paths are required, parse YAML and rewrite only `prompt_file`, `extra_params.action_file`, and media-reference `content` whose `format` is `path`; preserve prompts, URLs and base64 byte-for-byte. Do not use global `sed` replacement of `../`.

The bundled `media/image_with_red_background.png` and `media/landscape.png` at the audited revision are Git LFS pointer text, not decoded PNGs. Workloads needing them remain unavailable until the actual assets are materialized on the authorized server. Verify file signatures and decoding before launching; do not infer availability from a `.png` filename.


```
<run dir>/
└── vg-bench-<stamp>/                   # one per workload benchmarked
    ├── wan22-i2v-a14b.yaml                 # step 1: the workload as run, paths absolute
    ├── reference_media/                    # step 1: the media the server opens
    ├── result.json                         # step 3: --save-result [--save-detailed]
    ├── bench.log                           # step 3: the client's stdout and stderr
    ├── output_media/                       # step 3: --output-media-dir, one file per output
    └── serve/                              # step 2: only when this run started the server
        └── vg-serve-<stamp>/
            ├── wan22-i2v-a14b_2gpu.yaml
            └── server.log
```

Step 3 runs `RUN_WORKLOAD`; validate all resolved references before execution. Relative paths resolve from the workload document, as implemented in `benchmark_visual_gen.py:load_workload`. Keep prompts and action data with the snapshot, not merely mutable paths back to the skill.

Include all referenced inputs in the snapshot or a fixed dependency manifest. The client reads a prompt or action file into the request before sending, so those two need only to be readable by the client; a `format: path` reference is a path the **server** opens, and `reference_media/` is what keeps it beside the numbers it produced.

When reusing an authorized server, capture its model/checkpoint, process launch configuration, installed revision/image, GPU mapping and warmup state beside the result. Mark unresolved identity fields unknown.

One file per distinct workload shape, keyed by HF model id. [`workloads/README.md`](workloads/README.md) covers what is pinned and why.

A workload carries the whole request — prompt, shape, steps, guidance, seed. The client also takes those as flags of the same name for a one-off request, but **a benchmark you intend to compare belongs in a file**, which the run keeps as the record of what it measured.

## 2. Bring up its server

**Is the intended server already running?** A successful `/health` response establishes liveness only. Verify server ownership/authorization, model/checkpoint and effective configuration through its deployment record or supported model/info endpoint before reuse. An unrelated healthy process on the same port is not the requested benchmark target.

Otherwise [`serve/README.md`](serve/README.md), with `LOGDIR=$RUN/serve` so its tree lands inside this run. The workload supplies the pinned shape, which `scripts/serve_check.py --warmup-from` turns into the warmup config. Its first line names the checkpoint, but as a comment — serve step A is what resolves it.

`--model_path` settles serve step A and `--server_config` settles serve step B, so with both the
flow starts at serve step C. Serve step D still runs: this flow started the server, so this flow
stops it.

## 3. Run

```bash
python -m tensorrt_llm.serve.scripts.benchmark_visual_gen \
  --workload "$RUN_WORKLOAD" --port "$PORT" \
  --max-concurrency 1 --save-result --save-detailed \
  --num-gpus $NGPU --output-media-dir $RUN/output_media \
  --result-dir $RUN --result-filename result.json > $RUN/bench.log 2>&1
```

`--help` lists every flag; it is the authority, not this file. These five carry a reason the help text does not:

- **`--format mp4`**, added on `backend: openai-videos`. Where ffmpeg is installed the server's `auto` default already resolves to MP4; where it is missing, `mp4` fails at once while `auto` falls back to a pure-Python encoder that JPEGs every frame into an AVI, inside the measured window. The image routes take `png`/`webp`/`jpeg` and default to `png`.
- **`--max-concurrency 1`** measures a serial latency point. Throughput comparisons may use other fixed concurrency/rate points; keep those identical across baseline/candidate and report each point separately.
- **`--save-detailed`** for any server-side number — without it the result has no `timings.server_*`.
- **`--output-media-dir`** puts the run's media beside its numbers, which step 4 checks. The client creates the directory, and writes outside the measured window.
- **`--num-gpus $NGPU`** is the `n_workers` step 2 computed from the parallel sizes. The same model and workload at 1, 4 or 8 GPUs are different measurements, and nothing else in the result names the count.

## 4. Read the result

[`reference/wan22-ti2v-5b-t2v.json`](reference/wan22-ti2v-5b-t2v.json) is what `--save-result --save-detailed` wrote for one real run. Read it for the shape of the output; what follows is only what the numbers mean.

What each number measures, and how to read `e2e - gen` and `gen - server_gen`, is in
[`tensorrt_llm/serve/scripts/BENCHMARKING_VISUAL_GEN.md`](https://github.com/NVIDIA/TensorRT-LLM/blob/f405e10b730ea953961ecf5719242c581527587a/tensorrt_llm/serve/scripts/BENCHMARKING_VISUAL_GEN.md).
Two things to check before trusting a run:

**A run with failures is incomplete evidence.** Retain all attempts, failures and saved results; do not report successful-request-only latency as an equivalent full workload or use it to rank candidates.

**A completed request says nothing about the media.** Exit status and `success` describe the transfer: confirm the files in `output_media/` are non-zero, decode, and match the resolution the workload pins.

On a workload carrying `enable_audio`, confirm the track reached the file — the encoder drops audio with a warning and still returns 200:

```bash
ffprobe -v error -select_streams a -show_entries stream=codec_type -of csv=p=0 $RUN/output_media/0000_0.mp4
```

**One server at a time for anything you intend to compare.** Servers sharing a node contend for CPU and PCIe, so numbers taken while another is running measure the contention as much as the model.

## 5. Shut the server down

Only what step 2 started. [`serve/README.md`](serve/README.md) covers it.
