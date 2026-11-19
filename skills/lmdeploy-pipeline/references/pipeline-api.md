# Pipeline API Reference

Use this reference when the user asks about the `lmdeploy.pipeline` Python API, batch inference, streaming, chat sessions, PPL, reward scores, VLM input, `GenerationConfig`, or backend configs.

## Quick Start

```python
from lmdeploy import pipeline

model_path = '/path/to/model'
pipe = pipeline(model_path)

response = pipe('Hello, who are you?')
print(response.text)

responses = pipe(['Hello', 'Tell me a joke'])
for response in responses:
    print(response.text)
```

## Constructor

```python
from lmdeploy import ChatTemplateConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline

pipe = pipeline(
    model_path='internlm/internlm-chat-7b',
    backend_config=TurbomindEngineConfig(session_len=8192, tp=2),
    chat_template_config=ChatTemplateConfig(model_name='internlm-chat'),
    log_level='WARNING',
)

pipe = pipeline(
    model_path='internlm/internlm-chat-7b',
    backend_config=PytorchEngineConfig(session_len=8192, tp=2),
)
```

Key constructor parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | Local path, HuggingFace model id, or LMDeploy-quantized model id |
| `backend_config` | `TurbomindEngineConfig \| PytorchEngineConfig \| None` | Engine configuration |
| `chat_template_config` | `ChatTemplateConfig \| None` | Chat template settings |
| `log_level` | `str` | One of `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG` |
| `max_log_len` | `int \| None` | Max prompt chars/tokens printed in log |
| `speculative_config` | `SpeculativeConfig \| None` | Speculative decoding config |

## Backend Config Notes

- `TurbomindEngineConfig` and `PytorchEngineConfig` expose
  `language_model_only` for text-only use of hybrid VLM checkpoints without
  loading vision/multimodal encoder modules. The old
  `disable_vision_encoder` name is stale.
- Qwen3.5 VLM now has TurboMind native vision support. For native TurboMind
  vision models, the visual encoder runs in TurboMind instead of the legacy
  Python vision model path.

## Batch Inference

```python
from lmdeploy.messages import GenerationConfig

gen_config = GenerationConfig(max_new_tokens=256, temperature=0.8, top_p=0.9)

response = pipe.infer('What is AI?', gen_config=gen_config)
responses = pipe.infer(['prompt1', 'prompt2'], gen_config=gen_config, use_tqdm=True)

# pipe(...) is equivalent to pipe.infer(...)
response = pipe('What is AI?', gen_config=gen_config)
```

Important `pipe.infer()` parameters:

- `prompts`: `str | List[str] | List[Dict] | List[List[Dict]] | Tuple | List[Tuple]`
- `gen_config`: `GenerationConfig | List[GenerationConfig] | None`. For different limits or sampling settings in a batch, pass one config per prompt in the same order.
- `do_preprocess`: apply chat template
- `adapter_name`: LoRA adapter name
- `use_tqdm`: show progress bar

## Streaming

```python
for output in pipe.stream_infer('Tell me a story', gen_config=gen_config):
    print(output.text, end='', flush=True)
```

## Chat Sessions

Reuse the `Session` returned by `pipe.session()` for both turns. Save each completed response before the next call updates `session.response`. The following example uses the existing `pipe` and `gen_config`.

```python
session = pipe.session()
turns = []
try:
    for prompt in ('My name is Alice.', 'What is my name?'):
        pipe.chat(prompt, session=session, gen_config=gen_config)
        response = session.response
        turns.append({
            'prompt': prompt,
            'text': response.text,
            'token_ids': list(response.token_ids or []),
            'finish_reason': response.finish_reason,
        })
finally:
    session.close()
```

For a replay, retain the checkpoint and tokenizer revision, generation settings, ordered prompts and replies. With `stream_response=True`, exhaust the iterator before reading the completed turn. At LMDeploy 0.18.0, `chat` accumulates into the first yielded response. Copy `list(chunk.token_ids or [])` when each chunk arrives if saving its per-chunk IDs. Closing a session clears its history and response. Other sessions can continue using the pipeline.

## Perplexity

Use the tokenizer from the same checkpoint and revision as `pipe`. In the verified `AsyncEngine.async_get_ppl` implementation, both the Python result and the HTTP field named `ppl` are mean next-token negative log-likelihood. For a sequence of length `L`, the sum covers `L - 1` targets. Exponentiate the mean once for conventional perplexity. The example reuses `model_path` from Quick Start.

```python
import math

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)

ids = tokenizer.encode('The quick brown fox')
[mean_nll] = pipe.get_ppl(ids)
ppl = math.exp(mean_nll)

ids_batch = [
    tokenizer.encode('sentence one'),
    tokenizer.encode('sentence two'),
]
mean_nlls = pipe.get_ppl(ids_batch)
per_sequence_ppl = [math.exp(loss) for loss in mean_nlls]
```

Notes:

- Input must be `List[int]` or `List[List[int]]`.
- Each sequence must have length greater than 1.
- At LMDeploy `1cf5bd78`, `Pipeline.get_ppl` submits one async score per sequence through a concurrency limiter and returns results in input order. When comparing low-precision scores, record the batch composition and use the same token IDs and scoring boundary.
- For a corpus score, sum each sequence's mean loss multiplied by its scored token count, divide by the total scored tokens, then exponentiate.
- The async engine rejects PPL scoring when prefix caching or speculative
  decoding is enabled.

## Reward Score

```python
ids = tokenizer.encode('some text')
scores = pipe.get_reward_score(ids)
```

Reward scoring is only for supported reward model classes such as `InternLM2ForRewardModel` and `Qwen2ForRewardModel`.

## VLM Input

```python
from lmdeploy import ChatTemplateConfig, TurbomindEngineConfig, pipeline
from lmdeploy.vl import load_image

pipe = pipeline(
    'liuhaotian/llava-v1.5-7b',
    backend_config=TurbomindEngineConfig(session_len=8192),
    chat_template_config=ChatTemplateConfig(model_name='vicuna'),
)

image = load_image('https://example.com/image.jpg')
response = pipe(('describe this image', [image]))
print(response.text)
```

## Resource Management

```python
with pipeline('model_path') as pipe:
    response = pipe('Hello')

pipe = pipeline('model_path')
response = pipe('Hello')
pipe.close()
```
