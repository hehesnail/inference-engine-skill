# Serving Reference

Use this reference for normal LMDeploy serving, REST clients, OpenAI-compatible APIs, OpenAI Responses-compatible `/v1/responses`, Anthropic-compatible `/v1/messages`, and bundled sample scripts. For DP/EP serving, read [dp-ep.md](dp-ep.md).

## Start API Server

Use the job's assigned GPU visibility. On a shared server without an allocation, select an available GPU and set `CUDA_VISIBLE_DEVICES` before starting the service.

```bash
lmdeploy serve api_server /path/to/model --server-port 8100
```

The server has a background health monitor. Probe timing can be tuned with
`LMDEPLOY_HEALTH_POLL_INTERVAL`, `LMDEPLOY_HEALTH_PROBE_TIMEOUT`, and
`LMDEPLOY_HEALTH_UNHEALTHY_AFTER` when long backend stalls are expected.

## LoRA Adapter Selection

For a PyTorch server with named LoRA adapters, `--adapters red=/path/to/red blue=/path/to/blue` registers client-visible aliases on versions supporting this option. In `/v1/chat/completions`, set `model` to the intended base model or the exact alias returned by `/v1/models`. A Python pipeline uses `adapter_name=`. To check selection, compare the same prompt and generation settings and trace the requested alias into the target layer's `local_adapter_ids`. The order of `/v1/models` does not give those internal IDs.

## OpenAI-Compatible `v1/chat/completions`

Use the `openai` Python package and a base URL that includes `/v1`.

```python
import asyncio

from openai import AsyncOpenAI


async def chat_completions(base_url, model, messages, max_tokens=1024):
    client = AsyncOpenAI(api_key='YOUR_API_KEY', base_url=base_url)
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        top_p=1.0,
        max_completion_tokens=max_tokens,
    )


async def main():
    base_url = 'http://127.0.0.1:8100/v1'
    client = AsyncOpenAI(api_key='YOUR_API_KEY', base_url=base_url)
    model_ids = [item.id for item in (await client.models.list()).data]
    model = model_ids[0]  # Single-base example. For LoRA, select the registered alias.

    prompts = [[{'role': 'user', 'content': 'Say hello!'}]]
    responses = await asyncio.gather(*[
        chat_completions(base_url, model, prompt, max_tokens=256)
        for prompt in prompts
    ])
    for response in responses:
        print(response.choices[0].message.content)


asyncio.run(main())
```

From the skill directory, run the [chat sample](../samples/rest_api_chat.py) with `python samples/rest_api_chat.py --stream --max-tokens 8`. Omit `--stream` for a non-streaming request. Use `--base-url` and `--model` to select a different server or a specific served ID. The sample prints each assembled reply and its finish reason. To compare streaming with non-streaming output on LMDeploy 0.18.0, use the same model and message, set `temperature=0`, and pass `extra_body={'top_k': 1}` in both requests.

For a text stream, check the choice's `finish_reason`. A stop sequence or EOS can yield `stop`, and reaching the output budget yields `length`. With `include_usage`, a final usage-only chunk can have empty `choices`. Raw SSE ends with `data: [DONE]`. The OpenAI SDK consumes that sentinel. Inspect the raw stream if you need to verify its presence.

When prefix caching is enabled, OpenAI-compatible usage includes
`usage.prompt_tokens_details.cached_tokens` for accepted prefix hits.

## OpenAI Responses-Compatible `v1/responses`

Point the SDK at the LMDeploy server's `/v1` base URL. The model ID below comes from that server.

```python
from openai import OpenAI

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://127.0.0.1:8100/v1')
model = client.models.list().data[0].id

response = client.responses.create(
    model=model,
    input='Reply exactly: pong',
    max_output_tokens=32,
)
print(response.output_text)
```

Equivalent curl:

```bash
curl http://127.0.0.1:8100/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "served-model",
    "input": "Reply exactly: pong",
    "max_output_tokens": 32
  }'
```

LMDeploy behavior checked at `1cf5bd78`. For a different deployment, check its installed request adapter and parser. [LMDeploy Responses request adapter](https://github.com/InternLM/lmdeploy/blob/1cf5bd788a175b4d1c80dd49bd42377252439e5b/lmdeploy/serve/openai/responses/request.py)

- `input` accepts a string or supported message, `function_call`, and `function_call_output` items. This revision also accepts `input_image` URL/data parts in user messages and function outputs.
- The adapter builds one leading `system` message from top-level `instructions` followed by the `developer` and `system` content in input order. It then renders the conversation with the served model's chat template.
- Function tools are converted to LMDeploy OpenAI-compatible tools. Tool
  calling still requires `--tool-call-parser ...`.
- `parallel_tool_calls` defaults to `true`; when `false`, only the first parsed
  function call is returned.
- Hosted/non-function tools such as `web_search` are accepted but ignored.
- Responses-only logprob serialization and stream obfuscation options are
  accepted but ignored.
- `background` and `previous_response_id` are not supported.
- Streaming uses SSE events such as `response.created`,
  `response.output_text.delta`, `response.function_call_arguments.delta`, and
  `response.completed`.

## Anthropic-Compatible `v1/messages`

The `anthropic` Python package appends `/v1` to the server base URL. A raw HTTP client sends the `anthropic-version: 2023-06-01` header.

- Wrong: `http://127.0.0.1:8100/v1`, which becomes `/v1/v1/messages`.
- Correct: `http://127.0.0.1:8100`, which becomes `/v1/messages`.

```python
import asyncio

from anthropic import AsyncAnthropic


async def anthropic_messages(base_url, model, messages, max_tokens=1024):
    client = AsyncAnthropic(api_key='YOUR_API_KEY', base_url=base_url)
    return await client.messages.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )


async def main():
    base_url = 'http://127.0.0.1:8100'
    client = AsyncAnthropic(api_key='YOUR_API_KEY', base_url=base_url)
    model = (await client.models.list()).data[0].id

    prompts = [[{'role': 'user', 'content': 'Say hello!'}]]
    responses = await asyncio.gather(*[
        anthropic_messages(base_url, model, prompt, max_tokens=256)
        for prompt in prompts
    ])
    for response in responses:
        print(''.join(block.text for block in response.content if block.type == 'text'))
        print(response.stop_reason)


asyncio.run(main())
```

From the skill directory, run the [Messages sample](../samples/rest_api_messages.py) with `python samples/rest_api_messages.py --stream --max-tokens 8`. The same script accepts `--base-url`, `--model`, and non-streaming requests. Anthropic SDK 2.3.0 rejects direct `temperature` and `top_p` arguments before sending a request. For a deterministic comparison on LMDeploy 0.18.0, pass `extra_body={'temperature': 0, 'top_k': 1}` to the SDK.

For a Messages stream, keep the stop reason from `message_delta` and consume the final `message_stop`. An explicit stop sequence produces `stop_sequence`, and reaching the output budget produces `max_tokens`. If the connection ends before the required terminal event, report the response as incomplete.

On LMDeploy 0.18.0 both endpoints returned HTTP 404 for an unknown served model and HTTP 400 for a zero output budget or negative `top_p`. These errors arrived as JSON even with streaming requested. Check the HTTP status and body before parsing events. A valid output-budget stop returns HTTP 200 and its protocol-specific reason.

Extended LMDeploy behavior on `/v1/messages`:

- `messages` may be omitted or empty only when `input_ids` is supplied.
- Non-empty `messages` is mutually exclusive with `input_ids` and
  `image_data`.
- `image_data` is only accepted with `input_ids` when `messages` is empty.
- Raw `input_ids` bypass message rendering, so `system`, `tools`, and explicit
  non-auto `tool_choice` are rejected on that path.
- `return_token_ids`, `return_logprob`, and `return_routed_experts` can expose
  output ids, raw `(logprob, token_id)` pairs, and routed experts.
- `return_logprob` requires engine `logprobs_mode`; `return_routed_experts`
  requires starting api_server with `--enable-return-routed-experts`.
- Streaming usage follows the Anthropic event shape: `message_start` includes
  input and initial output token counts, while terminal `message_delta` carries
  output token usage.

## Perplexity Endpoint

`POST /get_ppl` returns mean next-token negative log-likelihood in the `ppl` field on the verified implementation. For conventional perplexity, exponentiate the returned value once. The [Python scoring example](pipeline-api.md#perplexity) also covers batch and corpus aggregation.
The request input may be raw text or token ids; raw text is tokenized directly
without applying a chat template.

The endpoint rejects requests when prefix caching or speculative decoding is
enabled, matching `AsyncEngine.async_get_ppl()`.

```bash
curl http://127.0.0.1:8100/get_ppl \
  -H "Content-Type: application/json" \
  -d '{"input": "The quick brown fox"}'
```

```bash
curl http://127.0.0.1:8100/get_ppl \
  -H "Content-Type: application/json" \
  -d '{"input": [1, 2, 3, 4]}'
```

## DP/EP Serving Reminder

For multiple DP/EP workers behind a proxy, check a completion on the proxy and identify the worker that handled it. For missing work in a worker trace, check the [DP/EP profiling window](dp-ep.md#dpep-profiler-pitfalls). If prefill and decode use separate services and the model lists successfully but generation stalls, check [PD connection warmup and the completion path](pd-disagg.md#smoke-test).
