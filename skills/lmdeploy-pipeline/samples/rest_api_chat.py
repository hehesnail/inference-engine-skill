"""Call an OpenAI-compatible LMDeploy server.

Run python samples/rest_api_chat.py, adding --stream for server-sent events.
"""

import argparse
import asyncio
from time import perf_counter

from openai import APIStatusError, AsyncOpenAI


async def chat_completions(base_url: str, model: str, messages: list,
                           max_tokens: int = 1024, stream: bool = False):
    async with AsyncOpenAI(api_key='YOUR_API_KEY', base_url=base_url) as client:
        options = dict(model=model, messages=messages, temperature=1.0, top_p=1.0,
                       max_completion_tokens=max_tokens)
        if not stream:
            return await client.chat.completions.create(**options)

        async with client.chat.completions.stream(**options, stream_options={'include_usage': True}) as events:
            saw_finish = False
            async for event in events:
                if event.type == 'chunk' and any(choice.finish_reason for choice in event.chunk.choices):
                    saw_finish = True
            if not saw_finish:
                raise RuntimeError('Incomplete chat stream: no choice supplied a finish_reason.')
            # The parsed final-completion helper raises on a valid length-limited reply.
            return events.current_completion_snapshot


async def async_main(base_url='http://127.0.0.1:8100/v1', model=None, max_tokens=4096, stream=False):
    if model is None:
        async with AsyncOpenAI(api_key='YOUR_API_KEY', base_url=base_url) as client:
            model = (await client.models.list()).data[0].id
    print(f'Using model: {model}')
    prompts = [
        'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
        'hakuna matata!',
        'The quick brown fox jumps over the lazy dog.',
    ]
    started = perf_counter()
    responses = await asyncio.gather(*[
        chat_completions(base_url, model, [{'role': 'user', 'content': prompt}], max_tokens, stream)
        for prompt in prompts
    ])
    for i, response in enumerate(responses):
        choice = response.choices[0]
        print(f'\nResponse {i} ({choice.finish_reason})')
        print(choice.message.content or '')
        if response.usage is not None:
            print(f'Tokens: input={response.usage.prompt_tokens}, output={response.usage.completion_tokens}')
    print(f'Duration: {perf_counter() - started:.3f} s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8100/v1')
    parser.add_argument('--model', help='Served model ID or registered adapter alias')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--stream', action='store_true')
    try:
        asyncio.run(async_main(**vars(parser.parse_args())))
    except (APIStatusError, RuntimeError) as error:
        raise SystemExit(str(error)) from None


if __name__ == '__main__':
    main()
