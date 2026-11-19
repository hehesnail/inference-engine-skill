"""Call an Anthropic-compatible LMDeploy server.

Run python samples/rest_api_messages.py, adding --stream for server-sent events.
The Anthropic SDK appends /v1 to the base URL.
"""

import argparse
import asyncio
from time import perf_counter

from anthropic import APIStatusError, AsyncAnthropic


async def anthropic_messages(base_url: str, model: str, messages: list,
                             max_tokens: int = 1024, stream: bool = False):
    async with AsyncAnthropic(api_key='YOUR_API_KEY', base_url=base_url) as client:
        options = dict(model=model, messages=messages, max_tokens=max_tokens)
        if not stream:
            return await client.messages.create(**options)

        async with client.messages.stream(**options) as events:
            saw_stop = False
            async for event in events:
                if event.type == 'message_stop':
                    saw_stop = True
            if not saw_stop:
                raise RuntimeError('Incomplete Messages stream: no message_stop event.')
            message = await events.get_final_message()
            if message.stop_reason is None:
                raise RuntimeError('Incomplete Messages stream: no stop_reason.')
            return message


async def get_model(base_url: str):
    async with AsyncAnthropic(api_key='YOUR_API_KEY', base_url=base_url) as client:
        return (await client.models.list()).data[0].id


async def async_main(base_url='http://127.0.0.1:8100', model=None, max_tokens=4096, stream=False):
    if model is None:
        model = await get_model(base_url)
    print(f'Using model: {model}')
    prompts = [
        'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
        'hakuna matata!',
        'The quick brown fox jumps over the lazy dog.',
    ]
    started = perf_counter()
    responses = await asyncio.gather(*[
        anthropic_messages(base_url, model, [{'role': 'user', 'content': prompt}], max_tokens, stream)
        for prompt in prompts
    ])
    for i, response in enumerate(responses):
        print(f'\nResponse {i} ({response.stop_reason})')
        print(''.join(block.text for block in response.content if block.type == 'text'))
        print(f'Tokens: input={response.usage.input_tokens}, output={response.usage.output_tokens}')
    print(f'Duration: {perf_counter() - started:.3f} s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8100')
    parser.add_argument('--model', help='Served model ID')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--stream', action='store_true')
    try:
        asyncio.run(async_main(**vars(parser.parse_args())))
    except (APIStatusError, RuntimeError) as error:
        raise SystemExit(str(error)) from None


if __name__ == '__main__':
    main()
