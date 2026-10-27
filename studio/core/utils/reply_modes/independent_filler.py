"""Independent optional wording through a configured non-thinking chat model."""
from __future__ import annotations

import asyncio
import os
import math
from urllib.parse import urlsplit


def generation_timeout():
    """Return the separate network-inclusive auxiliary text budget."""
    value = float(os.environ.get('STUDIO_FILLER_TIMEOUT_S', '3.0'))
    if not math.isfinite(value) or value <= 0:
        raise ValueError('STUDIO_FILLER_TIMEOUT_S must be positive and finite')
    return value


def configuration():
    """Resolve auxiliary settings without mutating the main reply provider."""
    from studio.core.utils.component_config.component import endpoint
    from studio.core.utils.llm.initialize import configuration as llm_config, credential, credential_name
    reply_provider = os.environ.get('VOICEMEM_STUDIO_PROVIDER') or 'deepseek'
    provider = os.environ.get('STUDIO_FILLER_PROVIDER') or reply_provider
    if provider not in {'deepseek', 'qwen', 'openai'}:
        raise ValueError('STUDIO_FILLER_PROVIDER must be deepseek, qwen or openai; local servers use openai')
    if provider == reply_provider:
        defaults = llm_config(provider)['config']
    else:
        defaults = {
            'deepseek': {'model': 'deepseek-v4-flash', 'base_url': 'https://api.deepseek.com'},
            'qwen': {'model': 'qwen3.6-flash', 'base_url': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'},
            'openai': {'model': 'gpt-4o', 'base_url': 'https://api.openai.com/v1'},
        }[provider]
    model = os.environ.get('STUDIO_FILLER_MODEL') or defaults['model']
    base = endpoint(os.environ.get('STUDIO_FILLER_BASE_URL') or defaults['base_url'])
    if 'STUDIO_FILLER_API_KEY' in os.environ:
        key = os.environ['STUDIO_FILLER_API_KEY']
    elif base != endpoint(defaults['base_url']):
        key = ''
    else:
        key = (credential(provider, 'reply') if provider == reply_provider
               else os.environ.get(credential_name(provider), ''))
    from ipaddress import ip_address
    host = urlsplit(base).hostname
    local = host == 'localhost'
    try:
        local = local or ip_address(host).is_loopback
    except ValueError:
        pass
    if not key and not local:
        raise ValueError('Independent work fillers require STUDIO_FILLER_API_KEY or the selected provider credential')
    if not model.strip():
        raise ValueError('Independent work fillers require a model')
    return provider, model, base, key or 'local'


async def generate(system, prompt, *, max_tokens=40, timeout_s=1.2):
    """Bound auxiliary text only; cancellation closes its independent HTTP stream."""
    if max_tokens <= 0 or timeout_s <= 0:
        return ''
    provider, model, base, key = configuration()
    from openai import AsyncOpenAI
    from .short_text import first_sentence

    async def request():
        options = {}
        if provider == 'deepseek':
            options['extra_body'] = {'thinking': {'type': 'disabled'}}
        elif provider == 'qwen':
            options['extra_body'] = {'enable_thinking': False}
        # Each request owns its client across startup/server event-loop lifetimes.
        async with AsyncOpenAI(api_key=key, base_url=base, timeout=timeout_s,
                               max_retries=0) as client:
            stream = await client.chat.completions.create(
                model=model, messages=[{'role': 'system', 'content': system},
                                       {'role': 'user', 'content': prompt}],
                max_tokens=max_tokens, temperature=.7, stream=True, **options)
            chunks = []
            try:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    text = chunk.choices[0].delta.content
                    if text:
                        chunks.append(text)
                        sentence = first_sentence(''.join(chunks))
                        if sentence:
                            return sentence
                return first_sentence(''.join(chunks))
            finally:
                await stream.close()

    try:
        return await asyncio.wait_for(request(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return ''
