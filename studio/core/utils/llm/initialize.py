"""Provider configuration shared by separate memory and visible-reply roles."""
from .component import ReplyModel

PROVIDER = "deepseek"
MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
CREDENTIALS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "openai": "OPENAI_API_KEY",
}

def credential_name(provider):
    """Return the conventional environment variable for a remote provider."""
    return CREDENTIALS.get(provider, "")

def credential(provider, purpose):
    """Resolve a role-specific key before falling back to provider conventions."""
    import os
    role = "VOICEMEM_MEMORY_API_KEY" if purpose == "memory" else "VOICEMEM_STUDIO_API_KEY"
    return os.environ.get(role) or os.environ.get(credential_name(provider), "")

def configuration(provider=PROVIDER):
    """Return provider settings without including credential values."""
    if provider == "deepseek":
        return {"provider": provider, "config": {"model": MODEL, "base_url": BASE_URL, "system": ""}}
    if provider == "qwen":
        return {"provider": provider, "config": {"model": "qwen3.6-flash", "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "system": ""}}
    return {"provider": provider, "config": {
        "model": "gpt-4o", "base_url": "https://api.openai.com/v1", "system": ""}}

def create(system, provider=PROVIDER, api_key=None):
    """Create the selected stream adapter; missing credentials raise immediately."""
    if provider == "local":
        from .local import LocalLLM
        from studio.paths import MODELS
        return LocalLLM(str(MODELS / "llm/Qwen3.5-4B-4bit"), system=system)
    from voicemem.reply import deepseek_reply, openai_reply
    cfg = configuration(provider)["config"]
    cfg["api_key"] = api_key or credential(provider, "reply")
    if provider == "qwen":
        if not cfg["api_key"]:
            raise ValueError("Qwen 回复需要 DASHSCOPE_API_KEY")
        return ReplyModel(deepseek_reply(**{**cfg, "system": system}, protocol="qwen"))
    factory = deepseek_reply if provider == "deepseek" else openai_reply
    return ReplyModel(factory(**{**cfg, "system": system}))
