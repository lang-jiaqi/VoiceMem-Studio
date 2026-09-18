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

def configuration(provider=PROVIDER, purpose="reply"):
    """Return the selected role's non-secret provider settings."""
    import os
    defaults = {
        "deepseek": (MODEL, BASE_URL),
        "qwen": ("qwen3.6-flash", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        "openai": ("gpt-4o", "https://api.openai.com/v1"),
        "local": ("mlx-community/Qwen3.5-4B-4bit", ""),
    }
    model, base_url = defaults[provider]
    prefix = "VOICEMEM_MEMORY" if purpose == "memory" else "VOICEMEM_STUDIO"
    return {"provider": provider, "config": {
        "model": os.environ.get(f"{prefix}_MODEL") or model,
        "base_url": os.environ.get(f"{prefix}_BASE_URL") or base_url,
        "system": "",
    }}

def create(system, provider=PROVIDER, api_key=None):
    """Create the selected stream adapter; missing credentials raise immediately."""
    cfg = configuration(provider, "reply")["config"]
    if provider == "local":
        from .local import LocalLLM
        from pathlib import Path
        from studio.paths import MODELS
        model = cfg["model"]
        bundled = MODELS / "llm/Qwen3.5-4B-4bit"
        return LocalLLM(str(bundled if model == "mlx-community/Qwen3.5-4B-4bit" else Path(model)),
                        system=system)
    from voicemem.reply import deepseek_reply, openai_reply
    cfg["api_key"] = api_key or credential(provider, "reply")
    if provider == "qwen":
        if not cfg["api_key"]:
            raise ValueError("Qwen 回复需要 DASHSCOPE_API_KEY")
        return ReplyModel(deepseek_reply(**{**cfg, "system": system}, protocol="qwen"))
    factory = deepseek_reply if provider == "deepseek" else openai_reply
    return ReplyModel(factory(**{**cfg, "system": system}))
