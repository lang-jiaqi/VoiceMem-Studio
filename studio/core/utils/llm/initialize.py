"""Fixed reply model selection; only credentials come from the environment."""
from .component import ReplyModel

PROVIDER = "deepseek"
MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"

def configuration(provider=PROVIDER):
    """Return visible provider settings without including credential values."""
    if provider == "deepseek":
        return {"provider": provider, "config": {"model": MODEL, "base_url": BASE_URL, "system": ""}}
    if provider == "qwen":
        return {"provider": provider, "config": {"model": "qwen3.6-flash", "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "system": ""}}
    return {"provider": provider, "config": {"model": "gpt-4o", "system": ""}}

def create(system, provider=PROVIDER):
    """Create the selected stream adapter; missing credentials raise immediately."""
    if provider == "local":
        from .local import LocalLLM
        from studio.paths import MODELS
        return LocalLLM(str(MODELS / "llm/Qwen3.5-4B-4bit"), system=system)
    from voicemem.reply import deepseek_reply, openai_reply
    cfg = configuration(provider)["config"]
    if provider == "qwen":
        import os
        cfg["api_key"] = os.environ.get("DASHSCOPE_API_KEY")
        if not cfg["api_key"]:
            raise ValueError("Qwen 回复需要 DASHSCOPE_API_KEY")
        return ReplyModel(deepseek_reply(**{**cfg, "system": system}, protocol="qwen"))
    factory = deepseek_reply if provider == "deepseek" else openai_reply
    return ReplyModel(factory(**{**cfg, "system": system}))
