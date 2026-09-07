"""人设：回复模型的 system prompt，一种语言一份。

**为什么不做成"一份中文 + 一句『用英文回答』"**：外挂的语言指令管得住用词，管不住
语感——停顿、语气词、留白的节奏都会照着中文的说话方式生成英文。两份各自按各自
语言的口语习惯写。

跟着**空间语言**走（``voicemem/lang.py``，建空间时定一次、写在空间 json 里），
不跟界面走，也不跟用户这一句用什么语言走：语言是库的属性，混语存会让一半记忆
检索不到。

    from voicemem import persona
    persona.system_prompt()          # 当前空间的语言
    persona.system_prompt("en")      # 指定

``VoiceMem(system=...)`` 传了自己的人设就完全不走这里。

生效正文在仓库根目录 prompt/，本模块只负责按语言选择。修改文件后重启服务。
"""
from __future__ import annotations
from voicemem.prompt_config import read_prompt, context_prompts


def by_lang(d: dict, lang: str = "") -> str:
    """双语文案里挑一份。语言只有 zh/en 两个值，取不到就按 en。"""
    if not lang:
        from voicemem.lang import memory_language
        lang = memory_language()
    return d.get(lang) or d["en"]


def system_prompt(lang: str = "") -> str:
    """默认人设。这是 system 里**唯一稳定的前缀**，每轮都一样——变的东西
    （记忆、历史）一律排在它后面拼，否则前缀缓存从第二轮起就断了，见 core.py。"""
    return by_lang(PERSONA, lang)


def stranger_note(lang: str = "") -> str:
    """声纹对不上：说话的不是这个库的主人，一条记忆都不能给他。"""
    return by_lang(STRANGER, lang)


def no_memory_note(lang: str = "") -> str:
    """这一轮一条都没检索到。不加这句模型会编——空库第一句就撞得上。"""
    return by_lang(NO_MEMORY_NOTE, lang)


def lang_note(lang: str = "") -> str:
    """回复语言。人设本身已经是目标语言了，这句管的是用户临时换语言问的情况。"""
    return by_lang(LANG_NOTE, lang)


PERSONA = {lang: read_prompt(f"llm_system_{lang}.md") for lang in ("zh", "en")}

STRANGER = context_prompts()["stranger"]

NO_MEMORY_NOTE = context_prompts()["no_memory"]

LANG_NOTE = context_prompts()["language"]
