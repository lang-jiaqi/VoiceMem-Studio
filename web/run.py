"""voicemem web demo —— 对话核心（EOU 0–300ms 投机预取）。管道在 utils.py，渲染在 index.html。

看点：本地 ASR+VAD 边听边算，partial 一到就后台起投机 Search（本地 E5 向量 + 本地 slot
分类，注入的 LocalQueryClassifier，0 LLM 0 网络），VAD 在 300ms 确认说完时记忆早已算好，
交给两条 ~10 行控制流的只剩「发 LLM / 发 Realtime」。说到一半停顿又续上（barge-in）→ 取消投机。

跑（参数见 ``--help``；每个都能用同名环境变量给默认值）::

    export OPENAI_API_KEY=sk-...
    python web/run.py                     # 默认 realtime
    python web/run.py \\
      --mode llm_tts \\                    # 没有 Realtime 权限时走这条
      --port 8787 \\
      --spec_min_chars 6 \\
      --gamble_ms 200 \\
      --confirm_ms 300

默认走 ``realtime``（OpenAI 原生语音）：一次往返直接出声，不像 llm_tts 那样要
"LLM 出文本(~1.0s) → 攒够一句 → TTS 合成(~1.2s)" 两段串行，体验差一截。
key 没有 Realtime 权限就用 ``--mode llm_tts``，那条路只要普通 chat + TTS，
TTS 还能换成本地离线模型（``TTS_BACKEND=local``）。

注意：记忆向量用本地 384 维 E5（投机预算内不能走网络）。换过旧 demo（OpenAI 1536 维）留了
记忆库的，维度不兼容——先清掉记忆目录再跑。
"""
import argparse
import asyncio
import base64
import json
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import uvicorn

HERE = Path(__file__).resolve().parent
_ROOT = HERE.parent
sys.path.insert(0, str(HERE))                       # 让 `import utils` 找到同目录管道层
sys.path.insert(0, str(_ROOT))
from echo_guard import UtteranceGuard
from voicemem.prompt_config import context_prompts, tts_prompts
os.environ.setdefault("VOICEMEM_MODELS_DIR", str(_ROOT / "models"))
# 记忆空间锚在**仓库根**，不跟当前目录走。否则 `cd web && python run.py` 会在
# web/ 底下另建一个空的 voicemem_memoryspace/demo，用户对着空库说半天话，
# 还以为记忆没生效（实测就这么踩过）。
os.environ.setdefault("VOICEMEM_MEMORYSPACE_ROOT", str(_ROOT / "voicemem_memoryspace"))


# ══════════════════ 命令行参数（同名环境变量给默认值，两种都行）══════════════════
# 放在下面那两个重 import 之前：utils / voicemem 会拉起 torch + sentence-transformers，
# 排在它们后面的话 `--help` 得先等模型库加载完。被 import 时不吃 sys.argv（传 []）。

def _parse(argv):
    p = argparse.ArgumentParser(description="voicemem web demo（脑图 + 0–300ms 投机预取）")
    p.add_argument("--mode", choices=["llm_tts", "realtime"],
                   default=os.environ.get("DEMO_MODE", "realtime"),
                   help="回复控制流：realtime=OpenAI 原生语音（默认，体验最好）；"
                        "llm_tts=LLM 流→TTS 流（不需要 Realtime 权限，可换本地 TTS）")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("VOICEMEM_PORT", 8787)))
    p.add_argument("--spec_min_chars", type=int, default=6,
                   help="partial 转写到几个字起投机预取")
    p.add_argument("--gamble_ms", type=int, default=200,
                   help="静音多久就赌你说完了，补投机一次")
    p.add_argument("--confirm_ms", type=int, default=300,
                   help="静音多久由 VAD 确认一轮结束，交出 Turn")
    p.add_argument("--config", default=os.environ.get("VOICEMEM_CONFIG"),
                   help="一个 .json，整体覆盖下面的 CONFIG")
    # 默认开中文空间：流式 ASR 按空间语言自动选（voicemem/utils/defaults.py 的
    # asr 工厂），zh → FunASR paraformer，en → sherpa zipformer-en。一个模型只认
    # 一种语言，选错就是"英文转出一串无意义的中文"。默认落在 demo（语言未定）上
    # 时这个选择没有依据，所以钉在 demo-zh；新建英文空间切过去会自动换成 en 那档。
    p.add_argument("--space", default=os.environ.get("VOICEMEM_SPACE", "demo-zh"),
                   help="用哪个 memory space（voicemem_memoryspace/<space>/）")
    p.add_argument("--memory_root", default=os.environ.get("VOICEMEM_MEMORY_ROOT", ""),
                   help="直接指定记忆库目录，给了就盖过 --space")
    p.add_argument("--llm", choices=["openai", "deepseek", "local"], default="openai",
                   help="回复模型：deepseek=API 非思考流式回复（DEEPSEEK_API_KEY）；"
                        "openai=原 API 配置；local=本机 MLX。TTS 独立配置。")
    p.add_argument("--eot", action=argparse.BooleanOptionalAction, default=True,
                   help="语义判「说完了没」（Smart Turn v3，32MB/15ms）。"
                        "开着时 --confirm_ms 只是兜底上限，可以放宽而不变慢；"
                        "--no-eot 退回纯掐表")
    p.add_argument("--backchannel", action=argparse.BooleanOptionalAction, default=True,
                   help="用户说到一半停顿时'嗯'一声（附和）。demo 里默认开，"
                        "--no-backchannel 关掉。"
                        "（库那侧默认仍是关的，见 harness/backchannel.py）")
    p.add_argument("--lang", choices=["en", "zh"],
                   default=os.environ.get("VOICEMEM_MEMORY_LANGUAGE", "en"),
                   help="新建 Memory Space 时用的语言：en（默认）/ zh。"
                        "已有空间用它自己建的时候定的那个")
    p.add_argument("--log-file", default=os.environ.get("VOICEMEM_LOG_FILE", ""),
                   help="日志文件路径；不传则自动写到 results/logs/")
    p.add_argument("--no-file-log", action="store_true",
                   default=os.environ.get("VOICEMEM_FILE_LOG", "1") == "0",
                   help="只输出到终端，不保存日志文件")
    p.add_argument("--verbose", action="store_true",
                   default=os.environ.get("VOICEMEM_VERBOSE", "0") == "1",
                   help="终端显示完整诊断日志；默认只显示时延、TTS 情绪提示和警告/错误")
    return p.parse_args(argv)


ARGS = _parse(None if __name__ == "__main__" else [])

# 必须放在重依赖 import 之前：模型加载、依赖库 warning、Uvicorn 日志和后面所有
# print 才能从进程启动第一刻起完整落盘。被别的模块 import 时不擅自创建日志文件。
# 语言：核心默认英文，demo 用 --lang zh 切中文。放在建 VoiceMem 之前，
# 因为抽取 prompt 是按它选中英两套示例的。
from voicemem.lang import set_memory_language as _set_lang   # noqa: E402
_set_lang(ARGS.lang)

LOG_FILE = None
if __name__ == "__main__" and not ARGS.no_file_log:
    from logging_utils import setup_file_logging
    LOG_FILE = setup_file_logging(_ROOT, ARGS.log_file, concise=not ARGS.verbose)

if __name__ == "__main__":
    from voicemem.prompt_trace import configure as _configure_prompt_trace
    print(f"[log] Prompt 请求记录：{_configure_prompt_trace(_ROOT / 'prompt' / 'logs')}", flush=True)

# Set the demo default before utils imports the TTS provider registry.
if ARGS.mode == "llm_tts":
    os.environ.setdefault("TTS_BACKEND", "breeze_mlx")
    _breeze_model = _ROOT / "models" / "tts" / "Breeze-TTS-2-mlx-4bit"
    if (_breeze_model / "config.json").is_file():
        os.environ.setdefault("VOICEMEM_BREEZE_MLX_MODEL", str(_breeze_model))
    os.environ.setdefault("VOICEMEM_BREEZE_REF_AUDIO", str(_ROOT / "voice" / "noctelle_ref_short.wav"))  # 3.7s 够锁音色；8.2s 的前缀是首帧里最大的一块
    if os.environ.get("TTS_BACKEND") == "breeze_mlx":
        # Short initial acknowledgements such as 在呢， can start synthesis.
        os.environ.setdefault("VOICEMEM_TTS_FIRST_MIN", "2")
        os.environ.setdefault("VOICEMEM_TTS_FIRST_SOFT", "1")

import utils                                         # noqa: E402  同目录管道层
from audio_timeline import AudioTimeline, SpeechRateEstimator  # noqa: E402
from session_context import SessionBuffer            # noqa: E402
from voicemem import VoiceMem                        # noqa: E402
from voicemem.audio_timing import TimedAudioChunk    # noqa: E402
from voicemem import gate                            # noqa: E402  轮次闸门（三路判定）
from voicemem import persona                         # noqa: E402  人设（双语，库和 demo 共用）
from harness import speak_tag                        # noqa: E402  回复模型自标语气
from voicemem.memory_api import build_memory_context # noqa: E402  提前生成时自己拼一份

BARGE_DEBUG = os.environ.get("BARGE_DEBUG", "1") != "0"
BARGE_THRESHOLD = float(os.environ.get("BARGE_THRESHOLD", "0.45"))  # 越小越容易被打断
#: 转写要比上次多出这么多个字，才算"他真的插话了"。
#: 之前这里是「连续人声 ≥280ms」，纯 VAD 判定太松——咳嗽、关门、AEC 没压干净的
#: 助手回声都算人声，日志里一串"连续人声 → 请求打断 / 助手没在说，忽略"在空转。
#: 换成等 ASR 真的吐出字，代价是多等一次出字（~200-300ms），换来不会自己掐自己。
#:
#: 这个数字直接决定"插话多久才被听见"：说满 N 个字要时间，实测 3 个字要 ~2 秒。
#: 曾经因为 2 个字会被回声骗到（漏出过 "an" —— 助手说的 Annie 回到麦克风里）
#: 才提到 3。现在回声改由 _is_echo() 按**助手正在说的原文**挡，不再靠字数硬扛，
#: 所以降回 2：插话少说一个字，大约快 300-500ms。
BARGE_MIN_CHARS = int(os.environ.get("BARGE_MIN_CHARS", "2"))
BARGE_STABLE_UPDATES = int(os.environ.get("BARGE_STABLE_UPDATES", "2"))
BARGE_REJECT_SILENCE_MS = int(os.environ.get("BARGE_REJECT_SILENCE_MS", "220"))
BARGE_CANDIDATE_TIMEOUT_MS = int(os.environ.get("BARGE_CANDIDATE_TIMEOUT_MS", "1200"))
#: 助手刚开口那一小段不允许被打断——那时候麦克风里几乎只有它自己的声音。
BARGE_GRACE_MS = int(os.environ.get("BARGE_GRACE_MS", "500"))
#: OpenAI 那侧用哪种回合/打断判定。semantic_vad 由模型判"这是不是真的在打断"，
#: 对 backchannel（"嗯""对""哦"）不敏感；server_vad 只看有没有声音，所以助手自己
#: 的回声、环境噪声都能把它掐了。模型或 SDK 不支持时会以 error 事件回来（不抛），
#: 日志里看到就改回 TURN_DETECTION=server_vad。
TURN_DETECTION = os.environ.get("TURN_DETECTION", "semantic_vad")
#: semantic_vad 的抢答倾向：low 更愿意等你说完，high 更爱抢。
VAD_EAGERNESS = os.environ.get("VAD_EAGERNESS", "low")
MIC_RATE = 24000                       # 前端上行的采样率（index.html 的 SAMPLE_RATE）
#: 按声纹拦陌生人。默认开——启动时预热过、又在后台线程算，实测对延迟零影响
#: （memory_hits 仍在 EOU 前 0.63s 到达，跟关掉时一样）。
SPEAKER_GATE = os.environ.get("VOICEMEM_SPEAKER_GATE", "1") != "0"   # 打断为什么没触发：看这几行日志
#: 连续几轮认成别人才判"陌生人"。1 = 一轮就翻脸（demo 里演"换个人说话"要的就是
#: 这个）；声纹库脏、老是把主人认成新人时，调到 2 能挡掉大部分误判。
STRANGER_MIN_TURNS = int(os.environ.get("STRANGER_MIN_TURNS", "1"))
#: 每轮都打一行说话人判定（默认只在判成陌生人时打）。
SPEAKER_DEBUG = os.environ.get("SPEAKER_DEBUG", "0") != "0"
MODE = ARGS.mode                                     # llm_tts | realtime
# 附和：demo 里默认开（--no-backchannel 关）。**库那侧默认仍是关的**——
# 让别人的产品在不知情时突然开始出声是另一回事。
# 必须在 import harness.backchannel **之前**设：那边的 ON 是模块级读的。
os.environ["VOICEMEM_BACKCHANNEL_EMIT"] = "1" if ARGS.backchannel else "0"
SPEC_MIN_CHARS = ARGS.spec_min_chars                 # partial 起投机
GAMBLE_S  = ARGS.gamble_ms / 1000                    # 赌说完
CONFIRM_S = ARGS.confirm_ms / 1000                   # VAD 静音多久算这一轮结束
# 曾经在开 EOT 时把兜底放宽到 800ms，想着"反正有语义判定顶着"。**那是错的**：
# EOT 拿不准的时候（升调疑问句实测只有 0.37）就得等满 800ms，比原来的 300ms 还慢，
# 而慢是听得见的。兜底就该是"最坏情况能接受的延迟"，不是"EOT 的备胎可以随便长"。
# 保持 300ms：EOT 判得出来的更快（约 100ms 就结束），判不出来的不比以前差。

#: 人设。**一种语言一份，不是把中文那份翻译过去再拼语言指令**——"用英文回答"
#: 这种外挂指令管得住用词，管不住语感：中文那份里的停顿、语气词、留白节奏，
#: 模型会照着中文的说话方式生成英文。两份各自按各自语言的口语习惯写。
#: 跟着**空间语言**（SPACE_LANG）选，不跟界面走，也不跟用户这一句走。




# 这一版人设换成了"好朋友"口径：先接住这句话的性质（分享/吐槽/玩笑/犹豫/求助），
# 用洞察而不是提问推进，情绪跨轮连续。上一版是"记忆助手"口径，全文在 git 里
# （ec15e08 之前）。两版都保留的硬约束，换人设时别丢：
#
# · emotion & characteristics 一个字都不许说出口——它是归因，不是事实，
#   念出来就成了当面分析用户。
# · 检索到 ≠ 相关；记忆里没写的细节一个字都不许补。
#   这两条是这套系统最贵的教训，见 _NO_MEMORY_NOTE。
# · 表演指示（语调、咬字、尾音）不写在这里。TTS 后端有 instruction 参数
#   （Breeze、gpt-4o-mini-tts 都收），隔着文字模型转述效果差两层。



#: 这一轮一条记忆都没检索到时追加的一句。
#:
#: 人设里那套"从含糊的一句话里猜出具体那件事"的指示，在有记忆时是这套系统最值钱
#: 的地方，可**一条都没检索到**时它就成了编造的许可证：新建一个空的 Memory Space、
#: 第一句问「我不能吃什么」，它张口就是「你的饮食禁忌里，辣椒和海鲜要注意，之前
#: 你提到过对这些过敏」——两样都是凭空来的。空库的第一句话就撞得上，而那正是别人
#: 第一次用这个 demo 的时刻。
#:
#: 跟 _STRANGER 的区别：那句是"你认识的是另一个人"，这句是"这件事你不知道"。
#: 界面选的语言。助手跟着它走——回复用这个语言，抽出来的记忆也用这个语言写。
#:
#: 记忆的语言必须跟着一起换，不然库里会中英混着长：同一件事今天记成中文、明天
#: 记成英文，检索时两边都只能命中一半。
UI_LANG = ARGS.lang          # 界面语言。右上角随时可切，跟记忆/回复语言无关

#: 助手说什么语言。原来是跟着**界面语言开关**走（前端把选择存在 localStorage 里，
#: 每次开页面自动 POST 给后端），结果是：后端默认值永远不生效，上次录英文 demo 切过
#: 一次，之后全程中文提问也照样英文回答，换库、重启都没用。
#: 现在改成跟着**用户这句话的语言**走——问什么语言答什么语言，跟界面无关。
#: 回复用哪种语言。跟**当前空间**走，不跟界面走，也不跟用户这一句用什么语言走。
#:
#: 原来是"用户说什么语言就回什么语言"。那在单语场景下没问题，但空间是有语言的：
#: 一个英文库里用户偶尔冒一句中文，助手跟着说中文、这轮记忆也就成了中文，
#: 库就混了。语言在建空间时定死，这里照着执行。


#: 这个空间用什么语言。**建空间时定一次，之后不再变**。
#:
#: 记忆和回复都跟着它，界面语言（UI_LANG）是另一回事，可以随便切。
#: 为什么不做成随时可切：语言是**库的属性**。检索是按向量做的，中文问句和英文
#: 记忆在向量空间里离得很远，一个库里中英混存的后果是一半记忆检索不到，而且
#: 不报错。中途切语言要么把库搞混，要么就得每条存两份——两种都不能接受。
SPACE_LANG = "en"


def space_language(name: str) -> str:
    """读这个空间建的时候定的语言；老空间没有这个字段就按 en。"""
    import json as _json
    d, safe = space_dir(name)
    f = d / f"{safe}.json"
    try:
        v = (_json.loads(f.read_text(encoding="utf-8"))
             .get("space", {}).get("language", ""))
        return "zh" if str(v).lower().startswith("zh") else "en"
    except Exception:
        return "en"


def _write_space_language(name: str, lang: str) -> None:
    import json as _json
    d, safe = space_dir(name)
    f = d / f"{safe}.json"
    try:
        doc = _json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        doc.setdefault("space", {})["language"] = lang
        f.write_text(_json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[space] 写语言失败（不影响使用）：{e}", flush=True)


def _space_is_empty(name: str) -> bool:
    """这个空间还一条记忆都没有吗。空库才允许改语言，见 set_lang。"""
    try:
        import sqlite3          # run.py 顶部没有导入它（其他用处都是函数内导入）
        from voicemem.utils.common import space as _sp
        d, _ = space_dir(name)
        db = _sp.db(d)
        if not Path(db).exists():
            return True
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            n = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        except sqlite3.OperationalError:
            # 表还没建 = 一条都没写过 = 空。刚建的空间正是这个状态，不区分的话
            # "空库可以改语言"这条对新建空间永远不成立——恰好把目标场景挡在外面。
            return True
        finally:
            c.close()
        return n == 0
    except Exception:
        return False          # 真读不出来（损坏/权限）才当非空，宁可不改


def set_lang(lang: str) -> str:
    """右上角那个选择器。返回**回复语言实际变成了什么**。

    以前这里只切界面，回复语言纹丝不动——而前端那句注释写的是"助手也跟着换语言"。
    两边说的是相反的事，于是就有了「界面中文、回答英文」，还没有任何提示。

    语言确实是**库的属性**：检索按向量做，一个库里中英混存会让一半记忆检索不到
    （见 voicemem/lang.py）。但这条约束只对**已经有记忆的库**成立——空库里没有
    任何东西会被搞乱，这时候拦着不让改纯粹是让人困惑。所以：

        空库    → 界面和记忆/回复语言一起切，并写进空间
        非空库  → 只切界面，**并把这件事告诉用户**（返回值让前端弹一句）
    """
    global UI_LANG, SPACE_LANG
    UI_LANG = "en" if str(lang).lower().startswith("en") else "zh"
    if UI_LANG == SPACE_LANG:
        return SPACE_LANG
    if _space_is_empty(ACTIVE_SPACE):
        _write_space_language(ACTIVE_SPACE, UI_LANG)
        SPACE_LANG = UI_LANG
        _set_lang(SPACE_LANG)
        print(f"[lang] 空间「{ACTIVE_SPACE}」还是空的 → 记忆和回复一起切到 {UI_LANG}",
              flush=True)
    else:
        print(f"[lang] 界面切到 {UI_LANG}，但空间「{ACTIVE_SPACE}」已有记忆、"
              f"仍是 {SPACE_LANG}——混语存会让一半记忆检索不到。"
              f"要换语言请新建一个空间。", flush=True)
    return SPACE_LANG


def _lang_note() -> str:
    return persona.lang_note(SPACE_LANG)




#: 问的是"一段声音"时才回放。故意做得很笨——这是个触发词表，不是意图分类器：
#: 多放一次听感上只是"它把当时那段放给你听"，判漏了也只是回到手动点 ▶。
_SOUND_WORDS = ("歌", "曲", "调子", "旋律", "音乐", "哼", "那段声音", "放来听",
                "放给我听", "播一下", "什么声音")


def _musical_memory_ids() -> set[str]:
    """音频里**真的有音乐**的那些记忆。

    ingest 时音乐识别命中过的一轮，会被打上 ``tune:<tune_id>`` 标签（跟
    ``scene:café`` / ``speaker:person_x`` 一样存在 memory_tags 里）。有了它就不用
    靠用户正好说过"我听到一首歌"——在咖啡馆随便聊，那一轮的背景音乐照样被记下来。

    读失败就返回空集：这只是个偏好排序，取不到就退回原来的"第一条有音频的"。
    """
    try:
        tunes = vm._o._audio._music_store().list_tunes()
        ids = [f"tune:{t['tune_id']}" for t in tunes]
        if not ids:
            return set()
        store = vm._o._get_repo()._cognitive_store
        return set(store.memory_ids_for_slots_v2(vm._o._user_id, ids))
    except Exception as e:
        print(f"[web] 读音乐标签失败（不影响回放）：{type(e).__name__}: {e}", flush=True)
        return set()


#: 回放正在进行到什么时候（单调时钟）。这段时间里的空文本轮次一律丢掉。
_REPLAY_UNTIL = 0.0


def _note_replay(memory_id: str) -> None:
    """记下这次回放大概会响多久。

    放出来的录音会从扬声器绕回麦克风，被 VAD 判成新的一轮——转写是空的、左右脑
    都没命中，模型手上什么都没有，于是回一句"抱歉，我没法直接重播"，日志里那段
    音乐还被认成"第 2 次听到"。页面那边已经在回放期间停了上行（见 voicemem.html
    的 holdMic），这里是第二道：万一前端没生效（别的客户端、老页面缓存），
    后端自己也认得出这一轮是自己放出去的。
    """
    global _REPLAY_UNTIL
    path = audio_of(memory_id)
    if not path:
        return
    try:
        import soundfile as sf
        dur = float(sf.info(path).duration)
    except Exception:
        dur = 15.0          # 读不出时长就按上限压一会儿，宁可多挡一轮
    _REPLAY_UNTIL = time.monotonic() + dur + 1.0


def _replaying_now() -> bool:
    return time.monotonic() < _REPLAY_UNTIL


#: 刚听过、还没来得及入库的那段音乐。``{"path": wav, "at": 单调时钟}``
#:
#: 为什么要单独记一份：入库走后台线程，一轮 15-20 秒才写完，而"刚听完就问"恰恰
#: 是最自然的用法——实测放完音乐立刻问「重播一下刚才那首歌」，那条记忆的
#: created_at 跟提问时间只差几秒，查库的时候还没有它。而 tune 识别是同步的
#: （Ingest 立刻带着 recognized_tune 返回），这份缓存在上一轮结束时就写好了，
#: 正好补上那十几秒空窗。
_LAST_TUNE: dict = {"path": "", "at": 0.0}

#: 用这个假 id 请求上面那段录音。真 memory_id 是 uuid，不会撞。
LAST_TUNE_ID = "last:tune"

#: 缓存多久算"刚才"。超过就不认了——半小时前听的歌不该被「刚才那首」捞出来。
LAST_TUNE_TTL_S = float(os.environ.get("VOICEMEM_LAST_TUNE_TTL", "1800"))


def _remember_tune(audio_path: str) -> None:
    _LAST_TUNE.update(path=str(audio_path or ""), at=time.monotonic())
    print(f"  [replay] 记住这段音乐：{audio_path}", flush=True)


def _last_tune_path() -> str:
    p = _LAST_TUNE.get("path") or ""
    if not p or time.monotonic() - float(_LAST_TUNE.get("at") or 0) > LAST_TUNE_TTL_S:
        return ""
    return p if Path(p).exists() else ""


#: 问句里的时间限定。记忆本来就绑了 created_at，「上周三下午在咖啡馆听的那首」
#: 这种要求靠语义检索是碰运气——"上周三""下午"对向量几乎没有影响，撞上哪条算哪条。
#: 所以时间和地点单独解析出来，当**硬条件**筛。
_DAY_WORDS = (
    (("大前天",), -3), (("前天",), -2), (("昨天", "昨晚", "昨儿"), -1),
    (("今天", "今早", "今晚", "今日"), 0),
)
_HOUR_WORDS = (
    (("凌晨", "半夜", "深夜"), (0, 6)),
    (("早上", "早晨", "今早", "一早", "清晨"), (6, 10)),
    (("上午",), (8, 12)),
    (("中午", "晌午"), (11, 14)),
    (("下午",), (12, 18)),
    (("傍晚", "黄昏"), (17, 20)),
    (("晚上", "晚间", "昨晚", "今晚", "夜里"), (18, 24)),
)
#: 星期几。「周三」单说指本周，配合「上周」就是上一周。
_WEEKDAYS = (("周一", "星期一", "礼拜一"), ("周二", "星期二", "礼拜二"),
             ("周三", "星期三", "礼拜三"), ("周四", "星期四", "礼拜四"),
             ("周五", "星期五", "礼拜五"), ("周六", "星期六", "礼拜六"),
             ("周日", "周天", "星期日", "星期天", "礼拜天"))

#: 中文地点说法 → 声学场景标签（scene_classifier.SceneTag）。
#: 场景是每轮录音自动分类出来的，存在 memory_tags 里（scene:café）。
_PLACE_WORDS = (
    (("咖啡馆", "咖啡店", "咖啡厅", "星巴克"), "café"),
    (("办公室", "公司", "工位", "单位"), "office"),
    (("家里", "家中", "在家", "屋里", "房间"), "home"),
    (("外面", "户外", "外边", "路上", "街上", "公园"), "outdoor"),
    (("车上", "地铁", "公交", "路上", "通勤", "火车"), "transit"),
    (("会议", "开会", "会上", "会议室"), "meeting"),
)


#: 「第几首」。候选按时间正序排，序号就是下标——同一段时间里听了好几首时，
#: 「第一首」「上一首」是最自然的说法，而这信息记忆里本来就有（created_at）。
_ORDINALS = (
    (("第一首", "第一段", "第1首", "头一首", "最早那首", "最先那首"), 1),
    (("第二首", "第二段", "第2首"), 2),
    (("第三首", "第三段", "第3首"), 3),
    (("第四首", "第四段", "第4首"), 4),
    (("第五首", "第五段", "第5首"), 5),
)
#: 从后往前数的说法。-1 = 最后一首，-2 = 倒数第二（也就是"上一首"）。
_ORDINALS_BACK = (
    (("最后一首", "最后那首", "最后一段", "最新那首", "最近那首"), -1),
    (("上一首", "前一首", "上一段", "前一段", "前面那首", "上一个"), -2),
)


def _ordinal_of(text: str):
    """问句里的「第几首」→ 1-based 序号（负数表示从后往前数）；没说返回 None。"""
    t = text or ""
    for words, n in _ORDINALS + _ORDINALS_BACK:
        if any(w in t for w in words):
            return n
    return None


def _place_of(text: str) -> str:
    """问句里提到的地点 → 场景标签；没提就返回 ""。"""
    t = text or ""
    return next((tag for words, tag in _PLACE_WORDS if any(w in t for w in words)), "")


def _time_window(text: str):
    """问句 → (起, 止) 两个 datetime；没提时间就返回 None。

    认得出：今天/昨天/前天/大前天、这周/上周/上上周、周一~周日、这个月/上个月、
    N月N号、N天前，以及凌晨/早上/上午/中午/下午/傍晚/晚上这些时段，可以组合
    （「上周三下午」）。认不出来就返回 None 交给"取最近的一条"——别自作聪明猜，
    猜错了放出来的是别的录音，比放不出来更糟。

    「刚才」「刚刚」故意不算时间限定：它们说的是"最近那次"，走严格分支反而会因为
    筛不到而什么都不放。
    """
    import re
    from datetime import datetime, timedelta
    t = text or ""
    now = datetime.now()
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    span = None                      # (起日, 止日)，止日不含

    m = re.search(r"(\d+)\s*天前", t)
    if m:
        d = day0 - timedelta(days=int(m.group(1)))
        span = (d, d + timedelta(days=1))

    if span is None:
        wd = next((i for i, words in enumerate(_WEEKDAYS) if any(w in t for w in words)), None)
        if wd is not None:
            # 本周一为基准；「上周」再往前推一周
            monday = day0 - timedelta(days=day0.weekday())
            if "上上周" in t or "上上星期" in t:
                monday -= timedelta(days=14)
            elif "上周" in t or "上星期" in t or "上礼拜" in t:
                monday -= timedelta(days=7)
            d = monday + timedelta(days=wd)
            span = (d, d + timedelta(days=1))

    if span is None:
        m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[号日]", t)
        if m:
            mth, dom = int(m.group(1)), int(m.group(2))
            year = now.year - 1 if mth > now.month else now.year
            try:
                d = datetime(year, mth, dom)
                span = (d, d + timedelta(days=1))
            except ValueError:
                span = None
    if span is None:
        m = re.search(r"(?<![0-9])(\d{1,2})\s*[号日](?!\s*[前后])", t)
        if m:
            dom = int(m.group(1))
            try:
                d = day0.replace(day=dom)
                if d > day0:                       # 这个月还没到，那说的是上个月
                    d = (day0.replace(day=1) - timedelta(days=1)).replace(day=dom)
                span = (d, d + timedelta(days=1))
            except ValueError:
                span = None

    if span is None:
        day = next((d for words, d in _DAY_WORDS if any(w in t for w in words)), None)
        if day is not None:
            d = day0 + timedelta(days=day)
            span = (d, d + timedelta(days=1))

    if span is None:
        if "上上周" in t or "上上星期" in t:
            monday = day0 - timedelta(days=day0.weekday() + 14)
            span = (monday, monday + timedelta(days=7))
        elif "上周" in t or "上星期" in t or "上礼拜" in t:
            monday = day0 - timedelta(days=day0.weekday() + 7)
            span = (monday, monday + timedelta(days=7))
        elif "这周" in t or "本周" in t or "这星期" in t:
            monday = day0 - timedelta(days=day0.weekday())
            span = (monday, monday + timedelta(days=7))
        elif "上个月" in t or "上月" in t:
            first = day0.replace(day=1)
            span = ((first - timedelta(days=1)).replace(day=1), first)
        elif "这个月" in t or "本月" in t:
            first = day0.replace(day=1)
            span = (first, (first + timedelta(days=32)).replace(day=1))

    hours = next((h for words, h in _HOUR_WORDS if any(w in t for w in words)), None)
    if span is None and hours is None:
        return None
    if span is None:                                  # 只说了时段，默认今天
        span = (day0, day0 + timedelta(days=1))
    if hours is None:
        return span
    # 时段只在跨度是一天时才细化——「上周下午」没有意义
    if (span[1] - span[0]).days > 1:
        return span
    return span[0] + timedelta(hours=hours[0]), span[0] + timedelta(hours=hours[1])


def _tune_memories() -> list[dict]:
    """所有能放出来的候选：带 tune: 标签、或者是纯声音轮，并且原声还在。

    为什么 sound_only 也算：直接对着麦克风放音乐、声学又没认出来时（外放、环境
    吵、片段短都会漏），那一轮既没有 tune 标签也没有话，只知道"有段录音、没人
    说话"。它恰恰最可能就是用户要找的那段。普通说话轮不会混进来。

    一条 = {"id", "at": datetime, "scenes": {...}, "text"}，新的在前。
    回放的候选池就是它——按时间、按地点筛都在这个池子里做，不依赖这一轮检索
    碰巧命中了什么。
    """
    from datetime import datetime
    try:
        import sqlite3
        from voicemem.utils.common import space as _space
        c = sqlite3.connect(_space.db(vm._o._memory_root))
        c.row_factory = sqlite3.Row
        rows = c.execute(
            """SELECT m.id, m.content, m.created_at,
                      (SELECT group_concat(s.slot) FROM memory_tags s
                        WHERE s.memory_id = m.id AND s.slot LIKE 'scene:%') scenes,
                      (SELECT u.slot FROM memory_tags u
                        WHERE u.memory_id = m.id AND u.slot LIKE 'tune:%' LIMIT 1) tune,
                      EXISTS (SELECT 1 FROM memory_tags o
                               WHERE o.memory_id = m.id AND o.slot = 'sound_only') sound_only
                 FROM memories m
                 WHERE EXISTS (SELECT 1 FROM memory_tags t
                                WHERE t.memory_id = m.id
                                  AND (t.slot LIKE 'tune:%' OR t.slot = 'sound_only'))
                 ORDER BY m.created_at DESC LIMIT 300""").fetchall()
        c.close()
    except Exception as e:
        print(f"[replay] 列音乐记忆失败：{type(e).__name__}: {e}", flush=True)
        return []

    out = []
    for r in rows:
        if not audio_of(r["id"]):
            continue
        try:
            at = datetime.fromisoformat(r["created_at"]).astimezone().replace(tzinfo=None)
        except Exception:
            continue
        out.append({"id": r["id"], "at": at, "text": r["content"] or "",
                    "sound_only": bool(r["sound_only"]),
                    "tune": (r["tune"] or "").split(":", 1)[-1],
                    "scenes": {x.split(":", 1)[1] for x in (r["scenes"] or "").split(",") if ":" in x}})
    return out


def _archived_memory_ids() -> list[str]:
    """所有存了原声的记忆 id，新的在前。

    按时间找录音时的兜底：音乐识别没命中的那些轮次没有 tune: 标签，但音频照样
    归档了。只认标签的话，用户明明刚放过一首歌，它却回一句"没存到"。
    """
    try:
        import sqlite3
        from voicemem.utils.common import space as _space
        c = sqlite3.connect(_space.db(vm._o._memory_root))
        rows = c.execute("SELECT id FROM memories ORDER BY created_at DESC LIMIT 200").fetchall()
        c.close()
        return [r[0] for r in rows]
    except Exception as e:
        print(f"[web] 列存档记忆失败（不影响回放）：{type(e).__name__}: {e}", flush=True)
        return []


def _created_at(mid: str):
    """这条记忆是什么时候写下的。查不到返回 None。

    向量库那边只存了 date（天粒度），要按"下午"筛得用 sqlite 里的 created_at。

    路径要用 vm 解析好的 _memory_root，**不能拿 ARGS.space**——``space.db()`` 收的
    是目录路径，给它一个裸名字（"musictest"）会当成相对路径，找不到就新建一个空
    库，于是每条记忆都查不到时间，按时间回放静默失效。
    """
    from datetime import datetime
    try:
        import sqlite3
        from voicemem.utils.common import space as _space
        c = sqlite3.connect(_space.db(vm._o._memory_root))
        row = c.execute("SELECT created_at FROM memories WHERE id=?", (mid,)).fetchone()
        c.close()
        if not row or not row[0]:
            return None
        return datetime.fromisoformat(row[0]).astimezone().replace(tzinfo=None)
    except Exception:
        return None


#: 同一首歌的两个片段最多隔多久还算"连着的一首"。VAD 在乐句停顿处断开、
#: 下一段重新起录，中间的空档就是这么来的。
TUNE_GAP_S = float(os.environ.get("VOICEMEM_TUNE_GAP_S", "90"))


#: 一组片段用这个前缀的假 id 请求，后面接用逗号分隔的 memory_id。
#: 走的还是既有的 /api/audio/{memory_id}，前端一个字都不用改。
GROUP_ID_PREFIX = "group:"

#: 拼好的整首放这儿。同一组只拼一次，之后直接命中。
_STITCH_CACHE: dict = {}


def _stitch(memory_ids: list) -> str:
    """把几段录音按顺序拼成一个 wav，返回路径；拼不了就返回第一段。

    采样率不一致时以第一段为准重采样——归档的都是 16k 单声道，真遇到不一致
    也不该让回放整个失败。
    """
    paths = [q for q in (audio_of(m) for m in memory_ids) if q]
    if not paths:
        return ""
    if len(paths) == 1:
        return paths[0]
    key = "|".join(paths)
    if key in _STITCH_CACHE and Path(_STITCH_CACHE[key]).exists():
        return _STITCH_CACHE[key]
    try:
        import numpy as np
        import soundfile as sf
        from voicemem.utils.audio.stream_io import resample as _resample
        chunks, sr0 = [], None
        for q in paths:
            x, sr = sf.read(q, dtype="float32", always_2d=False)
            if getattr(x, "ndim", 1) > 1:
                x = x.mean(axis=1)
            if sr0 is None:
                sr0 = sr
            elif sr != sr0:
                x = _resample(x, sr, sr0)
            chunks.append(x)
        out = TURN_AUDIO_DIR / f"stitch_{uuid.uuid4().hex[:12]}.wav"
        sf.write(out, np.concatenate(chunks), sr0)
        _STITCH_CACHE[key] = str(out)
        total = sum(len(c) for c in chunks) / float(sr0 or 16000)
        print(f"  [replay] 拼好 {len(paths)} 段 → {total:.1f}s", flush=True)
        return str(out)
    except Exception as e:
        print(f"  [replay] 拼接失败，只放第一段：{type(e).__name__}: {e}", flush=True)
        return paths[0]


def _same_song_group(pool: list, pick: dict) -> list:
    """挑中的这一段，连同它前后**同一首、时间相连**的片段，按时间正序。

    一轮录音在 VAD 判到静音时就结束了，而音乐里的乐句停顿、弱拍随便就够长——
    实测一段 16 秒的曲子被切成 7.8s + 5.0s 两轮，回放只放中的一段，听感就是
    "只放了开头"。这些碎片被音乐识别归成同一个 tune_id，把它们按时间拼回去
    就是原来那首。

    tune_id 认不出来（tune:unidentified）时只按时间相邻算——那时没有别的凭据，
    宁可少拼几段，也别把两首不同的歌接在一起。
    """
    tune = pick.get("tune") or ""
    same = [x for x in pool
            if (x.get("tune") or "") == tune and (tune != "unidentified" or x.get("sound_only"))]
    same.sort(key=lambda x: x["at"])
    if pick not in same:
        return [pick]
    i = same.index(pick)
    lo = i
    while lo > 0 and (same[lo]["at"] - same[lo - 1]["at"]).total_seconds() <= TUNE_GAP_S:
        lo -= 1
    hi = i
    while hi + 1 < len(same) and (same[hi + 1]["at"] - same[hi]["at"]).total_seconds() <= TUNE_GAP_S:
        hi += 1
    return same[lo:hi + 1]


def _group_id(group: list) -> str:
    """一组片段 → 一个 id。只有一段时就用它本身的 id，别绕。"""
    if not group:
        return ""
    if len(group) == 1:
        return group[0]["id"]
    return GROUP_ID_PREFIX + ",".join(x["id"] for x in group)


def _prefer_sound_only(cand: list) -> list:
    """候选里有"只有声音、没有说话"的那种，就只用它们。

    「给你听一首歌啊」和后面那段音乐是**两轮**——一轮录音从检测到人声开始、到 VAD
    判定说完为止，所以前一轮存的是他自己那句话（说的时候背景里已经有音乐，于是
    那轮也带 tune 标签），后一轮才是音乐本身。两条都在池子里，挑错了放出来是用户
    自己的声音，听感就是"音乐被截断了"。
    一条纯音乐轮都没有时原样返回——总比什么都不放强。
    """
    only = [x for x in cand if x.get("sound_only")]
    return only or cand


def _replay_id(text: str, result) -> str:
    """这一轮该不该把当时那段原声放回来，返回要放的 memory_id（不放就空串）。

    问的是声音（``_SOUND_WORDS``）才会放。候选池是**所有**带音乐标签、原声还在的
    记忆（``_tune_memories``），不是这一轮碰巧检索到的东西——「上周三在咖啡馆听的
    那首」靠语义检索是碰运气，而时间和地点记忆里本来就有。

      ① 说了时间或地点 → 当硬条件筛，取符合条件里最近的一条。
         筛空就是**真没有**，回空串让模型如实说，不退回去随便挑：退回去的话，
         问「昨天晚上那首歌」会把今天下午的录音放出来，用户以为它记错了时间，
         其实它压根没在按时间找。
      ② 什么条件都没说（"那首歌""刚才那段旋律"）→ 池子里最近的一条。
      ③ 池子是空的 → 退回这一轮检索命中的、和刚听过还没入库的那段。
    """
    if not _wants_sound(text):
        return ""

    pool = _tune_memories()
    window, place = _time_window(text), _place_of(text)
    nth = _ordinal_of(text)

    # 「第一首」没说是哪天的第一首时，默认今天——问"第一首"几乎都是指今天听的
    # 那批里的第一首，拿全库去数会数到几个月前那条。
    if nth is not None and window is None:
        from datetime import datetime, timedelta
        day0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        window = (day0, day0 + timedelta(days=1))

    if pool and (window or place):
        cand = pool
        if window:
            cand = [x for x in cand if window[0] <= x["at"] <= window[1]]
        if place:
            cand = [x for x in cand if place in x["scenes"]]
        cand = _prefer_sound_only(cand)
        if cand:
            cand.sort(key=lambda x: x["at"])          # 正序：第一首在最前
            if nth is None:
                pick = cand[-1]                        # 没说第几首就取最近的
            elif -len(cand) <= (nth - 1 if nth > 0 else nth) < len(cand):
                pick = cand[nth - 1 if nth > 0 else nth]
            else:
                print(f"  [replay] 那段时间只有 {len(cand)} 首，没有第 {nth} 首", flush=True)
                return ""
            print(f"  [replay] 按条件挑中 {pick['id'][:12]}（{pick['at']:%m-%d %H:%M}"
                  f"{' / ' + place if place else ''}"
                  f"{' / 第 %d 首' % nth if nth else ''}，共 {len(cand)} 首）", flush=True)
            return _group_id(_same_song_group(pool, pick))
        print(f"  [replay] 池里 {len(pool)} 段音乐，没有符合条件的"
              f"（{'时间 %s~%s ' % (window[0], window[1]) if window else ''}"
              f"{'地点 ' + place if place else ''}）", flush=True)
        return ""

    if pool:
        pick = max(_prefer_sound_only(pool), key=lambda x: x["at"])
        print(f"  [replay] 没说条件，放最近的 {pick['id'][:12]}"
              f"（{pick['at']:%m-%d %H:%M}）", flush=True)
        return _group_id(_same_song_group(pool, pick))

    # 池子是空的：音乐识别没命中，或者刚听完还没入库。
    playable = [h.memory_id for h in (getattr(result, "hits", None) or [])
                if audio_of(h.memory_id)]
    if playable:
        print(f"  [replay] 没有音乐标签，放检索到的 {playable[0][:12]}", flush=True)
        return playable[0]
    if _last_tune_path():
        print("  [replay] 还没入库，放刚听过的那段", flush=True)
        return LAST_TUNE_ID
    print(f"  [replay] 放不了：一段音乐记忆都没有，检索 "
          f"{len(getattr(result, 'hits', None) or [])} 条也都没音频，"
          f"缓存 path={_LAST_TUNE.get('path') or '(空)'}", flush=True)
    return ""


def _turn_detection() -> dict:
    """OpenAI 那侧的回合/打断判定。两种 provider 的参数**不通用**——semantic_vad
    不吃 threshold / prefix_padding_ms / silence_duration_ms，传了整段会被静默拒绝
    （只以 error 事件回来）。所以两套各写各的，别合并。

    create_response=False   什么时候回复由我们决定（本地判完一轮、记忆预取好）
    interrupt_response=True 用户一开口，服务端直接掐掉正在播的回复
    """
    # interrupt_response=False：**不让 OpenAI 那侧打断**。
    # 它的 VAD 是纯声学的，只要能量像人声就触发，对 AEC 残留毫无抵抗力——实测
    # 助手说到 1.4-1.6s 时被自己的回声打断，表现是"说半句突然卡住"
    # （日志：OpenAI VAD 听到人声 live=True 还在播=True 1570ms）。
    # 本地那条是 ASR 确认制，要真的转出几个字才算插话，回声转不出连贯的字。
    # 打断改成只走本地：on_speech 里我们自己发 response.cancel，效果一样。
    base = {"type": TURN_DETECTION, "create_response": False, "interrupt_response": False}
    if TURN_DETECTION == "semantic_vad":
        return {**base, "eagerness": VAD_EAGERNESS}
    # server_vad：默认 0.5 对插话太钝——人隔着扬声器说话，回声消除处理过之后
    # 信号本来就弱，够不到阈值就等于打不断。
    return {**base, "threshold": BARGE_THRESHOLD,
            "prefix_padding_ms": 200, "silence_duration_ms": 320}


#: 要回放时追加的一句。不加的话模型会去"描述"那段音频（"你说那是一首很轻快的
#: 钢琴曲…"）——它根本没听过那段音频，描述全是编的；而且用户马上就要亲耳听到。
_REPLAY_NOTE = context_prompts()["replay"]

#: 他在找一段声音、但那个时间段确实没有存档时追加的一句。
#: 不加的话模型会顺口答"当然，马上播放"——然后什么都不放。说要播却没播，
#: 比直接说没找到糟得多。
_NO_REPLAY_NOTE = context_prompts()["no_replay"]


def _wants_sound(text: str) -> bool:
    return any(w in (text or "") for w in _SOUND_WORDS)


#: 感知层判出来的情绪 → 一句**可执行的**表演指示。
#:
#: 光在人设里写"要有起伏"没用——那是形容词，模型没有对象可对。给它一个具体的
#: 目标（"他现在是焦虑的，你要放慢、压低、先接住"），语气才真的会变。
#: 情绪本身是声学感知算出来的（Qwen-Omni 归因 + 韵律 VAD），每轮都不一样。
_TONE = tts_prompts()["fallback_by_user_emotion"]


_STATE_LABEL = context_prompts()["state_label"]


def _tone_note(emotion: str) -> str:
    return _by_lang(_TONE).get((emotion or "").strip(), "")


#: 说话的基调，每一轮都带。跟 _TONE 拼起来就是这一轮给 TTS 的完整指示。
_SPEAK_BASE = tts_prompts()["base"]
_speak_base_env = os.environ.get("VOICEMEM_SPEAK_BASE", "")
#: 上一轮实际用的语气，给 speak_tag.smooth() 当锚点。进程级就够——它只是让相邻
#: 两轮听着接得上，跨会话不需要连续。
_LAST_TONE = {"tag": ""}


def _speak_instruction(emotion: str) -> str:
    """这一轮怎么念，直接给 TTS。

    _TONE 那 13 条本来就全是**发声指示**（"语速放慢""声音压低""语调扬上去"），
    以前拼在文本 prompt 里，等于让文字模型先理解一段发声描述、再指望 TTS 从
    字面上猜出来，中间隔了两层，实测基本没效果。TTS 的 instruction 参数就是
    收这个的（Breeze 有，gpt-4o-mini-tts 也有），直接送过去。
    """
    base = _speak_base_env or _by_lang(_SPEAK_BASE)
    tone = _tone_note(emotion)
    return f"{base}{tone}" if tone else base


# ── 短期对话历史 ──────────────────────────────────────────────────────────────
#: 回复模型是**无状态**的：reply.py 每轮只发 system + 用户这一句，没有前几轮。
#: 长期记忆管的是"关于这个人的事实"，管不了"我们刚才在聊什么"——于是聊完一个
#: 话题你说一句"ok ok"，它看到的就是一个孤零零的"ok ok"，重新打招呼
#: （"你好呀？有什么事我可以帮你的吗"）。这两种上下文缺一不可。
#: 每个 WebSocket 会话和 Memory Space 独立维护短期上下文。
#: ingest 确认已生成持久记忆后移除对应 turn。
#: 每句最多带这么多字进 prompt。回复有时很长，全塞进去会把记忆挤到后面。
_HISTORY_CHARS = int(os.environ.get("VOICEMEM_HISTORY_CHARS", "200"))
_SESSION_CONTEXT = SessionBuffer(text_limit=_HISTORY_CHARS)

def _by_lang(d: dict, lang: str = "") -> str:
    """demo 自己那几段双语文案（回放、语气）里挑一份。人设那几段在
    voicemem/persona.py，库和 demo 共用一份，读的都是空间语言。"""
    return persona.by_lang(d, lang or SPACE_LANG)


def _rt_persona(lang: str = "") -> str:
    """库里的人设 + 语气标注规则（**只在 llm_tts 那条路上加**）。

    语气标注是这套本地 TTS 管线特有的：模型在第一个 token 标 ``温和|``，
    harness/speak_tag.py 拿它挑发声指示、再把标签剥掉才送去合成。所以拼在这里
    而不是塞进库。

    realtime 那条路**不能加**：出声的是 OpenAI 自己，没有"剥掉标签再念"这一步，
    标签会被原样念出来，也会原样出现在字幕和记忆里（"温和|你喜欢尝新口味……"）。
    那条路的语气靠 instructions 里的文字描述控制，本来就不需要标签。

    这是 system 里**唯一稳定的前缀**，每轮都一样，所以拼接顺序不能变：
    变的东西（记忆、历史）一律排在它后面，见 core.py:248。"""
    lang = lang or SPACE_LANG
    base = persona.system_prompt(lang)
    if MODE == "realtime":
        return base
    return f"{base}\n\n{speak_tag.prompt_rule(lang)}"


def _history_block(session_id: str, space: str) -> str:
    from voicemem.lang import is_zh
    return _SESSION_CONTEXT.render(session_id, space, "zh" if is_zh() else "en")


def _push_history(session_id: str, space: str, user_text: str, reply_text: str,
                  interrupted: bool = False) -> str:
    return _SESSION_CONTEXT.add(
        session_id, space, user_text, reply_text, interrupted=interrupted)


def _finish_history_turn(turn_id: str, result: dict) -> None:
    committed = bool((result or {}).get("persistent_memory_created"))
    _SESSION_CONTEXT.mark_complete(turn_id, committed)
    if BARGE_DEBUG:
        state = "已进入长期记忆，移出 SessionBuffer" if committed else "未入库，保留短期上下文"
        print(f"[context] turn={turn_id[:8] or '-'} {state}", flush=True)


def _realtime_instructions(memory_context: str, stranger: bool = False,
                           replay: bool = False, emotion: str = "",
                           text: str = "", context_session: str = "",
                           context_space: str = "") -> str:
    """人设 + 这一轮检索到的记忆。要说清楚这是「你记得的事」，否则模型会把它当成
    背景资料念出来，而不是当成自己对这个用户的记忆自然地用。

    ``text``：用户这一轮说的话。只用来判断"他是不是在找一段录音而我们没找到"——
    那种情况要明说没找到，否则模型会顺口答"马上播放"然后什么都不放。"""
    if stranger:
        out = f"{_rt_persona()}\n\n{persona.stranger_note(SPACE_LANG)}"
        return f"{out}\n\n{_lang_note()}" if _lang_note() else out
    parts = [_rt_persona()]
    if memory_context:
        parts.append(memory_context)
    else:
        parts.append(persona.no_memory_note(SPACE_LANG))   # 一条都没检索到：别编
    session_context = _history_block(
        context_session, context_space or ACTIVE_SPACE)
    if session_context:
        parts.append(session_context)
    if _lang_note():
        parts.append(_lang_note())
    tone = _tone_note(emotion)
    if tone:
        parts.append(_by_lang(_STATE_LABEL) + tone)
    if replay:
        parts.append(_by_lang(_REPLAY_NOTE))
    elif _wants_sound(text):
        parts.append(_by_lang(_NO_REPLAY_NOTE))
    return "\n\n".join(parts)


# ══════════════════ 统一配置入口：一个 dict 配齐所有本地/api 模型 ══════════════════
# 打开这个 dict 就知道每个模型走本地还是 api。记忆侧（embedding/slots）走本地句向量
# → 整条 search 0 LLM、0 网络（实测 Search 本体 ~10ms）；reply 段（回复用的
# llm/tts/realtime）也在这一处配，省得分散在各处 env。缺省项走内置默认。
# 想外挂一份自定义 config：--config path.json（或 VOICEMEM_CONFIG）整体覆盖。
CONFIG = {
    "mode": "multi_modal",
    "memory_root": ARGS.memory_root or None,
    "space": ARGS.space,
    # 用哪个本地模型见 local_embedder.REGISTRY：默认 e5（多语）；加 "model": "bge"
    # 按空间语言选中/英那一版。换模型要 tools/reembed.py 重算老库，见 check_embedding。
    "embedding": {"provider": "local"},              # 记忆向量走本地模型（0 网络）
    "slots":     {"provider": "local"},              # slot 分类走同一个模型（0 LLM）
    # reply：回复用模型（核心不管，web 读）。默认全走 OpenAI api。
    "reply": {
        # system 在 get_space() 里按**这个空间的语言**重填。这里留空是故意的：
        # CONFIG 是模块级的，建它的时候还没打开任何空间，SPACE_LANG 还是默认值。
        "llm":      {"provider": "openai", "config": {"model": utils.CHAT_MODEL,
                                                      "system": ""}},
        # model 只在 openai 后端下传：TTS_MODEL 是 OpenAI 的模型名
        # （gpt-4o-mini-tts），本地后端拿它当仓库名/文件名去加载必然失败——
        # 而失败发生在回复任务里，表现是"文字和声音都没有"，看不出跟 TTS 有关。
        # 本地后端的模型各自有自己的环境变量（VOICEMEM_TTS_MODEL /
        # VOICEMEM_BREEZE_MLX_MODEL），默认值也各自合理。
        "tts":      ({"provider": utils.TTS_BACKEND,
                      "config": {"model": utils.TTS_MODEL}}
                     if utils.TTS_BACKEND == "openai"
                     else {"provider": utils.TTS_BACKEND, "config": {}}),
        "realtime": {"provider": "openai", "config": {"model": utils.RT_MODEL}},
    },
}

# --config / VOICEMEM_CONFIG 指向的 json 整体覆盖上面的 CONFIG（一个文件配齐）。
if ARGS.config:
    CONFIG = json.loads(Path(ARGS.config).read_text(encoding="utf-8"))

_LOCAL_LLM = None
if ARGS.llm == "deepseek":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("--llm deepseek 需要 DEEPSEEK_API_KEY（只在环境变量中设置）")
    CONFIG.setdefault("reply", {})["llm"] = {
        "provider": "deepseek", "config": {
            "model": os.environ.get("VOICEMEM_DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "system": ""}}
    print("[web] 文本回复：DeepSeek API / 非思考模式；不加载本地 LLM", flush=True)
if ARGS.llm == "local":
    from voicemem.local_llm import LocalLLM
    _LOCAL_LLM = LocalLLM(system=_rt_persona(ARGS.lang))
    if isinstance(CONFIG.get("reply"), dict) and "tts" in CONFIG["reply"]:
        CONFIG.setdefault("tts", CONFIG["reply"]["tts"])
    CONFIG["reply"] = _LOCAL_LLM          # VoiceMem(reply=...) 收可调用对象

REPLY = CONFIG.get("reply")                           # 传给 utils 的回复函数

# 声明式构造：from_config 是现有注入机制之上的糖（VoiceMem(embedding=fn, slots=fn,…)）。
#: 每个 Memory Space 一个 VoiceMem 实例，按需建、建好留着。
#:
#: 同一进程内建第二个实例几乎不花钱：模型是懒加载 + 进程内复用的，实测建实例
#: 0.0s、预热 2.5s（第一个是 6.8s + 4.9s）。所以切换空间不用重启服务。
_SPACES: dict = {}


def space_dir(name: str):
    """这个空间在磁盘上的目录。名字只允许字母数字和 - _，避免路径穿越。"""
    import re as _re
    safe = _re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]", "", (name or "").strip())[:32]
    if not safe:
        raise ValueError("空间名字不能为空")
    return _ROOT / "voicemem_memoryspace" / safe, safe


def get_space(name: str):
    """取（必要时创建）这个空间的 VoiceMem。"""
    _, safe = space_dir(name)
    if safe not in _SPACES:
        cfg = dict(CONFIG)
        cfg["space"] = safe
        # 人设按这个空间的语言选。实例是按空间缓存的，所以一个空间灌一次就够，
        # 不用每轮传——realtime 那条路走 _realtime_instructions()，每轮现拼。
        lang = space_language(safe)
        # reply 可能是**配置字典**（内置 provider），也可能是**可调用对象**
        # （--llm local 传进来的 LocalLLM）。后者自带人设，不能当字典展开——
        # 以前这里无条件 {**CONFIG["reply"]}，开本地模型直接 TypeError。
        if isinstance(CONFIG["reply"], dict):
            cfg["reply"] = {**CONFIG["reply"],
                            "llm": {**CONFIG["reply"]["llm"],
                                    "config": {**CONFIG["reply"]["llm"]["config"],
                                               "system": _rt_persona(lang)}}}
        elif hasattr(CONFIG["reply"], "set_system"):
            CONFIG["reply"].set_system(_rt_persona(lang))
        t0 = time.monotonic()
        inst = VoiceMem.from_config(cfg)
        inst.warmup(verbose=False)
        _SPACES[safe] = inst
        print(f"[space] 打开「{safe}」用了 {time.monotonic()-t0:.1f}s", flush=True)
    return _SPACES[safe]


def use_space(name: str) -> str:
    """切到这个空间。返回真正用的名字。

    ``vm`` 是模块级全局，下游全部按名字在运行时查找，所以这里重新绑定就够了——
    不用把实例一路传下去。注意 build_app 收的那几个回调必须是 lambda 而不是
    ``vm.classify`` 这种绑定方法，绑定方法会把切换前那个实例焊死。
    """
    global vm, ACTIVE_SPACE, SPACE_LANG
    vm = get_space(name)
    _, ACTIVE_SPACE = space_dir(name)
    SPACE_LANG = space_language(ACTIVE_SPACE)
    _set_lang(SPACE_LANG)            # 记忆语言跟着空间走
    if _LOCAL_LLM is not None:       # 本地 LLM 只有一个实例，人设得跟着换
        _LOCAL_LLM.system = _rt_persona(SPACE_LANG)
    return ACTIVE_SPACE


def list_spaces() -> list:
    """磁盘上有哪些 Memory Space，各有多少条记忆。"""
    import sqlite3
    root = _ROOT / "voicemem_memoryspace"
    out = []
    for d in sorted(p for p in root.glob("*") if p.is_dir()):
        n = 0
        try:
            from voicemem.utils.common import space as _sp
            db = _sp.db(d)
            if Path(db).exists():
                c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                n = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                c.close()
        except Exception:
            n = 0
        out.append({"id": d.name, "name": d.name, "count": n,
                    "active": d.name == ACTIVE_SPACE, "open": d.name in _SPACES,
                    "language": space_language(d.name)})
    return out


def create_space(name: str, language: str = "") -> dict:
    """建一个新的空 Memory Space：磁盘上出现这个名字的文件夹，里面是全新的空库。

    实例这里就建出来（顺带预热），这样点完"创建"立刻就能对话，不用等第一句话
    卡在模型加载上。
    """
    d, safe = space_dir(name)
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"「{safe}」已经存在了")
    d.mkdir(parents=True, exist_ok=True)
    lang = "zh" if str(language or ARGS.lang).lower().startswith("zh") else "en"
    # **必须先写语言再建实例**：get_space 里的人设、以及 VoiceMem 内部的 embedding
    # 模型和流式 ASR，都是照 `<space>.json` 的 space.language 挑的。先建实例的话那
    # 一刻 json 还不存在，全线按默认 en 挑完并缓存进 _SPACES，之后再写 zh 也追不回来
    # ——表现就是"新建了中文空间，却用英文模型转写、英文人设回复"。
    _write_space_language(safe, lang)    # 建的时候定一次，之后不再变
    get_space(safe)                      # 建库 + 预热（此时才读得到语言）
    print(f"[space] 新建「{safe}」（语言 {lang}）→ {d}", flush=True)
    return {"id": safe, "name": safe, "count": 0, "language": lang}


ACTIVE_SPACE = ""
vm = None
use_space(ARGS.space)


#: 语音轮的音频落在这儿。归档表存的是路径，文件本身得真的在。
TURN_AUDIO_DIR = _ROOT / "results" / "turn_audio"


def save_turn_audio(pcm16k) -> str:
    """把这一轮的 PCM 存成 wav，返回路径；存不下就返回 ""（不影响这一轮对话）。

    没有这一步，AudioArchive 里一条记录都不会有——它只在 ingest 收到 audio_path
    时才写。之前 demo 全程走 WS 流、从不落盘，所以"把当时那段原声放回来"做不到。
    """
    if pcm16k is None or not len(pcm16k):
        return ""
    try:
        import numpy as np
        import soundfile as sf
        TURN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        path = TURN_AUDIO_DIR / f"turn_{uuid.uuid4().hex[:12]}.wav"
        sf.write(path, np.asarray(pcm16k, dtype="float32"), 16000)
        return str(path)
    except Exception as e:
        print(f"[web] 存本轮音频失败（不影响对话）：{e}", flush=True)
        return ""


@dataclass
class Pending:
    """一轮说完时、投机预取早已算好的「预算记忆」——控制流拿来直接回复，不再搜。"""
    text: str
    memory_context: str
    result: object
    spoken: bool = True          # True=语音轮（音频已进 realtime 缓冲），False=打字轮
    audio_path: str = ""         # 这一轮落盘的 wav；ingest 拿它做场景/音乐/声纹感知，
                                 # 并在 audio_archive 里跟记忆绑定，之后能原样放回来
    stranger: bool = False       # 声纹认出说话的不是这个记忆库的主人
    replay: str = ""             # 该把哪条记忆当时那段原声放回来（memory_id），空=不放
    emotion: str = ""            # 上一轮感知到的情绪，用来给这一轮定语气
    route: str = gate.DEEP       # 轮次闸门判的路：deep 才注入事实记忆，见 voicemem/gate.py
    #: 提前生成的那一份还作数吗。判据是**下注之后你有没有接着说**，不是"文本一模
    #: 一样"——ASR 会边说边修正尾巴，人明明已经闭嘴了也会因为差一个词而白白作废。
    early_ok: bool = False
    #: VAD 最后一次判到人声的时刻（monotonic）。用来量"你闭嘴到助手出声"这整段——
    #: [lat] 原来的起点是"回合交出"，那已经在 300ms 静音确认 + 离线 ASR 复核之后了，
    #: 中间那截从没被量过，而用户感受到的等待正是从闭嘴那一刻开始算的。
    speech_end: float = 0.0


# ══════════════════ 两条控制流（各 ~10 行，只消费预取好的 Pending）══════════════════

# 分句规则（决定多久出第一声）在 voicemem/tts.py 的 cut_point。
_cut_point = utils.cut_point
#: 第一段过了这个长度就可以在词边界切（跟 voicemem/tts.py 的 _FIRST_MAX 同源）。
_FIRST_MAX_CHARS = int(os.environ.get("VOICEMEM_TTS_FIRST_MAX", "24"))


def _mentioned(name: str, text: str) -> bool:
    """实体名在这句话里被提到了吗。

    不能直接 `name in text`——图里的实体常带限定词（"Jiaqi的老板"），而人说的是
    "老板"。拆成词块（按"的"和非中文断开）逐个比，任一块出现就算提到。
    单字块不算：一个"我"、一个"歌"太容易误命中。
    """
    t = text or ""
    if not name:
        return False
    if name in t:
        return True
    for chunk in re.split(r"[的\s·、,，]+|[^\u4e00-\u9fffA-Za-z0-9]+", name):
        if len(chunk) >= 2 and chunk in t:
            return True
    return False


#: 声学模型至少要这么有把握才采纳（emotion2vec+ 的 softmax 分数）。
ACOUSTIC_MIN_SCORE = float(os.environ.get("VOICEMEM_ACOUSTIC_MIN", "0.92"))

#: 只有这几类才听声学的。
#:
#: 实测 emotion2vec+ 在真实录音上会**自信地判错**：「哎呀，早上好呀」判悲伤
#: 1.00、「我觉得挺好的」判悲伤 1.00——单靠提高阈值挡不住，它对错的答案给的
#: 就是满分。但它错的时候几乎全错在"悲伤"（对这个说话人的默认倾向）。
#: 高唤起的那几类（真笑出声、真发火）声学特征明显，恰恰是文本看不出来的，
#: 留给它；低唤起的交给语义。
ACOUSTIC_TRUST = set((os.environ.get("VOICEMEM_ACOUSTIC_TRUST")
                      or "开心,委屈,惊讶").split(","))

#: 情绪的语义样例。用本地 E5 把这句话跟这些比相似度——比关键词表宽
#: （"我觉得挺好的"里一个情绪词都没有），比声学准得多。0 网络、9ms。
_EMO_PROTO = {
    "开心": ["我今天特别开心", "太好了我很高兴", "真不错，我挺满意的", "哈哈太有意思了"],
    "悲伤": ["我很难过", "我心里特别难受", "我好失落", "这事让我挺沮丧的"],
    "委屈": ["我好生气", "太气人了", "凭什么这样对我", "我觉得很不公平"],
    "焦虑": ["我压力好大", "我有点紧张", "我很担心做不完", "这事儿让我睡不着"],
    "疲惫": ["我好累啊", "累死了，撑不住了", "一天下来人都空了"],
    # 「平静」要多给几句：日常陈述句和疑问句占了对话的大半，样例太少时它们
    # 撑不出足够的差距，就一路判成"(空)"——实测「我对花生过敏」「我不能吃什么」
    # 都因为这个没标上，加了样例之后 gap 从 0.002 涨到 0.08。
    "平静": ["今天天气不错", "我明天要去开会", "早上好", "我叫小明",
             "这个东西放在桌上", "我对花生过敏", "我在一家公司上班",
             "下周三下午三点有个会", "我不能吃什么", "这个怎么用",
             "帮我看一下", "我住在市中心"],
}
_PROTO = {}


def _emotion_by_meaning(text: str) -> str:
    """语义最近邻。够像**而且**跟第二名拉开差距才给标签，否则空——
    模棱两可时不标，比标错强。

    用的是**记忆库那一份**句向量模型（``utils.shared_embed_model()``），不是单独
    再加载一个。注意下面两个阈值（0.80 / 0.02）是跟着模型的余弦分布调的：换了
    embedding 模型要重调，不重调不会报错，只会变成"几乎不打标签"或"乱打标签"。
    """
    import numpy as np
    if not _PROTO:
        labels, sents = [], []
        for k, vs in _EMO_PROTO.items():
            labels += [k] * len(vs)
            sents += vs
        _PROTO["labels"] = labels
        _PROTO["V"] = np.array(
            utils.shared_embed_model().encode(sents, normalize_embeddings=True),
            dtype=np.float32)
    q = np.array(utils.shared_embed_model().encode([text], normalize_embeddings=True),
                 dtype=np.float32)[0]
    sims = _PROTO["V"] @ q
    i = int(np.argmax(sims))
    lab, best = _PROTO["labels"][i], float(sims[i])
    other = [float(sims[j]) for j in range(len(sims)) if _PROTO["labels"][j] != lab]
    gap = best - (max(other) if other else 0.0)
    return lab if best >= 0.80 and gap >= 0.02 else ""


#: emotion2vec+ 的英文标签 → 我们用的中文标签
_E2V_MAP = {"happy": "开心", "sad": "悲伤", "angry": "委屈", "fearful": "恐惧",
            "surprised": "惊讶", "disgusted": "厌恶", "neutral": ""}
_E2V = {}


def _acoustic_emotion(audio_path: str):
    """emotion2vec+（专门做语音情绪识别的模型）。返回 (标签, 分数)。"""
    if "m" not in _E2V:
        from funasr import AutoModel
        _E2V["m"] = AutoModel(model=os.environ.get("VOICEMEM_E2V_MODEL",
                                                   "emotion2vec/emotion2vec_plus_base"),
                              hub="hf", disable_update=True)
    r = _E2V["m"].generate(audio_path, granularity="utterance", extract_embedding=False)
    if not r:
        return "", 0.0
    lab, score = max(zip(r[0]["labels"], r[0]["scores"]), key=lambda x: x[1])
    en = str(lab).split("/")[-1].strip().lower()
    return _E2V_MAP.get(en, ""), float(score)


_SV = {}


def _sensevoice():
    """SenseVoiceSmall，一次推理同时出精转写和声学情绪。懒加载、进程内复用。

    注意别用 vm.utils.get("emotion")——那是韵律象限的启发式
    （PaperAlignedEmotionDetector），没有 run_with_emotion，而且判得不准。
    """
    if "t" not in _SV:
        from voicemem.utils.audio.asr import Transcriber, pick_device
        _SV["t"] = Transcriber(pick_device())
    return _SV["t"]


def _kick_acoustic(send, audio_path: str) -> None:
    """把声学情绪扔到后台算，算出可信结果再补一条 tag_update。

    emotion2vec 要跑整段音频，实测 2.3 秒。放在发 memory_hits 之前就等于把这
    2.3 秒加在"用户说完 → 助手开口"中间，而它给的结果十有八九还够不上信任阈值。
    放后台之后热路径一秒都不欠，真判准了 UI 上的情绪标签照样会更新。
    """
    if not audio_path or os.environ.get("VOICEMEM_ACOUSTIC_TAG", "1") == "0":
        return

    async def run():
        try:
            await wait_idle("声学情绪")
            t0 = time.monotonic()
            emo, score = await asyncio.to_thread(_acoustic_emotion, audio_path)
            take = bool(emo) and score >= ACOUSTIC_MIN_SCORE and emo in ACOUSTIC_TRUST
            if BARGE_DEBUG:
                print(f"  [emotion] 声学(后台) {(time.monotonic()-t0)*1000:.0f}ms "
                      f"-> {emo or '-'} {score:.2f}（{'采纳' if take else '不采纳'}）", flush=True)
            if take:
                await send({"type": "tag_update", "emotion": emo, "emotion_from": "acoustic"})
        except Exception as e:
            print(f"[web] 后台声学情绪跳过：{type(e).__name__}: {e}", flush=True)

    asyncio.create_task(run())


def fill_tags(payload: dict, text: str, audio_path: str = "",
              acoustic: bool = True) -> dict:
    """补上标签栏要的 emotion / entities——两样都是 0 LLM、0 网络。

    检索走的是本地 slot 分类器（投机预算内不能联网），它只出 slot，不出实体；
    情绪则要等这一轮 ingest 之后才算得出，而 memory_hits 是在回复之前就发的。
    结果就是标签栏上这两格一直空着。

    · 情绪：用 anchor_router 的关键词表现算一次（纯查表）。
    · 实体：这一轮命中的那几条记忆在认知图里挂了哪些实体，直接读（纯 sqlite）。
    """
    # ⓪ 右脑那几条换成第一人称的人话（只换显示，raw 原文照旧留着给脑图匹配）。
    #    还没改写好的这一轮先显示原文，同时排进后台队列——见 rb_human。
    for h in payload.get("right_brain_hits") or []:
        claim = h.get("claim") or ""
        if not claim:
            continue
        human = rb_human(claim)
        if human and human != claim:
            # 证据那半截（"｜他说过：…"）保留，它才是"你凭什么这么说"的支撑。
            tail = ""
            for sep in ("｜他说过：", " | he said: "):
                if sep in h.get("content", ""):
                    tail = sep + h["content"].split(sep, 1)[1]
                    break
            h["content"] = human + tail

    # ① 人明说了情绪就按他说的（查表，0 网络）——最准
    if not payload.get("emotion") and text.strip():
        try:
            from voicemem.rightbrain.anchor_router import normalize_emotion_strict
            payload["emotion"] = normalize_emotion_strict(text) or ""
        except Exception:
            pass

    # ② 没明说就看这句话的**意思**：本地 E5 跟每种情绪的样例句比语义相似度。
    #    9ms、0 网络，而且比关键词表宽（"我觉得挺好的"里一个情绪词都没有）。
    if not payload.get("emotion") and text.strip():
        try:
            emo = _emotion_by_meaning(text)
            if emo:
                payload["emotion"] = emo
                payload["emotion_from"] = "semantic"
        except Exception as e:
            print(f"[web] 语义情绪跳过：{type(e).__name__}: {e}", flush=True)

    # ③ 声学（emotion2vec+）：只在它**很有把握**时才盖过上面的判断。
    #
    #    为什么不让它当主力：实测在真实录音上它把「哎呀，早上好呀」判成难过
    #    （3.2 秒干净音频，不是静音太长的锅——裁剪过照样如此），SenseVoice 和
    #    韵律启发式也一样。这个麦克风/说话方式下，声学读不准。
    #    所以留着它，但要求 score ≥ ACOUSTIC_MIN_SCORE 才作数——真正带情绪地
    #    说话时它会给 0.95+，平淡说话时给的是 0.6、0.7 那种，正好挡掉。
    if acoustic and audio_path and os.environ.get("VOICEMEM_ACOUSTIC_TAG", "1") != "0":
        try:
            t0 = time.monotonic()
            emo, score = _acoustic_emotion(audio_path)
            take = bool(emo) and score >= ACOUSTIC_MIN_SCORE and emo in ACOUSTIC_TRUST
            if take:
                payload["emotion"] = emo
                payload["emotion_from"] = "acoustic"
            if BARGE_DEBUG:
                why = "采纳" if take else ("把握不够" if score < ACOUSTIC_MIN_SCORE
                                          else f"{emo} 不在信任名单")
                print(f"[emotion] 声学 {(time.monotonic()-t0)*1000:.0f}ms "
                      f"-> {emo or '-'} {score:.2f}（{why}）"
                      f"  最终={payload.get('emotion') or '-'}", flush=True)
        except Exception as e:
            print(f"[web] 声学情绪跳过：{type(e).__name__}: {e}", flush=True)

    # 实体：这里**不猜**。
    #
    # memory_hits 是在回复之前发的，而实体是跟抽事实同一次 LLM 调用出来的
    # （ingest 时，回复之后）。曾经在这儿做过一个"0 成本近似"——从命中的旧记忆
    # 里挑这句话提到过的实体——结果每句话都只剩说话人自己：
    #     说「我对花生过敏」→ 标签栏 ['小林']，真正抽出来的是 ['小林','花生']
    #     说「下周三在国金中心见客户」→ 标签栏空
    # 显示一个错的比先空着更糟。前端会在 ingest 落库后用真结果补上（见
    # voicemem.html 的 loadMemories）。

    rb = payload.get("right_brain_hits") or []
    inner = sum(1 for h in rb if h.get("internal"))
    print(f"[hits] 左脑 {len(payload.get('left_brain') or [])} 条  "
          f"右脑 {len(rb)} 条(内部 {inner}，页面显示 {len(rb)-inner})  "
          f"情绪={payload.get('emotion') or '-'}  "
          f"实体={'、'.join(payload.get('entities') or []) or '-'}", flush=True)
    return payload


class ReplySink:
    """回复的出口：先攒着，``commit()`` 之后转直发。

    提前生成靠它——用户还没说完时就照常跑一遍 ``voicemem_llm_tts``，只是文本和音频
    都落在这里，前端什么都收不到。等他真说完、且这一轮的文本跟当初赌的一致，
    ``commit()`` 把攒下的一次性放出去（音频已经现成，**立刻出声**），之后生成的
    部分直接走直发。赌错就整个丢掉，用户从头到尾没察觉。

    发送顺序必须保住：``answer_start`` 要排在音频前面，音频之间要按序。所以攒的是
    一条时间线（(kind, payload) 的列表），不是两个队列。
    """

    def __init__(self, send, send_audio):
        self._send, self._send_audio = send, send_audio
        self._buf: list[tuple[str, object]] = []
        self.live = False
        self._lock = asyncio.Lock()

    async def send(self, msg):
        async with self._lock:
            if self.live:
                await self._send(msg)
            else:
                self._buf.append(("json", msg))

    async def send_audio(self, pcm: bytes):
        async with self._lock:
            if self.live:
                await self._send_audio(pcm)
            else:
                self._buf.append(("pcm", pcm))

    @property
    def buffered_ms(self) -> float:
        n = sum(len(p) for k, p in self._buf if k == "pcm")
        return n / 2 / MIC_RATE * 1000

    async def commit(self) -> None:
        """赌对了：把攒下的按原顺序放出去，之后转直发。"""
        async with self._lock:
            for kind, payload in self._buf:
                if kind == "json":
                    await self._send(payload)
                else:
                    await self._send_audio(payload)
            self._buf.clear()
            self.live = True


def build_reply_context(memory_context: str, *, stranger: bool = False,
                        route=None, replay: str = "", text: str = "") -> str:
    """记忆 + 各种附加说明 → 交给回复模型的那段 context。

    **本地模型预热要跟它逐字一致**，所以只能有这一份实现。预热在用户还在说话时
    先把这段算进 KV 缓存，说完只剩他那句话要算；两边只要差一个字，前缀就对不上，
    整份白热（见 voicemem/local_llm.py 的 prewarm）。以前这段是内联在回复里的，
    预热那边照着抄一份，抄漏了 note 和语言提示——尾巴里多出五十多个 token，
    每轮多等一百多毫秒。
    """
    ctx = persona.stranger_note(SPACE_LANG) if stranger else (memory_context or "")
    if not stranger and gate.needs_memory(route) and not ctx.strip():
        ctx = persona.no_memory_note(SPACE_LANG)    # 该检索却一条都没有：别编
    note = (_by_lang(_REPLAY_NOTE) if replay
            else (_by_lang(_NO_REPLAY_NOTE) if _wants_sound(text) else ""))
    if note:
        ctx = f"{ctx}\n\n{note}" if ctx else note
    if _lang_note():
        ctx = f"{ctx}\n\n{_lang_note()}" if ctx else _lang_note()
    return ctx


def _mem_line() -> str:
    """进程 RSS / MLX 活跃与峰值 / 系统 swap，一行。峰值在哪一步涨上去的，靠它对日志。"""
    parts = []
    try:
        import psutil
        p = psutil.Process()
        parts.append(f"rss {p.memory_info().rss / 2**30:.1f}G")
        sw = psutil.swap_memory()
        parts.append(f"swap {sw.used / 2**30:.1f}/{sw.total / 2**30:.0f}G")
    except Exception:
        pass
    try:
        import mlx.core as mx
        parts.append(f"mlx {mx.get_active_memory() / 2**30:.1f}G 峰 {mx.get_peak_memory() / 2**30:.1f}G")
    except Exception:
        pass
    return " · ".join(parts)


#: 逐轮延迟统计：首轮单列（冷缓存、首次分配），后续轮次算中位数。
_LAT_HIST = {"n": 0, "first": None, "rest": []}


def _lat_note(total_ms: float) -> str:
    h = _LAT_HIST
    h["n"] += 1
    if h["first"] is None:
        h["first"] = total_ms
        return "首轮"
    h["rest"].append(total_ms)
    r = sorted(h["rest"])
    return f"第 {h['n']} 轮 · 后续 {len(r)} 轮中位 {r[len(r) // 2]:.0f}ms · 首轮 {h['first']:.0f}ms"


_REPLY_DISPLAY_LOCK = threading.Lock()


async def _send_reply_display(pending, send, ready, output_id, space, memory_vm):
    """展示不参与回复 prompt；首音频发送后才在工作线程上计算。

    跟随回复任务取消，过期结果不发给新回合。线程内的模型调用不能强杀，
    因此串行化展示计算，避免连续打断时堆出多份并发 embedding。
    """
    await ready.wait()
    if ACTIVE_SPACE != space or vm is not memory_vm:
        return
    started = time.monotonic()
    cancelled = threading.Event()

    def build():
        with _REPLY_DISPLAY_LOCK:
            if cancelled.is_set() or ACTIVE_SPACE != space or vm is not memory_vm:
                return None
            return fill_tags(
                utils.hits_payload(pending.result, has_audio=audio_of,
                                   cluster_of=hit_cluster),
                pending.text, pending.audio_path or "", acoustic=False)

    try:
        payload = await asyncio.to_thread(build)
        if payload is None or ACTIVE_SPACE != space or vm is not memory_vm:
            return
        if gate.needs_memory(pending.route):
            note_hits(pending.result)
        await send({"type": "memory_hits", "route": pending.route,
                    "output_id": output_id, **payload})
        print(f"[lat-ui] 展示标签后台耗时 {(time.monotonic()-started)*1000:.0f}ms"
              f"（已移出首音频关键路径，output={output_id}）", flush=True)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"[web] 回复展示跳过：{type(e).__name__}: {e}", flush=True)
    finally:
        cancelled.set()


async def voicemem_llm_tts(pending, send, send_audio, owner, timeline,
                           said=None, context_session="", context_space="",
                           memory_vm=None):
    memory_vm = memory_vm or vm
    context_space = context_space or ACTIVE_SPACE
    ready = asyncio.Event()
    display = asyncio.create_task(_send_reply_display(
        pending, send, ready, timeline.output_id, context_space, memory_vm))
    from voicemem.prompt_trace import prompt_scope
    try:
        with prompt_scope(output_id=timeline.output_id, session=context_session,
                          space=context_space, early=bool(getattr(pending, "early_ok", False))):
            return await _voicemem_llm_tts(
                pending, send, send_audio, owner, timeline, said=said,
                context_session=context_session, context_space=context_space,
                memory_vm=memory_vm, first_audio_ready=ready)
    finally:
        display.cancel()
        await asyncio.gather(display, return_exceptions=True)


async def _voicemem_llm_tts(pending, send, send_audio, owner, timeline,
                           said=None, context_session="", context_space="",
                           memory_vm=None, first_audio_ready=None):
    """记忆已在关键路径外预取好：LLM 流式回复 → TTS 流式语音。

    TTS 跟生成**并行**：LLM 吐满一句就丢进队列，另一条协程取出来合成、发音频。
    等全文生成完再开始合成的话，文本早打完了、音频还没起头（实测 TTS 首帧就要
    ~1.2s，加上生成那几秒，用户看着字干等）。

    ``said``：可选的 dict，边生成边把已说出的文本写进 said["text"]。打断判定要拿
    它挡回声（见 _is_echo）——助手说的话经麦克风绕回 ASR，转出来的字跟真人插话
    在字数上没区别，只能靠内容认。
    """
    memory_vm = memory_vm or vm
    context_space = context_space or ACTIVE_SPACE
    _entry = time.monotonic()
    await send({"type": "user_transcript", "text": pending.text})
    # memory_hits / 语义情绪只用于展示，由外层在首音频之后补发。
    if pending.replay:
        _note_replay(pending.replay)
        await send({"type": "play_memory", "memory_id": pending.replay})
    await send({"type": "answer_start", "output_id": timeline.output_id,
                "sample_rate": timeline.sample_rate})
    # 真实时延打点。之前只有离线基准（LLM 首字 768ms / TTS 首帧 800ms），
    # 实机上到底花在哪儿是猜的——这几行让日志能直接回答。
    _t0 = time.monotonic()
    _lat = {"llm": 0.0, "seg": 0.0, "audio": 0.0, "pre": 0.0}

    queue: asyncio.Queue = asyncio.Queue()

    # 走注入的那个 TTS（第九个可替换位）。--config 里换 provider、或库用户
    # VoiceMem(tts=lambda: MyTTS()) 传自己的实现，都在这儿生效；没配就是内置默认。
    tts = memory_vm.utils.get("tts")
    # 这一轮怎么念。默认按上一轮感知到的情绪；回复模型自己标了标签就用它的
    # （见 harness/speak_tag.py：它知道自己要说什么，比"上一轮用户什么心情"准）。
    # 标签是第一个 token，而 TTS 要等攒够第一段才开始，所以永远先到，不拖慢。
    speak_as = _speak_instruction(pending.emotion)
    tone = {"tag": "", "head": True, "buf": ""}   # head：还没剥过标签
    logged_instruction = object()

    def _synth_one(seg):
        """注入的 TTS 可能是用户自己写的、只认 stream(text)——那就退回去，
        少一层语气控制而已，不该因此整条链路报错。"""
        nonlocal logged_instruction
        import json
        effective = speak_as or getattr(tts, "instruction", "") or getattr(tts, "instructions", "")
        if effective != logged_instruction:
            logged_instruction = effective
            print(f"[tts-prompt] {timeline.output_id[:8]} "
                  f"{json.dumps(effective, ensure_ascii=False)}", flush=True)
        try:
            return tts.stream(seg, speak_as)
        except TypeError:
            return tts.stream(seg)

    # 一段回复被切成好几句、逐句合成，段与段之间会空一拍：合成完这段才发下一段的
    # 请求，中间要等「发请求 → 服务端 prefill → 第一个字节回来」，音频队列在这期间
    # 是空的，听感就是每句开头卡一下。TTS 在远端时（比如 Breeze 跑在 GPU 机器上，
    # 还隔着 SSH 隧道）这一拍尤其明显。
    # 所以拆成两级：拿到一段就**立刻**开合成、各自往自己的小队列里灌；播放那边按
    # 顺序一段段取。这样当前这段还在播的时候，下一段已经在算了。
    # 不限制并发：服务端是不是单并发由它自己决定（Breeze 就是），这边多发几个请求
    # 只是让它排上队，省掉每段一个来回的网络延迟。
    synths: list[asyncio.Task] = []
    streams: asyncio.Queue = asyncio.Queue()      # 每项是一段的 chunk 队列
    # 上面那段"不限制并发"只对远端成立。本地 Qwen-TTS 走的 gpu_loop 是**轮转**：
    # 两段同时合成 + LLM 还在解码后文，正在播的那段只分到 4/(4+4+1) 的 GPU，
    # 0.69x 实时变 1.57x——比播放慢，前端 80~320ms 的缓冲一秒内见底，就是长回复
    # 中间那一下卡。本地后端一段合成完再开下一段（见 tts.py 的 SERIAL），
    # 段与段之间不空拍：合成比播放快，下一段总在上一段播完前就算好了。
    _serial = asyncio.Semaphore(1) if getattr(tts, "SERIAL", False) else None

    async def synth():
        while (spec := await queue.get()) is not None:
            seg, text_start, text_end = spec
            chunks: asyncio.Queue = asyncio.Queue()
            state = {"complete": False}

            async def run(seg=seg, chunks=chunks, state=state):
                # LLM 解码和 TTS 合成都排在同一条 GPU 流上轮流推进(gpu_loop),
                # 不再需要在这里给谁让路——那套 hold/release 是为了错开两条并发
                # 流,而并发流正是驱动崩机的根因,现在从源头就没有了。
                try:
                    if _serial is not None:
                        await _serial.acquire()
                    try:
                        async for chunk in _synth_one(seg):
                            await chunks.put(chunk)
                    finally:
                        if _serial is not None:
                            _serial.release()
                    state["complete"] = True
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[web] 合成失败：{type(e).__name__}: {e}", flush=True)
                finally:
                    await chunks.put(None)        # 出错也要让播放那边收工

            synths.append(asyncio.create_task(run()))
            await streams.put((seg, text_start, text_end, chunks, state))
        await streams.put(None)

    async def _mark_first_audio():
        if not _lat["audio"]:
            _lat["audio"] = (time.monotonic() - _t0) * 1000
            # 两个起点都报：**闭嘴→出声**是用户真正感受到的等待，"回合交出→出声"
            # 只是它的后半段。前半段（静音确认 + 离线 ASR 复核 + 记忆确认）以前
            # 没人量过，可它是实打实压在体感上的。
            _vad = ((time.monotonic() - pending.speech_end) * 1000
                    if pending.speech_end else 0.0)
            _head = _vad - _lat["audio"] if _vad else 0.0
            total_label = (f"{_vad:.0f}ms" if pending.speech_end else
                           f"N/A（{'文字轮' if not pending.spoken else '提前生成那份' if pending.early_ok else '无VAD起点'}）")
            print(f"[lat] 闭嘴→首帧 {total_label}；生成前总等待 {_head:.0f}"
                  f" + LLM首字 {_lat['llm']:.0f}"
                  f" + 攒第一段 {_lat['seg'] - _lat['llm']:.0f}"
                  f" + TTS首帧 {_lat['audio'] - _lat['seg']:.0f}"
                  f"（回合交出→出声 {_lat['audio']:.0f}）｜{_lat_note(_lat['audio'])}",
                  flush=True)
            print(f"[mem] {_mem_line()}", flush=True)
            print(f"[lat-pre] 回复入口前 {max(0, (_entry-pending.speech_end)*1000) if pending.speech_end else 0:.0f}ms"
                  f" · 控制消息 {(_t0-_entry)*1000:.0f}ms"
                  f" · prompt准备 {_lat['pre']:.0f}ms"
                  f" · 等首个文字 {max(0, _lat['llm']-_lat['pre']):.0f}ms"
                  f" · VAD起点={'有' if pending.speech_end else '无'}", flush=True)

    async def speak():
        while (item := await streams.get()) is not None:
            seg, text_start, text_end, chunks, state = item
            segment_id = timeline.begin_segment(text_start, text_end)
            # 回声判定要拿**用户可能听到的**去比，不是已生成的——生成早跑到几段
            # 之后了。这里是音频真正开始发出去的时刻，最接近"说出口"。
            if said is not None:
                said["text"] = (said.get("text") or "") + seg
            try:
                while (chunk := await chunks.get()) is not None:
                    if isinstance(chunk, TimedAudioChunk):
                        if chunk.sample_rate != timeline.sample_rate:
                            raise ValueError(
                                f"TTS 输出采样率应为 {timeline.sample_rate}Hz，"
                                f"实际为 {chunk.sample_rate}Hz")
                        pcm = chunk.pcm
                        timeline.add_segment_timestamps(
                            segment_id, chunk.timestamps)
                    else:
                        pcm = chunk
                    timeline.append_audio(pcm)
                    await _mark_first_audio()
                    await send_audio(pcm)
                    if first_audio_ready is not None:
                        first_audio_ready.set()
            except asyncio.CancelledError:
                timeline.finish_segment(segment_id, complete=False)
                raise
            except Exception as e:                # 多半是听到一半关了页面，不是错误
                timeline.finish_segment(segment_id, complete=False)
                print(f"[web] 语音发送中断：{type(e).__name__}", flush=True)
                break
            else:
                timeline.finish_segment(
                    segment_id, complete=state["complete"])

    synther = asyncio.create_task(synth())
    speaker = asyncio.create_task(speak())
    reply, buf, sent = "", "", 0
    interrupted = False
    # 从这里到出声结束都算热路径：后台的声纹/情绪/入库看到这个标记就先不动手，
    # 别跟 LLM 和 TTS 抢那条唯一的 GPU 流（见 wait_idle）。
    _hot = hot_path_enter()
    try:
        # 跟 realtime 用同一份指令：两条路必须表现一致，否则换个 --mode
        # 人设和「右脑不许念出来」的约束就悄悄没了。
        # 走核心回复层（人设在 CONFIG.reply.llm.config.system，见 voicemem/reply.py
        # 的 compose_system：system + memory_context，和 realtime 那条拼出来的一样）。
        # 闸门判成浅/附和的轮次，核心那边压根没检索（见 voicemem/stream.py 的
        # _confirm），memory_context 天然是空的。这里只剩一件事：**别发那句"明说
        # 不知道"**——它是给"该查却查空了"用的，浅轮上发它，"讲个笑话"会被答成
        # "我不知道"。
        # 情绪不再拼进文本 prompt：那是**发声指示**（"压低、放软、留停顿"），
        # 让文字模型理解一遍再指望 TTS 猜出来，中间隔了两层。TTS 后端的 instruction
        # 参数就是收这个的，该搬过去。搬之前 pending.emotion 这一路暂时没有出口。
        ctx = build_reply_context(
            pending.memory_context, stranger=pending.stranger,
            route=pending.route, replay=pending.replay, text=pending.text)
        # 历史单独走消息数组，不再拼进 ctx（ctx 最终落在 system 里）。
        # 为的是 prompt 缓存：它复用最长公共前缀，而记忆每轮都变——记忆和历史
        # 挤在一起时，前缀从人设之后就断了，历史再长也一个 token 复用不上。
        hist = _SESSION_CONTEXT.messages(context_session, context_space,
                                         window=HISTORY_TURNS)
        _lat["pre"] = (time.monotonic() - _t0) * 1000
        async for d in memory_vm.reply_stream(pending.text, ctx, hist):
            if not _lat["llm"]:
                _lat["llm"] = (time.monotonic() - _t0) * 1000
            if tone["head"]:
                # 标签只可能在最前面，用**独立**的小缓冲攒（不能用 buf，那个是
                # 攒分句的，混用会把标签当成正文切出去）。攒到 12 个字符还没等到
                # 就放弃——一直攒着不发字会让首字延迟白白多几十毫秒。
                tone["buf"] += d
                tag, rest = speak_tag.split(tone["buf"])
                if tag:
                    # 跟上一轮拉平一下再用：模型每轮独立挑，日志里出现过
                    # 轻快→平静→抱歉 三连跳，单看每句都对、连起来像换了个人。
                    tag = speak_tag.smooth(_LAST_TONE["tag"], tag)
                    tone["tag"], tone["head"] = tag, False
                    _LAST_TONE["tag"] = tag
                    # **第二个参数要的是字符串，不是那个双语 dict。** 传 dict 进去
                    # 会把 "{'zh': …, 'en': …}" 整个塞进 TTS 的 instruction，
                    # 而这只发生在"认出标签"的那些轮——没认出标签的轮走
                    # _speak_instruction()，拼的是正常基调。同一段对话里两种指示
                    # 混着来，听感就是上一句下一句像两个人。
                    speak_as = speak_tag.instruction(
                        tag, _speak_base_env or _by_lang(_SPEAK_BASE))
                    d = rest
                    if BARGE_DEBUG:
                        print(f"[tone] 模型标的语气：{tag}", flush=True)
                elif len(tone["buf"]) < 26 and not any(
                        c in tone["buf"] for c in "]】"):
                    continue                  # 还可能是标签（含 "标签：轻快|" 这种
                    # 带前缀的），竖线/右括号还没出现就先攒着。18 给前缀留够空间。
                else:
                    tone["head"] = False      # 没标签，照旧
                    d = tone["buf"]
                if not d:
                    continue
            # 过了第一段的硬上限就在**词边界**切：新 token 以空格开头，说明前一个
            # 词刚好说完。MLX 的英文 token 是"空格开头"的（" have" 而不是 "have "），
            # 所以只看 buf 结尾是不是空格永远打不着——英文那种一句到底没逗号的回复
            # 会一直等到句号，实测第一段切出 70 个字、白等 200 多毫秒。
            if (sent == 0 and d[:1].isspace()
                    and len(buf.strip()) >= _FIRST_MAX_CHARS):
                segment = buf.strip()
                leading = len(buf) - len(buf.lstrip())
                start = len(reply) - len(buf) + leading
                if not _lat["seg"]:
                    _lat["seg"] = (time.monotonic() - _t0) * 1000
                    if BARGE_DEBUG:
                        print(f"[seg] 第一段 {len(segment)} 字（词边界切）"
                              f" → {segment!r}", flush=True)
                await queue.put((segment, start, start + len(segment)))
                buf, sent = "", sent + 1
            reply += d
            buf += d
            timeline.append_text(d)
            await send({"type": "answer_delta", "text": d})
            if _cut_point(buf, first=sent == 0):
                segment = buf.strip()
                leading = len(buf) - len(buf.lstrip())
                start = len(reply) - len(buf) + leading
                if not _lat["seg"]:
                    _lat["seg"] = (time.monotonic() - _t0) * 1000
                    if BARGE_DEBUG:
                        print(f"[seg] 第一段 {len(segment)} 字 → {segment!r}",
                              flush=True)
                await queue.put((segment, start, start + len(segment)))
                buf, sent = "", sent + 1
        if buf.strip():
            segment = buf.strip()
            leading = len(buf) - len(buf.lstrip())
            start = len(reply) - len(buf) + leading
            if not _lat["seg"]:
                _lat["seg"] = (time.monotonic() - _t0) * 1000
            await queue.put((segment, start, start + len(segment)))
    except asyncio.CancelledError:
        interrupted = True                      # 用户插话了，这一轮到此为止
    finally:
        await queue.put(None)                   # 生成出错也要让 speak() 收工

    async def _drop_pipeline():
        # 别把 speak() 留在后台继续往一条已经停播的连接上发音频。
        # 提前起跑的那几段合成也要一起停，否则它们会继续占着远端 TTS 的队列，
        # 下一轮的第一句得排在这些没人要的音频后面——听起来就是打断之后更卡。
        speaker.cancel()
        synther.cancel()
        for t in synths:
            t.cancel()
        await asyncio.gather(
            speaker, synther, *synths, return_exceptions=True)

    if interrupted:
        await _drop_pipeline()
    else:
        # 生成完了不等于说完了：音频还在一段段往外发，打断多半就落在这儿。
        # 不接住的话 CancelledError 会直接掀掉这个 Task，下面的存记忆一行都不跑，
        # 被打断的那一轮就永远进不了记忆。
        try:
            await synther
            await speaker
        except asyncio.CancelledError:
            interrupted = True
            await _drop_pipeline()
        else:
            # 合成完就放行后台活儿——**别等播放结束**。GPU 的活到这儿就干完了，
            # 剩下的几秒纯粹是音频在前端播；再堵着，声纹/情绪/入库白等好几秒，
            # 最后撞进下一轮的热路径里，等于没让。
            hot_path_exit(_hot)
            _kick_acoustic(send, pending.audio_path or "")
            timeline.mark_generation_complete()
            try:
                await send({"type": "answer_done", "output_id": timeline.output_id})
                timeout = max(2.0, min(
                    60.0, timeline.sent_samples / timeline.sample_rate + 2.0))
                await asyncio.wait_for(timeline.wait_playback_done(), timeout=timeout)
            except asyncio.TimeoutError:
                timeline.assume_drained()
            except asyncio.CancelledError:
                interrupted = True
                await _drop_pipeline()

    # 记录本轮上下文；记忆写入放到后台，避免阻塞下一轮收音。
    context_reply = timeline.heard_text() if interrupted else reply
    if interrupted and BARGE_DEBUG:
        print(f"[context] 打断于 {timeline.rendered_ms()}ms，保留回复 "
              f"{context_reply!r}", flush=True)
    history_turn_id = _push_history(
        context_session, context_space, pending.text, context_reply,
        interrupted=interrupted)
    hot_path_exit(_hot)                  # 出声结束，后台那些活儿可以动了
    queue_remember_turn(
        pending, context_reply, owner, history_turn_id, memory_vm=memory_vm)
    timeline.context_saved = True



async def start_realtime_turn(pending, conn, send, timeline,
                              context_session="", context_space=""):
    """把预取好的记忆注入 Realtime session，触发这一轮的原生语音。

    只负责"发起"；收音频/文本和收尾都在常驻的事件泵里（见 realtime_session）——
    OpenAI 的事件流只能有一个消费者，每轮各读各的会串台：上一轮被打断后残留的
    response.done 会被下一轮读到，当成自己说完了。
    """
    await send({"type": "user_transcript", "text": pending.text})
    if gate.needs_memory(pending.route):
        note_hits(pending.result)  # 让脑图快照保证这几条在图上。浅轮不记：这几条
                                   # 没进 prompt，算成"命中"会把记忆热度统计弄脏
    await send({"type": "memory_hits", "route": pending.route,
                **fill_tags(utils.hits_payload(pending.result, has_audio=audio_of,
                                              cluster_of=hit_cluster),
                            pending.text, pending.audio_path or "", acoustic=False)})
    _kick_acoustic(send, pending.audio_path or "")
    if pending.replay:
        # 前端收下先记着，等这一轮回复播完再放——助手的回复是排队播的，
        # 提前放会跟人声叠在一起。
        _note_replay(pending.replay)
        await send({"type": "play_memory", "memory_id": pending.replay})
    if pending.spoken:
        await conn.input_audio_buffer.commit()
    else:
        await conn.conversation.item.create(item={"type": "message", "role": "user",
                                                  "content": [{"type": "input_text", "text": pending.text}]})
    # 记忆走 response.create 的 per-response instructions，不是 session.update。
    # 后者是会话级设置，实测更新完模型这一轮根本读不到（问"我的猫叫什么"，库里
    # 明明检索到了"叫墨墨"，模型还答"你刚提过但我没听清"）。
    print(f"[lat] 本地判完说完 → 发 response.create", flush=True)
    await conn.response.create(response={
        "instructions": _realtime_instructions(
            pending.memory_context,      # 浅轮核心没检索过，这里本来就是空的
            pending.stranger,
            replay=bool(pending.replay),
            emotion=pending.emotion,
            text=pending.text,
            context_session=context_session,
            context_space=context_space),
    })
    await send({"type": "answer_start", "output_id": timeline.output_id,
                "sample_rate": timeline.sample_rate})


async def truncate_provider_output(conn, provider_item_id: str,
                                   timeline: AudioTimeline) -> None:
    truncate = getattr(conn, "truncate_output", None)
    if callable(truncate):
        await truncate(
            provider_output_id=provider_item_id,
            media_output_id=timeline.output_id,
            audio_end_samples=timeline.rendered_cutoff_samples(),
            sample_rate=timeline.sample_rate,
        )
        return
    conversation = getattr(conn, "conversation", None)
    item_api = getattr(conversation, "item", None)
    truncate = getattr(item_api, "truncate", None)
    if callable(truncate):
        await truncate(
            item_id=provider_item_id, content_index=0,
            audio_end_ms=timeline.rendered_ms())


async def _no_realtime(sock, err):
    """连不上 Realtime 时别让人对着 traceback 猜。

    要分清是**网络**还是**权限**：DNS/连接失败跟 key 没有关系，之前一律说成
    「key 可能没权限」，把人往错的方向指了。
    """
    name, text = type(err).__name__, str(err)
    network = (isinstance(err, (OSError, TimeoutError, ConnectionError))
               or "gaierror" in name.lower()
               or any(k in text.lower() for k in ("nodename", "temporary failure",
                                                  "name or service", "getaddrinfo",
                                                  "connection refused", "timed out")))
    if network:
        why = ("网络连不上 api.openai.com（DNS/代理/VPN 的问题，跟 key 无关）。"
               "确认能上网后重开；离线环境用 `--mode llm_tts` 也一样连不上，"
               "两条路都要访问 OpenAI。")
    elif any(k in text for k in ("401", "403", "invalid_api_key", "insufficient", "model_not_found")):
        why = ("这个 key 没有 Realtime 权限或模型不可用——改用 "
               "`python web/run.py --mode llm_tts`，那条路只要普通 chat + TTS。")
    else:
        why = ("先看这条报错本身；如果只是 Realtime 用不了，可以改用 "
               "`python web/run.py --mode llm_tts`（普通 chat + TTS）。")
    msg = f"连不上 OpenAI Realtime（{name}: {text}）。{why}"
    print(f"[web] {msg}", flush=True)
    try:
        await sock.send_json({"type": "error", "message": msg})
    except Exception:
        pass


# ══════════════════ 驱动 voicemem 核心流式会话（vm.stream()）══════════════════
# ASR + VAD + 投机预取（边说边预取 / 200ms 赌说完 / barge-in / 300ms 确认）
# 全在核心 VoiceStream 里。这里只做 demo 该做的：搬 socket 帧、发 partial、把说完
# 的一轮包成 Pending 交给控制流——demo 就是核心的使用示例，不再平行重写一套。

def remember_turn(pending, reply: str, owner: dict, history_turn_id: str = "",
                  memory_vm=None) -> None:
    """存这一轮，并顺手记下说话人是谁。

    说话人不用单独算：ingest 内部本来就要跑一次 preprocess（场景/声纹/情绪），
    返回值里直接带 speaker_id。之前我在热路径上又单独触发了一次完整感知——
    那套一次 424ms（AST 占 361ms），纯属重复劳动，而且挡在读 socket 前面。
    """
    # 没有转写文本的一轮 = 对着麦克风放了段声音（见 stream.py 的 MIN_SOUND_ONLY_S）。
    # 直接 ingest("") 抽不出任何事实，这段音频就白存了——之后问「刚才那首歌帮我
    # 重播」找不到任何记忆。给它一句话，让它能被检索到；音乐是什么由 ingest 内部
    # 的音乐识别打 tune: 标签，这里只负责让它有个记忆载体。
    # 判「有没有真的说话」不能只看空不空：音乐喂进 ASR 会硬转出一两个字母
    # （实测 cafe_song 转成 'i'），非空但毫无意义。要求至少两个中文字或三个
    # 英文字母才算说了话。
    text = pending.text
    meaningful = re.sub(r"[^\w\u4e00-\u9fff]", "", text or "")
    cjk = len(re.findall(r"[\u4e00-\u9fff]", meaningful))
    if pending.audio_path and cjk < 2 and len(meaningful) < 3:
        from voicemem.stream import SOUND_ONLY_TEXT
        text = SOUND_ONLY_TEXT          # 必须用这个常量，核心靠它认出没说话的那轮

    try:
        target_vm = memory_vm or vm
        r = target_vm.ingest(
            text, agent_reply=reply, async_facts=True,
            audio=pending.audio_path or None,
            on_complete=lambda result: _finish_history_turn(history_turn_id, result),
        ) or {}
    except Exception as e:
        print(f"[web] 存这一轮失败：{type(e).__name__}: {e}", flush=True)
        return
    # 上一轮的情绪留给下一轮的投机检索用。没有它右脑取不到情感记录，
    # 每轮只会返回同样那几条静态画像（见 voicemem/stream.py 的 emotion 说明）。
    affect = r.get("affect")
    if isinstance(affect, dict):
        affect = affect.get("emotion") or affect.get("label") or ""
    owner["emotion"] = str(affect or "").strip()

    # 这一轮听到音乐了就记住那段录音。tune 识别是同步的，这里拿到的时候后台
    # 入库才刚开始——"刚听完就问"能不能放出来，全靠这一句。
    if r.get("recognized_tune") and pending.audio_path:
        _remember_tune(pending.audio_path)
    elif BARGE_DEBUG and _wants_sound(pending.text or ""):
        print(f"  [replay] 这一轮没记住音乐："
              f"tune={bool(r.get('recognized_tune'))} audio={bool(pending.audio_path)}",
              flush=True)

    sid = r.get("speaker_id") or ""
    if not sid:
        # 这一轮太短，声纹压根没算（见 perceiver 的 VOICEMEM_SPEAKER_MIN_S）。
        # 不知道是谁 ≠ 换了个人，所以连 miss 计数都不动。
        return
    if not owner["id"]:
        owner["id"] = sid                  # 第一个开口的算这场对话的主人
    owner["last"] = sid
    owner["miss"] = 0 if sid == owner["id"] else owner.get("miss", 0) + 1


# 记忆写入在后台串行执行，避免阻塞实时事件循环和并发写入。
# 保存任务引用，确保会话结束后已排队的任务仍可完成。
#: 热路径（生成回复 / 出声）是不是正在跑。后台那些重活儿——声纹、emotion2vec、
#: 入库抽取——**都跟 LLM 和 TTS 抢同一块 GPU**，而它们一件都不急：记忆晚几秒写
#: 完没人察觉，回复晚几百毫秒出声人人都听得见。实测它们撞在一起时，真回合的
#: "排队等 GPU 线程"要 400~1300ms。所以让后台活儿等热路径空了再动手。
_HOT = {"n": 0}
_IDLE = asyncio.Event()
_IDLE.set()
#: 等不到空闲也不能无限等——用户一直说话就永远轮不到，记忆会积压到丢。
_IDLE_MAX_WAIT_S = float(os.environ.get("VOICEMEM_IDLE_MAX_WAIT", "8"))


def hot_path_enter() -> dict:
    """标记热路径开始。返回一个**只能被消费一次**的令牌，交给 hot_path_exit。

    令牌而不是裸计数：这一轮随时可能被打断（回复 task 直接被 cancel），配对的
    exit 就执行不到了，计数卡在 >0，之后每件后台活儿都要白等满超时。所以 exit
    做成幂等，再挂一个 task 完成回调兜底——两边都调也只生效一次。
    """
    token = {"open": True}
    _HOT["n"] += 1
    _IDLE.clear()
    try:
        asyncio.current_task().add_done_callback(lambda _: hot_path_exit(token))
    except Exception:                     # 不在 task 里（测试直接调）就算了
        pass
    return token


def hot_path_exit(token: dict) -> None:
    if not token.get("open"):
        return
    token["open"] = False
    _HOT["n"] = max(0, _HOT["n"] - 1)
    if _HOT["n"] == 0:
        _IDLE.set()


async def wait_idle(what: str = "") -> None:
    """后台重活儿开工前等一下热路径。最多等 _IDLE_MAX_WAIT_S。"""
    if _IDLE.is_set():
        return
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(_IDLE.wait(), timeout=_IDLE_MAX_WAIT_S)
    except asyncio.TimeoutError:
        pass
    if BARGE_DEBUG and what:
        print(f"  [idle] {what} 让路 {(time.monotonic()-t0)*1000:.0f}ms"
              f"（热路径计数 {_HOT['n']}）", flush=True)


_REMEMBER_LOCK = asyncio.Lock()
_REMEMBER_TASKS: set[asyncio.Task] = set()


async def _remember_background(pending, reply: str, owner: dict,
                               history_turn_id: str, memory_vm) -> None:
    queued_at = time.monotonic()
    async with _REMEMBER_LOCK:
        waited = time.monotonic() - queued_at
        if waited > 0.05 and BARGE_DEBUG:
            print(f"[memory] 入库排队 {waited:.2f}s", flush=True)
        await wait_idle("入库")
        started = time.monotonic()
        await asyncio.to_thread(
            remember_turn, pending, reply, owner, history_turn_id, memory_vm)
        if BARGE_DEBUG:
            print(f"[memory] 入库主流程 {time.monotonic()-started:.2f}s", flush=True)


def queue_remember_turn(pending, reply: str, owner: dict,
                        history_turn_id: str = "", memory_vm=None) -> None:
    memory_vm = memory_vm or vm
    task = asyncio.create_task(
        _remember_background(pending, reply, owner, history_turn_id, memory_vm))
    _REMEMBER_TASKS.add(task)

    def done(t: asyncio.Task) -> None:
        _REMEMBER_TASKS.discard(t)
        try:
            t.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[web] 后台记忆任务失败：{type(e).__name__}: {e}", flush=True)

    task.add_done_callback(done)


#: 比到多久以前。原来是 40 字——那是按「生成到哪儿就说到哪儿」估的，可生成比
#: 播放快得多（尤其现在会提前合成下一段），用户此刻听到的往往是好几秒前生成的
#: 内容，早滑出 40 字窗口了。改成按**已经说出口的**文本比，窗口也放宽。
ECHO_WINDOW = int(os.environ.get("VOICEMEM_ECHO_WINDOW", "300"))
#: 模糊匹配门槛：新出的字里，**连续**命中助手原话的那一段最长能占多大比例。
#:
#: 一开始用的是二元组重合率——那是错的：它不看连续性，用户插话只要词汇跟刚才
#: 聊的重合（"压力""GRE"这种，非常常见），散落的二元组就能凑过门槛，真插话被
#: 当回声吞掉，结果就是打断失灵。
#: 回声的特征是**一整段连续的原话**，真插话哪怕用词重合也接不成长串，所以改用
#: 最长公共子串。ASR 差一两个字只会把长串截短一点，仍然远高于真插话。
ECHO_RATIO = float(os.environ.get("VOICEMEM_ECHO_RATIO", "0.6"))
#: 短于这个长度只做精确匹配。两三个字的二元组太少，重合率动不动就是 1.0——
#: 用户跟着复述一个词（"GRE？"）就会被当成回声吞掉，那是真插话。
ECHO_FUZZY_MIN = int(os.environ.get("VOICEMEM_ECHO_FUZZY_MIN", "4"))


#: 附和词（backchannel）：听着的人随口应一声，不是要抢话。
#:
#: 打断判据是"转写比上次多出 ≥2 个字"，而中文的附和词正好两三个字——实测里
#: "对嗯""对的对"都把正在播的回复掐了。人一边听一边"嗯""对"是正常的对话行为，
#: 掐掉反而不自然。realtime 那条路靠 OpenAI 的 semantic_vad 判"这是不是真的在
#: 打断"来挡，llm_tts 这条没有，只能按词表挡。
#:
#: 判据是**新增的这一段整个都是附和词**：说"对，不过我想说的是…"时，新增里
#: 除了"对"还有别的，照样打断。代价只是纯附和的那次不打断，本来也不该打断。
#: **听**：用户说"嗯嗯"时不算一轮、不打断助手。助手自己**说**附和是另一个开关
#: （VOICEMEM_BACKCHANNEL_EMIT，见 harness/backchannel.py），两者互不相干。
BACKCHANNEL_ON = os.environ.get("VOICEMEM_BACKCHANNEL", "1") != "0"
# 附和词表搬进核心（voicemem/gate.py）：打断判定和"要不要检索"用的是同一张表，
# demo 和核心各留一份的话，改一处漏一处。这里只保留 demo 自己的开关。
_bc_norm = gate.norm

def _is_backchannel(new_chars: str) -> bool:
    """新增的这几个字是不是纯附和。"""
    return bool(BACKCHANNEL_ON) and gate.is_backchannel(new_chars)


def _is_echo(new_chars: str, said: str) -> bool:
    """ASR 新吐出的这几个字，是不是助手自己的声音绕回麦克风了。

    AEC 压不干净时，助手说的话会进 ASR，转出来的字跟真人插话在字数上没有区别。
    但内容上有：回声一定是助手**刚说过的原文**里的片段。

    ``said`` 要传**已经说出口的**文本（不是已生成的），两者能差好几秒。
    大小写要抹平——最早那个漏网的例子就是助手说 "Annie"、ASR 转出小写 "an"。
    """
    from echo_guard import is_echo
    return is_echo(new_chars, said, max(ECHO_WINDOW, min(len(said), 4096)),
                   ECHO_RATIO, ECHO_FUZZY_MIN)


_INTERRUPT_PREFIXES = tuple(_bc_norm(x) for x in (
    "停", "停一下", "先停", "暂停", "等等", "等一下", "先别说", "别说了", "打住",
    "stop", "wait", "hold on", "pause", "quiet",
))
_FILLER_PREFIX = re.compile(r"^[嗯呃啊哦噢喔欸诶唉哈哼]+")


def _barge_text(text: str) -> str:
    return _FILLER_PREFIX.sub("", _bc_norm(text))


def _is_explicit_interrupt(text: str) -> bool:
    clean = _barge_text(text)
    return bool(clean) and any(clean.startswith(prefix) for prefix in _INTERRUPT_PREFIXES)


def _has_barge_content(text: str) -> bool:
    """过滤单音节和纯语气词；只用于候选确认，不直接触发取消。"""
    clean = _barge_text(text)
    if not clean or len(set(clean)) == 1 or _is_backchannel(text):
        return False
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in clean)
    latin = sum(ch.isascii() and ch.isalnum() for ch in clean)
    return cjk >= BARGE_MIN_CHARS or latin >= max(3, BARGE_MIN_CHARS)


def _has_strong_final_barge(text: str) -> bool:
    clean = _barge_text(text)
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in clean)
    latin = sum(ch.isascii() and ch.isalnum() for ch in clean)
    return cjk >= 3 or latin >= 5


def _lcs_len(a: str, b: str) -> int:
    """最长公共**子串**（连续）长度。滚动一行的 DP，串都很短，开销可忽略。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ch in a:
        cur = [0] * (len(b) + 1)
        for j, cj in enumerate(b, 1):
            if ch == cj:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


# ── 附和（backchannel）─────────────────────────────────────────────────────
# 用户说到一半停 100ms 时"嗯"一声。什么时候出声由 harness/backchannel.py 的概率
# 模型定（句式/内容/情绪/韵律/不应期六个因子），这里只负责把声音送出去。
#
# 预合成放进程级：全部会话共用同一批音频，起服务后合成一次（有落盘缓存时更快）。
_BC_VOICE = {"obj": None, "task": None}

#: Realtime 和 TTS API 都有的音色名。
#:
#: **附和必须跟正文是同一个人的声音**，否则中间冒出来一声别人的"嗯"，比完全不附和
#: 突兀得多。两条路的音色来源不同：llm_tts 走注入的 TTS（音色就是它自己的），
#: realtime 走 OPENAI_REALTIME_VOICE。后者的默认值 marin（还有 cedar）**只有
#: Realtime 有**，TTS API 合不出来——那种情况下宁可不出声。
_TTS_SHARED_VOICES = {"alloy", "ash", "ballad", "coral", "echo",
                      "sage", "shimmer", "verse"}
#: 生成比实时慢的后端。它们不做附和预合成，理由见 _backchannel_tts。
_SLOW_TTS = {"VoxCPMTTS"}
#: 本地 TTS：启动时要预热，否则第一句要等模型加载。
_LOCAL_TTS = {"QwenTTS", "KokoroTTS", "VoxCPMTTS", "PiperTTS", "BreezeMLXTTS"}


def _backchannel_tts():
    """按当前 mode 拿一个**跟正文同音色**的 TTS；配不出同音色就返回 None。"""
    if MODE == "realtime":
        rt = str(getattr(utils, "RT_VOICE", "") or "")
        if rt not in _TTS_SHARED_VOICES:
            print(f"[backchannel] Realtime 音色 {rt!r} 在 TTS API 里没有对应的，"
                  f"合出来会是另一个人的声音 → 这条路关闭附和。"
                  f"想开就把 OPENAI_REALTIME_VOICE 换成 "
                  f"{'/'.join(sorted(_TTS_SHARED_VOICES))} 之一。", flush=True)
            return None
        from voicemem.tts import OpenAITTS
        return OpenAITTS(voice=rt)        # 同名音色，跟 Realtime 那侧是同一个人
    tts = vm.utils.get("tts")             # llm_tts：正文用哪个它就用哪个
    # **慢后端不做附和。** 附和要预合成 50 个词 × 3 种念法 = 150 段，而这些请求跟
    # 正文回复排在**同一个工作线程**里（本地模型只有一份，不能并发）。本地 TTS 一段
    # 要几秒，150 段就是好几分钟——正文的合成排在后面，表现就是"完全没有回复"。
    # 踩过这个，别再让它排队。
    if getattr(tts, "slow", False) or type(tts).__name__ in _SLOW_TTS:
        print(f"[backchannel] {type(tts).__name__} 合成比实时还慢，预合成 150 段会把"
              f"正文回复堵死 → 这条路关闭附和。想要附和请用 TTS_BACKEND=openai。",
              flush=True)
        return None
    return tts


def _backchannel_voice():
    """拿预合成好的那份；第一次调用时在后台起合成，没好之前返回 None。"""
    from harness.backchannel import BackchannelVoice
    if _BC_VOICE["obj"] is None:
        try:
            tts = _backchannel_tts()
            if tts is None:
                _BC_VOICE["obj"] = False
                return None
            _BC_VOICE["obj"] = BackchannelVoice(
                tts, lang=space_language(ACTIVE_SPACE),
                # key 只认**音色**，不带 mode：两条路现在用同一个音色（见 utils.RT_VOICE
                # 跟随 TTS_VOICE），带上 mode 等于同一个声音合两遍、缓存互不命中。
                voice_id=str(getattr(tts, "voice", "") or type(tts).__name__))
        except Exception as e:
            print(f"[backchannel] 拿不到 TTS，附和关闭：{type(e).__name__}: {e}", flush=True)
            _BC_VOICE["obj"] = False
            return None
    v = _BC_VOICE["obj"]
    if v is False:
        return None
    if _BC_VOICE["task"] is None and not v.primed:
        # 没人 await 这个 task，所以异常会被静默吞掉——ready 永远是 False，
        # 表现就是"一直说还没好"，而真正的原因（TTS 报错/没网/没 key）看不见。
        def _done(t):
            _BC_VOICE["task"] = None      # 失败了下次还能再试
            try:
                t.result()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[backchannel] 预合成失败：{type(e).__name__}: {e}", flush=True)
        _BC_VOICE["task"] = asyncio.create_task(v.prime(
            cache_only=type(v.tts).__name__ == "BreezeMLXTTS"))
        _BC_VOICE["task"].add_done_callback(_done)
    return v if v.ready else None


def _print_backchannel_status() -> None:
    """启动时把附和的状态一次说清楚。

    这东西不出声的原因有五六种（没开、音色对不上、还没合成、语言不是你以为的、
    概率没中），每一种的表现都是"没反应"。不在启动时讲明白，就只能靠猜。
    """
    from harness import backchannel as _bc
    if not _bc.emitting():
        print("[backchannel] 关闭。打开：VOICEMEM_BACKCHANNEL_EMIT=1", flush=True)
        return
    lang = space_language(ACTIVE_SPACE)
    words = "/".join(sorted({t for g in _bc._TOKENS[lang].values() for t in g})[:5])
    tts = None
    try:
        tts = _backchannel_tts()
    except Exception as e:
        print(f"[backchannel] 拿不到 TTS：{type(e).__name__}: {e}", flush=True)
    if tts is None:
        return                      # _backchannel_tts 自己已经说明了原因
    p = BackchannelPolicy_summary()
    print(f"[backchannel] 开启 · 空间「{ACTIVE_SPACE}」语言={lang} → 会说：{words} …",
          flush=True)
    print(f"[backchannel] 音色={getattr(tts, 'voice', '?')}（跟正文同一个）· {p}",
          flush=True)
    print("[backchannel] 想看每次判定：VOICEMEM_BC_DEBUG=1；"
          "先验通不通：VOICEMEM_BC_P0=0.9 VOICEMEM_BC_MAX_GAP=0.6 "
          "VOICEMEM_BC_REFRACTORY=0", flush=True)


def BackchannelPolicy_summary() -> str:
    from harness.backchannel import BackchannelPolicy
    p = BackchannelPolicy()
    return (f"基础概率={p.p0} 停顿窗口={p.gap_s*1000:.0f}~{p.max_gap_s*1000:.0f}ms "
            f"冷却={p.refractory_s}s")


_EOT = {"obj": None, "tried": False}


def _eot():
    """语义回合判定，进程级只建一次。关掉或建不起来时返回 None（退回纯掐表）。"""
    if not ARGS.eot:
        return None
    if not _EOT["tried"]:
        _EOT["tried"] = True
        try:
            from voicemem.utils.audio.eot import EndOfTurn, THRESHOLD
            _EOT["obj"] = EndOfTurn()
            print(f"[eot] 语义回合判定已启用（阈值 {THRESHOLD}，兜底 "
                  f"{CONFIRM_S*1000:.0f}ms）", flush=True)
        except Exception as e:
            print(f"[eot] 建不起来（{type(e).__name__}: {e}）→ 退回纯掐表 "
                  f"{CONFIRM_S*1000:.0f}ms。--no-eot 可关掉这条提示。", flush=True)
    return _EOT["obj"]


#: 提前起跑的阈值。比回合判定那个（0.6）高——赌错要白花一遍 LLM+TTS 的钱，
#: 所以只在模型很有把握时才赌。
#: 实测校准过：真实录音里句子**中间**稳定在 0.01~0.05（只有一个 0.28 的毛刺），
#: 而说完时是 0.42~0.73。所以 0.35 既够得着又不会在句中误触。原来的 0.75 是照
#: 模型文档定的，那个数在这批语音上一次都没达到过——等于提前生成从没生效。
EARLY_EOT = float(os.environ.get("VOICEMEM_EARLY_EOT", "0.5"))
#: 下注之后这段时间内说的话不算数（EOT 常在最后一个字的尾音里就触发）。
EARLY_GRACE_S = float(os.environ.get("VOICEMEM_EARLY_GRACE", "0.2"))
#: 附和播出去多久之内，转写里出现同样的词就当回声丢掉。
BC_ECHO_WINDOW_S = float(os.environ.get("VOICEMEM_BC_ECHO_WINDOW", "3.0"))
#: 助手说话时，要连续听到这么久的人声才暂停播放（挡自己的回声）。
#: 调大更不容易被回声误触，代价是真插话时暂停得晚一点。
CANDIDATE_MIN_SPEECH_S = float(os.environ.get("VOICEMEM_CANDIDATE_MIN_SPEECH", "0.12"))
#: 低于"当前响度的这个比例"就算词间空档。调大 → 更容易判成停顿，附和更密。
BC_QUIET_RATIO = float(os.environ.get("VOICEMEM_BC_QUIET_RATIO", "0.25"))
#: 下注（提前生成）之后过了这么久、你还在说、转写还在长，就重新允许附和。
#: 原来是下注即禁——而 EOT 开口不到两秒就常冲到 0.98 下注（'就'、'反正的一个是'
#: 这种半句），注一直挂到说完，整段话里附和窗口等于零。实测一场 20 条预合成
#: 一次没响。
BC_AFTER_EARLY_S = float(os.environ.get("VOICEMEM_BC_AFTER_EARLY", "1.0"))
#: 给回复模型看多少轮对话历史（滑窗，跟入没入库无关）。
HISTORY_TURNS = int(os.environ.get("VOICEMEM_HISTORY_TURNS", "6"))
#: 过了宽限期还说这么久，就判定「他还没说完」，那份提前生成作废。
#:
#: 放宽的代价要清楚：这段时间里说的话**没进那份生成**。所以 1.5s 意味着"你多说
#: 一句半，我仍然用之前那份回复"——赌的是那一句半没改变你的意思（补充、重复、
#: 语气词多半如此）。真改了意思就答非所问。想更保守就调小。
EARLY_MAX_SPEECH_S = float(os.environ.get("VOICEMEM_EARLY_MAX_SPEECH", "2.0"))
#: 下注时那句话，要覆盖最终文本的多大比例（按**内容重合**算，不是长度比）。
#:
#: EOT 判的是**音频**，而回复是拿**当时的流式文本**生成的——音频语义齐了，文本却
#: 常常还是残的。实测两种翻车，都得挡：
#:
#:   长了：下注 'o you have any recom' → 最终 'Do you have any recommendations
#:         of where should I study on weekends.' → 回复变成 "I didn't catch that."
#:   歪了：下注 'looking for a please' → 复核成 'Please.' → 回复答的是另一件事
#:
#: 所以不能用长度比（第二种的比值是 285%，照样通过），要看**最终文本有多少是
#: 下注时就已经说出来的**——用最长公共子串。
EARLY_MIN_COVER = float(os.environ.get("VOICEMEM_EARLY_MIN_COVER", "0.7"))
#: 手上已经有生成好的回复时，静音多久就结束回合。
#:
#: 比没有时短得多（300ms → 100ms）：等 300ms 的意义是"别白干活"——万一他还没说完，
#: 现在结束就得把 LLM+TTS 白跑一遍。可活都干完了，等就纯是浪费。
CONFIRM_READY_S = float(os.environ.get("VOICEMEM_CONFIRM_READY", "0.1"))


async def anticipate(sock, on_frame=None, on_speech=None, owner=None, is_busy=None,
                     said=None, on_candidate=None, on_candidate_reject=None,
                     on_playback_checkpoint=None, on_early=None,
                     on_speech_start=None, textless_confirm_s=None):
    """驱动核心流式会话，逐个 yield 确认回合的 Pending。
    on_frame(raw24k)：realtime 用它把原始音频平行喂给 OpenAI（方案 A）。
    on_speech()：本地 VAD 一听到人声就叫一次——realtime 拿它做打断（barge-in）。
    is_busy()：助手此刻是不是在说话。助手的声音会经麦克风回到 ASR，转出来的字
    照样会走 partial_transcript——用户就看见自己的输入框里冒出助手刚说的话。
    每轮保留开口时的播放状态和回声文本；非回声插话稳定后显示，纯附和只显示不回复。
    on_candidate()/on_candidate_reject()：疑似插话时可恢复地暂停/恢复播放；只有
    on_speech() 才是确认打断。"""
    stream = vm.stream(spec_min_chars=SPEC_MIN_CHARS, gamble_s=GAMBLE_S,
                       confirm_s=CONFIRM_S, eot=_eot(), textless_confirm_s=textless_confirm_s)
    utterance = UtteranceGuard()
    turn_finished = False
    last_partial = ""
    if owner is None:
        owner = {"id": "", "last": "", "miss": 0}   # 主人的声纹 / 上一轮是谁 / 连续认错几轮
    from harness.backchannel import Backchannel
    from harness import backchannel as _bc_mod
    bc = Backchannel()                    # 一路会话一个：它要记住上次什么时候附和过
    bc_speech_t0 = 0.0                    # 这一轮用户什么时候开的口
    prewarm_idle = True                   # 该趁空闲把「人设+历史」热一遍了
    prewarm_mem = None                    # 已经拿去续热过的那份投机记忆
    last_speak_t = 0.0                    # VAD 最后一次判到人声（量端到端延迟用）
    bc_rms_fast = 0.0                     # 最近一帧的能量（韵律里的"尾音"）
    bc_rms_slow = 0.0                     # 慢 EMA（韵律里的"之前的响度"）
    bc_gap = 0.0                          # 句中微停顿已经持续多久
    # 句中微停顿**只能用能量判**，VAD 不行。
    #
    # 试过给附和单独建一个 min_silence=0.08 的 VAD，以为参数调小就能看到词间空档。
    # 实测不行：同一段 7 秒的话，短参数 VAD 只报 3 处停顿（还都在头尾），能量法报
    # 7~8 处、均匀分布在句子中间（0.5/2.0/2.9/3.7/4.8/5.7 秒）。
    # 原因是 VAD 的任务本来就是"把一句话完整切出来"，词间空档是它主动桥接掉的，
    # 不是参数没调对。
    #
    # 阈值取"这段话当前响度的 25%"，不用绝对值——每个人音量不同、房间也不同。
    bc_recent: list = []                  # 最近播的附和词 [(时刻, 词)]，用来挡它自己的回声
    early_text = ""                       # 上次提前起跑时赌的那句话
    speak_run = 0.0                       # 连续听到人声多久（挡回声误触暂停）
    eot_peak = 0.0                        # 本轮 EOT 最高分（debug 用）
    early_at = 0.0                        # 什么时候下的注
    early_speech = 0.0                    # （保留字段，现在不参与判定）
    if _bc_mod.emitting():
        # 接通就起预合成。等第一次触发才合成的话，那几十秒里每次触发都拿不到音频、
        # 安静跳过——表现就是"开了但完全不响"，而且没有任何提示。
        _backchannel_voice()
    else:
        print("[backchannel] 未开启（VOICEMEM_BACKCHANNEL_EMIT=1 打开）", flush=True)

    def _reset_early():
        """**每条交出/丢弃回合的路径都要调**，否则 early_at 一直非空，
        `not early_at` 那个条件再也不成立——整场都不会下第二次注。
        实测日志：一轮被判回声丢弃之后，后面每轮都是"最高 0.99，没下注"。

        ``bc_speech_t0`` 同理，而且坏得更隐蔽：它是"这一轮什么时候开的口"，
        不归零就等于整场只开过一次口——本地模型只在第一轮预热（后面每轮都
        `没命中`，首字回到两秒），附和用的 speech_s 也会一路涨到几百秒。"""
        nonlocal early_at, early_speech, early_text, eot_peak, bc_speech_t0
        nonlocal prewarm_idle, prewarm_mem
        early_at, early_speech, early_text = 0.0, 0.0, ""
        eot_peak = 0.0
        bc_speech_t0 = 0.0
        prewarm_idle = True
        prewarm_mem = None

    def live_agent_text():
        """助手这一路已经出过声的文本：正文 + 附和词。

        ``said`` 是**函数**不是 dict（见调用方 said=lambda: ...），而附和词并不在
        它里面——那声"嗯"会经扬声器绕回麦克风被 ASR 转出来，不一起比对就会混进
        用户这一轮的文本里。
        """
        # 附和词只取**最近几秒**的（bc_recent 自带时间窗）。
        # 原来用的是整场累加的 bc_said——它越攒越长，而 _is_echo 只看最后
        # ECHO_WINDOW(300) 个字符，于是附和词把助手真正说的话整个挤出去了。
        # 后果是用户说的话越来越容易被判成"回声"，打断确认永远不成立：
        # 表现就是"我说话它还在说"。
        now = time.monotonic()
        recent = "".join(w for t, w in bc_recent if now - t < BC_ECHO_WINDOW_S)
        return (said() if said else "") + recent

    def heard_from_agent():
        return utterance.reference or live_agent_text()
    barge_base = 0                        # 上次触发打断时的转写长度
    barged = False                        # 这一轮是否已确认「人在插话」
    candidate = False                     # 已暂停播放、正在等更多证据
    candidate_updates = 0
    candidate_text = ""
    candidate_silence = 0.0
    candidate_age = 0.0
    discard_candidate_turn = False
    while True:
        msg = await sock.receive()
        if msg.get("type") == "websocket.disconnect":         # 关页面/刷新：收工
            return
        if msg.get("text"):                                   # 打字轮
            data = json.loads(msg["text"])
            if data.get("type") == "playback_checkpoint":
                if on_playback_checkpoint:
                    await on_playback_checkpoint(data)
                continue
            if data.get("type") == "user_text" and data.get("text", "").strip():
                stream.emotion = owner.get("emotion") or None
                turn = await stream.feed_text(data["text"])
                yield Pending(turn.text, turn.memory_context, turn.result, spoken=False,
                              replay=_replay_id(turn.text, turn.result),
                              emotion=owner.get("emotion", ""),
                              route=turn.route)
            continue
        if msg.get("bytes") is None:
            continue
        raw = msg["bytes"]
        if turn_finished:
            utterance = UtteranceGuard()
            turn_finished = False
            barge_base = 0
            barged = candidate = discard_candidate_turn = False
            candidate_updates = 0
            candidate_text = ""
            candidate_silence = candidate_age = 0.0
        # Snapshot before feed: final ASR awaits may outlive assistant playback.
        busy_at_capture = bool(is_busy and is_busy())
        reference_at_capture = live_agent_text()
        if on_frame:
            await on_frame(raw)                               # 方案 A：音频也进 OpenAI 缓冲
        stream.emotion = owner.get("emotion") or None         # 上一轮算出来的情绪
        st = await stream.feed(raw)                           # 核心：ASR + VAD + 投机预取
        turn_finished = bool(st.turn)
        utterance.observe(active=st.spoke or bool(st.turn) or st.state == "<speak>",
                          busy=busy_at_capture, reference=reference_at_capture)
        utterance.observe(active=False, busy=False, reference=live_agent_text())
        if getattr(st, "speech_end", 0):
            last_speak_t = st.speech_end
        cur = st.text.strip()
        busy = bool(is_busy and is_busy())
        input_echo = bool(heard_from_agent() and _is_echo(cur, heard_from_agent()))
        if st.state == "<speak>" and not getattr(st, "speech_end", 0):
            last_speak_t = time.monotonic()

        # 空闲时先把模型热上。**时机是这里，不是等他开口**：人设和历史在上一轮
        # 回复说完那一刻就定了，而预热要跑一秒多——挂在"开口"上，一句"who am i"
        # 说完预热还没跑完，等于白热。实测挂在开口上时每轮首字都是全量 prefill
        # 的 2100ms，一次都没命中。
        #
        # 条件是「不在说话 + 助手也没在说」：助手正播着的时候抢 GPU，会把正在
        # 合成的下一段拖慢。
        if (prewarm_idle and on_speech_start and not busy
                and st.state != "<speak>"):
            if on_speech_start():
                prewarm_idle = False
        # 说话中：投机检索把这一轮的记忆取回来了就立刻续进 KV。**只有真排上了
        # 才记下这一份**——开口那次预热还在跑时请求会被挡掉，记下来就等于"热过
        # 了"，再也不重试，尾巴里那一百多个 token 就一轮都省不掉。
        elif (on_speech_start and st.state == "<speak>"
              and st.memory is not None and st.memory is not prewarm_mem):
            if on_speech_start(st.memory, st.text):
                prewarm_mem = st.memory

        # 提前起跑：EOT 说这句已经齐了（哪怕人还在说），就先把回复生成出来存着。
        # 生成和合成加起来 1.4 秒，只有藏在用户还在说话的这段时间里，说完才可能
        # 立刻出声。赌错了取消重来——花的是钱，不是用户的等待。
        if st.eot_score > eot_peak:
            eot_peak = st.eot_score
        # **一轮只赌一次**：第一次 EOT 过线就下注，之后无论你还说多久都不再管，
        # 说完直接用那一份。
        #
        # 代价明说：那份回复是针对**下注那一刻为止**的话生成的。你后面补的内容
        # 它没听见——"我喜欢安静的地方"下注了，你接着说"但学校图书馆很吵"，
        # 回答就只针对前半句。换来的是延迟最低、逻辑最简单，赌错也不重来。
        if (on_early and not busy and not input_echo and st.spoke and cur
                and not (utterance.started_busy and _is_backchannel(cur))
                and st.eot_score >= EARLY_EOT and not early_at):
            early_text, early_at = cur, time.monotonic()
            await on_early(cur, st)
            # 回复已经在生成了 → 静音 100ms 就够，不必再等满 300ms。
            # 等 300ms 的意义是"别白干活"，活干完了就没意义了。
            stream.confirm_s = CONFIRM_READY_S

        # 附和用**能量**判停顿，不用 VAD 的静音计时。
        #
        # silero 为了不把一句话切碎，词间的小停顿一律当成还在说话——所以 st.silence
        # 只在你真的说完时才开始涨，附和永远落在"整句说完之后"。而人的附和是插在
        # 句子中间的。能量是逐帧真实的：说话时高，词间空档立刻掉下去。
        #
        # 阈值取"当前这段话响度的 25%"而不是绝对值：每个人音量不同，房间也不同。
        import numpy as _np
        from voicemem.utils.audio.stream_io import resample as _resample
        frame = _np.frombuffer(raw, _np.int16).astype(_np.float32) / 32768.0
        rms = float(_np.sqrt(_np.mean(frame * frame))) if len(frame) else 0.0
        frame_s = len(frame) / MIC_RATE
        if st.state == "<speak>":
            bc_rms_fast = rms
            bc_rms_slow = (0.9 * bc_rms_slow + 0.1 * rms) if bc_rms_slow else rms
            if not bc_speech_t0:
                bc_speech_t0 = time.monotonic()
        quiet = rms < max(0.008, BC_QUIET_RATIO * bc_rms_slow)
        bc_gap = bc_gap + frame_s if quiet else 0.0
        # 助手正在说话时不附和：那不是"我在听"，那是抢话。
        # 已经下过注（EOT 判这句齐了、回复正在生成）时也不附和——那声"嗯"会正好
        # 落在回复出声前一秒，听着像自言自语。附和是"我在听你说"，人都说完了就没
        # 意义了。
        # 刚下注不附和（那声"嗯"会正好落在回复出声前一秒，像自言自语）；但下注后
        # 又说了一秒还没停、字也多了，就是还在说——该应一声。
        bc_ok = (not early_at
                 or (time.monotonic() - early_at >= BC_AFTER_EARLY_S
                     and cur != early_text and st.state == "<speak>"))
        if not busy and not input_echo and not utterance.started_busy and bc_speech_t0 and bc_ok:
            voice = _backchannel_voice() if _bc_mod.emitting() else None
            token = bc.offer(text=cur, silence=bc_gap, spoke=st.spoke,
                             speech_s=time.monotonic() - bc_speech_t0,
                             emotion=owner.get("emotion", ""),
                             tail_rms=bc_rms_fast, prev_rms=bc_rms_slow,
                             lang=space_language(ACTIVE_SPACE),
                             available=voice.available if voice else set())
            if token:
                voice = _backchannel_voice()
                pcm = voice.get(token, bc.rng) if voice else None
                if pcm:
                    # 走**独立**的一路音频，不进回复那个播放器：进了的话 hearing()
                    # 会以为助手在说话，用户接着说就被判成插话——自己的"嗯"把自己掐了。
                    await sock.send_json({
                        "type": "backchannel", "token": token,
                        "sample_rate": 24000,
                        "pcm": base64.b64encode(pcm).decode()})
                    bc_recent.append((time.monotonic(), token))  # 见 heard_from_agent()
                    if BARGE_DEBUG:
                        print(f"[backchannel] {token!r}", flush=True)
                elif BARGE_DEBUG:
                    v = _BC_VOICE["obj"]
                    why = ("音色对不上，这条路不附和" if v is False
                           else "预合成还没好" if v is not None else "拿不到 TTS")
                    print(f"[backchannel] 判到该说 {token!r}，但{why} → 跳过", flush=True)
        frame_s = len(raw) / 2 / MIC_RATE

        # VAD 先触发可恢复暂停，后续 ASR 文本用于确认是否真正打断。
        #
        # **要连续听到人声才暂停**，一帧就停会被回声误触：助手自己的声音绕回麦克风
        # 也是真人声，VAD 当然判是。表现就是"它自己说着说着突然变 listening，
        # 我根本没开口，过一会又接着说"——那一停一起是听得见的。
        # AEC 之后的回声是断续的短促片段，真人插话是连着的，用持续时间就能分开。
        speak_run = speak_run + frame_s if st.state == "<speak>" else 0.0
        # "有人声" 是必要条件，别只看 speak_run——它在静音帧上是 0，而阈值也可能
        # 是 0，`0 >= 0` 成立，等于每一帧都触发暂停。
        if (busy and st.state == "<speak>" and speak_run >= CANDIDATE_MIN_SPEECH_S
                and not candidate and not barged
                and cur and not _is_echo(cur, heard_from_agent())
                and _has_barge_content(cur)):
            candidate = True
            discard_candidate_turn = False
            candidate_updates = 0
            candidate_text = ""
            candidate_silence = 0.0
            candidate_age = 0.0
            if BARGE_DEBUG:
                print("[barge] 疑似插话 → 暂停播放，等待 ASR 确认", flush=True)
            if on_candidate:
                await on_candidate()

        if candidate and not barged:
            candidate_age += frame_s
            candidate_silence = (candidate_silence + frame_s
                                 if st.state == "<silence>" else 0.0)
            looks_echo = bool(said is not None and cur and _is_echo(cur, heard_from_agent()))
            if looks_echo:
                candidate_updates = 0
                candidate_text = ""
            if cur and not looks_echo and not _is_backchannel(cur) and _has_barge_content(cur):
                normalized = _barge_text(cur)
                if normalized != candidate_text:
                    candidate_text = normalized
                    candidate_updates += 1

            confirmed = (_is_explicit_interrupt(cur) and not looks_echo)
            confirmed = confirmed or (not looks_echo and candidate_updates >= BARGE_STABLE_UPDATES)
            if confirmed:
                candidate = False
                barged = True
                barge_base = len(cur)
                if BARGE_DEBUG:
                    why = "明确停止指令" if _is_explicit_interrupt(cur) else "转写连续稳定增长"
                    print(f"[barge] {why} → 确认打断：{cur[-16:]!r}", flush=True)
                if on_speech:
                    await on_speech()
            elif (looks_echo or (candidate_silence * 1000 >= BARGE_REJECT_SILENCE_MS and not cur)
                  or (candidate_age * 1000 >= BARGE_CANDIDATE_TIMEOUT_MS
                      and candidate_updates == 0)):
                candidate = False
                discard_candidate_turn = True
                if BARGE_DEBUG:
                    print("[barge] 回声/无有效插话 → 恢复播放", flush=True)
                if on_candidate_reject:
                    await on_candidate_reject()
        # Keep filtering after playback ends AND after a confirmed interrupt:
        # both can happen before delayed ASR publishes the assistant's audio.
        echo = input_echo
        show_partial = utterance.allow_partial(
            cur, echo=echo, final=bool(st.turn), backchannel=_is_backchannel(cur),
            explicit=_is_explicit_interrupt(cur), confirmed=barged)
        if echo and last_partial:
            last_partial = ""
            await sock.send_json({"type": "partial_transcript", "text": "", "replace": True})
        if st.text.strip() and st.text != last_partial and show_partial:
            last_partial = st.text
            await sock.send_json({"type": "partial_transcript", "text": st.text, "replace": True,
                                  "non_interrupting": utterance.started_busy and _is_backchannel(cur)})
        if st.turn:                                           # VAD 确认说完 → 记忆早已预取好
            # A late ASR result may arrive after playback drains or a candidate
            # is rejected. Still discard the actual emitted phrase.
            if (heard_from_agent() and _is_echo(st.turn.text, heard_from_agent())):
                if BARGE_DEBUG:
                    print(f"[echo] 丢弃助手回声：{st.turn.text!r}", flush=True)
                if candidate and on_candidate_reject:
                    await on_candidate_reject()
                candidate = False
                candidate_updates = 0
                candidate_text = last_partial = ""
                _reset_early()
                continue
            if utterance.started_busy and _is_backchannel(st.turn.text):
                # Display acknowledgement, but do not create a reply, ingest it,
                # change speaker identity, or wait for hearing() to still be true.
                if candidate and on_candidate_reject:
                    await on_candidate_reject()
                await sock.send_json({"type": "user_backchannel", "text": st.turn.text})
                if BARGE_DEBUG:
                    print(f"[barge] 开口时助手在说话，附和只显示不回复：{st.turn.text!r}", flush=True)
                candidate = barged = discard_candidate_turn = False
                candidate_updates = 0
                candidate_text = last_partial = ""
                _reset_early()
                continue
            # 候选被否决过（那时 ASR 还没转出字），但**说完之后**离线复核出了一整句
            # ——那就是真人说的话，不能扔。
            #
            # 原来无条件扔，于是出现：流式在 1.2 秒里只憋出 'Place for me'，判成
            # "没形成文字"、恢复播放并标记作废；复核出
            # 'Yeah, I want you to recommend a place for me to study.' 时作废标记
            # 还在，整轮丢掉——表现就是"我说了一整句，它当没听见"。
            if discard_candidate_turn:
                discard_candidate_turn = False
                last_partial = ""
                barge_base = 0
                _t = (st.turn.text or "").strip()
                if _has_barge_content(_t) and not _is_echo(_t, heard_from_agent()):
                    if BARGE_DEBUG:
                        print(f"[barge] 候选虽被否决，但复核出完整内容 → 照常成一轮："
                              f"{_t!r}", flush=True)
                else:
                    if BARGE_DEBUG:
                        print(f"[barge] 丢弃未确认的声音回合：{st.turn.text!r}", flush=True)
                    _reset_early()
                    continue

            if candidate and not barged:
                final_text = st.turn.text.strip()
                looks_echo = bool(said is not None and final_text
                                  and _is_echo(final_text, heard_from_agent()))
                confirmed = (
                    not looks_echo
                    and not _is_backchannel(final_text)
                    and (_is_explicit_interrupt(final_text)
                         or candidate_updates >= BARGE_STABLE_UPDATES
                         or (_has_barge_content(final_text)
                             and _has_strong_final_barge(final_text)))
                )
                candidate = False
                if confirmed:
                    barged = True
                    if BARGE_DEBUG:
                        print(f"[barge] 完整回合确认插话：{final_text!r}", flush=True)
                    if on_speech:
                        await on_speech()
                else:
                    if BARGE_DEBUG:
                        print(f"[barge] 完整回合判为附和/回声/噪声 → 恢复：{final_text!r}",
                              flush=True)
                    if on_candidate_reject:
                        await on_candidate_reject()
                    last_partial = ""
                    barge_base = 0
                    candidate_updates = 0
                    candidate_text = ""
                    candidate_silence = 0.0
                    candidate_age = 0.0
                    continue

            # 我们正在放录音，而这一轮一个字都没转出来：那是自己的声音绕回来了。
            if _replaying_now() and not (st.turn.text or "").strip():
                if BARGE_DEBUG:
                    print("[replay] 回放期间的空白一轮，是自己的回声，丢掉", flush=True)
                last_partial = ""
                barge_base = 0
                barged = False
                candidate = False
                continue
            last_partial = ""
            barge_base = 0                                    # 新一轮，转写从头开始涨
            barged = False
            candidate = False
            candidate_updates = 0
            candidate_text = ""
            candidate_silence = 0.0
            candidate_age = 0.0
            # 这一轮是不是助手自己那声附和绕回来的。
            #
            # 附和走的是**独立音频路**（不进回复播放器，否则 hearing() 会把用户接
            # 下来的话判成插话），代价就是 busy 一直是 false——而整套回声防护是挂在
            # busy 上的，于是附和的回声一路畅通：被转写、成一轮、助手回复自己的
            # "mm-hmm"。这里单独挡一次。
            if BARGE_DEBUG:
                if _eot() is None:
                    print("[early] EOT 没启用 → 提前生成整个不工作。"
                          "多半是缺 onnxruntime：pip install onnxruntime", flush=True)
                else:
                    print(f"[early] 本轮 EOT 最高 {eot_peak:.2f}（下注线 {EARLY_EOT}）"
                          f"{'，下过注' if early_at else '，没下注'}", flush=True)
            eot_peak = 0.0
            _now = time.monotonic()
            bc_recent[:] = [(t, w) for t, w in bc_recent if _now - t < BC_ECHO_WINDOW_S]
            _bc_echo = bool(bc_recent) and _is_echo(
                st.turn.text, "".join(w for _, w in bc_recent))
            if _bc_echo:
                if BARGE_DEBUG:
                    print(f"[backchannel] 这一轮是自己那声附和的回声 → 丢弃："
                          f"{st.turn.text!r}", flush=True)
                _reset_early()
                continue
            _norm = lambda x: "".join(c for c in (x or "") if c.isalnum()).casefold()
            # 跟**同一个模型**的输出比：下注时是流式文本，这里也用流式文本
            # （st.turn.raw_text，复核前那份）。拿它比复核后的，比出来的是两个
            # ASR 的用词分歧，不是"他又说了话"——实测同一句话只能对上 57%。
            _f, _b = _norm(st.turn.raw_text or st.turn.text), _norm(early_text)
            _cover = (_lcs_len(_b, _f) / len(_f)) if _f else 0.0
            _early_ok = bool(early_at) and _cover >= EARLY_MIN_COVER
            if BARGE_DEBUG and early_at and not _early_ok:
                print(f"[early] 下注那句只覆盖了最终文本的 {_cover:.0%}"
                      f"（{early_text!r} vs {st.turn.text!r}）→ 作废重来", flush=True)
            early_at, early_speech, early_text = 0.0, 0.0, ""
            # 开口时刻也要归零，而且**这条成功路径最容易漏**——它不走
            # _reset_early()，自己就地清 early_at。漏了的后果不是"少一次下注"，
            # 是本地模型只在整场第一轮预热：prewarm_local 挂在"这一轮第一次开口"
            # 上，bc_speech_t0 非空就再也触发不了，后面每轮首字从 1.4s 回到 2.5s。
            bc_speech_t0 = 0.0
            prewarm_idle = True                 # 历史变了，空闲时重新热一遍
            prewarm_mem = None
            stream.confirm_s = CONFIRM_S        # 新一轮，回到正常兜底
            # 谁在说话。第一个开口的人算这场对话的主人；之后换了另一个声纹，
            # 就是陌生人——不能把主人的记忆讲给他听（"我是谁？"→"你是Jiaqi"
            # 这个 bug 就是因为检索从不看说话人）。
            # 用上一轮算出来的说话人做判断，这一轮的放后台算。
            # 直接读 st.speaker_id 会同步跑声纹+情绪+场景一整套模型——实测 2.1 秒，
            # 而且是在事件循环里，这期间连 socket 都不读，麦克风帧全堆着，
            # 表现就是"ASR 很卡"。代价是换人之后第一句仍按上一个人算。
            stranger = bool(SPEAKER_GATE and owner["id"]
                            and owner.get("miss", 0) >= STRANGER_MIN_TURNS)
            if stranger or SPEAKER_DEBUG:
                # "他怎么突然不认识我了"——看这一行。声纹把同一个人认成两个
                # person_* 时就会这样：记忆被清空，指令换成"就当第一次见面"。
                print(f"[speaker] owner={owner['id'] or '-'} last={owner['last'] or '-'}"
                      f" miss={owner.get('miss', 0)} stranger={stranger}", flush=True)
            yield Pending(st.turn.text,
                          "" if stranger else st.turn.memory_context,
                          st.turn.result, spoken=True,
                          audio_path=await asyncio.to_thread(
                              save_turn_audio, getattr(st, "_pcm", None)),
                          stranger=stranger,
                          replay="" if stranger else _replay_id(st.turn.text, st.turn.result),
                          emotion=owner.get("emotion", ""),
                          route=st.turn.route,
                          early_ok=_early_ok,
                          speech_end=last_speak_t)



# ══════════════════ 每种 mode 的会话循环 ══════════════════

async def _session_anticipate(session_id: str, sock, on_close=None, **kwargs):
    try:
        async for pending in anticipate(sock, **kwargs):
            yield pending
    finally:
        if on_close:
            await on_close()
        _SESSION_CONTEXT.clear_session(session_id)


async def llm_tts_session(sock):
    """llm_tts 这条路的打断。

    原来是 `async for pending in anticipate(sock): await voicemem_llm_tts(...)`——
    两个问题叠在一起，打断在结构上就不可能：
      · on_speech 没传进 anticipate，本地 VAD 听到人声也没人管；
      · voicemem_llm_tts 最后 `await speaker`，要等全部音频发完才返回。async for
        在这期间不会去拉下一个，anticipate 就停在那儿不读 socket 了，麦克风帧全
        堆在缓冲区里。听感就是"说什么都没用，他非要说完"。

    现在回复丢进后台任务，读 socket 的循环一刻不停；听到人声就取消那个任务。
    """
    # until：前端预计几点才把已发出去的音频播完（见 hearing()）。
    turn = {"task": None, "t0": 0.0, "until": 0.0, "echo_until": 0.0,
            "speech_end": 0.0, "play_started": False,
            "reply": {"text": ""}, "timeline": None}
    owner = {"id": "", "last": "", "miss": 0}
    speech_rate = SpeechRateEstimator()
    context_session = uuid.uuid4().hex
    candidate_paused = False
    candidate_paused_at = 0.0

    async def pause_candidate():
        nonlocal candidate_paused, candidate_paused_at
        if hearing() and not candidate_paused:
            candidate_paused = True
            candidate_paused_at = time.monotonic()
            await sock.send_json({"type": "answer_pause"})

    async def resume_candidate():
        nonlocal candidate_paused, candidate_paused_at
        if candidate_paused:
            candidate_paused = False
            if turn["until"]:
                turn["until"] += max(0.0, time.monotonic() - candidate_paused_at)
            candidate_paused_at = 0.0
            await sock.send_json({"type": "answer_resume"})

    def hearing() -> bool:
        """用户此刻还听不听得见助手。

        不能用 `task.done()` 代替：send_audio 是**不限速**的，TTS 出多快就往
        socket 里推多快，一段十几秒的回复两三秒就推完了。任务早就 done、前端
        那边还在播剩下的十几秒——这期间插话，原来的 stop_reply 直接 return，
        前端从没收到 answer_interrupt，表现正是"打断没反应，它非要念完"。
        所以按**已发出去的音频时长**算，跟 realtime 那条路一致：24k PCM16，
        一个样本 2 字节。
        """
        t = turn["task"]
        timeline = turn["timeline"]
        task_active = (t is not None and not t.done()
                       and not (timeline and timeline.playback_done))
        return (candidate_paused or task_active
                or time.monotonic() < turn["until"])

    async def playback_checkpoint(data):
        timeline = turn["timeline"]
        if timeline is None or data.get("output_id") != timeline.output_id:
            return
        timeline.update_checkpoint(
            data.get("rendered_samples", 0), data.get("sample_rate", MIC_RATE),
            data.get("state", "playing"))
        if data.get("event") == "started" and not turn["play_started"]:
            turn["play_started"] = True
            if turn["speech_end"]:
                elapsed = (time.monotonic() - turn["speech_end"]) * 1000
                print(f"[lat] 闭嘴→浏览器开始播放（含回报传输） {elapsed:.0f}ms", flush=True)
        # 前端缓冲见底会报 stalled（pcm-player-worklet 的 underflow），以前只记进
        # playback_state、从不打印，所以"偶尔卡一下"在日志里一条痕迹都没有。
        # 打出已播到第几毫秒和 GPU 线程上排着几个活儿——卡的时候谁在抢，一眼看出。
        if data.get("state") == "stalled" and BARGE_DEBUG:
            from voicemem.utils.gpu_loop import gpu_loop as _gl
            print(f"[play] 缓冲见底：已播 {timeline.rendered_ms():.0f}ms"
                  f"，GPU 线程活跃任务 {getattr(_gl(), 'active_count', '?')}", flush=True)
        if timeline.playback_done:
            turn["until"] = 0.0

    async def send_audio(pcm: bytes):
        """发音频，顺带记账。前端是排队播的（pcm-player-worklet），这里跟着算
        同一条时间线：上一块播完之后再接这一块。"""
        turn["until"] = max(turn["until"], time.monotonic()) + len(pcm) / 2 / MIC_RATE
        turn["echo_until"] = turn["until"] + 2.0
        if not turn["t0"]:
            # 宽限期从**助手真的出声**那一刻起算，不是从任务创建。TTS 首帧要
            # ~1.2s，按任务创建算的话宽限期在它开口前就过完了，等于没有。
            turn["t0"] = time.monotonic()
        await sock.send_bytes(pcm)

    async def stop_reply(force: bool = False):
        nonlocal candidate_paused, candidate_paused_at
        if not hearing():
            turn["task"] = None
            return
        since = (time.monotonic() - turn["t0"]) * 1000 if turn["t0"] else 0.0
        if not force and turn["t0"] and since < BARGE_GRACE_MS:
            if BARGE_DEBUG:
                print(f"[barge] 才说了 {since:.0f}ms，还在宽限期内，不打断", flush=True)
            return
        if BARGE_DEBUG:
            left = max(0.0, turn["until"] - time.monotonic()) * 1000
            print(f"[barge] ★ 打断：转写触发（前端还剩 {left:.0f}ms 没播完）", flush=True)
        task = turn["task"]
        timeline = turn["timeline"]
        heard_text = timeline.heard_text() if timeline else ""
        output_id = timeline.output_id if timeline else ""
        if timeline:
            timeline.mark_interrupted()
        if task is not None and not task.done():
            task.cancel()
        turn["task"], turn["until"] = None, 0.0
        candidate_paused = False
        candidate_paused_at = 0.0
        try:
            await sock.send_json({"type": "answer_interrupt",
                                  "output_id": output_id,
                                  "heard_text": heard_text})
        except Exception:
            pass
        if task is not None and not task.done():
            try:
                await task
            except asyncio.CancelledError:
                pass

    # 提前起跑的那一份：{"text","task","sink","timeline","pending"}。
    early = {"text": "", "task": None, "sink": None, "timeline": None,
             "pending": None, "said": None}

    async def drop_early(why: str = "") -> None:
        """赌错了：把提前生成的整个丢掉。它一个字节都没发出去，用户无感。"""
        t = early["task"]
        early.update(text="", task=None, sink=None, timeline=None, pending=None,
                     said=None)
        if t is not None and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        if why and BARGE_DEBUG:
            print(f"[early] 丢弃提前生成：{why}", flush=True)

    async def start_early(text, st):
        """EOT 说这句齐了（人可能还在说）→ 先把回复生成出来存进 sink。"""
        stop_prewarm()
        await drop_early("换了新的赌注")
        stop_prewarm()
        # 记忆可能没有——闸门判成浅内容时压根没检索（st.memory 是 None）。
        # 那不该妨碍提前生成：浅内容本来就不注入记忆，用空结果照样能生成。
        # 原来这里 return 掉了，等于浅问题永远享受不到提前生成。
        from voicemem.stream import empty_result
        result = st.memory if st.memory is not None else empty_result()
        pending = Pending(text, build_memory_context(result), result, spoken=True,
                          emotion=owner.get("emotion", ""), route=st.route)
        sink = ReplySink(sock.send_json, send_audio)
        timeline = AudioTimeline(prebuffer_seconds=0.16, rate_estimator=speech_rate)
        # 这一份要**留着**：提交之后它就是 turn["reply"]，防回声那套读的就是它。
        # 原来这里传的是个用完就扔的 {"text": ""}，于是助手自己说的话没进回声比对——
        # 它说 "Are you looking for a place to"，麦克风听回去转成 "looking for a
        # please"，被当成用户新说的一轮，还照着它下了注。
        said_state = {"text": ""}
        early.update(text=text, sink=sink, timeline=timeline, pending=pending,
                     said=said_state)
        if BARGE_DEBUG:
            print(f"[early] 提前起跑（EOT {st.eot_score:.2f}）：{text[-20:]!r}", flush=True)

        async def run():
            try:
                await voicemem_llm_tts(pending, sink.send, sink.send_audio, owner,
                                       timeline, said=said_state,
                                       context_session=context_session,
                                       context_space=ACTIVE_SPACE, memory_vm=vm)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[early] 提前生成失败：{type(e).__name__}: {e}", flush=True)
        early["task"] = asyncio.create_task(run())

    prewarm = {"task": None, "cancelled": None, "closed": False}

    def stop_prewarm():
        cancelled = prewarm["cancelled"]
        if cancelled is not None:
            cancelled.set()
        task = prewarm["task"]
        if task is not None and not task.done():
            task.cancel()

    def prewarm_local(result=None, text=""):
        """把「人设 + 历史 + （到货了的话）这一轮记忆」算进 KV 母本。

        空闲时调（见 _session_anticipate 的 prewarm_idle）：人设和历史在上一轮
        回复说完那一刻就定了，这时候算完，等他开口时早就是热的。人设那一截在
        开服时就已经进母本了，这里通常只是把新增的那一轮历史续上，几十毫秒。
        没开 --llm local 时什么都不做。
        """
        if _LOCAL_LLM is None or prewarm["closed"]:
            return False
        if prewarm["task"] and not prewarm["task"].done():
            return False
        # 回复还在生成/播放时绝对不能热：两边抢同一把锁和同一块 GPU，预热一旦
        # 先拿到锁，正在等第一个字的那次生成就要干等。
        # 提前生成还没接管到 turn；甚至 task 已完成、音频仍在 sink 等确认。
        # 这两种都不能继续热。hearing 也覆盖任务结束但前端还没播完的情况。
        if early["task"] is not None or hearing():
            return False
        hist = _SESSION_CONTEXT.messages(context_session, ACTIVE_SPACE,
                                         window=HISTORY_TURNS)
        # 记忆是投机检索**边说边取**回来的（_kick 在 silence==0 时就起），到货了
        # 就连整段 ctx 一起热掉——它是尾巴里最大的一块（实测 192 token 的尾巴里
        # 180 个是 5 条记忆）。说完时没算的就只剩他那句话本身。
        # route 一定是 DEEP：投机检索只对需要记忆的轮次起跑，有 result 就说明
        # 闸门放行了。ctx 的拼法必须跟回复那边**逐字一致**，所以共用同一个函数。
        ctx = ("" if result is None else
               build_reply_context(build_memory_context(result),
                                   route=gate.DEEP,
                                   replay=_replay_id(text, result), text=text))
        cancelled = threading.Event()
        prewarm["cancelled"] = cancelled
        model = _LOCAL_LLM

        async def run_prewarm():
            try:
                await asyncio.to_thread(model.prewarm, hist, ctx, cancelled=cancelled)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            except Exception as e:
                print(f"[llm] 后台预热失败：{type(e).__name__}: {e}", flush=True)

        prewarm["task"] = asyncio.create_task(run_prewarm())
        return True

    async def close_session():
        prewarm["closed"] = True
        stop_prewarm()
        await drop_early()
        task = turn["task"]
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[web] 回复收尾失败：{type(e).__name__}: {e}", flush=True)

    async for pending in _session_anticipate(
            context_session, sock, on_speech=stop_reply, owner=owner,
            is_busy=hearing, said=lambda: (turn["reply"]["text"]
                if hearing() or time.monotonic() < turn["echo_until"] else ""),
            on_candidate=pause_candidate, on_candidate_reject=resume_candidate,
            on_playback_checkpoint=playback_checkpoint, on_close=close_session,
            on_early=start_early, on_speech_start=prewarm_local, textless_confirm_s=0.2):
        # 整轮都是附和、而助手还在说：当没听见。
        #
        # _is_backchannel 原来只挡在"说到一半"那条路上（anticipate 里），可 VAD
        # 判完一整轮走的是**另一条**——下面这句 stop_reply(force=True)。所以只
        # 应一声"嗯"，VAD 认为你说完了一轮，新回合就以"顶掉旧回合"的名义把正在
        # 播的回复掐了，附和词表根本没被问到。实测日志里就是这样掐的。
        # 助手没在说话时的"嗯"照旧当正常一轮走。
        if _is_backchannel(pending.text) and hearing():
            if BARGE_DEBUG:
                print(f"[barge] 整轮都是附和 {pending.text!r}，不算一轮，继续说", flush=True)
            continue
        stop_prewarm()
        # 文本命中就接管提前生成；可能还没合成出音频，不能把命中等同于立即出声。
        if early["task"] is not None and pending.early_ok:
            await stop_reply(force=True)
            turn["t0"] = turn["until"] = 0.0
            turn["speech_end"], turn["play_started"] = pending.speech_end, False
            # 接管提前生成那份的 said——它记着助手已经说出口的话，防回声全靠它。
            # 原来这里给了个新的空 dict，等于把回声比对的依据清空了：助手说
            # "Are you looking for a place to"，麦克风听回去转成 "looking for a
            # please"，就被当成用户新说的一轮。
            turn["reply"] = early["said"] or {"text": ""}
            turn["timeline"] = early["timeline"]
            turn["task"] = early["task"]
            sink, ms = early["sink"], early["sink"].buffered_ms
            early.update(text="", task=None, sink=None, timeline=None, pending=None,
                     said=None)
            await sink.commit()
            if BARGE_DEBUG:
                state = "释放现成音频" if ms > 0 else "尚无音频，继续等待生成"
                print(f"[early] ★ 文本命中，已缓冲 {ms:.0f}ms 音频，{state}", flush=True)
            continue
        await drop_early("这一轮没赌成" if early["task"] else "")

        # 上一轮还没播完就被新的一轮顶掉。force：这里不能被宽限期挡下来，挡下来
        # 旧任务会继续往同一条 socket 里灌音频，两轮交织着播。
        await stop_reply(force=True)
        turn["t0"] = turn["until"] = 0.0
        turn["speech_end"], turn["play_started"] = pending.speech_end, False
        turn["reply"] = {"text": ""}            # 新一轮，回声比对从空的开始
        reply_state = turn["reply"]
        context_space = ACTIVE_SPACE
        memory_vm = vm
        timeline = AudioTimeline(
            prebuffer_seconds=0.16, rate_estimator=speech_rate)
        turn["timeline"] = timeline

        async def run_reply(pending=pending, timeline=timeline,
                            context_space=context_space, memory_vm=memory_vm,
                            reply_state=reply_state):
            try:
                await voicemem_llm_tts(
                    pending, sock.send_json, send_audio, owner, timeline,
                    said=reply_state, context_session=context_session,
                    context_space=context_space, memory_vm=memory_vm)
            finally:
                if not timeline.context_saved:
                    reply = timeline.heard_text()
                    history_turn_id = _push_history(
                        context_session, context_space, pending.text, reply,
                        interrupted=True)
                    queue_remember_turn(
                        pending, reply, owner, history_turn_id,
                        memory_vm=memory_vm)
                    timeline.context_saved = True

        task = asyncio.create_task(run_reply())
        turn["task"] = task

        def reply_done(done_task):
            try:
                done_task.result()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[web] 回复任务失败：{type(e).__name__}: {e}", flush=True)

        task.add_done_callback(reply_done)


async def realtime_session(sock):
    """方案 A：整段麦克风音频平行喂给 OpenAI Realtime；本地 ASR+VAD 只负责投机记忆 +
    用 500ms 判回合（关掉 OpenAI 自带 server_vad）。"""
    connected = False
    context_session = uuid.uuid4().hex
    try:
        async with utils.realtime_connect(REPLY) as conn:
            # 握手成功不代表能用：没权限/模型名不对时，OpenAI 是**连上之后**再发
            # close 4000（invalid_model / 权限错误）。所以要等第一次交互成功才算数。
            # turn_detection 只借 OpenAI 的 VAD 做**打断**，不让它接管回合：
            #   create_response=False    → 什么时候回复仍由我们决定（本地 VAD 判完
            #                              一轮、记忆预取好，才 response.create）
            #   interrupt_response=True  → 用户一开口，服务端直接掐掉正在播的回复
            # 打断判定放在 OpenAI 那侧，是因为它直接对着音频流做；本地 VAD 要等
            # 麦克风帧过完一整条链路（浏览器 AEC → ws → 重采样 → silero），真人
            # 隔着扬声器插话时信号本来就弱，很容易判不出来。
            # turn_detection 在 session.audio.input 下，**不是顶层**——写成顶层
            # 会被静默拒绝（"Unknown parameter: session.turn_detection"，只以 error
            # 事件回来，没人看就以为设上了）。原来那句 {"turn_detection": None} 一直
            # 没生效，于是 server_vad 始终开着、自动抢着回复：它生成的 response 不带
            # 我们注入的记忆，我们自己的 response.create 又撞上"已有 response 在跑"
            # 而失败——语音轮"没用上记忆"就是这么来的。
            #   create_response=False    → 什么时候回复由我们决定（本地 VAD 判完一轮、
            #                              记忆预取好，才 response.create）
            #   interrupt_response=True  → 用户一开口，服务端直接掐掉正在播的回复；
            #                              这条判定在 OpenAI 侧直接对着音频流做，比
            #                              本地 VAD（隔着 AEC + 网络 + 重采样）可靠
            await conn.session.update(session={
                "type": "realtime",
                "audio": {
                    "input": {"turn_detection": _turn_detection()},
                    "output": {"voice": utils.RT_VOICE},
                },
            })
            connected = True

            # 这一轮的状态：谁在说、说了什么、这轮用户的输入是什么。
            # until：前端预计几点才把已发出去的音频播完（见 hearing()）。
            turn = {"live": False, "reply": "", "pending": None,
                    "t0": 0.0, "until": 0.0, "first": False,
                    "timeline": None, "response_done": False,
                    "provider_item_id": "", "space": "", "memory_vm": None}
            owner = {"id": "", "last": "", "miss": 0}
            speech_rate = SpeechRateEstimator()
            timelines: dict[str, AudioTimeline] = {}
            playback_tasks: set[asyncio.Task] = set()
            candidate_paused = False
            candidate_paused_at = 0.0
            # Realtime 同一时刻只允许一个 response；取消完成后才能创建下一轮。
            response_idle = asyncio.Event()
            response_idle.set()

            def hearing() -> bool:
                """用户此刻还听不听得见助手。

                不能用 turn["live"] 代替：realtime 推音频比实时播放快得多，一段
                十几秒的回复两三秒就推完了，response.done 一到 live 就变 False，
                可前端那边还在播剩下的十几秒。这时候用户插话，on_speech 看到
                live=False 就"忽略"，前端从没收到 answer_interrupt——表现正是
                "打断没反应，它非要念完"。
                所以按**已发出去的音频时长**算：24k PCM16，一个样本 2 字节。
                """
                timeline = turn["timeline"]
                buffered = bool(timeline and not timeline.playback_done
                                and time.monotonic() < turn["until"])
                return (candidate_paused or turn["live"] or buffered
                        or time.monotonic() < turn["until"])

            async def pause_candidate():
                nonlocal candidate_paused, candidate_paused_at
                if hearing() and not candidate_paused:
                    candidate_paused = True
                    candidate_paused_at = time.monotonic()
                    await sock.send_json({"type": "answer_pause"})

            async def resume_candidate():
                nonlocal candidate_paused, candidate_paused_at
                if candidate_paused:
                    candidate_paused = False
                    if turn["until"]:
                        turn["until"] += max(0.0, time.monotonic() - candidate_paused_at)
                    candidate_paused_at = 0.0
                    await sock.send_json({"type": "answer_resume"})

            def close_turn(interrupted=False):
                p, reply = turn["pending"], turn["reply"]
                timeline = turn["timeline"]
                space = turn["space"] or ACTIVE_SPACE
                memory_vm = turn["memory_vm"] or vm
                turn.update(live=False, reply="", pending=None, timeline=None,
                            response_done=False, provider_item_id="",
                            space="", memory_vm=None)
                if p is None:
                    return
                if interrupted:
                    reply = timeline.heard_text() if timeline else ""
                    if BARGE_DEBUG and timeline:
                        print(f"[context] 打断于 {timeline.rendered_ms()}ms，保留回复 "
                              f"{reply!r}", flush=True)
                history_turn_id = _push_history(
                    context_session, space, p.text, reply,
                    interrupted=interrupted)
                queue_remember_turn(
                    p, reply, owner, history_turn_id, memory_vm=memory_vm)
                if timeline:
                    timelines.pop(timeline.output_id, None)

            async def playback_checkpoint(data):
                timeline = timelines.get(str(data.get("output_id") or ""))
                if timeline is None:
                    return
                timeline.update_checkpoint(
                    data.get("rendered_samples", 0),
                    data.get("sample_rate", MIC_RATE),
                    data.get("state", "playing"))
                if turn["timeline"] is timeline and timeline.playback_done:
                    turn["until"] = 0.0
                    if turn["response_done"]:
                        close_turn(interrupted=False)

            async def playback_fallback(timeline):
                delay = max(0.0, turn["until"] - time.monotonic()) + 0.5
                await asyncio.sleep(delay)
                if turn["timeline"] is timeline and turn["response_done"]:
                    timeline.assume_drained()
                    turn["until"] = 0.0
                    close_turn(interrupted=False)

            def schedule_playback_fallback(timeline):
                task = asyncio.create_task(playback_fallback(timeline))
                playback_tasks.add(task)
                task.add_done_callback(playback_tasks.discard)

            async def pump():
                """常驻事件泵：OpenAI 的事件流只有这一个消费者。

                turn["live"] 为假时一律不往前端转发——被打断后 OpenAI 还会吐一会儿
                残余音频，转过去的话前端刚 stopPlayback 又排上新的，打断就不干净了。
                """
                async for ev in conn:
                    t = getattr(ev, "type", "")
                    if t.endswith("output_audio.delta"):
                        if turn["live"]:
                            if not turn["first"]:
                                # 首帧音频延迟：response.create 发出去到 OpenAI 吐第一块
                                # 声音之间的时间。这是"它反应慢"里我们控制不了的那半。
                                turn["first"] = True
                                print(f"[lat] realtime 首帧 "
                                      f"{(time.monotonic()-turn['t0'])*1000:.0f}ms", flush=True)
                            pcm = base64.b64decode(ev.delta)
                            timeline = turn["timeline"]
                            if timeline:
                                timeline.append_audio(pcm)
                                timestamps = getattr(ev, "timestamps", ()) or ()
                                if timestamps:
                                    timeline.add_timestamps(tuple(timestamps))
                            turn["provider_item_id"] = (
                                getattr(ev, "item_id", "") or turn["provider_item_id"])
                            # 前端是排队播的（index.html 的 nextPlay），这里跟着算
                            # 同一条时间线：上一块播完之后再接这一块。
                            turn["until"] = (max(turn["until"], time.monotonic())
                                             + len(pcm) / 2 / 24000)
                            await sock.send_bytes(pcm)
                    elif t.endswith("output_audio_transcript.delta"):
                        if turn["live"]:
                            turn["reply"] += ev.delta
                            timeline = turn["timeline"]
                            if timeline:
                                timeline.append_text(ev.delta)
                                timestamps = getattr(ev, "timestamps", ()) or ()
                                if timestamps:
                                    timeline.add_timestamps(tuple(timestamps))
                            turn["provider_item_id"] = (
                                getattr(ev, "item_id", "") or turn["provider_item_id"])
                            await sock.send_json({"type": "answer_delta", "text": ev.delta})
                    elif t == "error" or t.endswith(".error"):
                        err = getattr(ev, "error", None)
                        code = getattr(err, "code", "")
                        # server_vad 判完一句会自己 commit 音频缓冲，我们随后那次
                        # commit 就撞上空缓冲。两种情况都得留着手动 commit（说得太短
                        # 时 server_vad 不会自动 commit），所以这条属于预期内，忽略。
                        # response_cancel_not_active：打断有两条路（本地 VAD 的
                        # on_speech + server_vad 的 interrupt_response），互为备份，
                        # 谁先到算谁的，慢的那个扑空是正常的。
                        if code not in (
                                "input_audio_buffer_commit_empty",
                                "response_cancel_not_active"):
                            print(f"[web] realtime 事件错误：{err or ev}", flush=True)
                    elif t.endswith("input_audio_buffer.speech_stopped"):
                        # OpenAI 判"你说完了"的时刻。跟本地 silero 判完（我们发
                        # response.create 那一刻）比，谁早谁晚——早的那个才是
                        # 真正的 EOU 下限，晚的那部分是白等的。
                        turn["stopped"] = time.monotonic()
                        if BARGE_DEBUG:
                            print("[lat] OpenAI 判说完", flush=True)
                    elif t.endswith("input_audio_buffer.speech_started"):
                        # OpenAI 的 VAD 听到人声：它那侧已经掐了回复，我们同步收尾
                        since = (time.monotonic() - turn["t0"]) * 1000
                        if BARGE_DEBUG:
                            print(f"[barge] OpenAI VAD 听到人声 (live={turn['live']}, "
                                  f"还在播={hearing()}, {since:.0f}ms)", flush=True)
                        # 宽限期，跟本地那条路一样。本地 VAD 判完一轮（静音 500ms）就
                        # 发 response.create，而 OpenAI 的 server_vad 只要 320ms 静音就
                        # 认为下一句开始了——用户的话尾、呼吸声、环境噪声都够触发。
                        # 没有这道门的话，speech_started 会在助手出声之前就到，
                        # 回复被掐在第一个音频块之前：一个字都听不见，还不报错。
                        # 只记录，不再据此打断——见 _turn_detection 里的说明。
                        # 留着这行日志是因为它是判断"回声压没压干净"最直接的证据：
                        # 助手说话期间频繁出现，就说明 AEC 有残留。
                    elif t.endswith("response.done") or t.endswith("response.cancelled"):
                        # done/cancelled 事件释放下一轮 response 的创建屏障。
                        response_idle.set()
                        if BARGE_DEBUG and not turn["live"]:
                            print("[barge] 旧 Realtime response 已退出", flush=True)
                        if turn["live"]:
                            timeline = turn["timeline"]
                            if timeline:
                                timeline.mark_generation_complete()
                            if t.endswith("response.cancelled"):
                                response_idle.clear()
                                heard = timeline.heard_text() if timeline else ""
                                provider_item_id = turn["provider_item_id"]
                                if timeline:
                                    timeline.mark_interrupted()
                                await sock.send_json({
                                    "type": "answer_interrupt",
                                    "output_id": timeline.output_id if timeline else "",
                                    "heard_text": heard,
                                })
                                close_turn(interrupted=True)
                                if provider_item_id and timeline:
                                    try:
                                        await truncate_provider_output(
                                            conn, provider_item_id, timeline)
                                    except Exception as e:
                                        if BARGE_DEBUG:
                                            print(f"[barge] Provider 上下文截断失败：{e}",
                                                  flush=True)
                                response_idle.set()
                            else:
                                turn["live"] = False
                                turn["response_done"] = True
                                await sock.send_json({
                                    "type": "answer_done",
                                    "output_id": timeline.output_id if timeline else "",
                                })
                                if timeline and timeline.playback_done:
                                    close_turn(interrupted=False)
                                elif timeline:
                                    schedule_playback_fallback(timeline)

            async def on_frame(raw):
                await conn.input_audio_buffer.append(audio=base64.b64encode(raw).decode())

            async def on_speech():
                """用户在助手说话时开口 → 打断。幂等：hearing() 一变假就不再触发。"""
                nonlocal candidate_paused, candidate_paused_at
                if not hearing():
                    if BARGE_DEBUG:
                        print("[barge] 有人声但助手没在说，忽略", flush=True)
                    return
                # 刚开口那一小段不许打断：那时候麦克风里几乎只有助手自己的声音，
                # 回声消除还没跟上，很容易一出声就把自己掐了。
                since = (time.monotonic() - turn["t0"]) * 1000
                if since < BARGE_GRACE_MS:
                    if BARGE_DEBUG:
                        print(f"[barge] 才说了 {since:.0f}ms，还在宽限期内，不打断", flush=True)
                    return
                if BARGE_DEBUG:
                    left = max(0.0, turn["until"] - time.monotonic()) * 1000
                    print(f"[barge] ★ 打断：转写触发（前端还剩 {left:.0f}ms 没播完）",
                          flush=True)
                active_response = not response_idle.is_set()
                timeline = turn["timeline"]
                heard_text = timeline.heard_text() if timeline else ""
                output_id = timeline.output_id if timeline else ""
                provider_item_id = turn["provider_item_id"]
                if timeline:
                    timeline.mark_interrupted()
                turn["live"], turn["until"] = False, 0.0
                candidate_paused = False
                candidate_paused_at = 0.0
                # 先通知前端停播——这是本地操作，立刻生效；而 response.cancel() 要
                # 等一次 OpenAI 往返。之前顺序反了，人插话后还得听完那一个往返的
                # 时间，听感就是"打断没用，他非要说完"。
                await sock.send_json({"type": "answer_interrupt",
                                      "output_id": output_id,
                                      "heard_text": heard_text})
                close_turn(interrupted=True)
                if active_response:
                    await conn.response.cancel()
                if provider_item_id and timeline:
                    try:
                        await truncate_provider_output(
                            conn, provider_item_id, timeline)
                    except Exception as e:
                        if BARGE_DEBUG:
                            print(f"[barge] Provider 上下文截断失败：{e}", flush=True)
                if active_response:
                    try:
                        await asyncio.wait_for(response_idle.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        # 本地播放已停止；下一轮继续等待 done/cancelled 事件。
                        print("[barge] 等待 Realtime 取消确认超时，下一轮暂缓创建", flush=True)

            pump_task = asyncio.create_task(pump())
            try:
                async for pending in anticipate(sock, on_frame=on_frame,
                                                on_speech=on_speech, owner=owner,
                                                # is_busy 不能漏：anticipate 里
                                                # 「助手在不在说话」全靠它，不传就恒为
                                                # False，候选暂停那一整套（if busy and…）
                                                # 永远不触发 —— realtime 打不断就是这个。
                                                is_busy=hearing,
                                                said=lambda: turn["reply"],
                                                on_candidate=pause_candidate,
                                                on_candidate_reject=resume_candidate,
                                                on_playback_checkpoint=playback_checkpoint):
                    # 跟 llm_tts 那条一致：整轮都是附和、助手还在说，就当没听见。
                    if _is_backchannel(pending.text) and hearing():
                        if BARGE_DEBUG:
                            print(f"[barge] 整轮都是附和 {pending.text!r}，不算一轮，继续说",
                                  flush=True)
                        continue
                    if hearing():                        # 上一轮还没播完就被新的一轮顶掉
                        await on_speech()
                    if not response_idle.is_set():
                        if BARGE_DEBUG:
                            print("[barge] 等旧 response 退出后再创建下一轮", flush=True)
                        await response_idle.wait()
                    context_space = ACTIVE_SPACE
                    memory_vm = vm
                    timeline = AudioTimeline(
                        prebuffer_seconds=0.08, rate_estimator=speech_rate)
                    timelines[timeline.output_id] = timeline
                    turn.update(
                        live=True, reply="", pending=pending,
                        t0=time.monotonic(), until=0.0, first=False,
                        timeline=timeline, response_done=False,
                        provider_item_id="", space=context_space,
                        memory_vm=memory_vm)
                    response_idle.clear()
                    try:
                        await start_realtime_turn(
                            pending, conn, sock.send_json, timeline,
                            context_session=context_session,
                            context_space=context_space)
                    except Exception:
                        response_idle.set()
                        raise
            finally:
                pump_task.cancel()
                try:
                    await pump_task
                except asyncio.CancelledError:
                    pass
                if turn["pending"] is not None:
                    timeline = turn["timeline"]
                    fully_played = bool(
                        turn["response_done"] and timeline and timeline.playback_done)
                    close_turn(interrupted=not fully_played)
                for task in playback_tasks:
                    task.cancel()
                if playback_tasks:
                    await asyncio.gather(
                        *list(playback_tasks), return_exceptions=True)
    except Exception as e:
        if connected:
            raise
        await _no_realtime(sock, e)
    finally:
        _SESSION_CONTEXT.clear_session(context_session)


#: 右脑 slot → 脑图三个簇。
#:
#: 右脑真正的分类单位是 rb_slots 里那 6 个 slot（情绪 / 喜好与厌恶 / 应对方式 /
#: 表达风格 / 思维模式 / 人物地点态度），不是 memory_class——那只有 heartnote /
#: response_experience 两种，分不出东西。脑图上只有三个簇，所以这里把 6 个 slot
#: 收敛成 3 个。
#:
#: 检索命中的内容里，slot 名就写在开头（"情绪：…""喜好与厌恶：…"）；
#: heartnote 是一条条的情绪时刻（"情感记录：…（内心OS：【难过】…）"），归 emotion。
SLOT_TO_CLUSTER = {
    "情绪":         "emotion",
    "情感记录":     "emotion",
    "内心OS":       "emotion",
    "喜好与厌恶":   "preference",
    "思维模式":     "preference",
    "应对方式":     "experiences",
    "表达风格":     "experiences",
    "人物地点态度": "experiences",
    "避免重复":     "experiences",
}
_CALM = ("", "平静", "中性")


def rb_cluster(content: str, memory_class: str = "", emotion: str = "") -> str:
    """一条右脑记忆归到脑图哪个簇。0 LLM，只看 slot 名。"""
    # 先剥掉 "[2026-06-20] " 这种日期前缀，否则它占满取来比对的那一小段，
    # heartnote 的"情感记录"就落到窗口外了。
    text = re.sub(r"^\s*\[[0-9-]{6,12}\]\s*", "", content or "")
    head = text[:14]
    for slot, cluster in SLOT_TO_CLUSTER.items():
        if slot in head:
            return cluster
    if str(memory_class) == "response_experience":
        return "experiences"
    if emotion not in _CALM:
        return "emotion"
    return "experiences"


def audio_of(memory_id: str) -> str:
    """这条记忆当时那段原声在哪；没归档过、或已过保留期被清掉，返回 ""。

    走核心的 GetOriginalAudio——它已经带了"文件还在不在"的检查，不用在这儿重写。
    ``LAST_TUNE_ID`` 是个例外：它指的是刚听过、还没来得及入库的那段（见 _LAST_TUNE）。
    """
    if memory_id == LAST_TUNE_ID:
        return _last_tune_path()
    if memory_id.startswith(GROUP_ID_PREFIX):     # 一组片段，现拼一个完整的
        return _stitch([m for m in memory_id[len(GROUP_ID_PREFIX):].split(",") if m])
    try:
        r = vm._o._audio.GetOriginalAudio(memory_id)
        return r.get("audio_path") or "" if r.get("found") else ""
    except Exception as e:
        print(f"[web] 查存档音频失败：{e}", flush=True)
        return ""


def hit_cluster(content: str, source: str) -> str:
    """给检索命中用：只有 content 和 source，没有 metadata。"""
    return rb_cluster(content, source, "")


def _rb_cluster(m) -> str:
    """给快照用：从 RightBrainMemory 对象取字段。"""
    meta = getattr(m, "metadata", None) or {}
    return rb_cluster(getattr(m, "content", ""),
                      str(getattr(m, "memory_class", "")),
                      meta.get("emotion", ""))


def fact_index(uid: str) -> dict:
    """左脑记忆 id → 事实原文。

    原文在向量库里，认知图的 memories 表只有 id/slot/热度这些，取不到文本——
    一开始用 get_memory_record 取，结果每条 heartnote 的起因都是空的。
    """
    try:
        entries = vm._o._get_repo()._vector_store.list_entries(user_id=uid)
        return {e["id"]: e["text"] for e in entries}
    except Exception as e:
        print(f"[web] 读左脑事实失败：{e}", flush=True)
        return {}


#: 右脑每个 slot 在脑图上最多画几个 entity。
RB_ENTITIES_PER_SLOT = int(os.environ.get("VOICEMEM_RB_GRAPH_PER_SLOT", "6"))
#: 左脑每个 slot 在脑图上最多画几条记忆。
LB_ENTRIES_PER_SLOT = int(os.environ.get("VOICEMEM_LB_GRAPH_PER_SLOT", "7"))


# ── 右脑判断的「人话」版本 ────────────────────────────────────────────────────
# 库里存的 claim 是给**模型**看的：紧凑、第三人称、像标签（"Facing a lot of
# pressure recently"）。那份不能动——prompt 需要的就是这种密度。
# 但页面上是给**人**看的，同一句话摆出来就很像在读档案。这里做一份只用于显示的
# 第一人称改写。
#
# 异步 + 缓存：memory_snapshot 是同步的，不能在里面等一次 LLM 往返。所以第一次
# 显示原文，后台改写完落进缓存，前端下一次轮询（watchMemories 本来就在轮）就换成
# 人话。改写只碰措辞，不新增任何事实。
#: 这个空间的主人叫什么。右脑那些话是**关于他**的，一律写"他"就少了那份认得他的
#: 感觉——"Jiaqi 一紧张就闷声不响"和"他一紧张就闷声不响"，前者才像认识他。
#: 名字来自声纹注册表（自报"我叫X"时绑定的），读不到就回落到"他"。
_OWNER_NAME_CACHE: dict = {}
#: 疑问词。"我叫什么名字？"被当成自我介绍绑进去过，registry 里真的躺着一条
#: name="什么名字"（见 voiceprint/speaker_identity.py 顶上那段）。显示前挡一道。
_BAD_NAME_CHARS = "什谁哪啥吗呢么?？"


def owner_name(space: str = "") -> str:
    """这个空间主人的名字；认不出来就返回 ""（调用方自己回落到"他"）。"""
    space = space or ACTIVE_SPACE
    if space in _OWNER_NAME_CACHE:
        return _OWNER_NAME_CACHE[space]
    name = ""
    try:
        import json
        from voicemem.utils.common import space as _sp
        d, _ = space_dir(space)
        p = _sp.mm(d, "voiceprint_registry.json")
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            cands = []
            for key, v in (data or {}).items():
                if not isinstance(v, dict) or v.get("role") != "user":
                    continue
                n = (v.get("name") or "").strip()
                # 挡掉疑问句绑进来的假名字，也挡掉 "user" 这种占位 key
                if not n or n.lower() == "user" or any(c in n for c in _BAD_NAME_CHARS):
                    continue
                cands.append((bool(v.get("entity_id")), n))
            if cands:
                # 有 entity_id 的更可信（真的在图里落过地）
                cands.sort(key=lambda t: not t[0])
                name = cands[0][1]
    except Exception as e:
        print(f"[rb] 主人姓名读取失败：{type(e).__name__}: {e}", flush=True)
    _OWNER_NAME_CACHE[space] = name
    return name


_RB_HUMAN: dict = {}          # claim 原文 -> 人话版本
_RB_HUMAN_PENDING: set = set()
#: 关掉就一直显示原始 claim（不想为显示花钱时）。
RB_HUMANIZE = os.environ.get("VOICEMEM_RB_HUMANIZE", "1") != "0"

_RB_HUMANIZE_PROMPT = (
    # 四版教训，别再退回去：
    # ① 只写"第一人称"不够——不说清是**谁**在说，模型只做同义替换。
    # ② "不许新增事实"会被读成"不许换说法"，于是退化成往原句前贴个"我注意到"。
    #    要把两件事拆开：事实不能加，措辞必须重说。
    # ③ "我发现/我注意到/在我看来"全是**报告动词**——语法上是第一人称，语气上还是
    #    观察员在汇报。要的是一个懂他的小东西在心疼他，所以这类开头得禁掉。
    # ④ 名字那条第一版写的是"和「他」换着用"，太软，十条全用了"他"——示例里也全是
    #    "他"，模型照着示例走。规则要硬，示例也得点名。
    #    后来改成**一律点名**：页面上每条是独立的一行，不是一段连贯的话，
    #    重复出现名字读起来是"这是关于谁的"，不是啰嗦。
    # ⑤ 显示语言跟**界面**走，不跟原文走。库里中英混着（翻译过一半），跟原文走
    #    页面上就中英混排。显示是给人看的，同一屏里不该夹生。
    "下面每行是一条关于 {who} 的判断，来自一个一直陪着 {who} 的小助手——"
    "像只很懂他的小动物，安静地待在旁边，什么都看在眼里。\n"
    "把每一条改写成这个小助手会说出来的话。\n"
    "\n"
    "语气：\n"
    "· 短。观察那半句十几个字最好，读起来是一句话，不是一条记录。\n"
    "· 有温度，带一点点护着他的意思——是心疼，不是分析。\n"
    "· **不要用「我发现」「我注意到」「在我看来」开头**，那是汇报的口气。"
    "直接说那件事，或者说你替他觉得怎么样。\n"
    "· 别肉麻、别撒娇、别堆感叹号、别讲道理也别安慰。\n"
    "{name_rule}"
    "\n"
    "内容：\n"
    "· **必须换一种说法**。原句是概括性的词（「简短回应」「寻求认同」），"
    "要还原成人会怎么形容（「话就变少了」「想有人接住他」）。\n"
    "· 但**不许新增任何事实**：不补细节、不猜原因、不加评价。换措辞，不换内容。\n"
    # 关键区分：往用户身上加事实=幻觉；说**我自己**打算怎么做=助手的立场，安全。
    # 而且这一半只在页面上显示，不进模型的 prompt，说错了也影响不到回答。
    "· 观察之后，可以再跟一句**你自己打算怎么做**，用「 —— 」隔开。"
    "只说你的做法（「我不追问」「我让他说完」），"
    "**不许再多说一句关于他的事**。\n"
    # 不加这条会写出"我帮他找个安静的地方复习"——语音助手做不到，读着就假。
    "· 你能做的只有**说话**这一件事：怎么开口、先提什么、避开什么、"
    "什么时候闭嘴。不要承诺现实世界里的行动（找地方、订闹钟、帮他做事），"
    "你做不到。\n"
    "· 这后半句是可选的：想不出自然的做法就只留观察，别硬凑。"
    "十条里有三四条带上就够了，条条都带会像在背守则。\n"
    "· 全部用{lang}输出，不管原文是什么语言。\n"
    "\n"
    "逐行输出，行数和顺序跟输入完全一致；不要编号、引号或多余的话。\n"
    "例：\n"
    "{examples}"
)

#: 示例。两件事都靠它带：**语气**和**输出语言**。
#: 模型跟示例走的力度远大于跟规则走——名字那条和语言这条都在这儿栽过：
#: 规则写了"每条点名"但示例用「他」，输出就全是「他」；规则写了"输出英文"
#: 但示例全中文，输出就全是中文。所以示例必须跟着目标语言换，而且都要点名。
_RB_HUMANIZE_EXAMPLES = {
    "zh": (
        "  输入  在焦虑时倾向于简短回应\n"
        "  输出  {who}一紧张就闷声不响 —— 我不追问，等他自己开口\n"
        "  输入  考试时容易走神\n"
        "  输出  {who}考试时容易走神 —— 这事我不主动提，怕他更焦虑\n"
        "  输入  在情绪低落时不喜欢被忽视\n"
        "  输出  {who}难过的时候，最怕没人理\n"
        "  输入  Feels accomplished when recognized\n"
        "  输出  夸{who}一句，他整个人就亮了\n"
        "  输入  Hates being interrupted\n"
        "  输出  打断{who}说话，他立刻就不说了 —— 我尽量让他讲完\n"
    ),
    "en": (
        "  输入  在焦虑时倾向于简短回应\n"
        "  输出  {who} goes quiet the moment he tenses up —— I don't push, I wait\n"
        "  输入  考试时容易走神\n"
        "  输出  {who}'s mind wanders in exams —— I won't bring it up, it'd only add pressure\n"
        "  输入  在情绪低落时不喜欢被忽视\n"
        "  输出  When {who} is down, being ignored is the worst of it\n"
        "  输入  Feels accomplished when recognized\n"
        "  输出  A little praise and {who} lights right up\n"
        "  输入  Hates being interrupted\n"
        "  输出  Cut {who} off mid-sentence and you'll lose him —— I let him finish\n"
    ),
}


def _rb_humanize_now(claims: list, name: str = "", lang: str = "zh") -> None:
    """后台线程里跑：一次 LLM 往返改写一批，结果落进 _RB_HUMAN。"""
    lang_name = "英文" if lang == "en" else "中文"
    try:
        from openai import OpenAI
        who = name or "他"
        sysmsg = _RB_HUMANIZE_PROMPT.format(
            who=who,
            lang=lang_name,
            # 规则要硬。写"换着用"的那版，十条全用了"他"。
            name_rule=(f"· **每条都直接叫他「{name}」**，别用「他」代替——"
                       "页面上每条是独立一行，点名是在说「这是关于谁的」。\n"
                       if name else ""),
            examples=_RB_HUMANIZE_EXAMPLES.get(lang, _RB_HUMANIZE_EXAMPLES["zh"])
                     .replace("{who}", who))
        r = OpenAI().chat.completions.create(
            model=utils.CHAT_MODEL, temperature=0.7,
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": "\n".join(claims)}],
        )
        lines = [x.strip() for x in (r.choices[0].message.content or "").splitlines() if x.strip()]
        if len(lines) != len(claims):      # 行数对不上就整批丢弃，别错位配对
            print(f"[rb] 改写行数不符（{len(lines)}≠{len(claims)}），这批跳过", flush=True)
            return
        for c, h in zip(claims, lines):
            _RB_HUMAN[(lang, name, c)] = h
    except Exception as e:
        print(f"[rb] 判断改写失败：{type(e).__name__}: {e}", flush=True)
    finally:
        _RB_HUMAN_PENDING.difference_update((lang, name, c) for c in claims)


def rb_human(claim: str) -> str:
    """显示用的那句话。还没改写好就先返回原文，同时排上队。"""
    if not RB_HUMANIZE or not claim:
        return claim
    name, lang = owner_name(), SPACE_LANG
    hit = _RB_HUMAN.get((lang, name, claim))
    if hit:
        return hit
    if (lang, name, claim) not in _RB_HUMAN_PENDING:
        _RB_HUMAN_PENDING.add((lang, name, claim))
        import threading
        threading.Thread(target=_rb_humanize_now,
                         args=([claim], name, lang), daemon=True).start()
    return claim


def rb_human_batch(claims: list) -> None:
    """一次把缺的都排上队——脑图一屏几十条，一条一个线程太浪费。"""
    if not RB_HUMANIZE:
        return
    name, lang = owner_name(), SPACE_LANG
    todo = [c for c in dict.fromkeys(claims)
            if c and (lang, name, c) not in _RB_HUMAN
            and (lang, name, c) not in _RB_HUMAN_PENDING]
    if not todo:
        return
    _RB_HUMAN_PENDING.update((lang, name, c) for c in todo)
    import threading
    threading.Thread(target=_rb_humanize_now, args=(todo, name, lang), daemon=True).start()


def right_brain_tree(uid, facts):
    """脑图右半球：slot → 判断 → 证据。

    读右脑的判断表（voicemem/rightbrain/traits_store.py）。旧的
    slot→entity→heartnote 结构已经不再写入，_right_brain_tree_v1 只用来看老数据。
    """
    try:
        store = vm._o._right._traits()
    except Exception as e:
        print(f"[web] 判断表读取失败：{type(e).__name__}: {e}", flush=True)
        return []

    traits = list(store.all(uid, per_slot=RB_ENTITIES_PER_SLOT))
    rb_human_batch([t.claim for t in traits])     # 缺的一次排队，见 rb_human
    out = []
    for t in traits:
        out.append({
            "cluster": t.cluster,
            "slot": t.slot,
            # 节点标题用人话版；还没改写好就是原文，下一次轮询会换上来。
            # raw 留着——前端拿它跟每轮命中做匹配，换成人话就对不上了。
            "raw": t.claim,
            "text": rb_human(t.claim),
            "desc": "",               # 判断本身就是概括，不再垫一行原始事实
            "notes": [{"text": e.quote, "emotion": e.emotion, "cause": e.cause}
                      for e in t.evidence],
        })
    return out


def _right_brain_tree_v1(uid: str, facts: dict) -> list:
    """右脑真实的三层结构：slot → entity → 挂在下面的 heartnote。

    脑图上的节点是 **entity**（"委屈""讨厌坚果和过敏""选择沉默忍耐"），不是
    一条条 heartnote —— entity 才是右脑归纳出来的那个"点"，heartnote 是支撑它
    的证据。每条 heartnote 再带上引发它的左脑事实，这样"为什么委屈"点两下就能看到。

    只读，全部走 graph_store 的公开方法。
    """
    graph = vm._o._right._rb_graph_store()
    repo = vm._o._right._rb_repo()
    notes = {}
    for m in repo.list_all(uid):
        # response_experience 记的是"助手上次怎么答的"，是给回复层看的内部笔记，
        # 不是对用户其人的认识。脑图上不该有它——它长出来的节点写着助手自己
        # 说过的话，看着像"系统把自己的回复当成了对你的了解"。
        if getattr(m, "memory_class", "") == "response_experience":
            continue
        meta = getattr(m, "metadata", None) or {}
        notes[m.id] = {
            "text": m.content,
            "emotion": meta.get("emotion", ""),
            "cause": facts.get(meta.get("left_memory_id", ""), ""),
        }

    out = []
    for slot in graph.list_slots(uid):
        # 没有 description 的 slot 跳过：右脑还没归纳过它，底下挂的就是原始实体
        # 倒进来的一堆东西（实测"人物地点态度"下面是 Jiaqi / 计算机本科 /
        # 新加坡国立大学NUS / 2026年9月 —— 人名、学历、机构、日期，不是态度）。
        # 画在脑图上只是噪声，还占着右半球的位置。
        # 这里原来是「slot 没有 description 就整个跳过」。
        # description 是**长期归因**（session 边界的 _summarize_slot）才生成的，
        # 新建的空间从没跑过 → 六个 slot 全是空描述 → 每个都被跳过 → 脑图右半球
        # 一个节点都没有。而节点画的是**实体**，跟这个 slot 有没有一句画像总结
        # 没关系：实体在、证据在，就该画出来。
        cluster = SLOT_TO_CLUSTER.get(slot.name, "experiences")
        # 每个 slot 只画前几个 entity——库里跑一阵就有 89 个，全画上去右半球糊成
        # 一片（左脑同期才 26 条）。
        #
        # 排序不能只看证据条数。原来是纯按条数倒序，结果「喜好与厌恶」下 24 个
        # 实体里新来的那个只有 1 条证据，永远排最后，**永远进不了图**——用户说
        # 一句全新的话，右脑一个新节点都不长，看着像没记住。
        # 现在留一半席位给最近新增的：一半按证据多少（稳定的画像），一半按新旧
        # （刚说的那句能立刻看见）。
        def rows(newest):
            out = []
            for ent in graph.get_entities_for_slot(uid, slot.id, newest_first=newest):
                mids = [i for i in graph.get_memories_for_entity(ent.id) if i in notes]
                out.append((len(mids), ent, mids))
            return out

        fresh_n = max(1, RB_ENTITIES_PER_SLOT // 2)
        by_recent = rows(True)[:fresh_n]                    # 真·最近新增（rowid 倒序）
        taken = {t[1].id for t in by_recent}
        by_evidence = [t for t in sorted(rows(False), key=lambda t: -t[0])
                       if t[1].id not in taken]
        picked = by_recent + by_evidence[:RB_ENTITIES_PER_SLOT - len(by_recent)]

        for _, ent, mids in picked:
            # 没有任何记忆挂在下面的节点不画。情绪 slot 建空间时会预置 8 个情绪词
            # 实体（悲伤/平静/孤独…），全新的空库打开就摆着六个孤零零的点，看着
            # 像已经记住了什么，其实背后一条证据都没有。
            if not mids:
                continue
            ns = [notes[i] for i in mids]
            # 实体描述只用巩固那一步（_summarize_entity）归纳出来的那句。
            #
            # 曾经在这儿加过兜底：没有描述就拿第一条证据的 cause 顶上。那是错的——
            # cause 是**左脑事实原文**，于是卡片变成
            #     标题：佳琪
            #     正文：用户的名字是佳琪，今年20岁，在NUS读书…   ← 原始事实
            #     note：【委屈】呃我是佳琪啊我今年二十岁了…       ← 原话
            # 而且同一句话抽出的几个实体（佳琪/计算机专业/NUS）cause 相同，三张卡片
            # 正文一模一样。宁可空着——标题本身就该是那句概括，中间不该再垫一行。
            desc = (getattr(ent, "description", "") or "").strip()
            out.append({
                "cluster": cluster,
                "slot": slot.name,
                "text": ent.name,                      # 脑图上显示的就是这个
                "desc": desc,
                "notes": ns,
            })
    return out


#: 最近一轮检索命中的记忆 id。快照要保证它们在图上——脑图上亮起来的必须是
#: 后端真正检索到的那几条，命中了却没画出来，"它记得这件事"就没演出来。
_LAST_HIT_IDS: set = set()


def note_hits(result) -> None:
    """记下这一轮检索命中了哪些左脑记忆。"""
    _LAST_HIT_IDS.clear()
    for h in (getattr(result, "hits", None) or []):
        mid = getattr(h, "memory_id", "")
        if mid:
            _LAST_HIT_IDS.add(str(mid))


def memory_snapshot(limit: int = 48) -> dict:
    """库里已有的记忆，供前端在打开页面时把脑图先铺满。

    只读，不碰模型：左脑走 list_entries + 认知图的 slot 标注，右脑走 list_all。
    库是空的（新用户）就返回空列表，前端照旧从空图开始长。
    """
    from voicemem.leftbrain.cognitive_graph.types import SlotV2

    uid = vm._o._user_id
    left, right = [], []
    try:
        repo = vm._o._get_repo()
        entries = repo._vector_store.list_entries(user_id=uid)
        # slot 标在认知图里，不在记忆条目上——先建 id -> slot 的反查表
        cog = repo._cognitive_store
        slot_of = {}
        for slot in SlotV2:
            for mid in cog.memory_ids_for_slots(uid, [slot]):
                slot_of.setdefault(mid, slot.value)
        # 助手自己说过的话也原样存在库里（见 3ed67f7），但脑图画的是"关于用户的
        # 记忆"——把助手的回复也长成节点，等于把它自己说的话当成对用户的认识。
        entries = [e for e in entries if e.get("role") != "assistant"]
        # 每个 slot 也限量。脑图上一个 slot 就是一块扇形，面积固定——daily_life
        # 攒到二十几条时那块就糊了，而别的 slot 才三四个点。限量之后各簇疏密一致。
        #
        # 但**这一轮检索命中的那几条必须留下**，哪怕它排在限量之外：图上亮起来的
        # 得是后端真检索到的东西，命中了却没画出来，看着就像"检索到 5 条只亮了 3 个"。
        # 命中的先排进去，剩下的名额再按原来的顺序填。
        per_slot, kept = {}, []
        hit_first = ([e for e in entries if str(e["id"]) in _LAST_HIT_IDS] +
                     [e for e in entries if str(e["id"]) not in _LAST_HIT_IDS])
        for e in hit_first:
            sl = slot_of.get(e["id"], "daily_life")
            hit = str(e["id"]) in _LAST_HIT_IDS
            per_slot[sl] = per_slot.get(sl, 0) + 1
            if hit or per_slot[sl] <= LB_ENTRIES_PER_SLOT:
                kept.append(e)
        entries = kept
        # 同理，命中的那几条不能被 limit 截掉
        head = [e for e in entries if str(e["id"]) in _LAST_HIT_IDS]
        rest = [e for e in entries if str(e["id"]) not in _LAST_HIT_IDS]
        for e in (head + rest)[:max(limit, len(head))]:
            # list_entries 的 date 直接截了 time_start 前 10 位，遇到纯时间串会切出
            # "09:20:37" 这种。不像日期就置空，别把垃圾送到前端。
            d = str(e.get("date", ""))
            # 带上这条记忆挂了哪些实体：前端据此在讲同一个人/同一件事的两条记忆
            # 之间连线——脑图上的连线才对应真实关系，而不是随便连。
            # 存的是实体 id（person_jiaqi_5ea413），前端要拿去当标签显示、也拿去
            # 按同名实体连线，所以在这儿换成名字。
            try:
                ents = []
                for eid in cog.entity_ids_for_memory(e["id"]) or []:
                    ent = cog.get_entity(eid)
                    nm = (getattr(ent, "name", "") if ent else "").strip()
                    ents.append(nm or eid)
            except Exception:
                ents = []
            # hit：这条是被这一轮检索命中、因而保底留在图上的。
            # 前端拿"图上多出了哪条"来判断"刚说的这句抽出了什么实体"，而保底进来
            # 的是**旧记忆**——不标出来的话，说一句「啦啦啦」也会让上次那些实体
            # （Jiaqi、室友、打游戏…）冒到标签栏上。
            left.append({"text": e["text"], "date": d if d[:4].isdigit() else "",
                         "slot": slot_of.get(e["id"], "daily_life"),
                         "hit": str(e["id"]) in _LAST_HIT_IDS,
                         "entities": list(ents)[:6]})
    except Exception as e:
        print(f"[web] 左脑快照读取失败：{e}", flush=True)
    try:
        right = right_brain_tree(uid, fact_index(uid))
    except Exception as e:
        print(f"[web] 右脑快照读取失败：{e}", flush=True)
    return {"left": left, "right": right}


# classify 必须包一层：直接传 vm.classify 会把**当前这个**实例焊进去，
# 切换空间之后脑图还在给旧空间分类。
app = utils.build_app(MODE, realtime_session if MODE == "realtime" else llm_tts_session,
                      lambda *a, **k: vm.classify(*a, **k), memory_snapshot, audio_of,
                      spaces=(list_spaces, create_space, use_space, lambda: ACTIVE_SPACE),
                      set_lang=set_lang)


def _warm_final_asr(memory_vm):
    """Load and actually decode before accepting connections, respecting opt-out."""
    if os.environ.get("VOICEMEM_FINAL_ASR", "1") == "0":
        print("[web] 离线复核预热跳过（已关闭）", flush=True)
        return
    started = time.monotonic()
    try:
        model = memory_vm.utils.get("asr_final")
        loaded = time.monotonic()
        if model is None:
            print("[web] 离线复核预热跳过（模型不可用）", flush=True)
            return
        import numpy as np
        model.transcribe(np.zeros(16000, dtype=np.float32))
        print(f"[web] 离线复核已预热：加载 {(loaded-started)*1000:.0f}ms"
              f" · 首次推理 {(time.monotonic()-loaded)*1000:.0f}ms", flush=True)
    except Exception as e:
        print(f"[web] 离线复核预热失败：{type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    print(f"[web] mode={MODE} spec≥{SPEC_MIN_CHARS}字 gamble={ARGS.gamble_ms}ms "
          f"confirm={ARGS.confirm_ms}ms -> http://localhost:{ARGS.port}/", flush=True)
    # 全部预热在这儿做完，别让第一句话去等模型加载。ASR(FunASR paraformer)
    # 是懒加载的，等用户开口才拉起来要好几秒——那几秒的音频堆在 socket 缓冲里，
    # 追赶时逐帧喂 VAD，静音会瞬间累计过 confirm_ms，第一句直接被截断（听感就是
    # "第一句又慢又不准"）。
    # TTS 后端在启动时就建一次：provider 名写错（或者用了已经删掉的后端）时，
    # 错误发生在**回复任务**里——表现是"文字和声音都没有"，看不出跟 TTS 有关。
    # 这里提前炸，把名字和可选项一起说清楚。
    # MLX 的缓冲池默认不设上限：每轮 fork 出来的 KV、TTS 的激活用完不还系统，攒着
    # 等复用。16GB 机器上这就是峰值 13G 对稳态 9.8G 那 3G 的来源之一。给它个上限，
    # 超过就真释放。VOICEMEM_MLX_CACHE_MB=0 关掉。
    try:
        import mlx.core as _mx
        _cap = int(os.environ.get("VOICEMEM_MLX_CACHE_MB", "512"))
        if _cap > 0:
            _mx.set_cache_limit(_cap * 2**20)
    except Exception:
        pass
    print(f"[mem] 启动 · {_mem_line()}", flush=True)
    try:
        _tts = vm.utils.get("tts")
        # 本地后端顺手合一句丢掉：模型加载要几十秒，不预热的话第一句话要等
        # 五秒才出声——而那正是最不该卡的位置。
        if type(_tts).__name__ in _LOCAL_TTS:
            import asyncio as _a

            async def _warm():
                # Kokoro 中英是两套 G2P 管线，各自首次建立要 1~3s（英文要加载
                # spaCy）；只热中文的话第一句英文回复照样冷。
                texts = ["你好", "hi"] if type(_tts).__name__ == "KokoroTTS" else ["你好"]
                for s in texts:
                    from contextlib import aclosing
                    _t = time.monotonic()
                    async with aclosing(_tts.stream(s)) as chunks:
                        async for _ in chunks:
                            break
                    print(f"[web] TTS 预热 {type(_tts).__name__} {s!r}：首块 "
                          f"{(time.monotonic() - _t) * 1000:.0f}ms（含加载）", flush=True)
            _a.run(_warm())
            print(f"[mem] TTS 就位 · {_mem_line()}", flush=True)
        if ARGS.backchannel and MODE == "llm_tts" and type(_tts).__name__ == "BreezeMLXTTS":
            # Prepare a small useful bank before accepting microphone sessions;
            # live turns only read it, never wait behind background synthesis.
            from harness.backchannel import BackchannelVoice
            _bc_voice = BackchannelVoice(_tts, lang=space_language(ACTIVE_SPACE))
            # 中文这 11 个词是逐条试听定下来的；Noctelle 音色直接从仓库 WAV 装入
            # 缓存，换成别的参考音色才会首次启动现合成。
            _bc_tokens = (["嗯", "嗯嗯", "对", "对啊", "是啊", "哦", "哦哦", "这样啊",
                           "不错", "我知道了", "挺好的"]
                          if _bc_voice.lang == "zh" else ["mm-hmm", "yeah", "right", "oh", "okay"])
            asyncio.run(_bc_voice.prime(tokens=_bc_tokens, variants=1))
            _BC_VOICE["obj"] = _bc_voice
    except Exception as e:
        from voicemem.tts import TTS_PROVIDERS
        print(f"[web] TTS 后端建不起来：{type(e).__name__}: {e}\n"
              f"      TTS_BACKEND={os.environ.get('TTS_BACKEND', '(没设)')!r}，"
              f"可选：{' / '.join(sorted(TTS_PROVIDERS))}", flush=True)
        raise SystemExit(1)

    print("[web] 预热本地模型（embedding / ASR / VAD / 感知）…", flush=True)
    vm.warmup(verbose=True)
    _warm_final_asr(vm)
    if _LOCAL_LLM is not None:
        # **在开门之前**把人设那一截算进 KV 母本。人设一个字都不会变，算一次就
        # 够用一整场；等第一个用户连上来再算，那一秒半就直接摊在他第一句话上
        # （实测第一句 5005ms，之后每句 2100ms）。
        _t = time.monotonic()
        _n = _LOCAL_LLM.prewarm()
        print(f"[web] 本地回复模型：人设 {_n} token 已进 KV 缓存"
              f"（{(time.monotonic() - _t) * 1000:.0f}ms）", flush=True)
        # 真生成几个 token。只填 KV 不算预热：第一次 stream_generate 还要编译
        # 采样图、首次分配 fork 的 cache、把权重真正拉进显存——这些原来全摊在第一个
        # 用户的第一句上（实测首轮 13s、后续 2s 的差就是它加换页）。这里跑掉，
        # 顺带把首字时间打出来：开服时就慢，说明机器已经在换页，别等到对话里发现。
        async def _gen_warm():
            from contextlib import aclosing
            t0 = time.monotonic(); first = 0.0; n = 0
            async with aclosing(_LOCAL_LLM("你好", "", [])) as gen:
                async for _tok in gen:
                    if not first:
                        first = (time.monotonic() - t0) * 1000
                    n += 1
                    if n >= 4:
                        break
            return first, (time.monotonic() - t0) * 1000, n
        try:
            _first, _tot, _n = asyncio.run(_gen_warm())
            print(f"[web] 本地回复模型：真生成预热 首字 {_first:.0f}ms · {_n} token {_tot:.0f}ms"
                  + ("  ⚠ 首字 >2s：多半在换页，看下面 [mem] 的 swap" if _first > 2000 else ""),
                  flush=True)
        except Exception as e:
            print(f"[web] 真生成预热失败（不影响运行）：{type(e).__name__}: {e}", flush=True)
    print(f"[mem] 全部就位 · {_mem_line()}", flush=True)
    print("[web] 可选模型："
          f"复核ASR(SenseVoice 0.9G)={'开' if os.environ.get('VOICEMEM_FINAL_ASR', '1') != '0' else '关'} "
          f"[VOICEMEM_FINAL_ASR=0] · "
          f"声学情绪(emotion2vec 1.0G)={'开' if os.environ.get('VOICEMEM_ACOUSTIC_TAG', '1') != '0' else '关'} "
          f"[VOICEMEM_ACOUSTIC_TAG=0] · "
          f"MLX缓存上限 {os.environ.get('VOICEMEM_MLX_CACHE_MB', '512')}MB", flush=True)
    _print_backchannel_status()
    print("[web] 就绪", flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
