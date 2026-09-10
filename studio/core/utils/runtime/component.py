from studio.harness.speaking_style.policy import RB_HUMANIZE_PROMPT, RB_HUMANIZE_EXAMPLES
"""Initialize one Studio process configuration and its shared state."""
import os
import asyncio
import json
import re
import threading
from pathlib import Path
from studio.core.utils.dialogue.component import CONTEXT
from voicemem.prompt_config import tts_prompts
from studio.web import transport as utils
from studio.core.utils.session_context.component import SessionBuffer
from voicemem import gate
from studio.paths import ROOT as _ROOT

def configure(self, args):
    self.ARGS = args
    self.BARGE_DEBUG = args.verbose
    self.BARGE_THRESHOLD = 0.45
    self.BARGE_MIN_CHARS = 2
    self.BARGE_STABLE_UPDATES = 2
    self.BARGE_REJECT_SILENCE_MS = 220
    self.BARGE_CANDIDATE_TIMEOUT_MS = 1200
    self.BARGE_GRACE_MS = 500
    self.TURN_DETECTION = 'semantic_vad'
    self.VAD_EAGERNESS = 'low'
    self.MIC_RATE = 24000
    self.SPEAKER_GATE = True
    self.STRANGER_MIN_TURNS = 1
    self.SPEAKER_DEBUG = args.verbose
    self.MODE = self.ARGS.mode
    from studio.core.utils.turn_taking import backchannel
    backchannel.ENABLED = args.backchannel
    backchannel.DEBUG = args.verbose
    self.SPEC_MIN_CHARS = self.ARGS.spec_min_chars
    self.GAMBLE_S = self.ARGS.gamble_ms / 1000
    self.CONFIRM_S = self.ARGS.confirm_ms / 1000
    self.UI_LANG = self.ARGS.lang
    self.SPACE_LANG = 'en'
    self._SOUND_WORDS = ('歌', '曲', '调子', '旋律', '音乐', '哼', '那段声音', '放来听', '放给我听', '播一下', '什么声音')
    self._REPLAY_UNTIL = 0.0
    self._LAST_TUNE: dict = {'path': '', 'at': 0.0}
    self.LAST_TUNE_ID = 'last:tune'
    self.LAST_TUNE_TTL_S = 1800.0
    self._DAY_WORDS = ((('大前天',), -3), (('前天',), -2), (('昨天', '昨晚', '昨儿'), -1), (('今天', '今早', '今晚', '今日'), 0))
    self._HOUR_WORDS = ((('凌晨', '半夜', '深夜'), (0, 6)), (('早上', '早晨', '今早', '一早', '清晨'), (6, 10)), (('上午',), (8, 12)), (('中午', '晌午'), (11, 14)), (('下午',), (12, 18)), (('傍晚', '黄昏'), (17, 20)), (('晚上', '晚间', '昨晚', '今晚', '夜里'), (18, 24)))
    self._WEEKDAYS = (('周一', '星期一', '礼拜一'), ('周二', '星期二', '礼拜二'), ('周三', '星期三', '礼拜三'), ('周四', '星期四', '礼拜四'), ('周五', '星期五', '礼拜五'), ('周六', '星期六', '礼拜六'), ('周日', '周天', '星期日', '星期天', '礼拜天'))
    self._PLACE_WORDS = ((('咖啡馆', '咖啡店', '咖啡厅', '星巴克'), 'café'), (('办公室', '公司', '工位', '单位'), 'office'), (('家里', '家中', '在家', '屋里', '房间'), 'home'), (('外面', '户外', '外边', '路上', '街上', '公园'), 'outdoor'), (('车上', '地铁', '公交', '路上', '通勤', '火车'), 'transit'), (('会议', '开会', '会上', '会议室'), 'meeting'))
    self._ORDINALS = ((('第一首', '第一段', '第1首', '头一首', '最早那首', '最先那首'), 1), (('第二首', '第二段', '第2首'), 2), (('第三首', '第三段', '第3首'), 3), (('第四首', '第四段', '第4首'), 4), (('第五首', '第五段', '第5首'), 5))
    self._ORDINALS_BACK = ((('最后一首', '最后那首', '最后一段', '最新那首', '最近那首'), -1), (('上一首', '前一首', '上一段', '前一段', '前面那首', '上一个'), -2))
    self.TUNE_GAP_S = 90.0
    self.GROUP_ID_PREFIX = 'group:'
    self._STITCH_CACHE: dict = {}
    self._REPLAY_NOTE = CONTEXT['replay']
    self._NO_REPLAY_NOTE = CONTEXT['no_replay']
    self._TONE = tts_prompts()['fallback_by_user_emotion']
    self._STATE_LABEL = CONTEXT['state_label']
    self._SPEAK_BASE = tts_prompts()['base']
    self._speak_base_env = ''
    self._LAST_TONE = {'tag': ''}
    self._HISTORY_CHARS = 200
    self._SESSION_CONTEXT = SessionBuffer(text_limit=self._HISTORY_CHARS)
    from studio.core.utils.llm.initialize import configuration
    provider = args.llm if args.mode == "llm_tts" else "openai"
    self.CONFIG = {
        "mode": "multi_modal", "memory_root": args.memory_root,
        "space": args.space, "embedding": {"provider": "local"},
        "slots": {"provider": "local"},
        "reply": {"llm": configuration(provider),
                  "tts": {"provider": "breeze_cuda" if args.backend == "cuda" else "breeze_mlx", "config": {}},
                  "realtime": {"provider": "openai", "config": {"model": "gpt-realtime"}}},
    }
    # Memory extraction and cleanup must use the same provider as visible reply.
    self.CONFIG["llm"] = self.CONFIG["reply"]["llm"]
    self.REPLY = self.CONFIG["reply"]
    self._LOCAL_LLM = None
    self._SPACES: dict = {}
    self.ACTIVE_SPACE = ''
    self.vm = None
    self.use_space(self.ARGS.space)
    self.TURN_AUDIO_DIR = _ROOT / 'results' / 'turn_audio'
    self._THINKING_ROUTER_ON = True
    self.ACOUSTIC_MIN_SCORE = 0.92
    self.ACOUSTIC_TRUST = set(('开心,委屈,惊讶').split(','))
    self._EMO_PROTO = {'开心': ['我今天特别开心', '太好了我很高兴', '真不错，我挺满意的', '哈哈太有意思了'], '悲伤': ['我很难过', '我心里特别难受', '我好失落', '这事让我挺沮丧的'], '委屈': ['我好生气', '太气人了', '凭什么这样对我', '我觉得很不公平'], '焦虑': ['我压力好大', '我有点紧张', '我很担心做不完', '这事儿让我睡不着'], '疲惫': ['我好累啊', '累死了，撑不住了', '一天下来人都空了'], '平静': ['今天天气不错', '我明天要去开会', '早上好', '我叫小明', '这个东西放在桌上', '我对花生过敏', '我在一家公司上班', '下周三下午三点有个会', '我不能吃什么', '这个怎么用', '帮我看一下', '我住在市中心']}
    self._PROTO = {}
    self._E2V_MAP = {'happy': '开心', 'sad': '悲伤', 'angry': '委屈', 'fearful': '恐惧', 'surprised': '惊讶', 'disgusted': '厌恶', 'neutral': ''}
    self._E2V = {}
    self._SV = {}
    self._LAT_HIST = {'n': 0, 'first': None, 'rest': []}
    self._REPLY_DISPLAY_LOCK = threading.Lock()
    self._HOT = {'n': 0}
    self._IDLE = asyncio.Event()
    self._IDLE.set()
    self._IDLE_MAX_WAIT_S = 8.0
    self._REMEMBER_LOCK = asyncio.Lock()
    self._REMEMBER_TASKS: set[asyncio.Task] = set()
    self.ECHO_WINDOW = 300
    self.ECHO_RATIO = 0.6
    self.ECHO_FUZZY_MIN = 4
    self.BACKCHANNEL_ON = True
    self._bc_norm = gate.norm
    self._INTERRUPT_PREFIXES = tuple((self._bc_norm(x) for x in ('停', '停一下', '先停', '暂停', '等等', '等一下', '先别说', '别说了', '打住', 'stop', 'wait', 'hold on', 'pause', 'quiet')))
    self._FILLER_PREFIX = re.compile('^[嗯呃啊哦噢喔欸诶唉哈哼]+')
    self._BC_VOICE = {'obj': None, 'task': None}
    self._TTS_SHARED_VOICES = {'alloy', 'ash', 'ballad', 'coral', 'echo', 'sage', 'shimmer', 'verse'}
    self._SLOW_TTS = {'VoxCPMTTS'}
    self._LOCAL_TTS = {'QwenTTS', 'KokoroTTS', 'VoxCPMTTS', 'PiperTTS', 'BreezeMLXTTS'}
    self._EOT = {'obj': None, 'tried': False}
    self.EARLY_EOT = 0.5
    self.BC_ECHO_WINDOW_S = 3.0
    self.CANDIDATE_MIN_SPEECH_S = 0.12
    self.BC_QUIET_RATIO = 0.25
    self.BC_AFTER_EARLY_S = 1.0
    self.HISTORY_TURNS = 6
    self.SLOT_TO_CLUSTER = {'情绪': 'emotion', '情感记录': 'emotion', '内心OS': 'emotion', '喜好与厌恶': 'preference', '思维模式': 'preference', '应对方式': 'experiences', '表达风格': 'experiences', '人物地点态度': 'experiences', '避免重复': 'experiences'}
    self._CALM = ('', '平静', '中性')
    self.RB_ENTITIES_PER_SLOT = 6
    self.LB_ENTRIES_PER_SLOT = 7
    self._OWNER_NAME_CACHE: dict = {}
    self._BAD_NAME_CHARS = '什谁哪啥吗呢么?？'
    self._RB_HUMAN: dict = {}
    self._RB_HUMAN_PENDING: set = set()
    self.RB_HUMANIZE = True
    self._RB_HUMANIZE_PROMPT = RB_HUMANIZE_PROMPT
    self._RB_HUMANIZE_EXAMPLES = RB_HUMANIZE_EXAMPLES
    self._LAST_HIT_IDS: set = set()
