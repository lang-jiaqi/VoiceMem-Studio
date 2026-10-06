"""Synthetic conversation, playback reports and memory callbacks for Studio tests."""
import asyncio
import types
from unittest.mock import AsyncMock, Mock

from studio.core.utils.conversation.component import Conversation
from studio.core.utils.contracts.component import Pending
from studio.core.utils.session_context.component import SessionBuffer
from voicemem.stream import empty_result
from tests.helpers.reply import ReplyFixture


class ConversationFixture:
    def setUp(self):
        f = ReplyFixture()
        f.setUp()
        self.agent = f.agent
        self.tts_instructions = []
        self.agent.BC_ECHO_WINDOW_S = 4
        self.agent.MIC_RATE = 24000
        self.agent.space_language = lambda _: 'zh'
        self.agent.BARGE_GRACE_MS = 500
        self.agent._LOCAL_LLM = None
        self.agent.route_pending_thinking = AsyncMock()
        self.agent.voicemem_llm_tts = self.agent._voicemem_llm_tts
        self.context = self.agent._SESSION_CONTEXT = SessionBuffer(text_limit=200)
        self.agent._push_history = Mock(side_effect=lambda *a, **k: self.context.add(*a, **k))
        self.memory = self.agent.vm
        self.messages, self.audio = [], []
        self.audio_ready = asyncio.Event()
        self.model_waiting = asyncio.Event()
        self.release_model = asyncio.Event()
        self.auto_playback = True
        self.block_model = False
        self.model_calls = 0
        self.model_args = []
        self.model_closed = asyncio.Event()

        async def model(*_):
            self.model_calls += 1
            self.model_args.append(_)
            try:
                yield '认真|这是生成的回复，后面还有更多说明。'
                self.model_waiting.set()
                if self.block_model:
                    await self.release_model.wait()
            finally:
                self.model_closed.set()

        async def tts(_text, instruction=None):
            self.tts_instructions.append(instruction)
            yield bytes(48000)

        self.memory.reply_stream = model
        self.memory.utils = types.SimpleNamespace(get=lambda _: types.SimpleNamespace(stream=tts))
        self.sock = types.SimpleNamespace(send_json=self.send, send_bytes=self.send_audio)
        self.session = Conversation(self.agent, self.sock)
        self.session.turn_taking.work_filler_enabled = False

    async def asyncTearDown(self):
        await self.session.close_session()

    def pending(self, text='最终确认的输入', **kwargs):
        return Pending(text, '', empty_result(), route='shallow', transcript_managed=True, **kwargs)

    async def send(self, message):
        self.messages.append(message)
        if message['type'] == 'answer_done' and self.auto_playback:
            timeline = self.session.turn['timeline']
            await self.session.playback_checkpoint(dict(
                output_id=message['output_id'], rendered_samples=timeline.sent_samples,
                sample_rate=24000, state='drained'))

    async def send_audio(self, pcm):
        self.audio.append(pcm)
        self.audio_ready.set()

    async def start(self, value=None):
        value = value or self.pending()
        await self.session.start_reply(value, asyncio.create_task(asyncio.sleep(0)))
        return self.session.turn['task']

    async def early(self):
        st = types.SimpleNamespace(memory=empty_result(), route='shallow', eot_score=.99)
        await self.session.start_early('设备重起怎么检查？', st)
        sink, timeline = self.session.early['sink'], self.session.early['timeline']
        await asyncio.wait_for(sink.wait_for_audio(), 1)
        return sink, timeline, self.session.early['task']

