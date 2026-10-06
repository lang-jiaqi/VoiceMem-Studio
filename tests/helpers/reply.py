"""Synthetic reply provider, output timeline and PCM sinks for Studio tests."""
import asyncio
import types
from unittest.mock import Mock

from studio.core.utils.audio_timeline.component import AudioTimeline
from studio.core.utils.contracts.component import Pending
from studio.core.utils.reply.component import Reply
from voicemem.stream import empty_result


class ReplyFixture:
    def setUp(self):
        self.messages, self.audio, self.tts_text = [], [], []
        self.tts_instructions = []
        self.timeline = AudioTimeline()
        self.agent = Reply()
        self.agent.ACTIVE_SPACE = 'fixture'
        self.agent._SESSION_CONTEXT = types.SimpleNamespace(messages=lambda *a, **k: [])
        self.agent.HISTORY_TURNS = 4
        self.agent._speak_instruction = lambda _, lang='': ''
        self.agent._speak_base_env = ''
        self.agent._SPEAK_BASE = {}
        self.agent._by_lang = lambda _, lang='': ''
        self.agent._LAST_TONE = {'tag': ''}
        self.agent.build_reply_context = lambda *a, **k: ''
        self.agent.BARGE_DEBUG = False
        self.agent._lat_note = lambda _: 'fixture'
        self.agent._mem_line = lambda: 'fixture'
        self.agent.hot_path_enter = lambda: None
        self.agent.hot_path_exit = lambda _: None
        self.agent._kick_acoustic = Mock(return_value=None)
        self.agent._push_history = Mock(return_value='fixture-history')
        self.agent.queue_remember_turn = Mock()

        async def tts(text, instruction=None):
            self.tts_text.append(text)
            self.tts_instructions.append(instruction)
            yield bytes(480)
        self.agent.vm = types.SimpleNamespace(
            utils=types.SimpleNamespace(get=lambda _: types.SimpleNamespace(stream=tts)))
        self.pending = Pending('合成测试问题', '', empty_result(),
                               route='shallow', reply_mode='direct', transcript_managed=True)

    async def send(self, message):
        self.messages.append(message)
        if message['type'] == 'answer_done':
            self.timeline.update_checkpoint(self.timeline.sent_samples, 24000, 'drained')

    async def send_audio(self, pcm):
        self.audio.append(pcm)

    def pipeline(self, model, sink=None, **kwargs):
        self.agent.vm.reply_stream = model
        return self.agent._voicemem_llm_tts(
            self.pending, sink.send if sink else self.send,
            sink.send_audio if sink else self.send_audio, {}, self.timeline,
            context_session='fixture-session', context_space='fixture',
            **kwargs)

    async def complete(self, chunks, **kwargs):
        async def model(*_):
            for chunk in chunks:
                yield chunk
        await asyncio.wait_for(self.pipeline(model, **kwargs), 1)

