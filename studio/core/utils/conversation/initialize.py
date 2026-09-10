"""Create state owned by one WebSocket conversation."""
import asyncio
import uuid
from studio.core.utils.dialogue.component import backchannel_policy
from studio.core.utils.turn_taking.initialize import Backchannel, TurnTakingStateMachine
from studio.core.utils.audio_timeline.component import SpeechRateEstimator

def initialize(self, agent, sock):
    self.agent = agent
    self.sock = sock
    self.turn_taking = TurnTakingStateMachine(backchannel=Backchannel(policy=backchannel_policy()), echo_window_s=self.agent.BC_ECHO_WINDOW_S)
    self.turn = {'task': None, 'continuation_task': None, 't0': 0.0, 'until': 0.0, 'echo_until': 0.0, 'speech_end': 0.0, 'play_started': False, 'reply': {'text': ''}, 'timeline': None, 'measure_started': 0.0, 'measure_recorded': False}
    self.owner = {'id': '', 'last': '', 'miss': 0}
    self.speech_rate = SpeechRateEstimator()
    self.context_session = uuid.uuid4().hex
    self.candidate_paused = False
    self.candidate_paused_at = 0.0
    self.filler_waiters: dict[str, asyncio.Event] = {}
    self.early = {'text': '', 'task': None, 'sink': None, 'timeline': None, 'pending': None, 'said': None, 'space': '', 'memory_vm': None, 'started': 0.0}
    self.prewarm = {'task': None, 'cancelled': None, 'closed': False}
    self.unfinished_wait = {'pending': None, 'task': None}
