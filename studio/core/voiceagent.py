"""Compose Studio sessions, shared state, and the browser service."""
from .utils.spaces.component import Spaces
from .utils.replay.component import Replay
from .utils.realtime_session.component import RealtimeSession
from .utils.context.component import Context
from .utils.memory.component import Memory
from .utils.routing.component import Routing
from .utils.perception.component import Perception
from .utils.reply.component import Reply
from .utils.diagnostics.component import Diagnostics
from .utils.interruption.component import Interruption
from .utils.speech.component import Speech
from .utils.capture.component import Capture
from .utils.visualization.component import Visualization
from .utils.startup.component import Startup
from .utils.runtime.component import configure

class VoiceAgent(
    Spaces, Replay, RealtimeSession, Context, Memory,
    Routing, Perception, Reply, Diagnostics, Interruption,
    Speech, Capture, Visualization, Startup,
):
    """Own application state while each utility retains its session/task boundary."""
    def __init__(self, args):
        configure(self, args)

    async def llm_tts_session(self, socket):
        from .core import converse
        await converse(self, socket)

    def build_app(self):
        from .core import build_app
        return build_app(self)
