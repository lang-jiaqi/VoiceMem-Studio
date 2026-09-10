"""Provider-neutral reply component; cancellation closes the underlying stream."""
from contextlib import aclosing

class ReplyModel:
    def __init__(self, stream):
        self.stream = stream

    async def __call__(self, text, memory_context="", history=None):
        async with aclosing(self.stream(text, memory_context, history)) as output:
            async for delta in output:
                yield delta

    async def aclose(self):
        close = getattr(self.stream, "aclose", None)
        if close:
            await close()
