"""Studio service lifecycle and chronological conversation controls."""
import asyncio
from contextlib import aclosing
from .utils.cli.component import parse_args


async def converse(agent, socket):
    """Listen, wait, interrupt, route, then release or generate one reply."""
    from .utils.conversation.component import Conversation
    from .utils.turn_taking.pause import needs_continuation
    session = Conversation(agent, socket)
    try:
        async with aclosing(session.listen()) as turns:
            async for pending in turns:
                if session.ignore(pending):
                    continue
                pending = await session.merge_continuation(pending)
                session.stop_prewarm()
                if pending.spoken and needs_continuation(pending.text):
                    await session.drop_early('等待用户补完半句')
                    await session.defer_unfinished_reply(pending)
                    continue
                ack = session.cached_ack(pending)
                if ack:
                    await session.emit_filler(*ack)
                routing = await session.route(pending)
                try:
                    if await session.commit_early(pending, routing):
                        continue
                    await session.stop_reply(force=True)
                    await session.start_reply(pending, routing)
                finally:
                    if not routing.done():
                        routing.cancel()
                    await asyncio.gather(routing, return_exceptions=True)
    finally:
        await session.close_session()


def build_app(agent):
    """Bind realtime or text/speech sessions and memory controls to the server."""
    from studio.web import transport
    async def session(socket):
        if agent.MODE == 'realtime':
            await agent.realtime_session(socket)
        else:
            await converse(agent, socket)
    return transport.build_app(
        agent.MODE, session, lambda *a, **k: agent.vm.classify(*a, **k),
        agent.memory_snapshot, agent.audio_of,
        spaces=(agent.list_spaces, agent.create_space, agent.use_space, lambda: agent.ACTIVE_SPACE),
        set_lang=agent.set_lang, title=transport.make_title_generator(agent.REPLY),
        pet_port=agent.ARGS.port,
    )


def main(argv=None):
    """Check all prerequisites, acquire weights, warm providers, then serve."""
    tts = None
    try:
        from .utils.environment.component import load_environment, prepare
        load_environment()
        args = parse_args(argv)
        prepare(args)
        from .utils.startup.initialize import inspect
        inspect(args)
        if args.check:
            return
        from .utils.models.initialize import acquire_all
        acquire_all(args)
        from studio.paths import ROOT
        if not args.no_file_log:
            from .utils.logging_utils.component import setup_file_logging
            setup_file_logging(ROOT, args.log_file, concise=not args.verbose)
        from voicemem.prompt_trace import configure
        configure(ROOT / 'prompt/logs')
        from .voiceagent import VoiceAgent
        agent = VoiceAgent(args)
        if args.mode == 'llm_tts':
            tts = agent.vm.utils.get('tts')
        agent.warmup()
        app = build_app(agent)
        import uvicorn
        app.router.add_event_handler(
            'startup',
            lambda: print(f'[startup] VoiceMem Studio 启动成功：http://localhost:{args.port}', flush=True),
        )
        uvicorn.run(app, host=args.host, port=args.port)
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        print(f'[startup] 无法启动：{exc}', flush=True)
        raise SystemExit(1) from None
    finally:
        if tts is not None and hasattr(tts, 'aclose'):
            asyncio.run(tts.aclose())
