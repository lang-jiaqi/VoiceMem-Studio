"""Studio service lifecycle and chronological conversation controls."""
import asyncio
import os
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
                await session.publish_user_input(pending)
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


def build_app(agent, demo_accounts=None):
    """Bind realtime or text/speech sessions and memory controls to the server."""
    from studio.web import transport
    from studio.core.utils.llm.initialize import credential

    async def session_for(active_agent, socket):
        if active_agent.MODE == 'realtime':
            await active_agent.realtime_session(socket)
        else:
            await converse(active_agent, socket)

    async def session(socket):
        await session_for(agent, socket)

    def components_for(active_agent):
        memory = active_agent.CONFIG["llm"]
        reply = active_agent.REPLY["llm"]
        memory_provider = memory["provider"]
        reply_provider = reply["provider"]
        return {
            "backend": active_agent.ARGS.backend,
            "mode": active_agent.MODE,
            "space": active_agent.ACTIVE_SPACE,
            "memory": {
                "provider": memory_provider,
                "configured": bool(credential(memory_provider, "memory")),
                "model": memory["config"].get("model", ""),
                "base_url": memory["config"].get("base_url", ""),
            },
            "reply": {
                "provider": reply_provider,
                "configured": reply_provider == "local" or
                bool(credential(reply_provider, "reply")),
                "model": reply["config"].get("model", ""),
                "base_url": reply["config"].get("base_url", ""),
            },
            "speech": {"provider": active_agent.REPLY["tts"]["provider"]},
        }

    return transport.build_app(
        agent.MODE, session, lambda *a, **k: agent.vm.classify(*a, **k),
        agent.memory_snapshot, agent.audio_of,
        spaces=(agent.list_spaces, agent.create_space, agent.use_space, lambda: agent.ACTIVE_SPACE),
        set_lang=agent.set_lang,
        title=transport.make_title_generator(agent.REPLY, agent.CONFIG["llm"]),
        components=lambda: components_for(agent),
        pet_port=agent.ARGS.port,
        demo_accounts=demo_accounts,
        demo_session=session_for,
        demo_components=components_for,
    )


def main(argv=None):
    """Check all prerequisites, acquire weights, warm providers, then serve."""
    tts = None
    try:
        from .utils.environment.component import load_environment, prepare
        load_environment()
        args = parse_args(argv)
        public_demo = os.environ.get('STUDIO_PUBLIC_DEMO') == '1'
        if public_demo and args.host not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('公开体验模式只监听本机；请通过 Tailscale Funnel 转发。')
        if public_demo and args.llm == 'local':
            raise ValueError('公开体验模式请使用 API 回复模型，避免每个账号加载一份本地 LLM。')
        if public_demo and args.mode == 'realtime':
            raise ValueError('公开体验模式请使用 llm_tts 模式，以保持账号记忆隔离。')
        prepare(args)
        from .utils.startup.initialize import inspect
        stage = args.prepare_stage or 'all'
        inspect(args, stage)
        if args.check:
            return
        from .utils.models.initialize import acquire_all
        acquire_all(args, stage)
        if args.prepare_stage:
            print('[startup] VoiceMem 基础模型准备完成。', flush=True)
            return
        from studio.paths import ROOT
        if not args.no_file_log:
            from .utils.logging_utils.component import setup_file_logging
            setup_file_logging(ROOT, args.log_file, concise=not args.verbose)
        from studio.core.utils.logging_utils.prompt_trace import configure
        configure(ROOT / 'prompt/logs')
        from .voiceagent import VoiceAgent
        agent = VoiceAgent(args)
        demo_accounts = None
        if public_demo:
            from studio.web.demo_accounts import DemoAccounts
            demo_accounts = DemoAccounts(args)
        if args.mode == 'llm_tts':
            tts = agent.vm.utils.get('tts')
        agent.warmup()
        app = build_app(agent, demo_accounts=demo_accounts)
        import uvicorn
        app.router.add_event_handler(
            'startup',
            lambda: print(f'[startup] VoiceMem Studio 启动成功：http://localhost:{args.port}', flush=True),
        )
        shutdown_timeout = 5 if os.environ.get('VOICEMEM_DESKTOP_INSTANCE') else None
        uvicorn.run(app, host=args.host, port=args.port,
                    timeout_graceful_shutdown=shutdown_timeout)
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        print(f'[startup] 无法启动：{exc}', flush=True)
        raise SystemExit(1) from None
    finally:
        if tts is not None and hasattr(tts, 'aclose'):
            asyncio.run(tts.aclose())
