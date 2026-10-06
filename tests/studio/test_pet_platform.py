"""Offline checks for desktop-pet process ownership by platform."""
import unittest
from unittest.mock import MagicMock, patch

from studio.web.pet_bridge import PetSupervisor, PetHub, TeeSocket


class PetPlatformTests(unittest.TestCase):
    def test_linux_and_wsl_do_not_spawn_desktop_processes(self):
        for enabled in ('0', '1'):
            with self.subTest(enabled=enabled), \
                    patch('studio.web.pet_bridge.sys.platform', 'linux'), \
                    patch.dict('os.environ', {'STUDIO_DESKTOP_PET': enabled}), \
                    patch('studio.web.pet_bridge.subprocess.Popen') as spawn:
                supervisor = PetSupervisor()
                supervisor.ensure_running('ws://127.0.0.1:8787/ws-pet')
                supervisor.shutdown()
                spawn.assert_not_called()

    def test_native_mac_keeps_optional_standalone_pet(self):
        for enabled in ('0', '1'):
            with self.subTest(enabled=enabled), \
                    patch('studio.web.pet_bridge.sys.platform', 'darwin'), \
                    patch.dict('os.environ', {'STUDIO_DESKTOP_PET': enabled}), \
                    patch('studio.web.pet_bridge.shutil.which', return_value='/fixture/node'), \
                    patch('studio.web.pet_bridge.subprocess.Popen') as spawn, \
                    patch('builtins.print'):
                spawn.return_value.poll.return_value = None
                PetSupervisor().ensure_running('ws://127.0.0.1:8787/ws-pet')
                self.assertEqual(spawn.call_count, int(enabled))


class PetObserverTests(unittest.IsolatedAsyncioTestCase):
    async def test_linux_still_broadcasts_to_remote_desktop_pets(self):
        from unittest.mock import AsyncMock
        hub = PetHub()
        observer = MagicMock()
        observer.send_text = AsyncMock()
        hub.add(observer)
        await hub.broadcast({'type': 'backchannel'})
        observer.send_text.assert_awaited_once_with('{"type": "backchannel"}')

    async def test_each_backchannel_gets_a_unique_pet_event(self):
        from unittest.mock import AsyncMock
        sock, hub = MagicMock(), MagicMock()
        sock.send_json = AsyncMock()
        hub.broadcast = AsyncMock()
        tee = TeeSocket(sock, hub, session_id='session')

        await tee.send_json({'type': 'backchannel', 'token': '嗯'})
        await tee.send_json({'type': 'backchannel', 'token': '嗯'})

        first, second = [call.args[0] for call in hub.broadcast.await_args_list]
        self.assertNotEqual(first['event_id'], second['event_id'])
        self.assertEqual(first['session_id'], 'session')


if __name__ == '__main__':
    unittest.main()
