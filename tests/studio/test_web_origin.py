"""Browser origins must be checked before transport callbacks run."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from studio.web.origin import same_origin
from studio.web.transport import build_app


class BrowserOriginTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('studio.web.transport.PetSupervisor'))
        self.entered = Mock()
        self.selected = Mock(side_effect=lambda name: name)

        async def session(socket):
            self.entered()
            await socket.send_json({'type': 'synthetic_fixture'})
            await socket.receive()

        self.app = build_app('llm_tts', session, lambda text: None,
                             spaces=(lambda: [], lambda *args: {}, self.selected, lambda: 'fixture'))
        self.client = self.enterContext(TestClient(self.app))

    def test_foreign_opaque_and_malformed_origins_never_enter_websockets(self):
        for route in ('/ws', '/ws-pet', '/ws/memories'):
            for origin in ('https://review.invalid', 'null', 'http://testserver:invalid',
                           'http://testserver.evil', 'http://testserver/path',
                           ' http://testserver', 'http://testserver\n'):
                with self.subTest(route=route, origin=origin):
                    with self.assertRaises(WebSocketDisconnect) as raised:
                        with self.client.websocket_connect(route, headers={'Origin': origin}):
                            self.fail('foreign origin accepted')
                    self.assertEqual(raised.exception.code, 1008)
        self.entered.assert_not_called()

    def test_same_origin_and_native_clients_keep_session_contract(self):
        for headers in ({}, {'Origin': 'http://testserver'}, {'Origin': 'http://testserver:80'}):
            with self.subTest(headers=headers), self.client.websocket_connect('/ws', headers=headers) as socket:
                self.assertEqual(socket.receive_json()['type'], 'session_ready')
                self.assertEqual(socket.receive_json()['type'], 'synthetic_fixture')
        self.assertEqual(self.entered.call_count, 3)
        for headers in ({}, {'Origin': 'http://testserver'}):
            for route in ('/ws-pet', '/ws/memories'):
                with self.subTest(route=route), self.client.websocket_connect(route, headers=headers):
                    pass

    def test_cross_site_post_cannot_switch_space_even_without_a_json_body(self):
        response = self.client.post('/api/spaces/other/use', headers={'Origin': 'https://review.invalid'})
        self.assertEqual(response.status_code, 403)
        self.selected.assert_not_called()
        for headers in ({}, {'Origin': 'http://testserver'}):
            response = self.client.post('/api/spaces/other/use', headers=headers)
            self.assertEqual(response.status_code, 200)
        self.assertEqual(self.selected.call_count, 2)

    def test_origin_comparison_handles_tls_ports_and_ipv6(self):
        for base, origin, accepted in (
                ('wss://service.example/', 'https://service.example:443', True),
                ('ws://[::1]:8787/', 'http://[::1]:8787', True),
                ('ws://localhost:8787/', 'http://localhost:8788', False),
                ('wss://service.example/', 'http://service.example', False),
                ('ws://localhost:8787/', 'http://name@localhost:8787', False)):
            with self.subTest(base=base, origin=origin):
                connection = SimpleNamespace(base_url=base, headers={'origin': origin})
                self.assertEqual(same_origin(connection), accepted)


if __name__ == '__main__':
    unittest.main()
