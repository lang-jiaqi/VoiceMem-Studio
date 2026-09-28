"""Account isolation and browser access checks for the opt-in public demo."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from studio.web.demo_accounts import DemoAccounts
from studio.web.transport import build_app
from studio.core.utils.spaces.component import Spaces


class FakeAgent:
    def __init__(self, args, root):
        self.root = root
        self.ACTIVE_SPACE = "default"
        self.vm = SimpleNamespace(classify=lambda query: SimpleNamespace(slots=[], entities=[]))

    def memory_snapshot(self):
        return {"left": [{"text": self.root.parent.name}], "right": []}

    def list_spaces(self):
        return [{"id": "default", "name": "default", "count": 0}]

    def create_space(self, name, lang):
        return {"id": name, "name": name}

    def use_space(self, name):
        self.ACTIVE_SPACE = name
        return name

    def audio_of(self, memory_id):
        return None

    def set_lang(self, lang):
        return "zh"


class PublicDemoTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        args = SimpleNamespace(space="unused", lang="en", memory_root="")
        self.accounts = DemoAccounts(args, root=self.temp.name,
                                     agent_factory=lambda config, root: FakeAgent(config, root))

        async def session(agent, socket):
            await socket.receive()

        self.app = build_app(
            "llm_tts", lambda socket: None, lambda query: None,
            spaces=(lambda: [], lambda name, lang: {}, lambda name: name, lambda: ""),
            demo_accounts=self.accounts, demo_session=session,
            demo_components=lambda agent: {"space": agent.ACTIVE_SPACE},
        )

    def client(self):
        client = TestClient(self.app, base_url="https://demo.ts.net")
        self.addCleanup(client.close)
        return client

    def register(self, client, name):
        response = client.post("/auth/register", json={"name": name, "password": "long-secret-123"},
                               headers={"origin": "https://demo.ts.net"})
        self.assertEqual(response.status_code, 200, response.text)

    def test_accounts_have_separate_memory_and_login_is_required(self):
        alice = self.client()
        bob = self.client()
        self.assertEqual(alice.get("/api/memories").status_code, 401)
        self.assertEqual(alice.get("/ui/login.html").status_code, 200)
        self.assertEqual(alice.get("/ui/digital.html", follow_redirects=False).status_code, 303)
        self.register(alice, "alice")
        self.register(bob, "bob")
        self.assertEqual(alice.get("/ui/digital.html").status_code, 200)
        a = alice.get("/api/memories").json()["left"][0]["text"]
        b = bob.get("/api/memories").json()["left"][0]["text"]
        self.assertNotEqual(a, b)
        self.assertEqual(alice.get("/api/spaces").status_code, 200)
        self.assertEqual(alice.get("/api/spaces").json()["spaces"][0]["id"], "default")
        self.assertEqual(alice.post("/api/spaces", json={"name": "other"},
                                    headers={"origin": "https://demo.ts.net"}).status_code, 403)
        self.assertEqual(alice.post("/auth/logout", headers={"origin": "https://demo.ts.net"}).status_code, 200)
        self.assertEqual(alice.get("/api/memories").status_code, 401)
        self.assertEqual(alice.post("/auth/login", json={"name": "alice", "password": "long-secret-123"},
                                    headers={"origin": "https://demo.ts.net"}).status_code, 200)
        self.assertEqual(alice.get("/api/memories").status_code, 200)
        self.assertEqual(bob.get("/api/memories").status_code, 200)

    def test_origin_and_websocket_auth(self):
        client = self.client()
        self.assertEqual(client.post("/auth/register", json={"name": "alice", "password": "long-secret-123"},
                                     headers={"origin": "https://other.test"}).status_code, 403)
        with self.assertRaises(WebSocketDisconnect):
            with client.websocket_connect("/ws", headers={"origin": "https://demo.ts.net", "host": "demo.ts.net"}):
                pass
        self.register(client, "alice")
        cookie = f"vm_demo_session={client.cookies['vm_demo_session']}"
        with self.assertRaises(WebSocketDisconnect):
            with client.websocket_connect("/ws", headers={"origin": "https://other.test", "host": "demo.ts.net"}):
                pass
        with client.websocket_connect("/ws", headers={"origin": "https://demo.ts.net", "host": "demo.ts.net", "cookie": cookie}) as ws:
            self.assertEqual(ws.receive_json()["type"], "session_ready")

    def test_space_uses_private_memory_root(self):
        roots = [Path(self.temp.name) / "a", Path(self.temp.name) / "b"]
        opened = []
        for root in roots:
            space = Spaces()
            space._SPACE_ROOT = root
            space._SPACES = {}
            space.ARGS = SimpleNamespace(lang="zh", llm="deepseek")
            space.CONFIG = {"reply": {"llm": {"config": {}}}}
            space._rt_persona = lambda lang: "persona"
            with patch("studio.core.utils.spaces.component.open_memory",
                       side_effect=lambda cfg: opened.append(cfg.copy()) or object()):
                space.get_space("default")
        self.assertEqual([item["memory_root"] for item in opened],
                         [str(root / "default") for root in roots])


if __name__ == "__main__":
    unittest.main()
