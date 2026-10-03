"""Account isolation and browser access checks for the opt-in public demo."""
import asyncio
import gzip
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from html.parser import HTMLParser
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urljoin

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from studio.web.demo_accounts import DemoAccounts
from studio.web.transport import build_app
from studio.web.pet_assets import (
    CORE_CDN, CORE_PATH, MODEL_PATH, MobilePetAssets, MOBILE_FILES, PET_SCRIPTS, PET_VENDORS,
)
from studio.core.utils.spaces.component import Spaces
from studio.core.utils.conversation.component import Conversation
from studio.core.utils.self_harness.component import SelfHarnessState


class FakeAgent:
    def __init__(self, args, root):
        self.root = root
        self.ACTIVE_SPACE = "default"
        self.SPEAKER_GATE = True
        self.UI_LANG = args.lang
        self.initial_language = args.lang
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
        self.UI_LANG = "en" if lang == "en" else "zh"
        return self.UI_LANG


class PageResources(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.preloads, self.scripts = [], []
        self.feed(html)

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if tag == "link" and values.get("rel") == "preload":
            self.preloads.append(values)
        if tag == "script" and "src" in values:
            self.scripts.append(values)


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

    def test_account_agent_preserves_single_owner_speaker_gate(self):
        _, token = self.accounts.register("alice", "long-secret-123")
        agent = self.accounts.agent(self.accounts.user(token)[0])
        self.assertTrue(agent.SPEAKER_GATE)

    def test_same_account_websocket_leases_survive_one_connection_closing(self):
        client = self.client()
        self.register(client, 'fixture-user')
        token = client.cookies['vm_demo_session']
        user_id = self.accounts.user(token)[0]
        headers = {'origin':'https://demo.ts.net', 'host':'demo.ts.net',
                   'cookie':f'vm_demo_session={token}'}
        with client.websocket_connect('/ws', headers=headers) as first:
            first.receive_json()
            agent = self.accounts._agents[user_id]
            with client.websocket_connect('/ws', headers=headers) as second:
                second.receive_json()
                self.assertEqual(self.accounts._active_agents[user_id], 2)
                self.assertEqual(self.accounts.prune_idle(idle_seconds=0, max_idle_agents=0), 0)
            self.assertEqual(self.accounts._active_agents[user_id], 1)
            self.assertIs(self.accounts._agents[user_id], agent)
        self.assertEqual(self.accounts._active_agents[user_id], 0)

    def test_failed_http_request_releases_account_lease(self):
        client = self.client()
        self.register(client, 'fixture-user')
        user_id = self.accounts.user(client.cookies['vm_demo_session'])[0]
        agent = self.accounts.agent(user_id)
        with patch.object(agent, 'memory_snapshot', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                client.get('/api/memories')
        self.assertEqual(self.accounts._active_agents[user_id], 0)

    def test_ui_language_update_keeps_other_accounts_and_space_state(self):
        alice, bob = self.client(), self.client()
        self.assertEqual(alice.post('/api/lang', json={'lang': 'zh'},
                                   headers={'origin': 'https://demo.ts.net'}).status_code, 401)
        self.register(alice, 'alice')
        self.register(bob, 'bob')
        agent = self.accounts.agent(self.accounts.user(alice.cookies['vm_demo_session'])[0])
        other = self.accounts.agent(self.accounts.user(bob.cookies['vm_demo_session'])[0])
        original_memory = agent.vm
        response = alice.post('/api/lang', json={'lang': 'zh'},
                              headers={'origin': 'https://demo.ts.net'})
        self.assertEqual(response.json(), {'lang': 'zh', 'reply_lang': 'auto'})
        self.assertEqual(agent.UI_LANG, 'zh')
        self.assertEqual(other.UI_LANG, 'en')
        self.assertEqual(agent.initial_language, 'en')
        self.assertIs(agent.vm, original_memory)
        self.assertEqual(agent.ACTIVE_SPACE, 'default')

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

    def test_mobile_pet_route_and_private_harness_preferences(self):
        alice = self.client()
        bob = self.client()
        self.register(alice, "alice")
        self.register(bob, "bob")
        mobile = {"user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)"}
        self.assertEqual(alice.get("/", headers=mobile, follow_redirects=False).headers["location"],
                         "/ui/pet-mobile.html")
        self.assertEqual(alice.get("/ui/digital.html", headers=mobile, follow_redirects=False).headers["location"],
                         "/ui/pet-mobile.html")
        self.assertEqual(alice.get("/ui/pet-mobile.html").status_code, 200)
        self.assertEqual(alice.get("/pet/avatar-controller.js").status_code, 200)
        self.assertEqual(alice.get("/pet/vendor/pixi.min.js").status_code, 200)
        self.assertEqual(self.client().get("/pet/avatar-controller.js").status_code, 401)
        alice_id = self.accounts.user(alice.cookies["vm_demo_session"])[0]
        bob_id = self.accounts.user(bob.cookies["vm_demo_session"])[0]
        self.accounts.save_harness(alice_id, "default", {"profile": {"speaking_style": {"speech_rate": "slow"}}, "persona": "先听我说"})
        self.assertEqual(self.accounts.load_harness(alice_id)["persona"], "先听我说")
        self.assertEqual(self.accounts.load_harness(bob_id), {})

    def test_mobile_fixed_resources_are_versioned_and_cacheable(self):
        client = self.client()
        self.register(client, "alice")
        page = client.get("/ui/pet-mobile.html")
        self.assertEqual(page.headers["cache-control"], "no-store")
        prefix = re.search(r"/pet/v/[0-9a-f]{16}", page.text).group()
        self.assertIn(f"new URL('{prefix}/',location.href)", page.text)
        for path in ("avatar-controller.js", "ui/studio-client.js", "ui/pet-mobile.js",
                     "vendor/pixi.min.js", "vendor/cubism4.min.js", "ui/pet-mobile.css",
                     "ui/markdown.js", "ui/markdown.css", "ui/vendor/katex/katex.min.js",
                     "ui/vendor/katex/katex.min.css", "ui/vendor/katex/fonts/KaTeX_Main-Regular.woff2"):
            with self.subTest(path=path):
                url = f"{prefix}/{path}"
                response = client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertIn("private", response.headers["cache-control"])
                self.assertIn("max-age=31536000", response.headers["cache-control"])
                self.assertIn("immutable", response.headers["cache-control"])
                head = client.head(url)
                self.assertEqual(head.status_code, 200)
                self.assertFalse(head.content)
                self.assertEqual(head.headers["etag"], response.headers["etag"])
                self.assertEqual(head.headers["content-length"], response.headers["content-length"])
                cached = client.get(url, headers={"if-none-match": response.headers["etag"]})
                self.assertEqual(cached.status_code, 304)
                self.assertFalse(cached.content)
                self.assertEqual(cached.headers["cache-control"], response.headers["cache-control"])
        css = client.get(f"{prefix}/ui/pet-mobile.css").text
        self.assertIn(f"url('{prefix}/assets/scene/call-background.png')", css)
        self.assertNotIn("url('/pet/assets/", css)

    def test_missing_versioned_runtime_is_not_cached_as_success(self):
        client = self.client()
        self.register(client, "alice")
        prefix = re.search(r"/pet/v/[0-9a-f]{16}", client.get("/ui/pet-mobile.html").text).group()
        original = Path.is_file
        with patch.object(Path, "is_file", lambda path: False if path.name == "pixi.min.js" else original(path)):
            response = client.get(f"{prefix}/vendor/pixi.min.js")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_mobile_math_runtime_and_font_urls_use_local_versioned_assets(self):
        client = self.client()
        self.register(client, "alice")
        page = client.get("/ui/pet-mobile.html")
        prefix = re.search(r"/pet/v/[0-9a-f]{16}", page.text).group()
        scripts = [entry["src"] for entry in PageResources(page.text).scripts]
        self.assertLess(scripts.index(f"{prefix}/ui/vendor/katex/katex.min.js"),
                        scripts.index(f"{prefix}/ui/markdown.js"))
        self.assertLess(scripts.index(f"{prefix}/ui/markdown.js"),
                        scripts.index(f"{prefix}/ui/pet-mobile.js"))
        css = client.get(f"{prefix}/ui/vendor/katex/katex.min.css").text
        fonts = set(re.findall(r"url\((fonts/[^)]+)\)", css))
        self.assertTrue(fonts)
        self.assertNotIn("https://", css)
        for font in fonts:
            response = client.get(f"{prefix}/ui/vendor/katex/{font}")
            self.assertEqual(response.status_code, 200, font)
            self.assertIn("immutable", response.headers["cache-control"])
            self.assertGreater(len(response.content), 1024, font)

    def test_first_visit_preloads_match_authenticated_runtime_and_model_urls(self):
        client = self.client()
        self.register(client, "alice")
        page = client.get("/ui/pet-mobile.html")
        prefix = re.search(r"/pet/v/[0-9a-f]{16}", page.text).group()
        resources = PageResources(page.text)
        self.assertEqual(len(resources.scripts), 9)
        self.assertTrue(all("defer" in script for script in resources.scripts))
        links = {item["href"]: item for item in resources.preloads}
        model_url = f"{prefix}/{MODEL_PATH}"
        refs = client.get(model_url).json()["FileReferences"]
        for name in (None, refs["Moc"], refs["Physics"], *refs["Textures"]):
            url = urljoin(model_url, name) if name else model_url
            self.assertIn(url, links)
            self.assertIn("crossorigin", links[url])
            self.assertEqual(links[url]["as"], "image" if name in refs["Textures"] else "fetch")
            self.assertEqual(client.head(url).status_code, 200)
        for name in PET_VENDORS:
            self.assertEqual(links[f"{prefix}/vendor/{name}"]["as"], "script")
        custom = PageResources(client.get("/ui/pet-mobile.html?model=/pet/assets/custom.model3.json").text)
        self.assertFalse(any(item["as"] in {"fetch", "image"} for item in custom.preloads))
        self.assertEqual(len(custom.preloads), 3)

    def test_gzip_is_lossless_prepared_once_and_has_separate_cache_validators(self):
        client = self.client()
        self.register(client, "alice")
        prefix = re.search(r"/pet/v/[0-9a-f]{16}", client.get("/ui/pet-mobile.html").text).group()
        paths = (f"{prefix}/assets/live2d/rattan/rattan.moc3", f"{prefix}/vendor/pixi.min.js",
                 f"{prefix}/ui/pet-mobile.css", f"{prefix}/live2d-renderer.js")
        with patch("studio.web.pet_assets.gzip.compress", side_effect=AssertionError("request compression")):
            for url in paths:
                with self.subTest(url=url):
                    original = client.get(url, headers={"accept-encoding": "identity"})
                    self.assertNotIn("content-encoding", original.headers)
                    with client.stream("GET", url, headers={"accept-encoding": "gzip"}) as encoded:
                        raw = b"".join(encoded.iter_raw())
                        headers = encoded.headers
                    self.assertEqual(headers["content-encoding"], "gzip")
                    self.assertEqual(headers["vary"], "Accept-Encoding")
                    self.assertEqual(int(headers["content-length"]), len(raw))
                    self.assertEqual(gzip.decompress(raw), original.content)
                    self.assertLess(len(raw), len(original.content) * .9)
                    self.assertNotEqual(headers["etag"], original.headers["etag"])
                    head = client.head(url, headers={"accept-encoding": "gzip"})
                    self.assertFalse(head.content)
                    self.assertEqual(head.headers["content-length"], str(len(raw)))
                    cached = client.get(url, headers={"accept-encoding": "gzip",
                                                      "if-none-match": f'W/{headers["etag"]}, "other"'})
                    self.assertEqual(cached.status_code, 304)
                    self.assertFalse(cached.content)
                    self.assertEqual(cached.headers["vary"], "Accept-Encoding")
                    self.assertEqual(client.get(url, headers={"accept-encoding": "gzip",
                                                              "if-none-match": original.headers["etag"]}).status_code, 200)
                    self.assertEqual(client.get(url, headers={"accept-encoding": "identity",
                                                              "if-none-match": headers["etag"]}).status_code, 200)

    def test_encoding_negotiation_range_requests_and_errors_keep_original_semantics(self):
        client = self.client()
        self.register(client, "alice")
        prefix = re.search(r"/pet/v/[0-9a-f]{16}", client.get("/ui/pet-mobile.html").text).group()
        url = f"{prefix}/assets/live2d/rattan/rattan.moc3"
        for encoding in ("identity", "gzip;q=0, *;q=1", "gzip;q=invalid", "br"):
            with self.subTest(encoding=encoding):
                self.assertNotIn("content-encoding", client.head(url, headers={"accept-encoding": encoding}).headers)
        for encoding in ("gzip;q=0.5", "br, GZIP;q=1", "*;q=1"):
            self.assertEqual(client.head(url, headers={"accept-encoding": encoding}).headers["content-encoding"], "gzip")
        original = client.get(url, headers={"accept-encoding": "identity"}).content
        response = client.get(url, headers={"accept-encoding": "gzip", "range": "bytes=0-15"})
        self.assertEqual(response.status_code, 206)
        self.assertNotIn("content-encoding", response.headers)
        self.assertEqual(response.content, original[:16])
        texture = client.head(f"{prefix}/assets/live2d/rattan/rattan.2048/texture_00.png")
        self.assertNotIn("content-encoding", texture.headers)
        for path in ("/api/memories", "/auth/me", "/ui/pet-mobile.html", f"{prefix}/assets/missing.moc3"):
            self.assertNotIn("content-encoding", client.get(path).headers)
        client.post("/auth/logout", headers={"origin": "https://demo.ts.net"})
        denied = client.get(url, headers={"accept-encoding": "gzip"})
        self.assertEqual(denied.status_code, 401)
        self.assertNotIn("content-encoding", denied.headers)
        self.assertEqual(denied.headers["cache-control"], "no-store")

    def test_model_relative_resources_stay_inside_versioned_cache(self):
        client = self.client()
        self.register(client, "alice")
        prefix = re.search(r"/pet/v/[0-9a-f]{16}", client.get("/ui/pet-mobile.html").text).group()
        model_url = f"{prefix}/assets/live2d/rattan/rattan.model3.json"
        response = client.get(model_url)
        self.assertEqual(response.status_code, 200)
        refs = response.json()["FileReferences"]
        for path in [refs["Moc"], refs["Physics"], *refs["Textures"], refs["Expressions"][0]["File"]]:
            url = urljoin(model_url, path)
            self.assertTrue(url.startswith(f"{prefix}/assets/"))
            resource = client.head(url)
            self.assertEqual(resource.status_code, 200)
            self.assertIn("immutable", resource.headers["cache-control"])
            self.assertEqual(client.head(url, headers={"if-none-match": resource.headers["etag"]}).status_code, 304)

    def test_cache_does_not_include_account_data_or_bypass_login(self):
        client = self.client()
        self.register(client, "alice")
        prefix = re.search(r"/pet/v/[0-9a-f]{16}", client.get("/ui/pet-mobile.html").text).group()
        url = f"{prefix}/avatar-controller.js"
        etag = client.get(url).headers["etag"]
        for path in ("/auth/me", "/api/memories", "/ui/pet-mobile.html", "/ui/studio-client.js"):
            self.assertEqual(client.get(path).headers["cache-control"], "no-store")
        response = client.post("/auth/logout", headers={"origin": "https://demo.ts.net"})
        self.assertEqual(response.headers["cache-control"], "no-store")
        denied = client.get(url, headers={"if-none-match": etag})
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(denied.headers["cache-control"], "no-store")
        self.assertEqual(client.get(f"{prefix}/assets/live2d/rattan/rattan.model3.json").status_code, 401)

    def test_versioned_routes_preserve_allowlists_and_reject_old_versions(self):
        client = self.client()
        self.register(client, "alice")
        prefix = re.search(r"/pet/v/[0-9a-f]{16}", client.get("/ui/pet-mobile.html").text).group()
        for path in (f"{prefix}/main.cjs", f"{prefix}/ui/settings.js", f"{prefix}/vendor/package.json",
                     f"{prefix}/assets/%2e%2e/main.cjs", "/pet/v/old/avatar-controller.js"):
            with self.subTest(path=path):
                response = client.get(path)
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.headers["cache-control"], "no-store")

    def test_explicit_harness_choices_survive_a_new_session(self):
        client = self.client()
        self.register(client, "alice")
        user_id = self.accounts.user(client.cookies["vm_demo_session"])[0]
        agent = self.accounts.agent(user_id)
        conversation = Conversation.__new__(Conversation)
        conversation.agent = agent
        conversation.self_harness = SelfHarnessState()
        conversation.sock = SimpleNamespace(send_json=lambda payload: asyncio.sleep(0))

        async def update():
            await conversation.update_self_harness({"speaking_style": {"speech_rate": "slow"}})
            await conversation.update_harness_prompt("persona", "先听我说")
            await conversation.update_backchannel_curve([0.5, 1.0, 1.5, 2.0])
        asyncio.run(update())
        saved = self.accounts.load_harness(user_id)
        self.assertEqual(saved["profile"]["speaking_style"]["speech_rate"], "slow")
        self.assertEqual(saved["persona"], "先听我说")
        self.assertEqual(saved["backchannel_curve"], [0.5, 1.0, 1.5, 2.0])
        self.assertEqual(saved["profile"]["turn_taking"]["backchannel"], "auto")

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


class MobileAssetVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.pet, self.ui, self.vendor = root / "pet", root / "ui", root / "node_modules"
        files = {
            **{self.pet / name: "// fixture script" for name in PET_SCRIPTS},
            **{self.ui / name: "// fixture client" for name in MOBILE_FILES},
            **{self.vendor / path: "// fixture runtime" for path in PET_VENDORS.values()},
            self.ui / "pet-mobile.html": '<head><link href="/ui/pet-mobile.css"></head><script>window.VM_PET_ASSET_ROOT="/pet/";</script><script src="/pet/avatar-controller.js"></script>',
            self.ui / "pet-mobile.css": "body {background:url('/pet/assets/scene/background.png')}",
            self.pet / "assets" / "model.moc3": "fixture model",
            self.pet / "assets" / "texture.png": "fixture texture",
        }
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def bundle(self):
        return MobilePetAssets(self.pet, self.ui, self.vendor)

    def test_version_is_stable_across_restarts_and_file_timestamp_changes(self):
        original = self.bundle()
        path = self.pet / "assets" / "texture.png"
        os.utime(path, (100, 100))
        self.assertEqual(self.bundle().version, original.version)

    def test_model_texture_scripts_styles_and_runtime_updates_change_version(self):
        for path in (self.pet / "assets" / "model.moc3", self.pet / "assets" / "texture.png",
                     self.pet / "avatar-controller.js", self.ui / "studio-client.js",
                     self.ui / "pet-mobile.css", self.vendor / PET_VENDORS["pixi.min.js"],
                     self.ui / "vendor/katex/katex.min.js", self.ui / "vendor/katex/katex.min.css",
                     self.ui / "vendor/katex/fonts/KaTeX_Main-Regular.woff2"):
            with self.subTest(path=path.name):
                before = self.bundle()
                path.write_bytes(path.read_bytes() + b"updated")
                after = self.bundle()
                self.assertNotEqual(after.version, before.version)
                self.assertIn(f'{after.prefix}/ui/pet-mobile.css', after.html)
                self.assertIn(f'{after.prefix}/assets/', after.css.decode())

    def test_installing_local_cubism_core_changes_version(self):
        before = self.bundle()
        self.assertEqual(before.core_url, CORE_CDN)
        path = self.pet / CORE_PATH
        path.parent.mkdir(parents=True)
        path.write_text("// fixture core")
        after = self.bundle()
        self.assertNotEqual(after.version, before.version)
        self.assertEqual(after.core_url, f"{after.prefix}/{CORE_PATH}")
        self.assertIn(f'window.VM_PET_CORE_URL={json.dumps(after.core_url)}', after.html)
        self.assertIn(after.core_url, [item["href"] for item in PageResources(after.html).preloads])
        self.assertNotIn(CORE_CDN, after.html)

    def test_missing_vendor_can_still_build_mobile_page(self):
        path = self.vendor / PET_VENDORS["pixi.min.js"]
        before = self.bundle().version
        path.unlink()
        after = self.bundle()
        self.assertNotEqual(after.version, before)
        self.assertNotIn(f"{after.prefix}/vendor/pixi.min.js", after.html)

    def test_preloads_reject_paths_outside_model_and_tolerate_invalid_models(self):
        model = self.pet / MODEL_PATH
        model.parent.mkdir(parents=True)
        model.write_text(json.dumps({"FileReferences": {"Moc": "../../../../avatar-controller.js",
                                                       "Textures": ["https://example.test/private.png", "missing.png"]}}))
        bundle = self.bundle()
        links = [item["href"] for item in PageResources(bundle.html).preloads]
        self.assertIn(f"{bundle.prefix}/{MODEL_PATH}", links)
        self.assertFalse(any("avatar-controller" in url or "example.test" in url or "missing.png" in url for url in links))
        for invalid in ("invalid json", '{"FileReferences": null}', '{"FileReferences": {"Textures": null}}'):
            with self.subTest(invalid=invalid):
                model.write_text(invalid)
                self.assertIn(f"{self.bundle().prefix}/{MODEL_PATH}", self.bundle().html)


if __name__ == "__main__":
    unittest.main()
