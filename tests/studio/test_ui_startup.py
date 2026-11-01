import contextlib
import io
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from studio.core.utils.cli.component import parse_args, select_provider
from studio.web.transport import build_app


class StartupTests(unittest.TestCase):
    def test_provider_selection_retries_and_supports_names(self):
        values = iter(['invalid', 'qwen'])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(select_provider('cuda', lambda prompt: next(values)), 'qwen')

    def test_mlx_allows_local(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(select_provider('mlx', lambda prompt: '4'), 'local')

    def test_interactive_launch_asks_once_and_shares_provider(self):
        with patch.dict('os.environ', {}, clear=True), patch('sys.argv', ['studio']), \
                patch('sys.stdin.isatty', return_value=True), patch(
                    'studio.core.utils.cli.component.select_provider', return_value='qwen') as choose:
            args = parse_args()
            self.assertEqual((args.memory_llm, args.llm), ('qwen', 'qwen'))
            choose.assert_called_once_with(args.backend)

    def test_explicit_check_and_noninteractive_never_prompt(self):
        with patch.dict('os.environ', {}, clear=True), patch('studio.core.utils.cli.component.select_provider') as choose:
            for options in (['--check'], ['--llm','openai'], ['--mode','realtime']):
                with patch('sys.argv', ['studio', *options]), patch('sys.stdin.isatty', return_value=True):
                    parse_args()
            with patch('sys.argv',['studio']), patch('sys.stdin.isatty', return_value=False):
                self.assertEqual(parse_args().llm,'deepseek')
            choose.assert_not_called()

    def test_ui_routes_and_worklets_are_served_without_models(self):
        async def session(socket):
            pass
        async def title(text):
            return 'fixture'
        app = build_app('llm_tts', session, lambda text:None, title=title)
        client = TestClient(app)
        response = client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(str(response.url), 'http://testserver/')
        self.assertIn('<base href="/ui/">', response.text)
        from html.parser import HTMLParser
        class Styles(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links = []
            def handle_starttag(self, tag, attrs):
                attributes = dict(attrs)
                if tag == 'a' and 'style-card' in attributes.get('class', '').split():
                    self.links.append(attributes.get('href'))
        styles = Styles()
        styles.feed(response.text)
        self.assertEqual(set(styles.links), {'technical.html', 'digital.html'})
        self.assertNotIn('id="uiFont"', response.text)
        self.assertNotIn('id="contentFont"', response.text)
        for name in ['technical.html','digital.html','studio-client.js','assets/avatar.jpg','assets/digital-background.jpg']:
            self.assertEqual(client.get('/ui/'+name).status_code, 200)
        for path in ['/legacy','/classic','/pcm-player-worklet.js','/mic-capture-worklet.js']:
            self.assertEqual(client.get(path).status_code, 200)
        self.assertEqual(client.get('/ui/%2e%2e/package.json').status_code, 404)

    def test_managed_desktop_root_identifies_its_backend(self):
        async def session(socket):
            pass
        async def title(text):
            return 'fixture'
        with patch.dict('os.environ', {'VOICEMEM_DESKTOP_INSTANCE': 'fixture-instance'}):
            app = build_app('llm_tts', session, lambda text: None, title=title)
        self.assertEqual(TestClient(app).get('/').headers['x-voicemem-desktop-instance'],
                         'fixture-instance')

if __name__ == '__main__':
    unittest.main()
