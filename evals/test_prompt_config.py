"""Edit actual prompt files, start a fresh process, inspect actual provider inputs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

PROBE = r'''
import ast, asyncio, json, os, queue, types
from pathlib import Path
from unittest.mock import patch
import httpx
from voicemem import persona
from voicemem.prompt_config import tts_prompts, context_prompts
from voicemem.reply import deepseek_reply
from voicemem.breeze_tts import BreezeMLXTTS
from harness import speak_tag, backchannel

# Actual web prompt assembly, without loading ASR/TTS models or memory DBs.
tree = ast.parse((Path(os.environ['TEST_REPO']) / 'web/run.py').read_text())
names = {'_rt_persona', '_by_lang', '_speak_instruction', '_tone_note'}
ns = dict(persona=persona, speak_tag=speak_tag, SPACE_LANG='zh', MODE='llm_tts',
          _SPEAK_BASE=tts_prompts()['base'], _TONE=tts_prompts()['fallback_by_user_emotion'],
          _speak_base_env='')
exec(compile(ast.Module(body=[n for n in tree.body if getattr(n,'name','') in names],
                        type_ignores=[]), 'web/run.py', 'exec'), ns)
system = ns['_rt_persona']('zh')
wire = []
original = httpx.AsyncClient
def handle(req):
    wire.append(json.loads(req.content))
    return httpx.Response(200, text='data: [DONE]\n\n'.replace('\\n','\n'))
async def main():
    with patch('httpx.AsyncClient', side_effect=lambda **kw: original(
            transport=httpx.MockTransport(handle), **kw)):
        reply = deepseek_reply(api_key='test-only', system=system)
        try:
            async for _ in reply('问题', persona.no_memory_note('zh')):
                pass
        finally:
            await reply.aclose()
    tts = BreezeMLXTTS(model='test-only')  # constructor does not load MLX
    q = queue.Queue(); q.put(None)
    job = types.SimpleNamespace(out=q, cancel=lambda: None)
    instruction = speak_tag.instruction('认真', ns['_SPEAK_BASE']['zh'])
    with patch('voicemem.utils.gpu_loop.gpu_loop', return_value=types.SimpleNamespace(
            iter=lambda *a,**kw: job)), patch('voicemem.prompt_trace.record_request') as record:
        async for _ in tts.stream('测试正文', instruction):
            pass
        request = record.call_args.args[2]
    print(json.dumps({'system':wire[0]['messages'][0]['content'],
                      'user':wire[0]['messages'][-1]['content'],
                      'tts':request, 'default':tts.instruction,
                      'fallback':ns['_speak_instruction']('焦虑'),
                      'styles':backchannel._STYLES}, ensure_ascii=False))
asyncio.run(main())
'''


class PromptConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        for name in ('llm_system_zh.md', 'llm_system_en.md', 'llm_tone_rule_zh.md',
                     'llm_tone_rule_en.md', 'llm_context.json', 'tts.json'):
            shutil.copyfile(ROOT / 'prompt' / name, self.directory / name)
        self.env = {**os.environ, 'VOICEMEM_PROMPT_DIR': str(self.directory),
                    'TEST_REPO': str(ROOT), 'PYTHONPATH': str(ROOT)}
        for key in ('VOICEMEM_BREEZE_REF_AUDIO', 'VOICEMEM_BREEZE_REF_TEXT',
                    'VOICEMEM_BREEZE_INSTRUCTION', 'VOICEMEM_SPEAK_BASE'):
            self.env.pop(key, None)

    def tearDown(self):
        self.tmp.cleanup()

    def run_probe(self):
        return subprocess.run([sys.executable, '-c', PROBE], cwd=self.directory,
                              env=self.env, text=True, capture_output=True, timeout=15)

    def test_file_edits_reach_deepseek_and_breeze_and_backchannel(self):
        (self.directory / 'llm_system_zh.md').write_text('学长修改的人设', encoding='utf-8')
        (self.directory / 'llm_tone_rule_zh.md').write_text('学长修改的标签协议', encoding='utf-8')
        cfg = json.loads((self.directory / 'tts.json').read_text())
        cfg['base']['zh'] = '自定义基调。'
        cfg['tones']['认真'] = '自定义认真语气。'
        cfg['fallback_by_user_emotion']['zh']['焦虑'] = '自定义焦虑回应。'
        cfg['breeze_default_instruction'] = '自定义默认音色指令。'
        cfg['backchannel_styles']['zh'][0] = '自定义附和语气。'
        (self.directory / 'tts.json').write_text(json.dumps(cfg), encoding='utf-8')
        context = json.loads((self.directory / 'llm_context.json').read_text())
        context['no_memory']['zh'] = '自定义无记忆提示。'
        (self.directory / 'llm_context.json').write_text(json.dumps(context), encoding='utf-8')
        result = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        out = json.loads(result.stdout)
        self.assertEqual(out['system'], '学长修改的人设\n\n学长修改的标签协议\n')
        self.assertEqual(out['user'], '自定义无记忆提示。\n\n问题')
        self.assertEqual(out['tts']['instruct'], '自定义基调。自定义认真语气。')
        self.assertEqual(out['default'], '自定义默认音色指令。')
        self.assertEqual(out['fallback'], '自定义基调。自定义焦虑回应。')
        self.assertEqual(out['styles']['zh'][0], '自定义附和语气。')

    def test_invalid_json_fails_with_filename_instead_of_using_old_defaults(self):
        (self.directory / 'tts.json').write_text('{invalid', encoding='utf-8')
        result = self.run_probe()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('tts.json', result.stderr)
        self.assertIn('配置格式错误', result.stderr)

    def test_missing_persona_fails_instead_of_using_hardcoded_persona(self):
        (self.directory / 'llm_system_zh.md').unlink()
        result = self.run_probe()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('llm_system_zh.md', result.stderr)

    def test_missing_tone_key_is_rejected(self):
        cfg = json.loads((self.directory / 'tts.json').read_text())
        cfg['tones'].pop('认真')
        (self.directory / 'tts.json').write_text(json.dumps(cfg), encoding='utf-8')
        result = self.run_probe()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('缺少字段', result.stderr)

    def test_templates_are_declared_as_wheel_data_and_not_ignored(self):
        import tomllib
        config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        files = config['tool']['setuptools']['data-files']['prompt']
        self.assertEqual(len(files), 6)
        for name in files:
            self.assertTrue((ROOT / name).is_file())
            ignored = subprocess.run(['git', 'check-ignore', name], cwd=ROOT,
                                     capture_output=True, text=True)
            self.assertEqual(ignored.returncode, 1, name)


if __name__ == '__main__':
    unittest.main()
