"""Numeric reading, marked-formula language and streamed source ownership."""
import asyncio
import unittest

from studio.core.utils.audio_timeline.component import AudioTimeline
from studio.core.utils.tts.markdown import MarkdownSpeech, prose_language
from studio.core.utils.tts.numbers import number_text
from tests.helpers.reply import ReplyFixture


class NumericSpeechTests(unittest.TestCase):
    def render(self, chunks, language='zh'):
        parser = MarkdownSpeech(language, cue_offset=0)
        timeline = AudioTimeline(track_delivery=True)
        for chunk in [*chunks, None]:
            if chunk is not None:
                timeline.append_source_text(chunk)
            speech = parser.feed(chunk or '', final=chunk is None)
            timeline.append_speech_text(speech.text, speech.source_ends, speech.consumed,
                                        previous_end=speech.previous_end)
        return timeline

    def test_large_values_preserve_place_value_sign_and_decimal_digits(self):
        cases = {
            '10000': '一万',
            '10001': '一万零一',
            '10010': '一万零一十',
            '10100': '一万零一百',
            '100000001': '一亿零一',
            '100010000': '一亿零一万',
            '100100001': '一亿零一十万零一',
            '123456789': '一亿二千三百四十五万六千七百八十九',
            '1,234,567': '一百二十三万四千五百六十七',
            '-12345.67890': '负一万二千三百四十五点六七八九，零',
            '+10000': '正一万',
            '3.1415926535': '三点一四一五，九二六五，三五',
            '0000012345': '零零零零，零一二三，四五',
            '0012': '零零一二',
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                spoken, ends = number_text(raw, 'zh')
                self.assertEqual(spoken, expected)
                self.assertEqual(len(ends), len(spoken))
                self.assertEqual(ends[-1], len(raw))
                self.assertEqual(tuple(sorted(ends)), ends)
                self.assertEqual(self.render(list(raw)).generated_text, expected)
                self.assertEqual(self.render(list('数值是' + raw + '。')).generated_text,
                                 '数值是' + expected + '。')

    def test_english_uses_english_number_words_and_preserves_zeros(self):
        cases = {
            '12345': 'twelve thousand, three hundred forty five',
            '100000001': 'one hundred million, one',
            '-10000.00001': 'negative ten thousand point zero zero zero zero, one',
            '00000': 'zero zero zero zero, zero',
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.render(list(raw), 'en').generated_text, expected)

    def test_long_numbers_are_chunk_invariant_and_have_monotonic_source_mapping(self):
        for language in ('zh', 'en'):
            for raw in ('1234567890123456789012345678901234567890',
                        '-1,234,567,890,123,456,789,012.345678901234567890',
                        '3.1415926535897932384626433832795028841971',
                        '编号：123456789012。', 'ID: 001234567890.'):
                baseline = self.render([raw], language)
                for split in range(1, len(raw)):
                    with self.subTest(language=language, raw=raw, split=split):
                        streamed = self.render([raw[:split], raw[split:]], language)
                        self.assertEqual(streamed.generated_text, baseline.generated_text)
                        self.assertEqual(streamed._source_ends, baseline._source_ends)
                streamed = self.render(list(raw), language)
                self.assertEqual(streamed.generated_text, baseline.generated_text)
                self.assertEqual(streamed._source_ends, baseline._source_ends)
                self.assertEqual(streamed._heard_prefix(len(streamed.generated_text)), raw)

    def test_thousand_digit_output_streams_without_waiting_for_eof_or_losing_digits(self):
        raw = '1234567890' * 100
        calls = []

        def language():
            calls.append(True)
            return 'zh'
        parser = MarkdownSpeech('zh', language_resolver=language)
        parts = []
        for char in raw:
            parts.append(parser.feed(char).text)
            self.assertLessEqual(len(parser.pending), 256)
        self.assertTrue(''.join(parts))
        parts.append(parser.feed('', final=True).text)
        expected = '，'.join(''.join('零一二三四五六七八九'[int(c)] for c in raw[i:i+4])
                            for i in range(0, len(raw), 4))
        self.assertEqual(''.join(parts), expected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.render([raw]).generated_text, expected)

    def test_ordinary_short_numbers_identifiers_and_urls_keep_existing_text(self):
        for raw in ('3.14，1.2，2 * 3，x_1。', '版本 v1234567890，GPT-4o。',
                    'https://example.test/12345678901234567890',
                    '1,2,3。', '价格 $5 和 $10。'):
            with self.subTest(raw=raw):
                self.assertEqual(self.render(list(raw)).generated_text, raw)

    def test_heard_numeric_prefix_never_claims_unspoken_digits_or_following_prose(self):
        timeline = self.render(list('123456789012345678901234567890。后续未播'))
        self.assertEqual(timeline._heard_prefix(len('一二三四')), '1234')
        self.assertEqual(timeline._heard_prefix(len('一二三四，五六七八')), '12345678')
        cardinal = self.render(list('金额12345，后续未播'))
        self.assertEqual(cardinal._heard_prefix(len('金额一万')), '金额')

    def test_pure_long_numbers_inside_math_markers_keep_all_digits(self):
        digits = '1234567890' * 60
        expected = self.render([digits]).generated_text
        for opening, closing in (('$', '$'), ('$$', '$$'), (r'\(', r'\)'), (r'\[', r'\]')):
            raw = opening + digits + closing
            for chunks in ([raw], list(raw)):
                with self.subTest(opening=opening, streamed=len(chunks) > 1):
                    timeline = self.render(chunks)
                    self.assertEqual(timeline.generated_text, expected)
                    self.assertEqual(timeline._heard_prefix(len(expected)), raw)

    def test_marked_math_and_currency_read_large_numbers(self):
        cases = {
            '$10000+10001=20001$': '一万加一万零一等于二万零一',
            '$1,234,567$': '一百二十三万四千五百六十七',
            r'\(-12.34+-1=-13.34\)': '负12.34加负1等于负13.34',
            '价格 $12345 USD': '价格 $一万二千三百四十五 USD',
            '| 数量 | 值 |\n| --- | --- |\n| 总数 | 12345 |': '数量，值\n总数，一万二千三百四十五',
            '[数量12345](https://example.test/12345)': '数量一万二千三百四十五',
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.render(list(raw)).generated_text, expected)


class FormulaLanguageTests(unittest.TestCase):
    def test_complete_and_unfinished_formula_markers_never_supply_english_evidence(self):
        for opening, closing in (('$', '$'), ('$$', '$$'), (r'\(', r'\)'), (r'\[', r'\]'),
                                 ('```latex\n', '\n```'), ('~~~math\n', '\n~~~')):
            raw = opening + r'E=mc^2+\frac{a}{b}' + closing
            for end in range(1, len(raw) + 1):
                with self.subTest(opening=opening, end=end):
                    self.assertEqual(prose_language(raw[:end], 'zh'), 'zh')
                    self.assertEqual(prose_language(raw[:end], 'en'), 'en')
                    self.assertEqual(prose_language('English explanation. ' + raw[:end], 'zh'), 'en')

    def test_projected_variables_and_tuples_use_prose_fallback(self):
        for raw in ('$x$', '$(x, y, z)$', '`cache`', '| 变量 | 值 |\n| --- | --- |\n| $x$ | 1 |',
                    '[查看$x$](https://example.test/)'):
            parser = MarkdownSpeech('zh', language_resolver=lambda: prose_language(raw, 'zh'))
            spoken = parser.feed(raw, final=True)
            for start, char in enumerate(spoken.text):
                with self.subTest(raw=raw, start=start):
                    self.assertEqual(parser.segment_language(char, start), 'zh')


class NumericReplyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = ReplyFixture()
        self.fixture.setUp()
        self.fixture.pending.language = 'zh'
        self.fixture.agent._SPEAK_BASE = {'zh': '中文语音。', 'en': 'English voice. '}
        self.fixture.agent._by_lang = lambda values, lang='': values.get(lang, '')

    async def test_chinese_formula_only_reply_keeps_chinese_instructions(self):
        for raw in ('$E=mc^2$', '$$E=mc^2$$', r'\(E=mc^2\)', r'\[E=mc^2\]', '$x$', '$(x,y,z)$'):
            with self.subTest(raw=raw):
                self.setUp()
                await self.fixture.complete(['平静|', *raw])
                self.assertTrue(self.fixture.tts_instructions)
                self.assertTrue(all('中文语音' in value for value in self.fixture.tts_instructions))
                self.assertFalse(any('English voice' in value for value in self.fixture.tts_instructions))

    async def test_genuine_english_reply_still_selects_english_instructions(self):
        raw = 'Here is the result: $12345+10000=22345$.'
        await self.fixture.complete(['平静|', *raw])
        self.assertTrue(all('English voice' in value for value in self.fixture.tts_instructions))
        self.assertIn('twelve thousand', ''.join(self.fixture.tts_text))

    async def test_numeric_speech_keeps_display_history_and_heard_source(self):
        raw = '结果是 **123456789**，圆周率是 3.1415926535897932384626。'
        await self.fixture.complete(['认真|', *raw])
        displayed = ''.join(event['text'] for event in self.fixture.messages if event['type'] == 'answer_delta')
        self.assertEqual(displayed, raw)
        self.assertEqual(self.fixture.agent._push_history.call_args.args[3], raw)
        self.assertEqual(self.fixture.timeline.heard_text(), raw)
        spoken = ''.join(self.fixture.tts_text)
        self.assertIn('一亿二千三百四十五万六千七百八十九', spoken)
        self.assertIn('三点一四一五', spoken)
        self.assertNotIn('123456789', spoken)

    async def test_cancellation_and_failure_do_not_flush_an_unfinished_number(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                self.setUp()
                entered = asyncio.Event()

                async def model(*_):
                    yield '认真|12345'
                    entered.set()
                    if failure:
                        raise RuntimeError('synthetic failure')
                    await asyncio.Event().wait()
                task = asyncio.create_task(self.fixture.pipeline(model))
                await entered.wait()
                if failure:
                    with self.assertRaises(RuntimeError):
                        await task
                else:
                    task.cancel()
                    await task
                self.assertEqual(self.fixture.tts_text, [])
                self.assertEqual(self.fixture.timeline.generated_text, '')


if __name__ == '__main__':
    unittest.main()
