"""Markdown speech, source mapping and real reply-pipeline regressions."""
import asyncio
import unittest

from tests.helpers.reply import ReplyFixture
from studio.core.utils.audio_timeline.component import AudioTimeline
from studio.core.utils.contracts.component import ReplySink
from voicemem.audio_timing import TextTimestamp
from studio.core.utils.tts.markdown import MarkdownSpeech


class MarkdownSpeechTests(unittest.TestCase):
    def test_lazy_language_preserves_speech_and_offsets_without_scanning_each_delta(self):
        from voicemem.lang import detect_language
        source, calls = '', []

        def language():
            calls.append(len(source))
            return detect_language(source, 'zh', keep_short=False)

        lazy = MarkdownSpeech('zh', cue_offset=0, language_resolver=language)
        eager = MarkdownSpeech('zh', cue_offset=0)
        raw = ('An English explanation. ' * 40 + '\n```python\nx = 1\n```\n'
               + '接下来我们看一个简单的算式。' * 120 + '$2+2=4$，以及 $(x, y)$。')
        chunks = [raw[i:i+8] for i in range(0, len(raw), 8)]
        for chunk in chunks:
            source += chunk
            eager.language = detect_language(source, 'zh', keep_short=False)
            self.assertEqual(lazy.feed(chunk), eager.feed(chunk))
        self.assertEqual(lazy.feed('', final=True), eager.feed('', final=True))
        self.assertGreater(len(calls), 0)
        self.assertLess(len(calls), len(chunks) // 4)

    def render(self, chunks, language="zh", cue_offset=0):
        parser = MarkdownSpeech(language, cue_offset=cue_offset)
        timeline = AudioTimeline(track_delivery=True)
        for chunk in chunks:
            timeline.append_source_text(chunk)
            parsed = parser.feed(chunk)
            timeline.append_speech_text(parsed.text, parsed.source_ends, parsed.consumed,
                                        previous_end=parsed.previous_end)
        parsed = parser.feed("", final=True)
        timeline.append_speech_text(parsed.text, parsed.source_ends, parsed.consumed,
                                    previous_end=parsed.previous_end)
        return timeline

    def test_common_formatting_and_chunk_boundaries(self):
        cases = {
            "结果是 **零**。": "结果是 零。",
            "# 标题\n- **重点**\n1. 下一项": "标题\n重点\n下一项",
            "这是 *斜体* 和 ~~删除~~。": "这是 斜体 和 删除。",
            "**重点 _内容_**结束": "重点 内容结束",
            "查看[**说明**](https://example.test/a(b))再继续。": "查看说明再继续。",
            "![白藤](https://example.test/image.png)": "白藤",
            "> 引用\n\n---\n结束": "引用\n\n\n结束",
            "```python\nx = 2 ** 3\n```\n完成。": "代码放在对话里了。\n完成。",
            "~~~js\nlet x = a_b;\n~~~": "代码放在对话里了。\n",
            "`x ** 2` 和 2 ** 3，变量 foo_bar，a*b。": "x ** 2 和 2 ** 3，变量 foo_bar，a*b。",
            "数组 [1, 2] 和 \\*星号、C#、C++、-1。": "数组 [1, 2] 和 *星号、C#、C++、-1。",
            "3.14，1.2，2 * 3，x_1。": "3.14，1.2，2 * 3，x_1。",
            "🐱 **你好**": "🐱 你好",
            "## 标题 ##\n正文。": "标题 \n正文。",
            "| 名称 | 值 |\n| --- | --- |\n| 结果 | **零** |": "名称，值\n结果，零",
            "| 算式 | 代码 |\n| --- | --- |\n| 2 ** 3 | `x|y` |": "算式，代码\n2 ** 3，x|y",
            "|x| + |y|\n这是绝对值。": "|x| + |y|\n这是绝对值。",
            r"先算 $2+2=4$。": "先算 2加2等于4。",
            r"负数 \(-2+-1=-3\)": "负数 负2加负1等于负3",
            r"令 $Q$ 为查询。": "令 Q 为查询。",
            r"坐标 $(x, y)$ 对应向量 $[a, b, c]$。": "坐标 x，y 对应向量 a，b，c。",
            r"$$A=\frac{QK^T}{\sqrt{d}}V$$" + "\n按权重汇总。": "公式写在对话里了。\n\n按权重汇总。",
            r"\[A=\sum_i w_i v_i\]": "公式写在对话里了。\n",
            r"参考 $\frac{a}{b}$ 的含义。": "参考 对话里的公式 的含义。",
            "执行 `softmax(Q @ K.T / sqrt(d)) @ V`。": "执行 对话里的代码。",
            "`code\nnext`后面的解释。": "对话里的代码后面的解释。",
            "$$" + "x" * 256 + "$$后面的解释。": "公式写在对话里了。\n后面的解释。",
            "`" + "x" * 49 + "`后面的解释。": "对话里的代码后面的解释。",
            "```python\n" + "secret_symbol = 1\n" * 30 + "```\n只讲结论。": "代码放在对话里了。\n只讲结论。",
            "````python\nx = 1\n````": "代码放在对话里了。\n",
            "```latex\nA=\\sum_i w_i v_i\n```\n解释。": "公式写在对话里了。\n解释。",
            "价格 $5 USD\n接着说。": "价格 $5 USD\n接着说。",
            "价格 $5 和 $10。": "价格 $5 和 $10。",
            "价格 $5": "价格 $5",
            "结论 $$" + "x_{1234567890}+" * 24 + "1$$结束。": "结论 公式写在对话里了。\n结束。",
        }
        for raw, expected in cases.items():
            baseline = self.render([raw])
            self.assertEqual(baseline.generated_text, expected, raw)
            for split in range(1, len(raw)):
                with self.subTest(raw=raw, split=split):
                    chunked = self.render([raw[:split], raw[split:]])
                    self.assertEqual(chunked.generated_text, expected)
                    self.assertEqual(chunked._source_ends, baseline._source_ends)
            character_chunks = self.render(list(raw))
            self.assertEqual(character_chunks.generated_text, expected, raw)
            self.assertEqual(character_chunks._source_ends, baseline._source_ends, raw)

    def test_viewing_cues_vary_but_dense_adjacent_blocks_do_not_repeat(self):
        code = "```python\nx=1\n```\n"
        narration = "这一段是在解释计算步骤和结果。" * 6 + "\n"
        spoken = self.render([code + narration + code + narration + code]).generated_text
        for cue in ("代码放在对话里了。", "这段实现可以在对话里查看。", "具体代码我写在对话里。"):
            self.assertIn(cue, spoken)
        dense = self.render([code * 8]).generated_text
        self.assertEqual(dense, "代码放在对话里了。\n")
        starts = {self.render([code], cue_offset=seed).generated_text for seed in range(5)}
        self.assertEqual(len(starts), 5)

    def test_english_cues_simple_math_and_short_identifiers(self):
        raw = "```js\nx=1;\n```\nThen $2-1=1$ and `cache.get` and $Q$."
        spoken = self.render(list(raw), language="en").generated_text
        self.assertEqual(spoken, "The code is in the conversation.\nThen 2 minus 1 equals 1 and cache.get and Q.")
        self.assertEqual(self.render(list('Use $(x, y)$.'), language='en').generated_text,
                         'Use x, y.')

    def test_unterminated_structures_have_bounded_buffers_and_never_leak_body(self):
        for prefix, body in (("```python\n", "code_body"), ("$$", "x_{12345}+"), ("`", "long_code_call(")):
            parser = MarkdownSpeech(cue_offset=0)
            spoken = parser.feed(prefix).text
            for _ in range(200):
                spoken += parser.feed(body).text
                self.assertLessEqual(len(parser.pending), 256)
                self.assertLessEqual(len(parser.math_content), 256)
            spoken += parser.feed("", final=True).text
            self.assertNotIn(body, spoken)
            self.assertTrue(spoken)

    def test_complex_formula_cue_does_not_wait_for_its_closing_delimiter(self):
        parser = MarkdownSpeech(cue_offset=0)
        speech = parser.feed(r"$$A=\frac{")
        self.assertEqual(speech.text, "公式写在对话里了。\n")
        self.assertEqual(parser.feed(r"QK^T}{\sqrt{d}}V$$继续解释。", final=True).text,
                         "继续解释。")

    def test_interruption_mapping_stops_before_unheard_explanation(self):
        raw = "```python\nx=1\n```\n这是尚未听到的解释。"
        timeline = self.render(list(raw))
        cue_end = timeline.generated_text.index("\n") + 1
        self.assertEqual(timeline._heard_prefix(cue_end), "```python\nx=1\n```")
        self.assertNotIn("尚未听到", timeline._heard_prefix(cue_end))
        self.assertEqual(timeline._heard_prefix(len(timeline.generated_text)), raw)

    def test_buffered_inline_content_is_not_marked_heard_before_it_is_spoken(self):
        parser = MarkdownSpeech(cue_offset=0)
        timeline = AudioTimeline()
        for chunk in ("开头。`", "cache.get"):
            timeline.append_source_text(chunk)
            speech = parser.feed(chunk)
            timeline.append_speech_text(speech.text, speech.source_ends, speech.consumed,
                                        previous_end=speech.previous_end)
        self.assertEqual(timeline.generated_text, "开头。")
        self.assertNotIn("cache", timeline._heard_prefix(3))
        timeline.append_source_text("`。")
        speech = parser.feed("`。", final=True)
        timeline.append_speech_text(speech.text, speech.source_ends, speech.consumed,
                                    previous_end=speech.previous_end)
        self.assertEqual(timeline._heard_prefix(4), "开头。`c")

    def test_normal_text_streams_immediately_and_ambiguous_tail_flushes_at_eof(self):
        parser = MarkdownSpeech()
        self.assertEqual(parser.feed("先把结论告诉你。").text, "先把结论告诉你。")
        self.assertEqual(parser.feed("2 *").text, "2 ")
        self.assertEqual(parser.feed("", final=True).text, "*")
        parser = MarkdownSpeech()
        self.assertEqual(parser.feed("[").text, "")
        self.assertEqual(parser.feed("1, 2]", final=True).text, "[1, 2]")
        parser = MarkdownSpeech()
        self.assertEqual(parser.feed("|").text, "")
        self.assertEqual(parser.feed("x|").text, "|x|")
        parser = MarkdownSpeech()
        parser.feed("| 名称 | 值 |\n| --- | --- |\n| 结果 | 零 |\n")
        self.assertEqual(parser.feed("接着正常说话。").text, "接着正常说话。")

    def test_provider_alignment_maps_back_to_original_markdown_and_freezes(self):
        timeline = self.render(["**甲乙**。后半句"])
        segment = timeline.begin_segment(0, len(timeline.generated_text))
        timeline.append_audio(bytes(48000))
        timeline.mark_sent(24000)
        timeline.add_segment_timestamps(segment, (
            TextTimestamp(0, 1, 0, 6000), TextTimestamp(1, 2, 6000, 12000),
            TextTimestamp(2, len(timeline.generated_text), 12000, 24000),
        ))
        timeline.finish_segment(segment)
        self.assertEqual(timeline.heard_text(), "")
        timeline.update_checkpoint(12000, 24000, "playing")
        self.assertEqual(timeline.heard_text(), "**甲乙**")
        timeline.mark_interrupted()
        timeline.update_checkpoint(24000, 24000, "drained")
        self.assertEqual(timeline.heard_text(), "**甲乙**")

    def test_duration_fallback_counts_spoken_text_and_never_includes_unheard_tail(self):
        timeline = self.render(["**甲乙**[丙丁](https://example.test/long-path)戊己"])
        segment = timeline.begin_segment(0, len(timeline.generated_text))
        timeline.append_audio(bytes(48000))
        timeline.mark_sent(24000)
        timeline.finish_segment(segment)
        timeline.update_checkpoint(12000, 24000)
        self.assertEqual(timeline.heard_text(), "**甲乙**[丙")
        timeline.update_checkpoint(24000, 24000, "drained")
        self.assertEqual(timeline.heard_text(), timeline.source_text)


class MarkdownReplyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = ReplyFixture()
        self.fixture.setUp()

    async def test_coordinate_reply_filters_private_control_and_keeps_math_source(self):
        raw = '坐标 $(x, y)$。'
        await self.fixture.complete(list('温和|<self_harness>{}</self_harness>' + raw),
                                    self_harness_profile={})
        displayed = ''.join(m['text'] for m in self.fixture.messages
                            if m['type'] == 'answer_delta')
        self.assertEqual(displayed, raw)
        self.assertEqual(''.join(self.fixture.tts_text), '坐标 x，y。')
        self.assertEqual(self.fixture.timeline.heard_text(), raw)
        self.assertEqual(self.fixture.agent._push_history.call_args.args[3], raw)

    async def test_structured_reply_keeps_raw_display_and_history_and_speaks_explanation(self):
        raw = ("先按相关程度分配权重。\n$$A=\\frac{QK^T}{\\sqrt{d}}V$$\n"
               "然后按这些权重汇总信息。\n```python\nsecret_call(Q, K, V)\n```\n"
               "这段实现对应刚才的计算步骤。")
        await self.fixture.complete(["认真|", *raw])
        displayed = "".join(m["text"] for m in self.fixture.messages if m["type"] == "answer_delta")
        self.assertEqual(displayed, raw)
        self.assertEqual(self.fixture.agent._push_history.call_args.args[3], raw)
        self.assertEqual(self.fixture.timeline.heard_text(), raw)
        spoken = "".join(self.fixture.tts_text)
        self.assertIn("然后按这些权重汇总信息。", spoken)
        self.assertIn("这段实现对应刚才的计算步骤。", spoken)
        for symbol in ("secret_call", "\\frac", "$$", "```", "QK^T"):
            self.assertNotIn(symbol, spoken)

    async def test_code_cue_speaks_while_provider_is_still_generating_the_block(self):
        release, audio_ready = asyncio.Event(), asyncio.Event()
        original = self.fixture.send_audio

        async def send_audio(pcm):
            await original(pcm)
            audio_ready.set()
        self.fixture.send_audio = send_audio

        async def model(*_):
            yield "认真|```python\n"
            await release.wait()
            yield "secret_call()\n```\n关键是先算权重。"
        task = asyncio.create_task(self.fixture.pipeline(model))
        try:
            await asyncio.wait_for(audio_ready.wait(), .5)
            self.assertTrue(self.fixture.tts_text)
            self.assertFalse(task.done())
            release.set()
            await asyncio.wait_for(task, 1)
            self.assertNotIn("secret_call", "".join(self.fixture.tts_text))
            self.assertIn("关键是先算权重。", "".join(self.fixture.tts_text))
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_raw_display_and_history_are_preserved_but_tts_is_plain(self):
        raw = "# 答案\n结果是 **零**。\n查看[说明](https://example.test/)。"
        await self.fixture.complete(["认真|", *raw])
        displayed = "".join(m["text"] for m in self.fixture.messages if m["type"] == "answer_delta")
        self.assertEqual(displayed, raw)
        self.assertEqual(self.fixture.timeline.source_text, raw)
        self.assertEqual(self.fixture.timeline.generated_text, "答案\n结果是 零。\n查看说明。")
        self.assertEqual("".join(self.fixture.tts_text), "答案结果是 零。查看说明。")
        self.assertEqual(self.fixture.agent._push_history.call_args.args[3], raw)
        self.assertEqual(self.fixture.timeline.heard_text(), raw)

    async def test_stalled_generation_speaks_before_closing_markdown_arrives(self):
        release = asyncio.Event()
        audio_ready = asyncio.Event()
        original = self.fixture.send_audio

        async def send_audio(pcm):
            await original(pcm)
            audio_ready.set()
        self.fixture.send_audio = send_audio

        async def model(*_):
            yield "温和|**这是先说的完整句子。"
            await release.wait()
            yield "**后面的内容。"
        task = asyncio.create_task(self.fixture.pipeline(model))
        try:
            await asyncio.wait_for(audio_ready.wait(), .5)
            self.assertEqual(self.fixture.tts_text, ["这是先说的完整句子。"])
            self.assertFalse(task.done())
            release.set()
            await asyncio.wait_for(task, 1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_cancelled_or_failed_reply_does_not_flush_pending_syntax(self):
        for prefix in ("[", "$QK+", "`cache.get"):
            for fail in (False, True):
                self.setUp()
                entered = asyncio.Event()

                async def model(*_):
                    yield "温和|" + prefix
                    entered.set()
                    if fail:
                        raise RuntimeError("fixture provider failure")
                    await asyncio.Event().wait()
                task = asyncio.create_task(self.fixture.pipeline(model))
                await entered.wait()
                if fail:
                    with self.assertRaises(RuntimeError):
                        await task
                else:
                    task.cancel()
                    await task
                self.assertEqual(self.fixture.tts_text, [])
                self.assertEqual(self.fixture.timeline.generated_text, "")

    async def test_markdown_speculation_remains_private_until_commit(self):
        sink = ReplySink(self.fixture.send, self.fixture.send_audio)

        async def model(*_):
            yield "温和|**四**。"
        task = asyncio.create_task(self.fixture.pipeline(model, sink))
        try:
            await asyncio.wait_for(sink.wait_for_audio(), .5)
            self.assertEqual(self.fixture.messages, [])
            self.assertEqual(self.fixture.audio, [])
            await sink.commit()
            await asyncio.wait_for(task, 1)
            self.assertEqual(self.fixture.tts_text, ["四。"])
            self.assertEqual(self.fixture.timeline.heard_text(), "**四**。")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
