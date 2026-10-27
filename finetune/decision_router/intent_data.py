"""Contrast instruction execution with quotation, negation scope and speaker intent."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .dialogue_data import _row
from .train import load_rows
from studio.core.utils.reply_modes.decision_features import texts, audio_vector


TERMS = ('深度思考', '深入思考', '深度推理', '深入推演', '深思', '深度计算', '深度想想', '严密推导')

# Each pair describes opposite current intentions while sharing the same term.
PAIRS = {
    'train': (
        ('请执行{d}来解决这道题。', '请解释{d}这个词，不执行它。'),
        ('我希望你{d}，不用介绍它的功能。', '我希望你介绍{d}这个功能，不用执行它。'),
        ('不是让我自己{d}，是让你来做。', '不是让你{d}，我会自己做。'),
        ('我没有让你解释术语，我让你真的{d}。', '我没有让你真的{d}，只让你解释术语。'),
        ('你不必说明什么叫{d}，先这样想过再答。', '你不必真的{d}，先说明这个词怎么理解。'),
        ('你先{d}，不要介绍设置入口。', '你先介绍{d}的设置入口，不要执行。'),
        ('我只要求你{d}后给答案，不要把过程念出来。', '我只要求你把{d}这个词念出来，不要实际思考。'),
        ('能不能进行{d}？请现在做一遍。', '能不能进行{d}？我只是在问是否有这个功能。'),
        ('本轮要{d}，前面说快速回答已经作废。', '本轮快速回答，前面说{d}已经作废。'),
        ('原来让你别{d}，现在我改变要求，请做。', '原来让你{d}，现在我改变要求，请停下。'),
        ('别只说自己能{d}，请真正用它来回答。', '别真的{d}，说一下自己是否支持这个功能。'),
        ('你刚才没有{d}，所以请重新想一遍。', '你刚才已经{d}，所以不用重新想一遍。'),
        ('请停止随口作答，切换为{d}。', '请停止{d}，切换为随口作答。'),
        ('我纠正一下，这里需要{d}。', '我纠正一下，这里不需要{d}。'),
        ('你不能先{d}再说吗？我是要你这样做。', '你不能先解释{d}怎么用吗？我没让你现在做。'),
        ('这里的引号表示模式名称，请用“{d}”回答。', '这里的引号表示引用，请翻译“{d}”。'),
        ('请{d}，但只展示最终一句结论。', '请只说最终一句结论，不用{d}。'),
        ('就算任务简单，这次也指定{d}。', '就算任务复杂，这次也指定不要{d}。'),
        ('任务很短，仍然要求你{d}。', '回答可以很长，仍然要求你不要{d}。'),
        ('别再给我{d}的定义了，我要你用起来。', '别再用{d}回答了，我只要它的定义。'),
        ('你说能{d}，那这道题就用它来解。', '你说能{d}，这个功能的名字用英语怎么说？'),
        ('我想看你{d}后的答案，别只复述旧结果。', '我想看{d}的字面意思，别开始推理。'),
        ('我问的不是{d}是什么意思，是要你这样回答。', '我问的是{d}是什么意思，不是要你这样回答。'),
        ('别误会，我让你{d}，不是让我等着自己想。', '别误会，我要自己{d}，你先听着。'),
        ('“{d}”是我指定你当前执行的方式。', '“{d}”是我要你分析语法的词语。'),
        ('不要取消{d}，这轮我确实要用。', '不要开启{d}，这轮我确实不用。'),
        ('我不是不要{d}，我就是要你深入想。', '我不是要{d}，我就是要个快答。'),
        ('请先开启{d}，之后给我一份简短清单。', '请先关闭{d}，之后给我一份简短清单。'),
        ('先说明白我的意图：需要你{d}。', '先说明白我的意图：只问{d}这个词的意思。'),
        ('我没有否定{d}这个要求，请照着做。', '我已经否定{d}这个要求，请别照着做。'),
        ('这句“请{d}”是我正在向你下的指令。', '这句“请{d}”是我引用来让你润色的文本。'),
        ('别只把{d}当成一句口头禅，真正想过再说。', '别把{d}这句口头禅当成命令，只帮我换种说法。'),
    ),
    'validation': (
        ('我要的是实际{d}，无需讲解这个按钮的作用。', '我要的是讲解{d}按钮的作用，无需实际执行。'),
        ('前面那句“不{d}”取消，本题请深入推演。', '前面那句“要{d}”取消，本题请直接简答。'),
        ('如果你有{d}能力，现在就用它回答。', '如果你有{d}能力，告诉我它叫什么名字即可。'),
        ('这次你必须{d}，不用把隐藏过程展示给我。', '这次你只须复述{d}这几个字，不须执行。'),
        ('不是在问你能否{d}，我是在请你现在使用。', '不是在请你{d}，我是在问你是否有这个能力。'),
        ('请恢复{d}模式，我想要深思后的判断。', '请暂停{d}模式，我想要普通判断。'),
        ('你还没有按要求{d}，请按要求重做。', '你还没有按要求停下{d}，请改成简答。'),
        ('这里说{d}是指你要去做，不是词语释义。', '这里说{d}是要你解释词语，不是要你去做。'),
    ),
    'calibration': (
        ('请把{d}作为这次的回答方式。', '请把{d}作为词语做一次解释。'),
        ('我现在撤销快速回答的要求，改用{d}。', '我现在撤销{d}的要求，改用快速回答。'),
        ('用上{d}，只把推敲后的结果告诉我。', '念出{d}，不要进行实际的推敲。'),
        ('我想让你{d}，而非介绍模式。', '我想让你介绍{d}，而非进入模式。'),
        ('这里不是引用，请真的开始{d}。', '这里只是引用，请别真的开始{d}。'),
        ('先{d}再说，前面取消它的要求无效。', '别{d}了，前面开启它的要求无效。'),
    ),
    'test': (
        ('本次采用{d}，功能介绍可以省掉。', '本次只做{d}的功能介绍，执行可以省掉。'),
        ('我不是在咨询{d}的概念，我希望你实践它。', '我不是在要求{d}，我只是咨询相关概念。'),
        ('把当前题用{d}重新做，结果简短即可。', '把{d}这句话重新写得简短即可。'),
        ('我说过别急，现在请用{d}认真检查答案。', '我说过别推了，现在请停止{d}直接报答案。'),
        ('不需要说明模式名称，我指定你实际{d}。', '不需要真的运行，我指定你说明{d}的名称。'),
        ('请按“{d}”的方式处理；引号只是标记模式。', '请改写“{d}”这句话；引号表示待编辑的文本。'),
        ('你上一轮理解反了，我现在确实要求{d}。', '你上一轮理解反了，我现在没有要求{d}。'),
        ('我愿意等，请把{d}用在这个小问题上。', '我不用等，请省略{d}直接回答这个小问题。'),
    ),
}

HISTORIES = {
    'train': ((), (('user', '前面给我的建议可以再讨论。'), ('assistant', '我刚才只给了简短建议。')),
              (('user', '请分析一个分布式协议。'), ('assistant', '之前的论证已经完成。'),
               ('user', '接下来我换个问题。'), ('assistant', '好的，请说出新的要求。'))),
    'validation': ((), (('user', '我们刚说到出行清单。'), ('assistant', '我已经列了常见物品。')),
                   (('user', '为并发更新建立正确性证明。'), ('assistant', '证明已经完整给出。'),
                    ('user', '我这次问的不是上一题。'), ('assistant', '明白，请继续。'))),
    'calibration': ((), (('user', '我们刚聊到书架的摆放。'), ('assistant', '我提到了两种普通摆法。')),
                    (('user', '评估多地区灾备方案的边界。'), ('assistant', '各类故障情况已经论证完毕。'),
                     ('user', '接下来换一个要求。'), ('assistant', '好的。'))),
    'test': ((), (('user', '我们在讨论博物馆参观安排。'), ('assistant', '刚才列了一份普通路线。')),
             (('user', '推导一个优化问题的充分条件。'), ('assistant', '完整推导已经结束，答案也给出了。'),
              ('user', '我准备再问别的事情。'), ('assistant', '我会以这次的话为准。'))),
}

TOPICS = ('', '我说的是一个简单的加法。', '我问的是几种常见植物的区别。', '我想要一张日常用品的小表格。')

ENGLISH_PAIRS = {
    'train': (
        ('Do not just define {d}; use it to solve the question.', 'Do not use {d}; just define the expression.'),
        ('I want the result after {d}, without showing your work.', 'I want a definition of {d}, without performing it.'),
        ('I did not ask for a description; I asked you to use {d}.', 'I did not ask you to use {d}; I asked for a description.'),
        ('Stop giving a quick reply and switch to {d}.', 'Stop using {d} and switch to a quick reply.'),
        ('The quotation marks name a mode: answer using "{d}".', 'The quotation marks enclose text: translate "{d}".'),
        ('My earlier request to skip {d} is cancelled; use it now.', 'My earlier request to use {d} is cancelled; skip it now.'),
        ('I mean you should use {d}, not explain how it works.', 'I mean you should explain {d}, not use it.'),
        ('Apply {d} even for this easy question.', 'Skip {d} even for this difficult question.'),
    ),
    'validation': (
        ('Please engage {d}; a short final answer is enough.', 'Please describe {d}; no actual reasoning is needed.'),
        ('I am requesting actual {d}, rather than a feature tutorial.', 'I am requesting a tutorial about {d}, rather than its execution.'),
        ('Treat "{d}" as the mode I want you to use.', 'Treat "{d}" as the phrase I want you to edit.'),
    ),
    'calibration': (
        ('Use {d} before answering; keep the reasoning private.', 'Give the meaning of {d}; do not carry it out.'),
        ('Replace your quick answer with one based on {d}.', 'Replace {d} with a quick answer.'),
    ),
    'test': (
        ('I want you to do {d}, without explaining the name of the feature.', 'I want you to explain {d}, without doing it.'),
        ('Begin {d} for the current problem; I can wait.', 'Avoid {d} for the current problem; be brief.'),
        ('Use the mode called "{d}" for your reply.', 'Correct the spelling of the expression "{d}" in my document.'),
    ),
}


def build(legacy):
    """Add independent scope contrasts; previous test rows stay regression-only."""
    original = [json.loads(line) for line in Path(legacy).read_text().splitlines()]
    rows, sampled = [], Counter()
    for row in sorted(original, key=lambda row: hashlib.sha256(row['id'].encode()).hexdigest()):
        if row['split'] == 'test':
            continue
        category = row['category']
        if row['split'] == 'train' and category in {'explicit_deep', 'intent_negative'}:
            if sampled[category] >= 650:
                continue
            sampled[category] += 1
        rows.append(row)
    for split, pairs in PAIRS.items():
        for family, pair in enumerate(pairs):
            group = f'intent-scope-v3/{split}/{family:02d}'
            for index, term in enumerate(TERMS):
                for label, template in enumerate(pair):
                    instruction = template.format(d=term)
                    for variant, suffix in enumerate(TOPICS):
                        context = (index+variant) % len(HISTORIES[split])
                        text = instruction+suffix
                        if variant == 3:
                            text = '嗯，'+text.translate(str.maketrans('', '', '，。？！、'))
                        row = _row(group, f'{index}-{label}-{variant}', text,
                                   HISTORIES[split][context], label == 0, split,
                                   'explicit_deep' if label == 0 else 'intent_negative')
                        row['source'] = 'authored-scope-contrast-v3'
                        rows.append(row)
    for split, pairs in ENGLISH_PAIRS.items():
        for family, pair in enumerate(pairs):
            group = f'intent-scope-v3/{split}/en-{family:02d}'
            for index, term in enumerate(('deep reasoning', 'deep thinking', 'in-depth reasoning', 'deliberate thought')):
                for label, template in enumerate(pair):
                    for context, history in enumerate(((), (('user', 'We discussed an earlier problem.'),
                                                           ('assistant', 'The previous answer was only preliminary.')))):
                        row = _row(group, f'{index}-{label}-{context}', template.format(d=term), history,
                                   label == 0, split, 'explicit_deep' if label == 0 else 'intent_negative')
                        row['source'] = 'authored-scope-contrast-v3'
                        rows.append(row)
    ids, families, cases, result = set(), {}, {}, []
    for row in rows:
        if row['id'] in ids:
            raise ValueError('Duplicate observation identity')
        ids.add(row['id'])
        if row['group'] in families and families[row['group']] != row['split']:
            raise ValueError('Wording family crosses partitions')
        families[row['group']] = row['split']
        key = (*texts(row['text'], row.get('history')), audio_vector(row.get('audio_state')).tobytes())
        if key in cases:
            prior = cases[key]
            if prior['needs_deep'] != row['needs_deep'] or prior['split'] != row['split']:
                raise ValueError('Conflicting semantic input')
            continue
        cases[key] = row
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.legacy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False)+'\n')
    load_rows(args.out)
    report = {'rows': len(rows), 'sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
              'previous_test_role': 'regression only, excluded from every new partition',
              'splits': {split: {'rows': len(selected := [row for row in rows if row['split'] == split]),
                                  'deep': sum(row['needs_deep'] for row in selected),
                                  'with_history': sum(bool(row.get('history')) for row in selected),
                                  'observed_audio': sum(bool(row.get('audio_state')) for row in selected),
                                  'groups': len({row['group'] for row in selected})}
                         for split in ('train', 'validation', 'calibration', 'test')}}
    args.out.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
