"""Separate private reasoning, spoken detail and completion of prior work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .dialogue_data import _row, audit
from .train import load_rows
from studio.core.utils.reply_modes.decision_features import texts, audio_vector


FRAMES = {
    'train': (
        ('请先{d}，但不用{v}{p}。', '这次不用{d}，只把已有结果{v}给我。'),
        ('{p}不必{v}出来，你仍要{d}后再答。', '不必{d}，直接{v}上次的结论。'),
        ('别{v}{p}，我要求的是先{d}得到答案。', '别再{d}，我要求的是{v}已有答案。'),
        ('不让你{v}{p}并不是取消{d}，请照常思考。', '不让你{d}就是取消思考要求，请直接{v}结果。'),
        ('先启动{d}，回答只留三句话，不用{v}{p}。', '先关闭{d}，只留三句话，不用重新推导。'),
        ('{d}需要保留，只是不要{v}{p}。', '{d}本轮不要保留，直接{v}之前的答案。'),
        ('你不是可以{d}一遍吗，我需要一个', '你不是提过“{d}”这个名称吗，我只是问它的意思'),
    ),
    'validation': (
        ('{p}可以不{v}，但这轮请{d}后回答。', '{p}可以省掉，这轮请不要{d}。'),
        ('我想要先{d}，不需要你{v}{p}。', '我想要直接报数，不需要进行{d}。'),
    ),
    'calibration': (
        ('请在内部{d}，不用对我{v}{p}。', '请省略内部{d}，直接回答就好。'),
    ),
    'test': (
        ('这轮先用{d}把问题想透，答复里省略{p}。', '这轮省略{d}，直接重复上次的结果即可。'),
        ('请进入{d}模式，只需说结论，不必{v}怎么想的。', '请退出{d}模式，只需读出原有结论。'),
    ),
}

SCENES = {
    'train': (
        ('严密推导消息乱序时重试协议的正确性。', '我只分析了第一种时序，还没证明其他交错。', '所有时序已经证明完毕，结论是协议正确，需要三个副本。'),
        ('在多种约束下求解设备排班的最优值。', '刚列出约束，最优方案与完整推导尚未给出。', '比较所有可行情况得出最优值为八，完整推导到此结束。'),
    ),
    'validation': (
        ('证明有界缓存更新规则始终保持一致。', '刚写出一个猜测值，证明过程还没有给出。', '所有边界与交错都已逐一验证，完整证明已给出，答案是五。'),
    ),
    'calibration': (
        ('推导弹性任务分配算法的最坏延迟界。', '目前只得到候选数字，论证仍然没有完成。', '以上各步覆盖了全部情况，完整推导结束，延迟上界是十二。'),
    ),
    'test': (
        ('论证多站点协作更新不会覆盖已确认的数据。', '只给出了猜测的结果，后续论证仍待补齐。', '上面的推演已经覆盖所有顺序，完整步骤和结论都给出了，答案是七。'),
    ),
}

FOLLOWUPS = {
    'train': ('后面的步骤呢？', '给我展开完整推法。', '讲讲你怎么得到的。', '把过程补上。',
              '再把未完成的处理完。', '继续做下去。', '结果怎么得到的？', '那答案呢？',
              '你给我个完整结果。', '你刚才的过程再讲一下。'),
    'validation': ('把刚才完整的推演过程交代一下。', '最后结果是多少呀？'),
    'calibration': ('后续求解步骤说给我听。', '所以最终数字是多少？'),
    'test': ('我想了解详细步骤，别单报数。', '那最终的答案是什么呢？'),
}

ENGLISH = {
    'train': (
        ('Use {d}, but do not show the reasoning process.', 'Do not use {d}; just show the previous result.'),
        ('Use {d} privately. You need not reveal the intermediate work.', 'Explain what {d} means without enabling that feature.'),
        ('I want actual {d}, not a display of your thoughts.', 'I want the meaning of {d}, not its activation.'),
        ('Do not turn off {d}; only hide the thinking steps.', 'Turn off {d}; give a direct answer.'),
        ('Keep {d} enabled and omit your reasoning from the response.', 'Describe the option called {d} without turning it on.'),
    ),
    'validation': (('Reason with {d} internally, then state only the conclusion.', 'Define the words {d} without performing it.'),),
    'calibration': (('Perform {d}, but keep the detailed reasoning private.', 'Explain the term {d} without switching it on.'),),
    'test': (('Use {d} before replying and omit the intermediate steps.', 'Please describe {d} without activating that mode.'),),
}


def build(previous):
    rows = [json.loads(line) for line in Path(previous).read_text().splitlines()]
    rows = [row for row in rows if row['split'] != 'test']
    terms = ('深度思考', '深入思考', '深思', '深度推理', '深度计算', '深入推演')
    for split, pairs in FRAMES.items():
        for family, pair in enumerate(pairs):
            group = f'scope-state/{split}/intent-{family}'
            for term, d in enumerate(terms):
                for verb, v in enumerate(('展示', '念出', '输出', '说出')):
                    for process, p in enumerate(('推理过程', '思考步骤', '中间的分析')):
                        for label, template in enumerate(pair):
                            text = template.format(d=d, v=v, p=p)
                            history = () if verb % 2 == 0 else (('user', '我希望得到更准确的判断。'),
                                                                 ('assistant', '我先给了一个初步结果。'))
                            rows.append(_row(group, f'{term}-{verb}-{process}-{label}', text, history,
                                             label == 0, split, 'explicit_deep' if label == 0 else 'intent_negative'))
    for split, scenes in SCENES.items():
        for family, text in enumerate(FOLLOWUPS[split]):
            group = f'scope-state/{split}/followup-{family}'
            for scene, (request, unfinished, complete) in enumerate(scenes):
                for state, answer in enumerate((unfinished, complete)):
                    asks_derivation = any(word in text for word in ('过程', '推法', '怎么', '推演', '步骤'))
                    deep = state == 0 or asks_derivation
                    rows.append(_row(group, f'{scene}-{state}', text,
                                     (('user', request), ('assistant', answer)), deep, split,
                                     'context_deep' if deep else 'context_ordinary'))
                rows.append(_row(group, f'{scene}-simple', text,
                    (('user', '解释一下为什么会有彩虹。'), ('assistant', '阳光通过水滴发生折射，后面还可以补充一些常识。')),
                    False, split, 'context_ordinary'))
        for scene, (request, unfinished, _) in enumerate(scenes):
            group = f'scope-state/{split}/ordinary-{scene}'
            for index, query in enumerate(('我先自己想一想，你等我一下。', '上次说好的集合时间是几点？',
                                            '只解释一下这里的名词是什么意思。')):
                rows.append(_row(group, str(index), query, (('user', request), ('assistant', unfinished)),
                                 False, split, 'ordinary'))
        group = f'scope-state/{split}/complex'
        rows.append(_row(group, '0', scenes[0][0], (('user', '我提出一个新问题。'),
                                                  ('assistant', '请说出需要满足的条件。')), True,
                         split, 'implicit_complex'))
    for split, pairs in ENGLISH.items():
        for family, pair in enumerate(pairs):
            group = f'scope-state/{split}/intent-en-{family}'
            for term, d in enumerate(('deep reasoning', 'deep thinking', 'in-depth reasoning')):
                for label, template in enumerate(pair):
                    for context, history in enumerate(((), (('user', 'I want a careful answer.'),
                                                           ('assistant', 'The previous result was preliminary.')))):
                        rows.append(_row(group, f'{term}-{label}-{context}', template.format(d=d), history,
                                         label == 0, split, 'explicit_deep' if label == 0 else 'intent_negative'))
    result, cases = [], {}
    for row in rows:
        if row['group'].startswith('scope-state/'):
            row['source'] = 'authored-scope-and-completion-v6'
        key = (*texts(row['text'], row.get('history')), audio_vector(row.get('audio_state')).tobytes())
        if key in cases:
            prior = cases[key]
            if prior['split'] != row['split'] or prior['needs_deep'] != row['needs_deep']:
                raise ValueError('Conflicting scope or completion example')
            continue
        cases[key] = row
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.previous)
    report = audit(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False)+'\n')
    load_rows(args.out)
    report['data_sha256'] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    report['previous_test_role'] = 'untrained regression, excluded from this dataset'
    args.out.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
