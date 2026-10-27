"""Promote inspected synthetic examples into training and keep fresh evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .dialogue_data import _row, audit
from .train import load_rows
from studio.core.utils.reply_modes.decision_features import texts, audio_vector


PAIRS = {
    'validation': (
        ('我这回想实际用{d}来处理，菜单入口不用介绍。', '我这回只问{d}在菜单哪个位置，不用拿它处理。'),
        ('先进行{d}，推演细节你可以不说。', '先说{d}的含义，实际推演这次省掉。'),
        ('上一句说别{d}是口误，现在我希望你认真执行。', '上一句说要{d}是口误，现在我希望你省掉它。'),
        ('你给的太草率，请通过{d}重新得到结论。', '你给的太冗长，请跳过{d}直接复述结论。'),
        ('我用引号写“{d}”是给当前回答选模式。', '我用引号写“{d}”是让你修改这几个字。'),
        ('回答短一点可以，但请先{d}。', '回答详细点可以，但本次不要{d}。'),
        ('不用讲解功能用途，这次我要你真正{d}。', '不用真正执行，这次我要你讲解{d}的功能用途。'),
        ('我是在请你{d}，并非询问你有没有这能力。', '我是在询问你有没有{d}能力，并非要求你现在使用。'),
    ),
    'calibration': (
        ('我指定先{d}，然后只告诉我最终答案。', '我指定只读出{d}这个名称，然后普通回答。'),
        ('先前的快答要求撤回，现在请{d}。', '先前的{d}要求撤回，现在请快答。'),
        ('在这个小问题上也请使用{d}，不要只介绍术语。', '只介绍{d}这个术语，不要在这题上使用。'),
        ('我希望你进入{d}状态，而不是解释设置。', '我希望你解释{d}的设置，而不是进入状态。'),
        ('可以不展示过程，但你需要先{d}。', '可以展示之前的结论，但这回不需要{d}。'),
        ('请按我现在的话来，重新{d}一遍。', '请按我现在的话来，停止{d}。'),
    ),
}

HISTORY = {
    'validation': (
        ('论证不确定能源输入下的资源调度满足所有限制。', '只验证了常规输入，还没有证明异常输入时可行。'),
        ('给我讲讲小松鼠在图书馆找书的故事。', '小松鼠刚走到第一排书架，故事还没结束。'),
    ),
    'calibration': (
        ('推导多租户共享设备时的公平分配算法并证明界限。', '只推到了一般情况，极端负载下的界限尚未证明。'),
        ('介绍几种常见的收纳工具。', '已经说了收纳盒，还能继续介绍袋子和挂钩。'),
    ),
}


def build(sources, context_source):
    rows, identities = [], {}
    for path in sources:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row['id'] in identities:
                previous = identities[row['id']]
                if any(previous.get(key) != row.get(key) for key in ('text', 'history', 'audio_state', 'needs_deep')):
                    raise ValueError('Repeated example identity changed')
                continue
            identities[row['id']] = row
            rows.append({**row, 'split': 'train', 'prior_role': row['split'],
                         'evaluation_role': 'inspected-synthetic-hard-example'})
    for line in Path(context_source).read_text().splitlines():
        row = json.loads(line)
        if row['id'] in identities:
            continue
        identities[row['id']] = row
        rows.append(row if row['split'] == 'test' else {**row, 'split': 'train', 'prior_role': row['split']})
    terms = ('深度思考', '深入思考', '深思', '深度推理', '深度计算', '深入推演')
    for split, pairs in PAIRS.items():
        scenes = HISTORY[split]
        for family, pair in enumerate(pairs):
            group = f'fresh-v5/{split}/scope-{family:02d}'
            for label, template in enumerate(pair):
                for index, term in enumerate(terms):
                    for context in range(3):
                        history = () if context == 0 else (('user', scenes[context-1][0]),
                                                           ('assistant', scenes[context-1][1]))
                        rows.append(_row(group, f'{label}-{index}-{context}', template.format(d=term),
                                         history, label == 0, split,
                                         'explicit_deep' if label == 0 else 'intent_negative'))
        queries = ('先解释刚才那个术语的含义就好。', '我之前说的预约时刻是几点？',
                   '我想自己整理思路，你先等等。', '现在换个轻松的话题吧。')
        for family, text in enumerate(queries):
            group = f'fresh-v5/{split}/ordinary-{family}'
            for context, (request, answer) in enumerate(scenes):
                rows.append(_row(group, str(context), text, (('user', request), ('assistant', answer)),
                                 False, split, 'ordinary'))
        for context, (request, answer) in enumerate(scenes):
            group = f'fresh-v5/{split}/continue-{context}'
            rows.append(_row(group, '0', '接下来余下部分继续完成吧。',
                             (('user', request), ('assistant', answer)), context == 0, split,
                             'context_deep' if context == 0 else 'context_ordinary'))
        group = f'fresh-v5/{split}/complex'
        rows.append(_row(group, '0', scenes[0][0], (), True, split, 'implicit_complex'))
        rows.append(_row(group, '1', scenes[0][0], (('user', '还有一个需要推导的新问题。'),
                                                  ('assistant', '请说出完整要求。')), True, split, 'implicit_complex'))
    extra = (
        ('Could you think deeply about this and only show the final result?', True),
        ('Tell me what deep thinking is called in the settings menu.', False),
        ('I am quoting someone: "think deeply". Please translate that quotation.', False),
        ('I am asking for deeper reasoning now, rather than a definition of it.', True),
        ('Do not show the intermediate work, but please reason deeply first.', True),
        ('Please do not reason deeply this time; restate the result.', False),
    )
    for index, (text, label) in enumerate(extra):
        group = f'fresh-v5/test/english-{index}'
        rows.append(_row(group, '0', text, (), label, 'test', 'explicit_deep' if label else 'intent_negative'))
    result, cases = [], {}
    for row in rows:
        key = (*texts(row['text'], row.get('history')), audio_vector(row.get('audio_state')).tobytes())
        if key in cases:
            previous = cases[key]
            if previous['needs_deep'] != row['needs_deep'] or previous['split'] != row['split']:
                raise ValueError('An evaluated input overlaps training or has conflicting labels')
            continue
        cases[key] = row
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--context-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.sources, args.context_source)
    report = audit(rows)
    report['prior_synthetic_test_role'] = 'training hard examples; not independent evaluation'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False)+'\n')
    load_rows(args.out)
    report['data_sha256'] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
