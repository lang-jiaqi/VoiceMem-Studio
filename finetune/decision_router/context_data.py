"""Keep basic current questions ordinary even after unfinished complex dialogue."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .dialogue_data import _row, audit
from .train import load_rows
from studio.core.utils.reply_modes.decision_features import texts, audio_vector


SCENES = {
    'train': (
        ('证明并发队列在多个读写者下仍然线性一致。', '现在只证明了单个写者，多个写者交错的情况尚未论证。'),
        ('在不同资源约束下推导生产线的最优排程。', '还没有处理设备临时故障时的边界条件。'),
        ('讲讲小狐狸寻找朋友的故事。', '小狐狸走到了山脚下，故事还有后半段。'),
        ('介绍常见的文具种类。', '我只说了笔和本子，接下来还可以介绍其他文具。'),
    ),
    'validation': (
        ('论证消息乱序时事务恢复协议仍然正确。', '刚分析完正常顺序，乱序到达和重放尚未证明。'),
        ('在多种风险和成本约束下设计异地备份的切换。', '目前还没有解决多个地点同时故障时的冲突。'),
        ('讲一个小海豚出海探险的故事。', '小海豚发现了岛屿，后面的故事还没有讲完。'),
        ('说说常见的室内盆栽。', '刚讲了绿萝，还可以继续介绍其他植物。'),
    ),
    'calibration': (
        ('推导非线性控制器稳定的充分条件。', '只处理了无扰动情形，还要推导有扰动时的稳定条件。'),
        ('为配送系统制定兼顾人手、时间和故障的排班。', '刚建完约束，尚未证明排班可行并分析最坏情况。'),
        ('讲一个小鹿穿过雨林的故事。', '小鹿找到了一条小路，故事仍有下半段。'),
        ('介绍日常厨房用具。', '刚提了锅和碗，还有其他普通用具可以接着说。'),
    ),
    'test': (
        ('为共享相册建立不会丢失修改的合并协议并证明正确性。', '刚说明单人编辑，跨设备并发修改的证明还没有完成。'),
        ('论证自主设备在电源与通信同时异常时的安全边界。', '单个故障已分析完，两个故障同时出现还需要继续推导。'),
        ('给我讲一个小刺猬寻找雨伞的故事。', '小刺猬到了树洞外，余下的故事还没说。'),
        ('介绍一些适合野餐的普通食物。', '只说了面包和水果，还有几类食物没有介绍。'),
    ),
}

ASKS = {
    'train': (
        '先说明{term}这个名词指什么。', '你刚刚说的{term}是什么意思呀？',
        '不用往下做，先给{term}一个简单定义。', '这里提到{term}，用一句大白话说说。',
        '我原先记下的{item}是什么？', '提醒我一下，上次提过的{item}。',
        '先只报已经得到的数值，不用重新论证。', '刚才的结果再念一遍就行。',
        '这题先放一边，我自己再琢磨琢磨。', '我先组织语言，你等我一下。',
        '我不想聊这个了，陪我随便说两句。', '我们换个轻松话题，讲个小笑话。',
    ),
    'validation': (
        '等等，{term}的字面含义是什么？', '先别推进，简单解释一下{term}。',
        '回忆一下，我当时说的{item}是哪一个？', '不用再演算，最后数字是几？',
        '我想自己静一静再考虑。', '这个话题暂时打住，陪我闲聊一会儿。',
    ),
    'calibration': (
        '你把{term}这个术语说得通俗一点。', '我只问以前记录的{item}，不展开分析。',
        '请重复现成的结论，别再算。', '我自己先考虑，你先别往下讲。',
    ),
    'test': (
        '你提到的{term}，到底是个什么概念？', '用最简单的一句话告诉我{term}的意思。',
        '我之前留的{item}，你能提醒我一下吗？', '前面说过的{item}是什么来着？',
        '把已算好的结果报一下即可。', '让我自己理理思路，先别继续了。',
        '我现在只想找个人聊会儿天。',
    ),
}

CONTINUE = {
    'train': ('把还没讲完的接着说下去。', '接着完成后面的内容。', '现在继续往后吧。'),
    'validation': ('那剩下的呢，请接着来。', '刚才没完成的你继续。'),
    'calibration': ('沿着之前未完的内容继续吧。',),
    'test': ('你接着把后半部分讲下去。', '那后面的情况接着说呀。'),
}

EXPLICIT = {
    'train': (('先用深思把本轮问题想透，不用讲功能介绍。', '只介绍深思的功能，这轮不用真的运行。'),
              ('我需要你深入思考后作答，题目简单也如此。', '我需要你简单作答，题目复杂也别开启深思。')),
    'validation': (('请先深度推理，然后给出精简的回答。', '请只解释深度推理这个词，保持普通回答。'),),
    'calibration': (('这次需要的方式是深入思考，请执行。', '这次只解释深入思考这四个字，别执行。'),),
    'test': (('我指定这一题使用深思，请实际推敲后再答。', '我指定这一题快速回答，请把深思关掉。'),
             ('不是在让你介绍模式，请真正深度思考。', '不是在让你开启模式，请介绍深度思考这个名词。')),
}


def build(previous):
    rows = [json.loads(line) for line in Path(previous).read_text().splitlines()]
    rows = [row for row in rows if row['split'] != 'test']
    terms = ('冗余', '重放', '回滚', '一致性', '边界条件', '冲突合并')
    items = ('见面地点', '预计工期', '预算上限', '偏好的颜色', '联系时间', '预约日期')
    for split, scenes in SCENES.items():
        for family, template in enumerate(ASKS[split]):
            group = f'context-v4/{split}/ordinary-{family:02d}'
            for topic, (term, item) in enumerate(zip(terms, items)):
                for scene, (request, answer) in enumerate(scenes):
                    text = template.format(term=term, item=item)
                    history = (('user', request), ('assistant', answer))
                    rows.append(_row(group, f'{topic}-{scene}', text, history, False, split, 'ordinary'))
        for family, text in enumerate(CONTINUE[split]):
            group = f'context-v4/{split}/continue-{family:02d}'
            for scene, (request, answer) in enumerate(scenes):
                rows.append(_row(group, str(scene), text, (('user', request), ('assistant', answer)),
                                 scene < 2, split, 'context_deep' if scene < 2 else 'context_ordinary'))
        for family, pair in enumerate(EXPLICIT[split]):
            group = f'context-v4/{split}/intent-{family:02d}'
            for label, text in enumerate(pair):
                for scene, (request, answer) in enumerate(scenes):
                    rows.append(_row(group, f'{label}-{scene}', text,
                                     (('user', request), ('assistant', answer)), label == 0, split,
                                     'explicit_deep' if label == 0 else 'intent_negative'))
        for scene, (request, answer) in enumerate(scenes[:2]):
            group = f'context-v4/{split}/complex-{scene}'
            for form, history in enumerate(((), (('user', '我有个需要认真推导的新问题。'),
                                                 ('assistant', '请把具体约束说出来。')))):
                rows.append(_row(group, str(form), request, history, True, split, 'implicit_complex'))
    result, cases = [], {}
    for row in rows:
        if row['group'].startswith('context-v4/'):
            row['source'] = 'authored-context-contrast-v4'
        key = (*texts(row['text'], row.get('history')), audio_vector(row.get('audio_state')).tobytes())
        if key in cases:
            prior = cases[key]
            if prior['split'] != row['split'] or prior['needs_deep'] != row['needs_deep']:
                raise ValueError('Conflicting semantic input')
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
    report['previous_test_role'] = 'regression-only; excluded from all new partitions'
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
