"""Author contrastive dialogue decisions with explicitly held-out wording families."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .train import load_rows
from studio.core.utils.reply_modes.decision_features import texts, audio_vector


# Each pair is a single wording family, including both intent polarities.
REQUESTS = {
    'train': (
        ('这道题请深度思考之后再说。', '这一题不用深度思考，直接作答。'),
        ('你能开启深思来处理吗？', '你能介绍深思功能在哪里吗？'),
        ('别急着作答，先深入推理。', '别深入推理，先给我个简答。'),
        ('能不能深度计算一遍？', '“深度计算”这个说法是什么意思？'),
        ('你不能先深入想清楚再回答吗？', '你不能把“深入想清楚”翻译一下吗？'),
        ('我的要求是深度推理以后给答案。', '我刚才并没有要求你深度推理。'),
        ('请认真做完整的深度推演。', '我自己会认真推演，你先随便聊聊。'),
        ('不是让你随口答，是让你深入推理。', '不是让你深入推理，是要一句概括。'),
        ('刚才没想透，这次启动深度思考重新做。', '刚才已经想透了，这次不用再推导。'),
        ('先别赶时间，请从头严密推演。', '先别严密推演，报一下先前的结论。'),
        ('这回请打开深度思考模式。', '这回只解释“打开深度思考模式”这句话。'),
        ('我想要你仔细深入考虑这件事。', '我想要的是我自己仔细考虑，你只听着。'),
        ('刚才说不用想，现在改成要深入分析。', '刚才说深入分析，现在改成不用展开。'),
        ('你不是可以深思吗，麻烦这题用上。', '你是不是有个叫深思的功能？'),
        ('别只重复结论，请深入推导根据。', '别再重复推导，请告诉我最后结论。'),
        ('请深度思索后，交代假设和推理过程。', '请转述“深度思索后交代假设”这段文字。'),
        ('我不是要快答，我要你用深思来回答。', '我不是要深思，我要你快答。'),
        ('你怎么没开深度思考？这次开着重新回答。', '你怎么会把“深度思考”这个词读错？'),
        ('用你的深入思考能力，重新检查后再回答。', '用一句话说明深入思考能力是指什么。'),
        ('呃，这个，麻烦先深度思考再回答。', '呃，这个，麻烦别深度思考，简短说。'),
        ('这次我明确要求你做深度推理。', '这次我明确没有要求你做深度推理。'),
        ('请切到深思模式，再把问题处理一遍。', '请读出“切到深思模式”，不执行这条指令。'),
        ('不要省略思考，先严密论证再作答。', '不要再做论证，只复述已有答案。'),
        ('我改口了，这次要充分深入思考。', '我改口了，这次不要充分深入思考。'),
    ),
    'validation': (
        ('麻烦用深度思考处理，简单的题也请这么做。', '麻烦保持快速回答，题目里提到深思也不用开启。'),
        ('难道不能先深度推理完，再把答案告诉我吗？', '难道“深度推理”不就是个功能名称吗？解释一下。'),
        ('我说的不是补充几句，我是要求你深入思考再重答。', '我说的不是深入思考，我只需要你补充一句结论。'),
        ('你刚才只给了直觉，这次请启用深思重做。', '你刚才已经推过了，这次给直觉回答就行。'),
        ('可以先深度想想，等想清楚了再开口吗？', '可以先别深度想，只把刚才的结果念出来吗？'),
        ('所谓认真回答，我这里指的是实际开启深度思考。', '所谓“实际开启深度思考”，这几个字怎么翻译？'),
        ('这个请求不复杂，我仍然希望你深入推演。', '这个请求可能复杂，不过本轮先不要推演。'),
        ('先别沿用前面的快答，请改用深思完成。', '先别沿用前面的深思，请改成快答。'),
    ),
    'calibration': (
        ('请调用深度思考，把理由充分想透再讲。', '请解释“调用深度思考”这句话，不用照着做。'),
        ('我需要你花心思深度推导，不是立刻报数。', '我需要的是立刻报数，不是再花心思推导。'),
        ('可否把深思打开，然后重新判断这件事？', '可否告诉我深思开关的名字是什么？'),
        ('先前说不用深思作废，这次认真深入推理。', '先前要求深思作废，这次普通回答即可。'),
        ('就算答案很短，也请先深入思考。', '就算题目很长，也不用展开深度思考。'),
        ('你能否真正深思一轮，而不是复述结论？', '你能否复述“真正深思一轮”，不执行它？'),
    ),
    'test': (
        ('这一次请采用深度思考来作答。', '这一次请介绍“采用深度思考”的含义。'),
        ('别再直接猜了，先把深度推理用起来。', '别再深度推理了，我现在只想知道原结论。'),
        ('我是在要求你深思，不是在询问功能介绍。', '我是在询问深思的功能介绍，不是在要求你执行。'),
        ('难不成你不能深度想过以后再给我结论吗？', '难不成你把我自己要想一想理解成让你深思了？'),
        ('认真做一遍深度计算，并核查中间步骤。', '只说明“深度计算并核查步骤”这几个词的意思。'),
        ('先想透再答，这轮我指定使用深入推理。', '先不用想透，这轮我指定使用普通回答。'),
        ('你没有按我的要求深思，请深思后重来。', '你没有按我的要求简答，请停止深思再重来。'),
        ('把这回的回答切成深思，我要推敲过的结果。', '把这回的回答切成简答，我要之前的结果。'),
    ),
}

TOPICS = (
    '几种常见的果汁口味', '四种家用电池的区别', '八十以内偶数的个数',
    '周末活动的一句宣传语', '刚才提到的那件事情', '一个简短的旅行清单',
    '两款入门路由器的规格', '一句英文问候语的翻译',
)

ENGLISH_REQUESTS = {
    'train': (
        ('Please think deeply before answering this question.', 'Please answer briefly without deep reasoning.'),
        ('Can you use deep reasoning and work it through?', 'Can you explain what the term deep reasoning means?'),
        ('I want you to reason carefully in depth this time.', 'I will think carefully myself; just listen for now.'),
        ('Do not rush. Use deep thought to reconsider your answer.', 'Do not reconsider it; repeat the result you already obtained.'),
        ('Could you enable deep thinking for this simple request?', 'Could you translate the words enable deep thinking?'),
        ('I changed my mind: please reason deeply now.', 'I changed my mind: skip deep reasoning now.'),
        ('Why not think this through deeply before replying?', 'Why is this feature called deep thinking?'),
        ('Use deliberate reasoning and verify the assumptions.', 'Just define deliberate reasoning in a sentence.'),
    ),
    'validation': (
        ('This time I explicitly request a deeply reasoned answer.', 'This time I only want the meaning of deeply reasoned answer.'),
        ('Would you switch on deep reasoning and try again?', 'Would you tell me where the deep reasoning option is?'),
        ('Even though it is easy, please think deeply about it.', 'Even though it is hard, give a quick answer without reasoning.'),
    ),
    'calibration': (
        ('Take time to reason in depth before you respond.', 'Do not take extra time; just state your earlier conclusion.'),
        ('I am asking you to use deep thought, not just describe it.', 'I am asking you to describe deep thought, not perform it.'),
    ),
    'test': (
        ('Please activate deep thinking and reconsider the problem.', 'Please explain the phrase activate deep thinking.'),
        ('Could you actually think it through in depth this time?', 'Could you quote the words think it through in depth?'),
        ('Set aside the quick reply and use deep reasoning instead.', 'Set aside deep reasoning and give me a quick reply instead.'),
    ),
}

CONTEXTS = (
    (),
    (('user', '我们讨论一下日常生活。'), ('assistant', '可以，告诉我你现在的问题。')),
    (('user', '刚才那件事情有什么结论？'), ('assistant', '我先给了一个简短的回答。')),
    (('user', '前面的问题需要在费用、可靠性、效率和上线工期之间权衡，请先分析约束。'),
     ('assistant', '已经深入分析了各个方案，刚才给出的长回答涵盖了推导、风险和失败条件。'),
     ('user', '前一个问题先告一段落，接下来我会提出另一个要求。'),
     ('assistant', '好的，前面的结论已经保留。接下来我会以你当前的要求为准。')),
)

COMPLEX = {
    'train': (
        '建立离线同步协议，分析重复提交与断网重连的所有交错，并证明数据不会丢失。',
        '预算和停机时间都受限，比较三种迁移路径的风险并设计验证与回滚。',
        '证明递推算法在所有允许参数下收敛，找出反例并分类处理边界。',
        '服务随机卡死，综合锁顺序、重试和超时构造故障时序并提出区分实验。',
        '同时考虑人手、资金、依赖冲突和截止日期，推导可执行的最优排期。',
        '推导含不等式约束的最优控制条件，证明每一步满足可行性。',
        '在不能丢消息又要限制延迟的条件下设计幂等协议，并验证重复与乱序。',
        '分析两个区域同时故障时的一致性保证，论证恢复策略成立的条件。',
    ),
    'validation': (
        '为长时间离线的终端设计冲突解决机制，证明合并后所有副本最终一致。',
        '一个任务只能偶发复现超卖，重建可能的并发顺序并设计能排除各假设的实验。',
        '资源与容错要求冲突时，推导最少部署数量的下界并证明可达性。',
        '在多个项目争抢人员时设计调度方案，评估最坏延迟与公平性之间的权衡。',
    ),
    'calibration': (
        '异地系统在时钟不准与消息延迟同时发生时应如何保证互斥？请构造并验证协议。',
        '业务峰值期间迁移不可暂停，制定满足费用、可恢复性和数据完整性的分阶段方案。',
        '请证明随机选择算法的误差上界，并据此推导足够的样本数。',
        '考虑断电、重放和部分写入，建立日志恢复模型并论证不会损坏已确认记录。',
    ),
    'test': (
        '当节点重启与网络延迟叠加时，论证租约协议仍然互斥，并找出不成立的边界。',
        '多人并发修改同一文档，设计不丢更新的合并规则，证明最终结果与消息顺序无关。',
        '预算缩减且上线日期不能改，重新分配资源并量化风险、收益和失败回退路径。',
        '推导该数列在参数变化时的极限，分类讨论不收敛的区域并构造反例。',
    ),
}

ORDINARY = {
    'train': (
        '帮我想想晚餐选面还是饭。', '请认真说，柠檬水是什么味道？',
        '列个常见蔬菜的小表格。', '十二乘以七，给个数就行。',
        '我刚才说的约会在周几？', '解释一下什么叫网络延迟。',
        '我有点烦，你陪我说两句。', '给我写一句儿童节祝福。',
        '我明天的日程是什么？', '帮我简单比较一下铅笔和圆珠笔。',
        '求平方函数的导数，直接说结果。', '我自己考虑一会儿再继续。',
    ),
    'validation': (
        '认真回答一下，乌龙茶和花茶有什么区别？', '请把十一加上十四的结果告诉我。',
        '给我一个简短的下班问候。', '我前面提到的地点在哪里？',
        '解释“推导过程”是什么意思。', '我今天有点累，想随便聊聊。',
    ),
    'calibration': (
        '想一想明天早餐买点什么。', '介绍一下保温杯的用途。',
        '之前我们约的时间是几点？', '用一句话解释什么是平均数。',
        '别继续分析，我只是想倾诉。', '六十五减去二十七，只报结果。',
    ),
    'test': (
        '认真讲讲雨伞和遮阳伞的区别。', '我提过的电影名字是什么？',
        '想个简短的周年庆口号。', '把十七乘以九的答案念出来。',
        '用一句话解释“严谨论证”的字面含义。', '先等我组织好语言，我在自己想事情。',
    ),
}

FOLLOWUPS = {
    'train': ('继续往下说。', '把后面接着讲。', '接下去呢？', '再接着来。',
              '后面还没说完吧。', '然后怎么做？', '你把剩下的讲完。', '再往下推一段。'),
    'validation': ('接上刚刚没完成的部分。', '后续部分请继续。', '那下一步呢？'),
    'calibration': ('下面接着说下去吧。', '把还没讲的部分说完。', '往后还有什么？'),
    'test': ('顺着刚才继续往下。', '余下的内容呢？', '后续你接着处理吧。'),
}

REASONS = (
    ('证明一个排序算法在所有输入下都正确。', '目前只讨论了一般情况，边界情况的证明还未完成。'),
    ('设计在费用与可靠性约束下的升级方案。', '刚建好约束模型，还需要推导方案并验证失败路径。'),
    ('分析某个服务随机丢失更新的原因。', '只查过单线程，还要分析并发交错以及设计区分实验。'),
    ('请推导数列收敛所需的条件。', '必要条件已经得到，还没证明充分性和处理边界。'),
)

SIMPLE = (
    ('讲一个小兔子的故事。', '小兔子走到河边，看到一艘小船，故事还没有讲完。'),
    ('介绍三种常见的水果。', '刚说了苹果和香蕉，接下来还要介绍第三种。'),
    ('讲讲普通周末可以做些什么。', '刚提到散步和看电影，还可以继续说几种日常活动。'),
    ('给我介绍一个旅游景点。', '我才讲了入口的位置，接着可以讲里面有哪些普通设施。'),
)


def _row(group, identity, text, history, deep, split, category):
    return {'id': f'{group}/{identity}', 'group': group, 'split': split,
            'text': text, 'history': [{'role': role, 'content': content} for role, content in history],
            'needs_deep': deep, 'source': 'authored-contrastive-dialogue-v2',
            'category': category, 'label_provenance': 'authored-intent-policy'}


def build():
    """Keep complete wording families, polarities and contexts in one partition."""
    rows = []
    for split, pairs in REQUESTS.items():
        for family, pair in enumerate(pairs):
            group = f'dialogue-v2/{split}/intent-{family:02d}'
            for label, instruction in enumerate(pair):
                for context, history in enumerate(CONTEXTS):
                    forms = (instruction, '嗯那个，'+instruction.translate(str.maketrans('', '', '，。？！、')))
                    for channel, text in enumerate(forms):
                        rows.append(_row(group, f'bare-{label}-{context}-{channel}', text, history,
                                         label == 0, split, 'explicit_deep' if label == 0 else 'intent_negative'))
                for topic, name in enumerate(TOPICS):
                    for context, history in enumerate(CONTEXTS):
                        forms = (f'关于{name}，{instruction}', f'{instruction}我问的是{name}。')
                        for order, text in enumerate(forms):
                            rows.append(_row(group, f'{label}-{topic}-{context}-{order}', text,
                                             history, label == 0, split, 'explicit_deep' if label == 0 else 'intent_negative'))
    for split, pairs in ENGLISH_REQUESTS.items():
        contexts = ((), (('user', 'We were comparing a few everyday options.'),
                         ('assistant', 'I gave a short preliminary answer.')))
        for family, pair in enumerate(pairs):
            group = f'dialogue-v2/{split}/intent-en-{family:02d}'
            for label, instruction in enumerate(pair):
                for context, history in enumerate(contexts):
                    for form, suffix in enumerate(('', ' The topic is a short list of common household items.')):
                        rows.append(_row(group, f'{label}-{context}-{form}', instruction+suffix,
                                         history, label == 0, split,
                                         'explicit_deep' if label == 0 else 'intent_negative'))
    for split, tasks in COMPLEX.items():
        for family, task in enumerate(tasks):
            group = f'dialogue-v2/{split}/complex-{family:02d}'
            for context, history in enumerate(CONTEXTS):
                for style, prefix in enumerate(('', '嗯，麻烦你，', '呃，就是，')):
                    rows.append(_row(group, f'{context}-{style}', prefix+task, history, True, split, 'implicit_complex'))
    for split, tasks in ORDINARY.items():
        for family, task in enumerate(tasks):
            group = f'dialogue-v2/{split}/ordinary-{family:02d}'
            for context, history in enumerate(CONTEXTS):
                for style, prefix in enumerate(('', '嗯，', '那个，我想问，')):
                    rows.append(_row(group, f'{context}-{style}', prefix+task, history, False, split, 'ordinary'))
    for split, followups in FOLLOWUPS.items():
        for family, text in enumerate(followups):
            group = f'dialogue-v2/{split}/followup-{family:02d}'
            for deep, contexts in ((True, REASONS), (False, SIMPLE)):
                for index, (request, answer) in enumerate(contexts):
                    history = (('user', request), ('assistant', answer))
                    for prefix, prior in enumerate(((), (('user', '我们接着前面的话题聊。'), ('assistant', '好的。')))):
                        rows.append(_row(group, f'{int(deep)}-{index}-{prefix}', text, prior+history,
                                         deep, split, 'context_deep' if deep else 'context_ordinary'))
    return rows


def audit(rows):
    """Fail on missing coverage, duplicate inputs and wording-family leakage."""
    cases, families = {}, {}
    for row in rows:
        family = row['group']
        if family in families and families[family] != row['split']:
            raise ValueError('A wording family crosses partitions')
        families[family] = row['split']
        key = (*texts(row['text'], row.get('history')), audio_vector(row.get('audio_state')).tobytes())
        if key in cases:
            previous = cases[key]
            if previous['split'] != row['split'] or previous['needs_deep'] != row['needs_deep']:
                raise ValueError('A semantic input crosses splits or contradicts its label')
            raise ValueError('Duplicate input needs explicit deduplication')
        cases[key] = row
    splits = {}
    for split in ('train', 'validation', 'calibration', 'test'):
        selected = [row for row in rows if row['split'] == split]
        categories = Counter(row['category'] for row in selected)
        expected = {'explicit_deep', 'intent_negative', 'implicit_complex', 'ordinary', 'context_deep', 'context_ordinary'}
        if not expected <= set(categories):
            raise ValueError(f'{split}: missing decision categories')
        for category in expected:
            if not any(row.get('history') for row in selected if row['category'] == category):
                raise ValueError(f'{split}: missing contextual {category}')
        splits[split] = {'rows': len(selected), 'groups': len({row['group'] for row in selected}),
                         'deep': sum(row['needs_deep'] for row in selected),
                         'with_history': sum(bool(row.get('history')) for row in selected),
                         'categories': dict(categories)}
    return {'rows': len(rows), 'splits': splits, 'synthetic_only': True,
            'split_unit': 'complete authored wording family; both polarities/all contexts/all audio views stay together'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--speech-out', type=Path)
    parser.add_argument('--speech-per-split', type=int, default=24)
    args = parser.parse_args()
    rows = build()
    report = audit(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False)+'\n')
    load_rows(args.out)
    report['data_sha256'] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    if args.speech_out:
        selected = []
        for split in ('train', 'validation', 'calibration', 'test'):
            candidates = [row for row in rows if row['split'] == split]
            candidates.sort(key=lambda row: hashlib.sha256(row['id'].encode()).hexdigest())
            counts = Counter()
            for row in candidates:
                category = row['category']
                if counts[category] >= args.speech_per_split//6:
                    continue
                identity = hashlib.sha256(row['id'].encode()).hexdigest()[:20]
                selected.append({**row, 'id': f'speech-v2-{split}-{identity}-0'})
                counts[category] += 1
        with args.speech_out.open('x', encoding='utf-8') as output:
            for row in selected:
                output.write(json.dumps(row, ensure_ascii=False)+'\n')
        report['speech_source_rows'] = len(selected)
    args.out.with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
