"""Generate a small synthetic pilot; this is not a production voice dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


NORMAL = (
    '你好，今天过得怎么样？', '给我讲个简短的睡前故事。', '帮我想想晚饭吃什么。',
    '查一下我之前记录的会议时间。', '我上次说喜欢什么口味的咖啡？', '解释一下什么是数据库。',
    '苹果和梨的味道有什么不同？', '求三乘五的结果。', '给我一句生日祝福。',
    '不用深入思考，简单解释一下这个概念。', '解释“深度思考模式”这个词是什么意思。',
    '我先想一想，稍后再继续。', '帮我查一下之前存的快递地址。', '介绍一下北京的天气特点。',
    '简单比较一下跑步和散步。', '把这句话翻译成英文。', '列举三种常见的水果。',
    '我有点难过，陪我聊一会儿。', '请认真回答，我明天约了谁？', '设计一句简短的广告语。',
    '证明题通常是什么意思？', '十七加二十五是多少？', '求平方函数的导数，只要答案。',
    '不要推导，直接告诉我已经算出的结果。', '不用开启深思模式，告诉我怎么打开设置。',
    '解释一下为什么会有彩虹。', '帮我把这段通知改得礼貌一点。', '推荐一首适合放松的歌。',
    '我记得上周讨论过旅行，最后定的是哪里？', '总结一下刚才已经讨论出的结论。',
    '对“请深入思考”这句话做个语法分析，不用执行它。', '我只是说我自己要仔细想想。',
)

DEEP = (
    '请深入思考这个问题，再给我完整的分析。', '请用深度推理重新检查这个方案。',
    '证明这个递推关系对所有正整数成立，并处理边界条件。',
    '推导优化算法的收敛条件，并证明每一步满足约束。',
    '设计跨区域容灾架构，权衡成本、一致性和恢复时间。',
    '分析高并发下偶发死锁的多种时序，给出修复和验证实验。',
    '在预算有限且不能停机的条件下，制定数据库迁移与回滚方案。',
    '综合长期收益、风险承受能力和资金约束比较这几种方案。',
    '为分布式锁建立故障模型，证明网络分区时的互斥性。',
    '考虑资源、工期和依赖冲突，设计可执行的项目调度方案。',
    '推导这个积分的完整求解过程，并说明换元成立的条件。',
    '逐步证明这个几何结论，检查退化情形并寻找反例。',
    '设计重试、幂等和超时协同工作的协议，分析重复执行风险。',
    '分析服务只在特定负载下丢消息的多种可能原因，设计区分实验。',
    '请仔细深入想一遍，不用急着给答案。', '请严密论证这个算法的正确性和终止性。',
    '比较两种部署策略在峰值延迟、费用和维护复杂度上的长期取舍。',
    '给出同时满足安全、兼容和性能约束的升级方案并评估失败路径。',
    '假设两个节点同时宕机，推导一致性协议仍然成立的充分条件。',
    '请构造反例检验这个猜想，并证明哪些参数范围内它成立。',
    '根据历史需求和现实限制，制定多阶段实施计划并论证优先级。',
    '深入分析这两个选择之间的利弊，说明假设变化如何影响结论。',
    '推导带约束控制问题的最优性条件，检查不可行的边界情况。',
    '设计并发缓存失效协议，证明更新和读取交错时不会返回旧版本。',
    '分析网络抖动、熔断和重试的相互作用，设计稳定性验证方法。',
    '在容错要求和资源限制下，推导最少副本数并证明下界。',
    '请深入论证一下，不要只给我一个直觉结论。',
    '证明这个随机算法的误差上界，并推导所需样本量。',
    '为异构设备设计任务分配算法，分析最坏情况下的延迟和公平性。',
    '构建容量与故障恢复的联合模型，比较两套扩容方案的风险。',
    '制定不丢数据的跨云切换策略，并分析每个失败阶段如何回滚。',
    '请从不同假设出发深入推演这件事的可能结果。',
)

CONTEXT = (
    ('继续。', '严密证明这个协议的正确性。', '只证明了单节点故障，还需要处理双节点故障。', True),
    ('继续讲。', '讲个睡前故事。', '小猫找到了森林里的小屋。', False),
    ('所以结果是多少？', '请推导这个积分。', '推导已经完成，结果为四分之一。', False),
    ('接着推导。', '求带约束优化问题的解。', '刚写出拉格朗日函数，还没处理可行性条件。', True),
    ('我上次说的是哪一天？', '深入分析系统可靠性。', '需要考虑三个失效模型。', False),
    ('那么两个节点同时失效呢？', '证明这个算法在故障时仍然正确。', '目前仅处理单个节点失效。', True),
    ('别展开了，给个结论。', '比较这两种复杂架构。', '完整比较已经完成，第二种更合适。', False),
    ('把剩余的证明完成。', '证明这个递推式收敛。', '还没有排除参数等于零的边界情况。', True),
    ('我先组织一下语言。', '推导最优调度算法。', '我们先列出了资源约束。', False),
    ('那不同约束之间冲突时呢？', '设计高可用容灾方案。', '目前假定延迟和成本要求可以同时满足。', True),
    ('它是什么意思？', '分析这个系统的故障时序。', '副本表示同一份数据的额外拷贝。', False),
    ('请把这个问题深入想一遍。', '解释一下这个方案。', '刚刚只给出了初步直觉。', True),
)


def build():
    """Keep all paraphrases of one seed in the same data split."""
    seeds = [(text, [], False) for text in NORMAL] + [(text, [], True) for text in DEEP]
    seeds += [(text, [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}], deep)
              for text, user, assistant, deep in CONTEXT]
    rows = []
    for index, (text, history, deep) in enumerate(seeds):
        split = ('test' if index % 5 == 0 else 'calibration' if index % 5 == 1 else 'train')
        group = f'synthetic-seed-{index:03d}'
        for variation, prefix in enumerate(('', '嗯，', '麻烦你，', '你好，')):
            rows.append({'id': f'{group}-{variation}', 'group': group, 'split': split,
                         'text': prefix + text, 'history': history, 'needs_deep': deep,
                         'source': 'synthetic-pilot-v2'})
    topics = (
        '数据库复制', '网络缓存', '消息队列', '任务调度', '访问控制', '日志检索',
        '网络路由', '服务扩容', '故障恢复', '文件同步', '视频传输', '存储压缩',
        '旅行路线', '活动排期', '库存管理', '课程安排', '配送计划', '人力分配',
        '设备维护', '能源调配', '实验设计', '项目验收', '备份轮换', '上线切换',
    )
    ordinary = (
        '{topic}是什么，简单解释一下。', '用一句话概括{topic}，不用展开推导。',
        '我之前提到的{topic}是什么？', '把“{topic}”翻译成英文。',
        '“深入分析{topic}”是什么意思？只解释这个表达，不执行分析。',
        '不用深度思考，告诉我{topic}的基本用途。', '请认真回答，{topic}通常用在什么地方？',
        '简单列举{topic}的三个常见术语。', '只给{topic}刚才已经得出的结论。',
        '不要再分析{topic}，我只是想查之前的记录。',
        '请给我一个关于{topic}的简单例子。', '为{topic}拟一个简短标题。',
    )
    complex_tasks = (
        '深入分析{topic}，综合成本、时效和可靠性给出可验证的方案。',
        '在资源受限且不能中断服务的前提下，比较{topic}的两套方案并分析风险。',
        '为{topic}设计满足多项冲突约束的实施策略，论证取舍。',
        '对{topic}建立约束模型，推导最优配置并检查可行性边界。',
        '请用深度思考模式完整审查{topic}的边界情况和失败路径。',
        '分析{topic}中多个因素交互导致的故障，设计区分原因的验证实验。',
        '重新深入推理{topic}的方案，寻找反例并说明结论何时不成立。',
        '深入比较{topic}的长期成本和故障代价，给出分阶段实施与回退计划。',
    )
    for index, topic in enumerate(topics):
        split = ('test' if index % 5 == 0 else 'calibration' if index % 5 == 1 else 'train')
        group = f'synthetic-domain-{index:03d}'
        for item, template in enumerate(ordinary + complex_tasks):
            for variation, prefix in enumerate(('', '嗯，', '麻烦你，', '你好，')):
                rows.append({'id': f'{group}-{item}-{variation}', 'group': group,
                             'split': split, 'text': prefix + template.format(topic=topic),
                             'history': [], 'needs_deep': item >= len(ordinary),
                             'source': 'synthetic-pilot-v2'})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as file:
        for row in build():
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'{len(build())} synthetic rows; no recorded voice or teacher API')


if __name__ == '__main__':
    main()
