const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../studio/apps/ui/markdown.js'), 'utf8');

function environment({ math = false, mathFails = false } = {}) {
  class Node {
    constructor(tag, text = '') {
      this.tag = tag; this.children = []; this.text = text; this.attributes = {};
      this.isConnected = true;
      this.classList = { add() {} };
    }
    append(...nodes) {
      for (const node of nodes) {
        if (node.tag === '#fragment') this.append(...node.children);
        else this.children.push(node);
      }
    }
    replaceChildren(...nodes) { this.text = ''; this.children = []; this.append(...nodes); }
    set textContent(text) { this.text = String(text); this.children = []; }
    get textContent() { return this.text + this.children.map(child => child.textContent).join(''); }
    setAttribute(key, value) { this.attributes[key] = value; }
    cloneNode(deep) {
      const copy = new Node(this.tag, this.text);
      copy.attributes = { ...this.attributes }; copy.className = this.className;
      if (deep) copy.children = this.children.map(child => child.cloneNode(true));
      return copy;
    }
  }
  const frames = new Map(); let next = 0;
  const mathCalls = [];
  const window = {};
  if (math) window.katex = { render(text, node, options) {
    mathCalls.push({ text, options });
    if (mathFails) throw new Error('Invalid fixture expression');
    node.append(new Node('math', text));
  }};
  const context = vm.createContext({
    URL, location: { href: 'https://fixture.test/' }, window,
    document: {
      createElement: tag => new Node(tag),
      createTextNode: text => new Node('#text', text),
      createDocumentFragment: () => new Node('#fragment'),
    },
    requestAnimationFrame: fn => { frames.set(++next, fn); return next; },
    cancelAnimationFrame: id => frames.delete(id),
  });
  vm.runInContext(source, context);
  return {
    host: new Node('div'), render: context.window.VMMarkdown.render, frames, mathCalls,
    flush() { const jobs = [...frames.values()]; frames.clear(); jobs.forEach(fn => fn()); },
  };
}

function tags(node, tag) {
  return [...(node.tag === tag ? [node] : []), ...node.children.flatMap(child => tags(child, tag))];
}

test('headings, emphasis, lists, quotes, links and code use semantic DOM', () => {
  const e = environment();
  e.render(e.host, '# 标题\n**粗体**和*斜体*、~~删除~~、`x_1`\n\n- 一\n- 二\n\n1. 三\n2. 四\n\n> 引用\n\n[说明](https://example.test/a(b))\n\n```python\nx = 2 ** 3\n```');
  for (const tag of ['h1', 'strong', 'em', 'del', 'code', 'ul', 'ol', 'blockquote', 'a', 'pre']) {
    assert(tags(e.host, tag).length, tag);
  }
  assert.equal(tags(e.host, 'a')[0].href, 'https://example.test/a(b)');
  assert.equal(tags(e.host, 'pre')[0].textContent, 'x = 2 ** 3');
  assert(!e.host.textContent.includes('**粗体**'));
});

test('tables remove separator syntax and keep escaped/code pipes within cells', () => {
  const e = environment();
  e.render(e.host, '| 名称 | 内容 |\n| --- | --- |\n| A | **值** |\n| B | `a|b` |');
  assert.equal(tags(e.host, 'th').length, 2);
  assert.equal(tags(e.host, 'td').length, 4);
  assert.equal(tags(e.host, 'code')[0].textContent, 'a|b');
  assert(!e.host.textContent.includes('---'));
});

test('raw HTML stays text, images do not fetch, and unsafe URLs are not links', () => {
  const e = environment();
  e.render(e.host, '<script>alert(1)</script>\n\n![描述](https://example.test/image.png) [危险](javascript:alert(1)) [邮件](mailto:help@example.test)');
  assert.equal(tags(e.host, 'script').length, 0);
  assert.equal(tags(e.host, 'img').length, 0);
  assert(e.host.textContent.includes('<script>alert(1)</script>'));
  assert(e.host.textContent.includes('描述 危险 邮件'));
  assert.equal(tags(e.host, 'a').length, 1);
  assert.equal(tags(e.host, 'a')[0].rel, 'noopener noreferrer');
});

test('ordinary mathematical operators and variable names remain intact', () => {
  const e = environment();
  const text = '2 ** 3，2**3，a*b，x_1，foo_bar';
  e.render(e.host, text);
  assert.equal(e.host.textContent, text);
});

test('streaming paints coalesce and open emphasis can preview a heard prefix', () => {
  const e = environment();
  e.render(e.host, '**已', { streaming: true });
  e.render(e.host, '**已经说过', { streaming: true });
  assert.equal(e.frames.size, 1);
  e.flush();
  assert.equal(tags(e.host, 'strong')[0].textContent, '已经说过');
  assert.equal(e.host.textContent, '已经说过');
});

test('completion/reset cancels stale streaming paints', () => {
  const e = environment();
  e.render(e.host, '旧回复', { streaming: true });
  e.render(e.host, '**完成**');
  e.flush();
  assert.equal(e.host.textContent, '完成');
  e.render(e.host, '旧回复', { streaming: true });
  e.render(e.host, '');
  e.flush();
  assert.equal(e.host.textContent, '');
});

test('detached chat nodes are not painted and callbacks run after paint', () => {
  const e = environment(); let called = 0;
  e.render(e.host, '内容', { streaming: true, onRender: () => called++ });
  e.host.isConnected = false;
  e.flush();
  assert.equal(called, 0);
  e.host.isConnected = true;
  e.render(e.host, '内容', { streaming: true, onRender: () => called++ });
  e.flush();
  assert.equal(called, 1);
});

test('inline and display LaTeX are parsed before Markdown escapes and emphasis', () => {
  const e = environment({ math: true });
  e.render(e.host, String.raw`查询 $q$，键 $k_i$。

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]
最后乘 $V$。`);
  assert.equal(e.mathCalls.length, 4);
  assert.equal(e.mathCalls[1].text, 'k_i');
  assert(e.mathCalls[2].text.includes(String.raw`\frac{QK^\top}{\sqrt{d_k}}`));
  assert.equal(e.mathCalls[2].options.displayMode, true);
  assert.equal(tags(e.host, 'em').length, 0);
  for (const { options } of e.mathCalls) {
    assert.equal(options.trust, false);
    assert.equal(options.maxExpand, 200);
    assert.equal(options.maxSize, 10);
  }
  assert.notEqual(e.mathCalls[0].options.macros, e.mathCalls[1].options.macros);
});

test('dollar blocks, parenthesis math and math fences retain surrounding prose', () => {
  const e = environment({ math: true });
  e.render(e.host, String.raw`前文。
$$x^2+1$$后文。

行内 \(x-y\) 和 $\alpha$。

\[z=1\]

\begin` + '\n\n```latex\n\\frac{a}{b}\n```\n解释。');
  assert.equal(e.mathCalls.length, 5);
  assert(e.host.textContent.includes('后文。'));
  assert(e.host.textContent.includes('解释。'));
  assert.equal(tags(e.host, 'pre').length, 0);
});

test('code and prices stay literal while table math may contain unescaped pipes', () => {
  const e = environment({ math: true });
  e.render(e.host, String.raw`金额 $5 和 $10；代码 \$5，\[不是公式\]` + '\n\n```python\nvalue = "$x$"\n```\n\n| 量 | 定义 |\n| --- | --- |\n| A | $a|b$ |');
  assert(e.host.textContent.includes('金额 $5 和 $10'));
  assert(tags(e.host, 'pre')[0].textContent.includes('"$x$"'));
  assert.equal(tags(e.host, 'td').length, 2);
  assert(e.mathCalls.some(call => call.text === 'a|b'));
});

test('coordinate tuples render as math without changing prices or code', () => {
  const e = environment({ math: true });
  e.render(e.host, '坐标 $(x, y)$，向量 $[a, b, c]$，数值 $(-1, 2.5)$；价格 $5 和 $10。代码 `$(x, y)$`。');
  assert.deepEqual(e.mathCalls.map(call => call.text), ['(x, y)', '[a, b, c]', '(-1, 2.5)']);
  assert(e.host.textContent.includes('价格 $5 和 $10'));
  assert.equal(tags(e.host, 'code')[0].textContent, '$(x, y)$');
  e.render(e.host, '坐标 $(x, y)$', { streaming: true }); e.flush();
  assert(!e.host.textContent.includes('$'));
});

test('incomplete streaming formulas wait for closure and completed formulas are cached', () => {
  const e = environment({ math: true });
  e.render(e.host, String.raw`先看 $$\frac{a`, { streaming: true }); e.flush();
  assert.equal(e.mathCalls.length, 0);
  assert.equal(e.host.textContent, '先看 …');
  e.render(e.host, String.raw`先看 $$\frac{a}{b}$$。`);
  assert.equal(e.mathCalls.length, 1);
  e.render(e.host, String.raw`先看 $$\frac{a}{b}$$。继续解释`, { streaming: true }); e.flush();
  assert.equal(e.mathCalls.length, 1);
  assert(e.host.textContent.includes('继续解释'));
});

test('malformed or unavailable math falls back to text without failing the reply', () => {
  for (const math of [true, false]) {
    const e = environment({ math, mathFails: true });
    e.render(e.host, String.raw`前文 $\badcommand{x}$ 后文。`);
    assert(e.host.textContent.includes(String.raw`$\badcommand{x}$`));
    assert(e.host.textContent.includes('后文。'));
  }
  const e = environment({ math: true });
  e.render(e.host, '$$' + 'a'.repeat(8193) + '$$');
  assert.equal(e.mathCalls.length, 0);
});

test('bundled KaTeX typesets attention and rejects trusted HTML/resource commands', () => {
  const katex = require('../studio/apps/ui/vendor/katex/katex.min.js');
  const options = { displayMode: true, trust: false, strict: 'ignore', maxExpand: 200, maxSize: 10 };
  const html = katex.renderToString(String.raw`\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V`, options);
  assert(html.includes('katex-html'));
  assert(html.includes('<math'));
  assert(html.includes('<mfrac>'));
  for (const input of [String.raw`\includegraphics{https://fixture.test/a.png}`, String.raw`\href{javascript:alert(1)}{x}`, String.raw`\htmlStyle{position:fixed}{x}`]) {
    const rendered = katex.renderToString(input, options);
    assert(!/<(?:img|a)\b/.test(rendered));
    assert(!/style=["'][^"']*position:fixed/.test(rendered));
  }
  assert.throws(() => katex.renderToString(String.raw`\def\a{\a}\a`, options), /expansion/);
});
