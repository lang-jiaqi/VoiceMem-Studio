'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../ui/index.html'), 'utf8');
const inline = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];

test('desktop launch window stays on the home page; portrait touch phone opens digital', () => {
  assert.ok(inline);
  for (const [width, coarse, expected] of [
    [680, false, null],
    [1440, false, null],
    [390, true, '/ui/digital.html'],
  ]) {
    let destination = null;
    vm.runInNewContext(inline, {
      matchMedia(query) {
        assert.equal(query, '(max-width: 760px) and (pointer: coarse)');
        return { matches: width <= 760 && coarse };
      },
      location: { replace(value) { destination = value; } },
    });
    assert.equal(destination, expected);
  }
});
