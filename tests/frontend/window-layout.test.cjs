'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { studioWindowLayout } = require('../../studio/apps/window-layout.cjs');

test('the home window has room for the language selector and both style cards', () => {
  const layout = studioWindowLayout(true, { x: 0, y: 30, width: 1920, height: 1050 });
  assert.deepEqual(layout, { x: 600, y: 275, width: 720, height: 560, minWidth: 480, minHeight: 380 });
});

test('Studio bounds and minimum sizes fit small work areas and offset displays', () => {
  for (const area of [
    { x: 0, y: 30, width: 1512, height: 850 },
    { x: 0, y: 24, width: 1280, height: 656 },
    { x: -1280, y: -240, width: 1280, height: 720 },
    { x: 1920, y: 100, width: 800, height: 480 },
  ]) for (const home of [true, false]) {
    const layout = studioWindowLayout(home, area);
    assert.ok(layout.x >= area.x + 16);
    assert.ok(layout.y >= area.y + 16);
    assert.ok(layout.x + layout.width <= area.x + area.width - 16);
    assert.ok(layout.y + layout.height <= area.y + area.height - 16);
    assert.ok(layout.minWidth <= layout.width);
    assert.ok(layout.minHeight <= layout.height);
  }
});

test('a large work area retains the full conversation size', () => {
  const layout = studioWindowLayout(false, { x: 0, y: 30, width: 2560, height: 1320 });
  assert.equal(layout.width, 1440);
  assert.equal(layout.height, 920);
  assert.equal(layout.minWidth, 1080);
  assert.equal(layout.minHeight, 680);
});

test('packaged desktop includes the window layout owner', () => {
  assert.ok(require('../../studio/apps/package.json').build.files.includes('window-layout.cjs'));
});
