'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '../ui/index.html'), 'utf8');
test('home page does not redirect normal browser sessions by viewport', () => {
  assert.doesNotMatch(html, /location\.replace\('\/ui\/digital\.html'\)/);
  assert.match(html, /<section id="welcome"/);
});
