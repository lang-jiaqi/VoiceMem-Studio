// Exercise runtime dependencies, shared requests, fallback and retries without WebGL.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../studio/pet/live2d-renderer.js'), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));

function environment(options = {}) {
  const scripts = [];
  const context = vm.createContext({
    URL, clearTimeout, setTimeout, location: { href: 'https://demo.test/ui/pet-mobile.html' },
    VM_PET_ASSET_ROOT: 'https://demo.test/pet/v/fixture/',
    VM_PET_CORE_URL: options.coreUrl,
    VM_PET_MOBILE: Boolean(options.mobile),
    document: {
      createElement() { return { remove() { this.removed = true; } }; },
      head: { appendChild(tag) { scripts.push(tag); } },
    },
    ...options.globals,
  });
  vm.runInContext(source, context);
  const canvas = { addEventListener() {} };
  function request(fragment, index = 0) {
    const matches = scripts.filter(tag => tag.src.includes(fragment));
    assert(matches[index], `missing request: ${fragment} (${index})`);
    return matches[index];
  }
  function succeed(fragment, index = 0) {
    const tag = request(fragment, index);
    if (fragment.includes('core')) context.Live2DCubismCore = {};
    else if (fragment.includes('pixi')) context.PIXI = {};
    else if (fragment.includes('cubism4')) context.PIXI.live2d = { Live2DModel: {} };
    tag.onload();
  }
  return { context, scripts, request, succeed, renderer: () => new context.Live2DRenderer(canvas) };
}

test('Core and Pixi start together; shared callers load the plugin once after both', async () => {
  const env = environment();
  const first = env.renderer().ensureRuntime();
  const second = env.renderer().ensureRuntime();
  assert.equal(env.scripts.length, 2);
  env.succeed('pixi.min.js');
  await flush();
  assert.equal(env.scripts.length, 2, 'plugin must wait for Core');
  env.succeed('live2dcubismcore');
  await flush();
  assert.equal(env.scripts.length, 3);
  env.succeed('cubism4');
  await Promise.all([first, second]);
  await env.renderer().ensureRuntime();
  assert.equal(env.scripts.length, 3);
});

test('native local-Core failure falls back to the official CDN while Pixi proceeds', async () => {
  const env = environment();
  const ready = env.renderer().ensureRuntime();
  env.request('live2dcubismcore').onerror();
  env.succeed('pixi.min.js');
  await flush();
  assert(env.request('live2dcubismcore', 0).removed);
  assert(env.request('live2dcubismcore', 1).src.startsWith('https://cubism.live2d.com/'));
  env.succeed('live2dcubismcore', 1);
  await flush();
  env.succeed('cubism4');
  await ready;
});

test('mobile selects a preloaded Core URL without requesting a missing local copy', async () => {
  const coreUrl = 'https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js';
  const env = environment({ coreUrl, mobile: true });
  const ready = env.renderer().ensureRuntime();
  assert.equal(env.request('live2dcubismcore').src, coreUrl);
  env.succeed('live2dcubismcore');
  env.succeed('pixi.min.js');
  await flush();
  env.succeed('cubism4');
  await ready;
  assert.equal(env.scripts.filter(tag => tag.src.includes('live2dcubismcore')).length, 1);
});

test('exhausted Core fallbacks report an error and can retry without stale script listeners', async () => {
  const env = environment();
  const renderer = env.renderer();
  const rejected = assert.rejects(renderer.ensureRuntime(), error => error.code === 'CUBISM_CORE_MISSING');
  env.request('live2dcubismcore').onerror();
  env.succeed('pixi.min.js');
  await flush();
  env.request('live2dcubismcore', 1).onerror();
  await rejected;
  const ready = renderer.ensureRuntime();
  env.succeed('live2dcubismcore', 2);
  await flush();
  env.succeed('cubism4');
  await ready;
  assert.equal(env.scripts.filter(tag => tag.src.includes('pixi.min.js')).length, 1);
});

test('failed Pixi can retry while sharing a still-pending Core request', async () => {
  const env = environment();
  const renderer = env.renderer();
  const rejected = assert.rejects(renderer.ensureRuntime(), /Unable to load/);
  env.request('pixi.min.js').onerror();
  await rejected;
  const ready = renderer.ensureRuntime();
  assert.equal(env.scripts.filter(tag => tag.src.includes('live2dcubismcore')).length, 1);
  env.succeed('pixi.min.js', 1);
  env.succeed('live2dcubismcore');
  await flush();
  env.succeed('cubism4');
  await ready;
});

test('failed display plugin can retry with already-loaded Core and Pixi', async () => {
  const env = environment();
  const renderer = env.renderer();
  const rejected = assert.rejects(renderer.ensureRuntime(), /Unable to load/);
  env.succeed('live2dcubismcore');
  env.succeed('pixi.min.js');
  await flush();
  env.request('cubism4').onerror();
  await rejected;
  const ready = renderer.ensureRuntime();
  await flush();
  env.succeed('cubism4', 1);
  await ready;
  assert.equal(env.scripts.length, 4);
});

test('mobile texture loading matches CORS preloads; desktop options stay compatible', async () => {
  for (const mobile of [false, true]) {
    let receivedOptions;
    const model = { internalModel: { motionManager: { on() {} }, on() {} }, autoUpdate: true };
    const globals = {
      Live2DCubismCore: {},
      PIXI: {
        Application: class { constructor() { this.stage = { addChild() {} }; } },
        live2d: { Live2DModel: { async from(url, options) { receivedOptions = options; return model; } } },
      },
    };
    const env = environment({ mobile, globals });
    const renderer = env.renderer();
    renderer.resize = () => {};
    await renderer.loadModel('https://demo.test/pet/v/fixture/model.model3.json');
    assert.equal(receivedOptions.crossOrigin, mobile ? 'anonymous' : undefined);
    assert.equal(receivedOptions.autoUpdate, false);
    assert.equal(receivedOptions.autoInteract, false);
    assert.equal(env.scripts.length, 0);
    assert.equal(renderer.active, true);
  }
});
