'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');

const hiyoriFiles = [
  'assets/live2d/hiyori/README-LICENSE.txt',
  'assets/live2d/hiyori/hiyori_pro_t11.moc3',
  ...['model3.json', 'physics3.json', 'pose3.json', 'cdi3.json']
    .map(suffix => `assets/live2d/hiyori/hiyori_pro_t11.${suffix}`),
  ...['texture_00.png', 'texture_01.png']
    .map(file => `assets/live2d/hiyori/hiyori_pro_t11.2048/${file}`),
  ...Array.from({ length: 10 }, (_, index) =>
    `assets/live2d/hiyori/motion/hiyori_m${String(index + 1).padStart(2, '0')}.motion3.json`),
];

const rattanFiles = [
  'assets/live2d/rattan/README-SOURCE.md',
  ...['model3.json', 'moc3', 'physics3.json', 'cdi3.json']
    .map(suffix => `assets/live2d/rattan/rattan.${suffix}`),
  'assets/live2d/rattan/idle.motion3.json',
  ...Array.from({ length: 20 }, (_, index) =>
    `assets/live2d/rattan/exp/${index + 1}.exp3.json`),
  'assets/live2d/rattan/exp/loli.exp3.json',
  'assets/live2d/rattan/exp/大Q.exp3.json',
  ...Array.from({ length: 5 }, (_, index) =>
    `assets/live2d/rattan/rattan.2048/texture_0${index}.png`),
];

const sourceFiles = [
  'style.css', 'state.cjs', 'renderer.js', 'voicemem-link.js',
  'audio-lip-sync.js', 'avatar-behavior-controller.js', 'avatar-controller.js',
  'avatar-parameter-controller.js', 'live2d-renderer.js',
  'assets/scene/call-background.png', 'assets/live2d/README.md',
  ...rattanFiles,
  'node_modules/pixi.js/dist/browser/pixi.min.js', 'node_modules/pixi.js/LICENSE',
  'node_modules/pixi-live2d-display/dist/cubism4.min.js', 'node_modules/pixi-live2d-display/LICENSE',
  'THIRD_PARTY_NOTICES.md',
];

// Recognize the previous generated layout only to preserve it during migration.
const legacyFiles = [
  'debug-panel.js',
  // The Hiyori model bundled by the previous Live2D release.
  ...hiyoriFiles,
  // Canvas/PNG pet bundled by the immediately preceding App release.
  'style.css', 'state.cjs', 'renderer.js', 'scene.js', 'voicemem-link.js', 'avatar-rig.js',
  'assets/avatar/calm.png', 'assets/avatar/talk.png', 'assets/avatar/blink.png',
  'assets/avatar/squint-smile.png', 'assets/scene/curtains.png', 'THIRD_PARTY_NOTICES.md',
  // Older generated Live2D payload retained for safe one-time migration.
  'style.css', 'state.cjs', 'renderer.js', 'rig.js', 'voicemem-link.js',
  'tilted-smile.js', 'transparency.js', 'face-calibration.js', 'motion-calibration.js',
  'assets/sit.png', 'assets/lie.png', 'vendor/live2dcubismcore.min.js', 'THIRD_PARTY_NOTICES.md',
  ...['sit', 'lie'].flatMap(pose => [
    ...['model3.json', 'moc3', 'cdi3.json', 'idle.motion3.json', 'blink.motion3.json', 'nod.motion3.json', 'shake.motion3.json']
      .map(suffix => `models/${pose}/noctelle-${pose}.${suffix}`),
    `models/${pose}/noctelle-${pose}.2048/texture_00.png`,
  ]),
  'vendor/pixi.min.js', 'vendor/unsafe-eval.min.js', 'vendor/cubism4.min.js',
  'vendor/pixi.js.LICENSE.txt', 'vendor/@pixi-unsafe-eval.LICENSE.txt', 'vendor/pixi-live2d-display.LICENSE.txt',
];

function desktopHtml(html) {
  const policy = /connect-src [^;]+;/;
  if (!policy.test(html)) throw new Error('Pet connection policy is missing.');
  // The dedicated Electron session further restricts this to the selected observer URL.
  return html.replace(policy, "connect-src 'self' ws://127.0.0.1:* ws://localhost:* ws://[::1]:* wss:; object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none';");
}

async function listFiles(directory) {
  const info = await fs.lstat(directory);
  if (!info.isDirectory()) throw new Error('Pet build directory must be a real directory.');
  const actual = [];
  async function visit(relative = '') {
    for (const entry of await fs.readdir(path.join(directory, relative), { withFileTypes: true })) {
      const file = path.posix.join(relative, entry.name);
      if (entry.isDirectory()) await visit(file);
      else if (entry.isFile()) actual.push(file);
      else throw new Error(`Unexpected pet resource type: ${file}`);
    }
  }
  await visit();
  return actual;
}

async function petSource(apps, env = process.env) {
  if (env.STUDIO_PET_DIR) {
    const explicit = path.resolve(env.STUDIO_PET_DIR);
    if (!(await fs.stat(explicit)).isDirectory()) throw new Error(`Studio pet source is not a directory: ${explicit}`);
    return explicit;
  }
  const bundled = path.resolve(apps, '../pet');
  try {
    if ((await fs.stat(bundled)).isDirectory()) return bundled;
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  return path.resolve(apps, '../../pet');
}

async function preparePet({ apps = path.resolve(__dirname, '..'), destination = path.join(apps, '.pet-runtime') } = {}) {
  const source = await petSource(apps);
  const sourceOf = file => path.join(file.startsWith('node_modules/') ? apps : source, file);
  const inventory = ['index.html', ...sourceFiles];
  const html = desktopHtml(await fs.readFile(path.join(source, 'index.html'), 'utf8'));
  for (const file of sourceFiles) await fs.access(sourceOf(file));
  let actual;
  try { actual = await listFiles(destination); }
  catch (error) { if (error.code !== 'ENOENT') throw error; actual = []; }
  const known = new Set([...inventory, ...legacyFiles]);
  if (actual.some(file => !known.has(file))) throw new Error('Unexpected files in generated .pet-runtime; inspect the build directory before packaging.');
  if (actual.some(file => !inventory.includes(file))) {
    const backup = await fs.mkdtemp(`${destination}.previous-`);
    await fs.rename(destination, path.join(backup, 'resources'));
    console.log(`[desktop] Previous pet resources preserved at ${backup}`);
  }
  await fs.mkdir(destination, { recursive: true });
  for (const file of sourceFiles) {
    await fs.mkdir(path.dirname(path.join(destination, file)), { recursive: true });
    await fs.copyFile(sourceOf(file), path.join(destination, file));
  }
  await fs.writeFile(path.join(destination, 'index.html'), html);
  if ((await listFiles(destination)).sort().join('\n') !== [...inventory].sort().join('\n')) throw new Error('Pet resource inventory mismatch.');
  return inventory;
}

if (require.main === module) preparePet().then(files => console.log(`[desktop] Pet resources prepared: ${files.length} files`))
  .catch(error => { console.error(error.message); process.exitCode = 1; });
module.exports = { sourceFiles, desktopHtml, petSource, preparePet };
