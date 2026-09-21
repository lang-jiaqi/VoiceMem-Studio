'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const promises = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const dependencies = require('../scripts/ensure-dependencies.cjs');

test('Node version preflight rejects unsupported runtimes before installing dependencies', () => {
  assert.doesNotThrow(() => dependencies.assertNodeVersion('22.12.0'));
  assert.doesNotThrow(() => dependencies.assertNodeVersion('24.0.0'));
  assert.throws(() => dependencies.assertNodeVersion('22.11.99'), /Node\.js >=22\.12\.0/);
  assert.throws(() => dependencies.assertNodeVersion('invalid'), /当前为 invalid/);
});

async function temporary(t) {
  const directory = await promises.mkdtemp(path.join(process.env.VOICEMEM_TEST_TMP || os.tmpdir(), 'studio-dependencies-test-'));
  t.after(() => promises.rm(directory, { recursive: true, force: true }));
  await promises.writeFile(path.join(directory, 'package-lock.json'), '{}');
  return directory;
}

function provideDependencies(directory) {
  for (const [name, relative] of dependencies.required) {
    const file = path.join(directory, relative);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    if (name === 'Electron') {
      fs.writeFileSync(file, 'fixture-electron');
      const executable = path.join(directory, 'node_modules/electron/dist/fixture-electron');
      fs.mkdirSync(path.dirname(executable), { recursive: true });
      fs.writeFileSync(executable, 'fixture');
    } else fs.writeFileSync(file, 'fixture');
  }
}

test('missing App runtime dependencies trigger one locked dev install', async t => {
  const apps = await temporary(t), calls = [];
  const installed = dependencies.ensureDependencies({
    apps, env: {
      npm_execpath: '/fixture/npm-cli.js', NODE_ENV: 'production',
      NPM_CONFIG_IGNORE_SCRIPTS: 'true', NPM_CONFIG_OMIT: 'dev',
      ELECTRON_SKIP_BINARY_DOWNLOAD: '1', npm_config_production: 'true',
    },
    run: (file, args, options) => {
      calls.push({ file, args, options });
      provideDependencies(apps);
      return { status: 0 };
    },
  });
  assert.equal(installed, true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].file, process.execPath);
  assert.deepEqual(calls[0].args, [
    '/fixture/npm-cli.js', 'ci', '--include=dev', '--ignore-scripts=false', '--no-audit', '--no-fund',
  ]);
  assert.equal(calls[0].options.cwd, apps);
  assert.equal(calls[0].options.env.npm_config_omit, '');
  assert.equal(calls[0].options.env.npm_config_ignore_scripts, 'false');
  assert.equal('NODE_ENV' in calls[0].options.env, false);
  assert.equal('NPM_CONFIG_OMIT' in calls[0].options.env, false);
  assert.equal('NPM_CONFIG_IGNORE_SCRIPTS' in calls[0].options.env, false);
  assert.equal('ELECTRON_SKIP_BINARY_DOWNLOAD' in calls[0].options.env, false);
  assert.equal('npm_config_production' in calls[0].options.env, false);
});

test('an Electron package without its downloaded executable is repaired', async t => {
  const apps = await temporary(t); const calls = [];
  provideDependencies(apps);
  await promises.rm(path.join(apps, 'node_modules/electron/dist/fixture-electron'));
  assert.deepEqual(dependencies.missingDependencies(apps).map(([name]) => name), ['Electron']);
  assert.equal(dependencies.ensureDependencies({
    apps,
    run: () => { calls.push('npm ci'); provideDependencies(apps); return { status: 0 }; },
  }), true);
  assert.deepEqual(calls, ['npm ci']);
});

test('complete App runtime dependencies skip npm install', async t => {
  const apps = await temporary(t);
  provideDependencies(apps);
  assert.equal(dependencies.ensureDependencies({
    apps, run: () => { throw new Error('npm should not run'); },
  }), false);
});
