'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const promises = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const dependencies = require('../scripts/ensure-dependencies.cjs');

async function temporary(t) {
  const directory = await promises.mkdtemp(path.join(process.env.VOICEMEM_TEST_TMP || os.tmpdir(), 'studio-dependencies-test-'));
  t.after(() => promises.rm(directory, { recursive: true, force: true }));
  await promises.writeFile(path.join(directory, 'package-lock.json'), '{}');
  return directory;
}

function provideDependencies(directory) {
  for (const [, relative] of dependencies.required) {
    const file = path.join(directory, relative);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, 'fixture');
  }
}

test('missing App runtime dependencies trigger one locked dev install', async t => {
  const apps = await temporary(t), calls = [];
  const installed = dependencies.ensureDependencies({
    apps, env: {
      npm_execpath: '/fixture/npm-cli.js', NODE_ENV: 'production',
      NPM_CONFIG_IGNORE_SCRIPTS: 'true', NPM_CONFIG_OMIT: 'dev',
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
  assert.equal('NPM_CONFIG_OMIT' in calls[0].options.env, false);
  assert.equal('NPM_CONFIG_IGNORE_SCRIPTS' in calls[0].options.env, false);
});

test('complete App runtime dependencies skip npm install', async t => {
  const apps = await temporary(t);
  provideDependencies(apps);
  assert.equal(dependencies.ensureDependencies({
    apps, run: () => { throw new Error('npm should not run'); },
  }), false);
});
