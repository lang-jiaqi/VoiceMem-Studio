const { spawn } = require('node:child_process');
const env = { ...process.env, NODE_TLS_REJECT_UNAUTHORIZED: '1' };
delete env.ELECTRON_RUN_AS_NODE;
const child = spawn(require('electron'), ['.', ...process.argv.slice(2)], { cwd: __dirname, env, stdio: 'inherit', windowsHide: true });
child.on('error', error => { console.error(error); process.exitCode = 1; });
child.on('exit', code => { process.exitCode = code ?? 1; });
