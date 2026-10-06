#!/usr/bin/env node
'use strict';
// Test-only Docker replacement. It never contacts a real Docker daemon.
const fs = require('node:fs');
const args = process.argv.slice(2);
fs.appendFileSync(process.env.VOICEMEM_SMOKE_DOCKER_LOG, `${JSON.stringify(args)}\n`);
if (args[0] === 'context') console.log('"unix:///synthetic/docker.sock"');
else if (args.includes('port')) console.log(`127.0.0.1:${process.env.VOICEMEM_SMOKE_PORT}`);
else if (!args.includes('up')) process.exitCode = 1;
