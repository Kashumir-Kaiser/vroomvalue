#!/bin/sh
set -eu

rm -rf .next
npm run build

mkdir -p .next/standalone/.next
cp -r .next/static .next/standalone/.next/static

exec env HOSTNAME=0.0.0.0 PORT="${PORT:-3000}" node .next/standalone/server.js
