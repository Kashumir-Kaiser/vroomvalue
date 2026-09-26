#!/bin/sh
set -eu

# /app/.next is a Docker named-volume mount in development. Remove its
# contents, not the mount point itself, or Linux returns "Resource busy".
mkdir -p .next
find .next -mindepth 1 -maxdepth 1 -exec rm -rf {} +

npm run build

mkdir -p .next/standalone/.next
cp -r .next/static .next/standalone/.next/static

exec env HOSTNAME=0.0.0.0 PORT="${PORT:-3000}" node .next/standalone/server.js
