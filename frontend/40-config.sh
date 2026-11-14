#!/bin/sh
# Runs at container start (nginx image executes /docker-entrypoint.d/*.sh).
# Writes the backend URL from the environment into the static config, so the same image works
# against any backend: loose coupling via a configurable REST endpoint.
set -eu
: "${API_BASE_URL:=http://localhost:8000}"
printf 'window.APP_CONFIG = { API_BASE_URL: "%s" };\n' "$API_BASE_URL" > /usr/share/nginx/html/js/config.js
echo "frontend: API_BASE_URL=$API_BASE_URL"
