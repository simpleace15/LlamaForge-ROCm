#!/usr/bin/env bash
# LlamaForge-ROCm container entrypoint.
#
# The image ships a pre-built llama-server (ROCm/HIP) at /usr/local/bin/llama-server,
# so there is no llama.cpp source checkout to build inside the container. This
# script writes a config.json pointing at the mounted volumes, then starts the
# llama.cpp router and the LlamaForge dashboard.
set -e

CONFIG_DIR="${CONFIG_DIR:-/app/config}"
MODELS_DIR="${MODELS_DIR:-/app/models}"
LOG_DIR="${LOG_DIR:-/app/logs}"
CFG="$CONFIG_DIR/config.json"
INI="$CONFIG_DIR/models.ini"

# Point the backend at the mounted config volume (config.json + models.ini).
export LLAMAFORGE_CONFIG_DIR="$CONFIG_DIR"

mkdir -p "$CONFIG_DIR" "$MODELS_DIR" "$LOG_DIR"

# Pin AMD GPUs to their high performance level. On passive/datacenter cards
# (e.g. Radeon Pro V620 / Navi 21) the driver's "auto" perf level parks the
# cards at low clocks during inference, which tanks memory-bandwidth-bound
# throughput (~5x slower). rocm-smi is present in the -complete runtime image.
# Best-effort: ignore failure so the container still starts on hosts without
# ROCm tooling or with non-AMD GPUs.
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --setperflevel high >/dev/null 2>&1 || true
  echo "set AMD GPU performance level to high"
fi

# Write config.json on first run (never overwrite a user's existing one).
if [ ! -f "$CFG" ]; then
  cat > "$CFG" <<EOF
{
  "llama_src": "",
  "build_dir": "",
  "server_bin": "/usr/local/bin/llama-server",
  "models_ini": "$INI",
  "model_dirs": ["$MODELS_DIR"],
  "router_port": 8080,
  "panel_port": 8090,
  "panel_host": "0.0.0.0",
  "router_host": "0.0.0.0",
  "router_api_key": "",
  "cmake_flags": {},
  "amd_gpu_targets": "",
  "git_remote": "https://github.com/ggml-org/llama.cpp"
}
EOF
  echo "Wrote $CFG (edit it, or set ROUTER_API_KEY, to change ports/host/key)."
fi

# Ensure a models.ini with a global section exists (llama-server refuses to
# start without it).
if [ ! -f "$INI" ]; then
  cat > "$INI" <<'INI'
version = 1

[*]
ctx-size = 8192
flash-attn = on
jinja = true
n-gpu-layers = 99
load-on-startup = false
INI
  echo "Created $INI"
fi

# Optional API key from the environment (recommended when exposing the router).
if [ -n "$ROUTER_API_KEY" ]; then
  python3 - "$CFG" <<'PY'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p))
cfg["router_api_key"] = __import__("os").environ["ROUTER_API_KEY"]
json.dump(cfg, open(p, "w"), indent=2)
PY
fi

# Start the llama.cpp router (if not already up).
if ! lsof -ti tcp:8080 -sTCP:LISTEN >/dev/null 2>&1; then
  args=(--models-preset "$INI" --models-max 1 --offline
        --host 0.0.0.0 --port 8080 --metrics)
  [ -n "$ROUTER_API_KEY" ] && args+=(--api-key "$ROUTER_API_KEY")
  /usr/local/bin/llama-server "${args[@]}" \
    >>"$LOG_DIR/router.out.log" 2>>"$LOG_DIR/router.err.log" &
  echo "started llama.cpp router on 0.0.0.0:8080"
fi

# Start the LlamaForge dashboard.
cd /app/backend
exec python3 server.py
