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

# RADV GTT-spill fix (measured 5-6x on RDNA2: 8-10 t/s -> 57-62 t/s on a
# 62 GB workload). Mesa < 26.x defaults to spilling large allocations through
# GTT; nogttspill disables that. The Dockerfile already sets this ENV — the
# export here keeps it true even when the image env is overridden by
# compose/run without an explicit value. Set RADV_PERFTEST explicitly in
# compose to override.
export RADV_PERFTEST="${RADV_PERFTEST:-nogttspill}"
echo "RADV_PERFTEST=$RADV_PERFTEST"

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
  "amd_backend": "rocm",
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

# Resolve the max concurrent resident models from config.json (default 5).
# config.json is written above only on first run and never overwritten, so a
# user's hand-edited models_max must be read back here rather than re-derived.
MODELS_MAX="$(python3 - "$CFG" <<'PY'
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
except Exception:
    cfg = {}
print(int(cfg.get("models_max", 5)))
PY
)"

# Resolve the AMD backend (rocm|vulkan) and the matching --device list. On the
# dual-backend image this is the on-instance ROCm-vs-Vulkan switch; the device
# list names every AMD GPU of that backend so offloading is deterministic.
# Empty when no AMD GPUs are detected (llama.cpp auto-selects then).
#
# Per-model backend mode: if ANY model section in the ini carries its own
# `device` key, the router-level --device is OMITTED entirely — llama.cpp's
# router merges its own CLI args into every child preset, so a global --device
# would silently override each section's per-model choice (verified against
# llama.cpp master 85c5522). Sections win by the router not passing one.
DEVICE="$(python3 - "$CFG" "$INI" <<'PY'
import json, os, sys
try:
    cfg = json.load(open(sys.argv[1]))
except Exception:
    cfg = {}

# Parse models.ini just enough to find per-section device= keys (a `device`
# key in [*] is a global default, not a per-model selection, and doesn't
# suppress the router's own flag).
per_model = False
try:
    cur = None
    with open(sys.argv[2]) as f:
        for line in f:
            s = line.strip()
            if s.startswith("[") and s.endswith("]"):
                cur = s[1:-1]
                continue
            if cur in (None, "*") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            if k.strip() == "device" and v.strip():
                per_model = True
                break
except OSError:
    pass

if per_model:
    print("")          # sections own the device; router CLI must stay silent
else:
    backend = cfg.get("amd_backend", "rocm")
    prefix = "Vulkan" if backend == "vulkan" else "ROCm"
    # Count AMD GPUs via the DRM render nodes (Vulkan needs no /dev/kfd).
    try:
        n = len([x for x in os.listdir("/dev/dri") if x.startswith("renderD")])
    except Exception:
        n = 0
    print(",".join(f"{prefix}{i}" for i in range(n)) if n else "")
PY
)"

# Start the llama.cpp router (if not already up).
if ! lsof -ti tcp:8080 -sTCP:LISTEN >/dev/null 2>&1; then
  args=(--models-preset "$INI" --models-max "$MODELS_MAX" --offline
        --host 0.0.0.0 --port 8080 --metrics)
  [ -n "$DEVICE" ] && args+=(--device "$DEVICE")
  [ -n "$ROUTER_API_KEY" ] && args+=(--api-key "$ROUTER_API_KEY")
  /usr/local/bin/llama-server "${args[@]}" \
    >>"$LOG_DIR/router.out.log" 2>>"$LOG_DIR/router.err.log" &
  echo "started llama.cpp router on 0.0.0.0:8080"
fi

# Start the LlamaForge dashboard.
cd /app/backend
exec python3 server.py
