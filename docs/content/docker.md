---
title: Docker & ROCm
section: guides
order: 5
---

# Docker & ROCm

LlamaForge-ROCm adds two things upstream LlamaForge lacks: **ROCm (AMD GPU)
support** and a **Docker deployment**. This page covers both — how to build the
ROCm variant, and how to run it in a container on Unraid (or any Docker host
with AMD GPUs).

## ROCm support

On a Linux host with AMD GPUs, the Setup and Build tabs detect them and build
llama.cpp with `GGML_HIP=ON` instead of CUDA. CUDA/CPU/Metal are unchanged —
ROCm is additive.

### AMD device detection

`backend/hardware.py`'s `detect_amd_gpus()` reads the ROCm KFD topology
(`/sys/class/kfd/kfd/topology/nodes/*/properties` for the GPU name and
`gfx_target_version`, plus each node's `mem_banks/*/properties` for VRAM size)
and the DRM render nodes (`/dev/dri/renderD*`). This works with just the
`amdgpu` kernel driver loaded — no `rocm-smi` binary required. Live
util/temp/used telemetry (`routes._amd_telemetry()`) uses `rocm-smi --json`
when present and falls back to VRAM-only KFD data otherwise.

### Configurable AMDGPU_TARGETS

The build targets a broad default set of AMD architectures:

```
gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201
```

| Arch | GPUs |
|------|------|
| `gfx908` | Instinct MI100 |
| `gfx90a` | Instinct MI200 (MI210/MI250/MI250X) |
| `gfx942` | Instinct MI300 (MI300X/MI300A) |
| `gfx1030` | RX 6000 series, Radeon Pro V620 |
| `gfx1100`/`gfx1101`/`gfx1102` | RX 7000 series |
| `gfx1150`/`gfx1151` | RX 8000 series |
| `gfx1200`/`gfx1201` | RX 9000 series |

> Vega (`gfx900`/`gfx906`) and RX 5000 (`gfx1010`) are omitted — ROCm 7.x
> dropped them. Use a ROCm 6.x base image if you need those older GPUs.

Narrow the list to your own GPU(s) for a much faster build:

- **Bare metal:** set `amd_gpu_targets` in `config.json`, e.g. `"gfx1030;gfx1100"`.
  When left blank, the detected archs are used automatically.
- **Docker:** pass `--build-arg AMDGPU_TARGETS=gfx1030` (or edit the
  `docker-compose.yml` build arg).

### AMD-aware VRAM-fit ratings

The Discover tab's FITS/TIGHT/OFFLOAD ratings and the "Will it run?" tok/s
estimates now include AMD GPUs. `backend/vram_predict.py` carries memory
bandwidth presets for Instinct (MI50–MI300X), Radeon Pro (W6600–W7900, V620),
and RX 5000/6000/7000 parts, keyed off the KFD node name. Unknown AMD parts
fall back to the default bandwidth like any unknown GPU.

## Vulkan support (AMD RDNA2)

On **RDNA2** cards (gfx1030 — Radeon Pro V620, RX 6000) with no hardware matrix
cores, ROCm's multi-GPU layer-split forces activations across PCIe every token
(no NVLink on these cards), collapsing a 3-GPU split to ~7 tok/s. Vulkan's
multi-GPU path does not suffer this penalty — it holds ~16 tok/s regardless of
split or context length.

Select Vulkan by setting `"amd_backend": "vulkan"` in `config.json` (default
`"rocm"`), or from the Setup tab's **AMD backend** dropdown. The router is
restarted with `--device Vulkan0,Vulkan1,Vulkan2` (or `HIP0,HIP1,HIP2`).
Vulkan needs no `AMDGPU_TARGETS` and no `/dev/kfd` — only the DRM render nodes,
which are already passed through. Device detection prefers `vulkaninfo --json`
and falls back to the DRM card vendor/device IDs.

### Build the image

```bash
# Dual-backend (default): HIP + Vulkan in one binary, switch at runtime
docker build -t llamaforge-rocm .

# Narrow to a single backend for a smaller/faster build
docker build --build-arg AMD_BACKEND=rocm -t llamaforge-rocm .
docker build --build-arg AMD_BACKEND=vulkan -t llamaforge-vulkan .
```

The default image builds on the ROCm toolchain image (24.04 + ROCm 7.2.1) with
the Vulkan headers + loader added on top, so a single `llama-server` carries
both backends. The runtime stage keeps `rocm-smi` (perf-level pinning +
telemetry) and adds `libvulkan1` so the Vulkan backend can reach the host's
RADV ICD. `--device` selects the backend at runtime, so one instance serves
both ROCm and Vulkan without a rebuild or a second container.

## Docker deployment

### Build

```bash
# Broad default targets (slow build, works on most AMD GPUs)
docker build -t llamaforge-rocm .

# Narrow to your GPU for a much faster build
docker build --build-arg AMDGPU_TARGETS=gfx1030 -t llamaforge-rocm .

# Pin a specific llama.cpp ref
docker build --build-arg LLAMACPP_REF=master -t llamaforge-rocm .
```

The image is a two-stage build on `rocm/dev-ubuntu-24.04:7.2.1` (matching
llama.cpp's own ROCm image): stage 1 clones and compiles llama.cpp with
`GGML_HIP=ON` + `AMDGPU_TARGETS` on the `-complete` toolchain image, stage 2
copies the `llama-server` binary and its HIP backend `.so` files into a runtime
image.

### Run (docker-compose)

```bash
# 1. Edit docker-compose.yml: set AMDGPU_TARGETS and the renderD* device list
#    to match your GPUs (see below).
# 2. Bring it up:
docker compose up -d
```

The compose file mounts two volumes — `./config` (config.json, models.ini,
logs) and `./models` (your GGUF files) — and exposes the router on `8080` and
the dashboard on `8090`.

### GPU passthrough (mandatory)

Three things must line up or the container sees **zero GPUs**:

1. **`/dev/kfd`** — the ROCm kernel-fusion driver device.
2. **one render node per GPU** — `/dev/dri/renderD128`, `renderD129`,
   `renderD130`, … Add one `devices:` line per physical GPU.
3. **`group_add: video`** — render nodes are owned `root:video` mode `0660`,
   so the container user must be in the `video` group.

```yaml
devices:
  - /dev/kfd:/dev/kfd
  - /dev/dri/renderD128:/dev/dri/renderD128
  - /dev/dri/renderD129:/dev/dri/renderD129
  # ... one line per GPU
group_add:
  - video
ipc: host
shm_size: "16gb"
```

`ipc: host` and a large `shm_size` matter: llama.cpp uses big shared memory for
multi-GPU and large-context KV caches.

### Verify the GPUs are visible

```bash
docker exec llamaforge-rocm rocm-smi
# or, if rocm-smi isn't in the image:
docker exec llamaforge-rocm ls /dev/dri/renderD*
```

If `rocm-smi` lists no GPUs, check the three passthrough items above — a
missing render node is the usual cause.

### Unraid

On Unraid, add the container via **Docker → Add Container** (or drop the
`docker-compose.yml` into a folder and use the Compose plugin):

1. Set the repository to your built image (or build from the repo).
2. Add the device mappings: `/dev/kfd` and one `/dev/dri/renderD*` per GPU.
3. Add `--group-add video` (or set the extra parameter `--group-add=video`).
4. Add `--ipc=host` and `--shm-size=16g`.
5. Map ports `8080` and `8090`, and mount two paths: one for config, one for
   models.

The container runs the router and dashboard on `0.0.0.0` internally, so the
port mappings expose them to your LAN. Set `ROUTER_API_KEY` to require a key on
the exposed router.

## Networking notes

The container binds the dashboard and router to `0.0.0.0` (via `panel_host` and
`router_host` in the generated `config.json`), which is the right choice for a
Docker host: the dashboard's Host/Origin guard is relaxed when `panel_host` is
not `127.0.0.1`, so the host can reach it through the port mapping. If you
prefer host networking, set `network_mode: host` in the compose file and drop
the `ports:` block — the container then binds the host's interfaces directly.
