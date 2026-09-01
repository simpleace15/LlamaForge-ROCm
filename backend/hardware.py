"""Detect CPU/GPU and recommend CMake build flags + runtime defaults.

Windows + Linux: NVIDIA CUDA and AMD ROCm/HIP are the primary accelerators,
with a CPU-only fallback. macOS: Apple Silicon unified memory with a Metal
build (no CUDA/ROCm). Platform branching lives in osplat; this module just
asks it.

AMD detection reads the ROCm KFD topology (sysfs) plus the DRM render nodes,
so it works on any Linux host with the amdgpu driver loaded — no rocm-smi
binary required. The AMDGPU_TARGETS list is configurable (config.json
`amd_gpu_targets`) so a user can compile for one GPU or a broad set.
"""
import os, re, subprocess

import osplat

# Broad-but-sane default set of AMD GPU architectures for a ROCm/HIP build.
# Covers Vega (gfx900/906), MI50/MI60 (gfx906), MI100 (gfx908), MI200 (gfx90a),
# MI300 (gfx942), RX 5000 (gfx1010), RX 6000 (gfx1030), RX 7000 (gfx1100/1101/1102).
# Users should narrow this to their own GPU(s) via config.json `amd_gpu_targets`
# for faster builds; this default trades build time for "just works" coverage.
AMDGPU_TARGETS_DEFAULT = ("gfx900;gfx906;gfx908;gfx90a;gfx942;"
                          "gfx1010;gfx1030;gfx1100;gfx1101;gfx1102")

def _run(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""

def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except Exception:
        return ""

# Compute-capability -> CUDA arch number used by CMAKE_CUDA_ARCHITECTURES
def detect_gpus():
    if osplat.IS_MAC:
        return []                       # no NVIDIA on Apple Silicon; Metal instead
    out = _run(["nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap",
                "--format=csv,noheader,nounits"])
    gpus = []
    for ln in out.strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) >= 3:
            cc = f[3] if len(f) > 3 and f[3] and f[3] != "[N/A]" else ""
            gpus.append({"index": int(f[0]), "name": f[1],
                         "vram_mib": int(f[2]) if f[2].isdigit() else None,
                         "compute_cap": cc})
    return gpus

# gfx_target_version (KFD) -> gfxNNN[letter] string. AMD's encoding is not a
# clean positional decode (gfx906 vs gfx90a vs gfx942 disagree), so this is a
# lookup table for the common archs, with a best-effort positional fallback for
# anything unknown. "" if unparseable.
_AMD_GFX_NAMES = {
    90000: "gfx900", 90010: "gfx90a", 90011: "gfx90c",
    90600: "gfx906", 90800: "gfx908", 90402: "gfx942",
    101000: "gfx1010", 101002: "gfx1012", 103000: "gfx1030",
    110000: "gfx1100", 110100: "gfx1101", 110200: "gfx1102",
    120000: "gfx1200", 120100: "gfx1201",
}

def _amd_gfx_name(ver):
    try:
        v = int(ver)
    except (TypeError, ValueError):
        return ""
    if v in _AMD_GFX_NAMES:
        return _AMD_GFX_NAMES[v]
    # Best-effort positional decode for unknown values: major*10000 + minor*100 + step.
    major, minor, step = v // 10000, (v // 100) % 100, v % 100
    letter = {10: "a", 11: "b", 12: "c"}.get(step, str(step))
    return f"gfx{major}{minor}{letter}"

def _amd_kfd_nodes():
    """Yield (name, gfx_arch, vram_bytes) per AMD GPU from the KFD topology.

    Reads /sys/class/kfd/kfd/topology/nodes/*/properties (name + gfx target)
    and each node's mem_banks/*/properties (size_in_bytes). GPU nodes carry a
    gpu_id; CPU nodes do not, so they are skipped. Never raises.
    """
    base = "/sys/class/kfd/kfd/topology/nodes"
    if not os.path.isdir(base):
        return
    for node in sorted(os.listdir(base)):
        props = _read(os.path.join(base, node, "properties"))
        # GPU nodes carry a non-zero gfx_target_version; CPU/IO nodes report 0.
        # (Older kernels exposed a "gpu_id" field here — do not rely on it, it
        # is absent on modern kernels such as Unraid 7.x's.)
        if not props:
            continue
        m = re.search(r"gfx_target_version\s+(\d+)", props)
        if not m or int(m.group(1)) == 0:
            continue
        name = ""
        gfx = ""
        m = re.search(r"name\s+(\S+)", props)
        if m:
            name = m.group(1)
        m = re.search(r"gfx_target_version\s+(\d+)", props)
        if m:
            gfx = _amd_gfx_name(m.group(1))
        vram = 0
        banks = os.path.join(base, node, "mem_banks")
        if os.path.isdir(banks):
            for bank in sorted(os.listdir(banks)):
                bp = _read(os.path.join(banks, bank, "properties"))
                m = re.search(r"size_in_bytes\s+(\d+)", bp)
                if m:
                    vram += int(m.group(1))
        yield name, gfx, vram

def detect_amd_gpus():
    """AMD GPUs via ROCm KFD + DRM render nodes. Returns [{index, name,
    vram_mib, gfx_arch}]. Empty on non-Linux or when no amdgpu driver is loaded."""
    if not osplat.IS_LINUX:
        return []
    gpus = []
    for i, (name, gfx, vram_bytes) in enumerate(_amd_kfd_nodes()):
        gpus.append({
            "index": i,
            "name": name or f"AMD GPU {i}",
            "vram_mib": int(vram_bytes / (1024 * 1024)) if vram_bytes else None,
            "gfx_arch": gfx,
        })
    # Fallback: if KFD topology is unavailable but render nodes exist, report
    # them so the user at least sees *something* (VRAM/arch unknown).
    if not gpus:
        render = sorted(glob_render_nodes())
        gpus = [{"index": i, "name": f"AMD GPU (render node {r})",
                 "vram_mib": None, "gfx_arch": ""}
                for i, r in enumerate(render)]
    return gpus

def glob_render_nodes():
    """DRM render node paths (/dev/dri/renderD128, renderD129, ...)."""
    try:
        return [os.path.join("/dev/dri", n) for n in sorted(os.listdir("/dev/dri"))
                if n.startswith("renderD")]
    except Exception:
        return []

# ---------------------------------------------------------------- Vulkan
# Vulkan has no KFD dependency (that's ROCm-only), so AMD GPUs are enumerated
# from the Vulkan ICDs + the DRM card vendor/device IDs instead. This is what
# lets a Vulkan build see the same cards a ROCm build does, without /dev/kfd.

# AMD PCI vendor id (0x1002) and the device ids we can name. 0x73a1 = Radeon
# Pro V620 (Navi 21 / gfx1030) — Tyler's cards. Unknown AMD device ids still
# count as Vulkan-capable GPUs; only the name is generic.
_AMD_DEVICE_NAMES = {
    "0x73a1": "Radeon Pro V620",
    "0x73bf": "Radeon RX 6900 XT",
    "0x73a5": "Radeon RX 6900 XT",
    "0x73a2": "Radeon Pro W6800",
    "0x744c": "Radeon RX 7900 XTX",
    "0x7448": "Radeon RX 7900 XT",
    "0x74a0": "Radeon Pro W7900",
}

def _drm_amd_cards():
    """Yield (device_id_hex, name) per AMD DRM card from /sys/class/drm.

    Reads each card's device/vendor sysfs files; only AMD (0x1002) cards are
    returned. Never raises. This is the Vulkan fallback when `vulkaninfo` is
    absent — it counts AMD Vulkan-capable GPUs without any Vulkan tooling.
    """
    base = "/sys/class/drm"
    if not os.path.isdir(base):
        return
    for card in sorted(os.listdir(base)):
        if not card.startswith("card"):
            continue
        dev = _read(os.path.join(base, card, "device", "device"))
        vendor = _read(os.path.join(base, card, "device", "vendor"))
        if not dev or not vendor:
            continue
        if vendor.lower() != "0x1002":
            continue
        dev = dev.lower()
        yield dev, _AMD_DEVICE_NAMES.get(dev, "AMD GPU")

def _vulkaninfo_gpus():
    """Parse `vulkaninfo --json` for AMD device name + VRAM, if present.

    Returns a list of {name, vram_mib} for AMD devices, or [] when vulkaninfo
    is absent or reports nothing. Best-effort: any parse failure yields [].
    """
    out = _run(["vulkaninfo", "--json"], timeout=20)
    if not out:
        return []
    try:
        import json as _json
        data = _json.loads(out)
    except Exception:
        return []
    gpus = []
    # vulkaninfo --json nests devices under VkPhysicalDeviceProperties; the
    # exact path varies by version, so walk the tree for dicts that carry both
    # a deviceName and a vendorID of 0x1002 (AMD).
    def _walk(node):
        if isinstance(node, dict):
            name = node.get("deviceName")
            vid = node.get("vendorID")
            if name and vid == 0x1002:
                vram = 0
                # VRAM lives in memoryHeaps (VK_MEMORY_HEAP_DEVICE_LOCAL_BIT=1).
                for heap in (node.get("memoryHeaps") or []):
                    if isinstance(heap, dict) and heap.get("flags", 0) & 1:
                        vram = max(vram, int(heap.get("size", 0) or 0))
                gpus.append({"name": name,
                             "vram_mib": int(vram / (1024 * 1024)) if vram else None})
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)
    _walk(data)
    return gpus

def detect_vulkan_gpus():
    """AMD GPUs visible to a Vulkan build. Returns [{index, name, vram_mib,
    gfx_arch}], same shape as detect_amd_gpus() so the dashboard's GPU
    telemetry and VRAM-fit ratings keep working.

    Prefers `vulkaninfo --json` (device name + VRAM); falls back to the DRM
    card vendor/device IDs (count + name only, VRAM unknown). gfx_arch is
    left "" — Vulkan doesn't expose the gfx target the way KFD does, and the
    build doesn't need it (no AMDGPU_TARGETS for Vulkan).
    """
    if not osplat.IS_LINUX:
        return []
    gpus = []
    for i, g in enumerate(_vulkaninfo_gpus()):
        gpus.append({"index": i, "name": g["name"], "vram_mib": g["vram_mib"],
                     "gfx_arch": ""})
    if gpus:
        return gpus
    # Fallback: count AMD DRM cards. VRAM/arch unknown, but the user sees the
    # right number of Vulkan-capable GPUs.
    return [{"index": i, "name": name, "vram_mib": None, "gfx_arch": ""}
            for i, (dev, name) in enumerate(_drm_amd_cards())]


def device_list(backend, count):
    """Comma-separated `--device` list for a backend: 'ROCm0,ROCm1' or
    'Vulkan0,Vulkan1'. Empty string when count <= 0 (caller should leave the
    device unset and let llama.cpp auto-select).

    Device naming matches `llama-server --list-devices` exactly: llama.cpp
    names HIP devices ROCm* (GGML_CUDA_NAME = "ROCm" when built with
    GGML_USE_HIP, ggml-cuda.h) and Vulkan devices Vulkan* — verified live on
    the 3x V620 rig; the historical 'HIP*' prefix here made children fail
    instantly with "invalid device: HIP0".

    A dual-backend binary (GGML_HIP + GGML_VULKAN) needs an explicit list to
    offload deterministically — otherwise auto-select may pick the wrong
    backend.
    """
    if count <= 0:
        return ""
    prefix = "Vulkan" if backend == "vulkan" else "ROCm"
    return ",".join(f"{prefix}{i}" for i in range(count))


def ini_defines_per_model_device(ini):
    """True when any model section carries its own `device` key.

    The router overlays its own CLI args onto every child preset (upstream
    server-models.cpp: preset.merge(base_preset)) and LLAMA_ARG_DEVICE is not
    stripped from that overlay - so a router-level --device would silently
    clobber every per-section device (verified live against llama.cpp master
    85c5522). Callers therefore skip the router-level --device entirely when
    any section defines its own, letting sections decide per model.
    A `device` key in [*] is a global default, not a per-model selection, and
    a blank/comment-only value means "unset" - neither counts.
    """
    for name, sect in (ini or {}).items():
        if name in ("*",):
            continue
        val = (sect or {}).get("device")
        if val is not None and val.strip() and not str(val).lstrip().startswith("#"):
            return True
    return False


def router_device_for(backend, count, per_model=False):
    """The --device list for the router launch, honoring per-model mode.

    per_model=True (any models.ini section sets its own device=) returns ""
    even with GPUs present, so llama.cpp never receives a router-level
    --device that would override the sections. per_model=False keeps the
    historical global list (device_list(), empty when count <= 0).
    """
    if per_model:
        return ""
    return device_list(backend, count)

def detect_all_gpus():
    """NVIDIA + AMD GPUs combined, for VRAM-fit ratings and auto-tune. Each row
    carries `vram_mib` and `name`; NVIDIA rows add `compute_cap`, AMD rows add
    `gfx_arch`."""
    return detect_gpus() + detect_amd_gpus()

def _amd_targets_for(amd_gpus):
    """Semicolon-joined, sorted, deduped gfx archs from detected AMD GPUs, or
    "" when none could be read (caller falls back to the configured default)."""
    archs = sorted({g["gfx_arch"] for g in amd_gpus if g.get("gfx_arch")})
    return ";".join(archs)

def detect_ram_gb():
    """Total system RAM in GB (SI, /1e9 to match GPU vendor GB units), 0 if unknown."""
    return round(osplat.total_ram_bytes() / 1e9, 1)


def _detect_cpu_windows():
    # wmic is removed on recent Windows 11; use PowerShell CIM.
    out = _run(["powershell", "-NoProfile", "-Command",
                "$c=Get-CimInstance Win32_Processor|Select-Object -First 1;"
                "'{0}|{1}|{2}' -f $c.Name,$c.NumberOfCores,$c.NumberOfLogicalProcessors"])
    info = {"name": "", "cores": None, "threads": None}
    parts = out.strip().split("|")
    if len(parts) == 3:
        info["name"] = parts[0].strip()
        info["cores"] = int(parts[1]) if parts[1].strip().isdigit() else None
        info["threads"] = int(parts[2]) if parts[2].strip().isdigit() else None
    # crude AVX-512 hint: recent AMD Zen4/5 and Intel server/HEDT
    n = info["name"].lower()
    info["avx512_hint"] = any(x in n for x in ["ryzen 7 9", "ryzen 9 9", "ryzen 7 7", "ryzen 9 7", "xeon", "threadripper"])
    return info

def detect_cpu():
    if osplat.IS_LINUX:
        c = osplat.linux_cpu()
        return {"name": c["name"], "cores": c["cores"], "threads": c["threads"],
                "avx512_hint": c["avx512"]}      # real flag, not a name heuristic
    if osplat.IS_MAC:
        c = osplat.mac_cpu()
        return {"name": c["name"], "cores": c["cores"], "threads": c["threads"],
                "avx512_hint": False}
    return _detect_cpu_windows()

def recommend(gpus=None, cpu=None, amd_gpus=None, amd_targets=None, amd_backend=None):
    """Return {cmake_flags:{...}, notes:[...], runtime:{...}} for this machine.

    amd_targets overrides the AMDGPU_TARGETS list (config.json `amd_gpu_targets`);
    when None, detected archs are used, falling back to AMDGPU_TARGETS_DEFAULT.

    amd_backend selects the AMD accelerator: "rocm" (default, GGML_HIP) or
    "vulkan" (GGML_VULKAN). Vulkan is the right choice on RDNA2 (gfx1030) cards
    with no matrix cores, where ROCm's multi-GPU layer-split collapses to ~7
    tok/s but Vulkan holds ~16 tok/s regardless of split or context length.
    """
    gpus = detect_gpus() if gpus is None else gpus
    amd_gpus = detect_amd_gpus() if amd_gpus is None else amd_gpus
    cpu  = detect_cpu()  if cpu  is None else cpu
    flags, notes = {}, []

    if osplat.IS_MAC:
        flags["GGML_METAL"] = "ON"
        notes.append("Apple Silicon detected - Metal build (uses unified memory as VRAM).")
        runtime = {"n-gpu-layers": "99", "flash-attn": "on"}
        flags["GGML_NATIVE"] = "ON"
        return {"cmake_flags": flags, "notes": notes, "runtime": runtime,
                "gpus": gpus, "cpu": cpu}

    if amd_gpus:
        if amd_backend == "vulkan":
            flags["GGML_VULKAN"] = "ON"
            notes.append(f"Vulkan build (RADV) for AMD RDNA2 ({len(amd_gpus)} GPU(s)).")
            notes.append("Vulkan avoids ROCm's multi-GPU PCIe penalty on RDNA2 — "
                         "holds ~16 tok/s at 100k context with a 3-GPU split.")
        else:
            flags["GGML_HIP"] = "ON"
            targets = amd_targets or _amd_targets_for(amd_gpus) or AMDGPU_TARGETS_DEFAULT
            flags["AMDGPU_TARGETS"] = targets
            notes.append(f"ROCm/HIP build for AMD arch(s) {targets.replace(';', ', ')} "
                         f"({len(amd_gpus)} GPU(s)).")
            notes.append("Set config.json `amd_gpu_targets` to narrow this to your GPU(s).")
    elif gpus:
        archs = sorted({g["compute_cap"].replace(".", "") for g in gpus if g["compute_cap"]})
        flags["GGML_CUDA"] = "ON"
        if archs:
            flags["CMAKE_CUDA_ARCHITECTURES"] = ";".join(archs)
            notes.append(f"CUDA build for arch(s) {', '.join(archs)} ({len(gpus)} GPU(s)).")
        flags["GGML_CUDA_FA_ALL_QUANTS"] = "ON"   # quantized-KV flash attention
        notes.append("Enabled flash-attention for all quant KV combos.")
    else:
        notes.append("No GPU detected - configuring a CPU-only build.")

    flags["GGML_NATIVE"] = "ON"
    if cpu.get("avx512_hint"):
        for f in ("GGML_AVX512", "GGML_AVX512_VNNI", "GGML_AVX512_VBMI", "GGML_AVX512_BF16"):
            flags[f] = "ON"
        notes.append("Enabled AVX-512 (+VNNI/VBMI/BF16) for this CPU.")

    has_gpu = bool(gpus or amd_gpus)
    runtime = {
        "n-gpu-layers": "99" if has_gpu else "0",
        "flash-attn": "on" if has_gpu else "off",
    }
    return {"cmake_flags": flags, "notes": notes, "runtime": runtime,
            "gpus": gpus, "cpu": cpu}

if __name__ == "__main__":
    import json
    print(json.dumps(recommend(), indent=2))
