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
        if not props or "gpu_id" not in props:
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

def recommend(gpus=None, cpu=None, amd_gpus=None, amd_targets=None):
    """Return {cmake_flags:{...}, notes:[...], runtime:{...}} for this machine.

    amd_targets overrides the AMDGPU_TARGETS list (config.json `amd_gpu_targets`);
    when None, detected archs are used, falling back to AMDGPU_TARGETS_DEFAULT.
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
