"""Model-aware ini defaults: GLM family gets min-p = 0.01.

llama.cpp's default min-p (0.05) measurably harms GLM's output quality; 0.01
is the measured-good value for the GLM family. Applied additively — only
fills a missing key, never overrides one the user set.

Wired into config.apply_ctx_defaults (monkeypatched wrapper) so every caller —
panel startup, scan-apply, autotune — inherits the default without each one
needing to know about it.
"""
import config

def apply_glm_defaults(ini, set_keys=None):
    """Fill min-p = 0.01 into models.ini sections for GLM-family models.

    `ini` is the read_sections() dict. `set_keys` is config.set_keys or any
    (section, updates) writer — injectable for tests. A section counts as
    GLM when its id or its model path contains "glm" (case-insensitive).
    A section that already carries min-p is left untouched (the user's value
    wins, same contract as ctx defaults).

    Returns the list of changed section names.
    """
    set_keys = set_keys or config.set_keys
    changed = []
    for name, kv in (ini or {}).items():
        if name == "*":
            continue
        blob = f"{name} {(kv or {}).get('model', '')}".lower()
        if "glm" not in blob:
            continue
        if (kv or {}).get("min-p"):
            continue
        set_keys(name, {"min-p": "0.01"})
        changed.append(name)
    return changed


_orig_apply_ctx = config.apply_ctx_defaults


def apply_ctx_defaults_with_glm(path=None):
    res = _orig_apply_ctx(path)
    try:
        changed = apply_glm_defaults(config.read_sections(path))
        if changed:
            res.setdefault("changed", []).extend(changed)
    except Exception:
        pass  # advisory default; never break the ctx pass
    return res


# Every existing caller (startup, scan-apply, autotune) goes through
# config.apply_ctx_defaults — patch it once at import so they all inherit.
config.apply_ctx_defaults = apply_ctx_defaults_with_glm


if __name__ == "__main__":
    import json, sys
    ini_path = sys.argv[1] if len(sys.argv) > 1 else config.ini_path()
    print(json.dumps({"changed": apply_glm_defaults(config.read_sections(ini_path))}))