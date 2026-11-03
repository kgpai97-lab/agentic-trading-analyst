"""Run manifest — the reproducibility record written next to every run.

Phase 2: any run should be auditable and (data permitting) reproducible. The
manifest captures the inputs that determine a run's behaviour:

  * model ids + reasoning flags for both tiers,
  * SHA-256 hashes of every prompt file in prompts/ (frozen-prompt evidence for
    the forward test — Phase 3 records these so we can prove the prompts didn't
    change),
  * the effective config (secrets stripped),
  * the as_of_date and snapshot directory pointer,
  * package + key dependency versions, and a UTC timestamp.

Written by PortfolioGraph.propagate() alongside the report/transcript, sharing
the same `<label>_<date>_<stamp>` stem.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

# Keys that must never be persisted (they can carry secrets / machine paths).
_REDACT_KEYS = {"ollama_base_url"}


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_prompts(prompts_dir: str | Path) -> dict[str, str]:
    """Return {filename: sha256} for every *.txt prompt, sorted by name."""
    d = Path(prompts_dir)
    if not d.is_dir():
        return {}
    return {p.name: _hash_file(p) for p in sorted(d.glob("*.txt"))}


def _safe_config(config: dict) -> dict:
    """Copy of config with secret/path-ish keys redacted; JSON-serialisable."""
    out = {}
    for k, v in config.items():
        if k in _REDACT_KEYS:
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def _dep_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for mod in ("langgraph", "langchain_core", "langchain_ollama", "yfinance", "pydantic"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[mod] = "not-installed"
    return versions


def build_manifest(
    config: dict,
    prompts_dir: str | Path,
    as_of_date: str,
    *,
    label: str = "",
    extra: dict | None = None,
) -> dict:
    """Assemble the manifest dict (does not write it)."""
    manifest = {
        "label":       label,
        "as_of_date":  as_of_date,
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "models": {
            "deep_think_llm":  config.get("deep_think_llm"),
            "quick_think_llm": config.get("quick_think_llm"),
            "deep_reasoning":  config.get("deep_reasoning"),
            "quick_reasoning": config.get("quick_reasoning"),
            "llm_provider":    config.get("llm_provider"),
        },
        "prompt_hashes": hash_prompts(prompts_dir),
        "snapshot_dir":  config.get("snapshot_dir", ""),
        "config":        _safe_config(config),
        "dependencies":  _dep_versions(),
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(
    out_dir: str | Path,
    stem: str,
    config: dict,
    prompts_dir: str | Path,
    as_of_date: str,
    *,
    label: str = "",
    extra: dict | None = None,
) -> Path | None:
    """Write ``<out_dir>/<stem>.manifest.json``. Returns the path (or None)."""
    if not out_dir:
        return None
    try:
        manifest = build_manifest(config, prompts_dir, as_of_date, label=label, extra=extra)
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{stem}.manifest.json"
        path.write_text(json.dumps(manifest, indent=2, default=str, ensure_ascii=False),
                        encoding="utf-8")
        return path
    except Exception as exc:  # noqa: BLE001 — manifest is best-effort
        print(f"[manifest] WARNING: failed to write manifest: {exc}")
        return None
