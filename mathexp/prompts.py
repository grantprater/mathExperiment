"""Prompt templates, menus, seeds, JSON schemas and response parsing."""

from __future__ import annotations

import json
import re
import tomllib
from functools import lru_cache

from . import PKG

PROMPTS = PKG / "prompts"
STEPS = ("predict", "weave_menu", "weave_free", "mutate", "method", "explain")
#: Run 3, A5b: the explanation note is at most this many words (longer text is cut).
EXPLAIN_MAX_WORDS = 80


@lru_cache
def template(name: str) -> tuple[str, str]:
    """(version, body) of prompts/<name>.md; the first line is ``version: ...``."""
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    first, _, body = text.partition("\n")
    if not first.startswith("version:"):
        raise ValueError(f"prompt {name} lacks a version line")
    return first.split(":", 1)[1].strip(), body


def render(name: str, **vars) -> tuple[str, str]:
    version, body = template(name)

    def sub(m):
        key = m.group(1)
        if key not in vars:
            raise KeyError(f"prompt {name} needs {key}")
        return str(vars[key])
    return version, re.sub(r"\{\{(\w+)\}\}", sub, body)


def system_prompt() -> tuple[str, str]:
    return template("system")


@lru_cache
def bridges() -> list[dict]:
    return tomllib.loads((PKG / "menus" / "bridges.toml").read_text(encoding="utf-8"))["bridge"]


@lru_cache
def lemmas() -> list[dict]:
    return tomllib.loads((PKG / "menus" / "lemmas.toml").read_text(encoding="utf-8"))["lemma"]


@lru_cache
def seeds() -> list[dict]:
    return tomllib.loads((PKG / "seeds.toml").read_text(encoding="utf-8"))["seed"]


def seed(name: str) -> dict:
    for s in seeds():
        if s["name"] == name:
            return s
    raise KeyError(f"unknown seed {name!r}; known: {[s['name'] for s in seeds()]}")


def bridge_menu(only: list[str] | None = None) -> str:
    out = []
    for b in bridges():
        if only is not None and b["id"] not in only:
            continue
        flag = " (statement unverified)" if b.get("unsure") else ""
        out.append(f"- [{b['id']}] {b['name']}{flag}: {' '.join(b['dictionary'].split())}")
    return "\n".join(out)


def lemma_menu() -> str:
    out = []
    for lm in lemmas():
        flag = " (statement unverified)" if lm.get("unsure") else ""
        out.append(f"- [{lm['id']}] {lm['name']}{flag}. If: {lm['hypotheses']} Then: {lm['conclusion']}")
    return "\n".join(out)


# ---- JSON schemas (passed to --json-schema where supported) ------------------
MAX_CANDIDATES = 5
NUM01 = {"type": "number", "minimum": 0, "maximum": 1}
SCHEMAS = {
    "predict": {
        "type": "object",
        "properties": {
            "predictions": {"type": "array", "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"},
                               "candidates": {"type": "array", "maxItems": MAX_CANDIDATES, "items": {
                                   "type": "array", "minItems": 2, "maxItems": 2,
                                   "items": {"type": "number"}}},
                               "other": NUM01},
                "required": ["n", "candidates", "other"]}},
            "recognized_as": {"type": ["string", "null"]},
            "p_recognized": NUM01},
        "required": ["predictions", "recognized_as", "p_recognized"]},
    "weave": {
        "type": "object",
        "properties": {"bounds": {"type": "array", "items": {
            "type": "object",
            "properties": {"bridge_id": {"type": "string"},
                           "lemma_ids": {"type": "array", "items": {"type": "string"}},
                           "translation": {"type": "string"}, "bound_expr": {"type": "string"},
                           "proof_sketch": {"type": "string"}, "p_holds": NUM01},
            "required": ["bridge_id", "lemma_ids", "translation", "bound_expr", "proof_sketch", "p_holds"]}}},
        "required": ["bounds"]},
    "mutate": {
        "type": "object",
        "properties": {"mutations": {"type": "array", "items": {
            "type": "object",
            "properties": {"spec": {"type": "object"}, "what_if": {"type": "string"},
                           "predicted_effect": {"type": "string", "enum": ["larger", "equal", "smaller"]},
                           "p_effect": NUM01},
            "required": ["spec", "what_if", "predicted_effect", "p_effect"]}}},
        "required": ["mutations"]},
    "method": {
        "type": "object",
        "properties": {"bridge_id": {"type": ["string", "null"]},
                       "bands": {"type": "object",
                                 "properties": {b: NUM01 for b in ("tight", "close", "moderate", "loose", "failed")},
                                 "required": ["tight", "close", "moderate", "loose", "failed"]}},
        "required": ["bridge_id", "bands"]},
    "explain": {
        "type": "object",
        "properties": {"explanation": {"type": "string"}},
        "required": ["explanation"]},
    "ping": {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"]},
}


class ParseError(ValueError):
    pass


def extract_json(text: str):
    """First JSON object or array in ``text`` (code fences and prose tolerated)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(t[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ParseError("no JSON object found")


def _prob(x) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 <= x <= 1:
        raise ParseError(f"probability out of range: {x!r}")
    return float(x)


def normalize_distribution(candidates, other) -> dict:
    """Validate one n's distribution. Candidates are [value, prob] pairs (or
    {value, prob} objects); at most MAX_CANDIDATES after merging repeated values.
    Integer values only (a float with no fractional part is accepted).

    Masses: if candidates + other sum to more than 1, everything is divided by the
    sum (PROTOCOL). If they sum to less than 1, the shortfall is added to ``other``
    (our choice: unassigned mass is residual mass)."""
    if not isinstance(candidates, list):
        raise ParseError("predict: candidates must be a list")
    merged: dict[int, float] = {}
    for c in candidates:
        if isinstance(c, dict):
            c = [c.get("value"), c.get("prob", c.get("p"))]
        if not isinstance(c, (list, tuple)) or len(c) != 2:
            raise ParseError(f"predict: bad candidate {c!r}")
        v, pr = c
        if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v):
            raise ParseError(f"predict: candidate value must be an integer, got {v!r}")
        merged[int(v)] = merged.get(int(v), 0.0) + _prob(pr)
    if len(merged) > MAX_CANDIDATES:
        raise ParseError(f"predict: more than {MAX_CANDIDATES} candidates")
    other = _prob(other if other is not None else 0.0)
    total = sum(merged.values()) + other
    if total <= 0:
        raise ParseError("predict: zero total probability")
    normalized = total > 1 + 1e-9
    if normalized:
        merged = {v: pr / total for v, pr in merged.items()}
        other /= total
    else:
        other = 1.0 - sum(merged.values())
    cands = sorted(([v, pr] for v, pr in merged.items()), key=lambda x: (-x[1], x[0]))
    return {"candidates": cands, "other": max(0.0, other), "normalized": normalized}


def parse_predict(text: str, ns: list[int]) -> dict:
    obj = extract_json(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("predictions"), list):
        raise ParseError("predict: missing predictions list")
    preds = {}
    for p in obj["predictions"]:
        try:
            n = int(p["n"])
            cands, other = p["candidates"], p.get("other", 0.0)
        except (KeyError, TypeError, ValueError):
            raise ParseError(f"predict: bad item {p!r}") from None
        if n in ns:
            preds[n] = normalize_distribution(cands, other)
    if not preds:
        raise ParseError("predict: no predictions for the requested n")
    rec = obj.get("recognized_as")
    rec = rec if isinstance(rec, str) and rec.strip() and rec.strip().lower() not in ("null", "none") else None
    p_rec = obj.get("p_recognized")
    p_rec = _prob(p_rec) if p_rec is not None else None
    return {"predictions": preds, "recognized_as": rec, "p_recognized": p_rec}


def parse_weave(text: str, k: int) -> list[dict]:
    obj = extract_json(text)
    items = obj if isinstance(obj, list) else obj.get("bounds") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        raise ParseError("weave: missing bounds list")
    out = []
    for b in items[:k]:
        if not isinstance(b, dict):
            raise ParseError("weave: bound is not an object")
        lem = b.get("lemma_ids") or []
        if isinstance(lem, str):
            lem = [lem]
        out.append({"bridge_id": str(b.get("bridge_id", "")), "lemma_ids": [str(x) for x in lem],
                    "translation": str(b.get("translation", "")), "bound_expr": b.get("bound_expr"),
                    "proof_sketch": str(b.get("proof_sketch", "")), "p_holds": _prob(b.get("p_holds"))})
    return out


def parse_mutate(text: str, m: int) -> list[dict]:
    obj = extract_json(text)
    items = obj if isinstance(obj, list) else obj.get("mutations") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        raise ParseError("mutate: missing mutations list")
    out = []
    for it in items[:m]:
        if not isinstance(it, dict) or not isinstance(it.get("spec"), dict):
            raise ParseError("mutate: item without a spec object")
        eff = it.get("predicted_effect")
        if eff not in ("larger", "equal", "smaller"):
            raise ParseError(f"mutate: bad predicted_effect {eff!r}")
        out.append({"spec": it["spec"], "what_if": str(it.get("what_if", "")),
                    "predicted_effect": eff, "p_effect": _prob(it.get("p_effect"))})
    return out


def parse_method(text: str, choose_from: list[str] | None = None) -> dict:
    """Run 3 method forecast: a probability for each of the five bands (missing bands
    are an error; masses are divided by their sum). With ``choose_from`` (P was empty),
    ``bridge_id`` must name one of those bridges; otherwise it is ignored."""
    from .method import BANDS

    obj = extract_json(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("bands"), dict):
        raise ParseError("method: missing bands object")
    raw = obj["bands"]
    missing = [b for b in BANDS if b not in raw]
    if missing:
        raise ParseError(f"method: missing bands {missing}")
    dist = {b: _prob(raw[b]) for b in BANDS}
    total = sum(dist.values())
    if total <= 0:
        raise ParseError("method: zero total probability")
    normalized = abs(total - 1) > 1e-6
    dist = {b: p / total for b, p in dist.items()}
    bridge = obj.get("bridge_id")
    if choose_from is not None:
        if not isinstance(bridge, str) or bridge.strip() not in choose_from:
            raise ParseError(f"method: bridge_id {bridge!r} is not one of the menu's bridges")
        bridge = bridge.strip()
    else:
        bridge = None
    return {"bands": dist, "bridge_id": bridge, "normalized": normalized, "raw_total": total}


def parse_explain(text: str) -> dict:
    """Run 3 explanation note: a non-empty string, cut to EXPLAIN_MAX_WORDS words."""
    obj = extract_json(text)
    exp = obj.get("explanation") if isinstance(obj, dict) else None
    if not isinstance(exp, str) or not exp.strip():
        raise ParseError("explain: missing explanation")
    words = exp.split()
    return {"explanation": " ".join(words[:EXPLAIN_MAX_WORDS]), "words": len(words),
            "truncated": len(words) > EXPLAIN_MAX_WORDS}


def parse_ping(text: str) -> dict:
    obj = extract_json(text)
    if not isinstance(obj, dict) or obj.get("ok") is not True:
        raise ParseError(f"ping: expected {{\"ok\": true}}, got {text[:200]!r}")
    return obj


def recognition_matches(recognized: str | None, known_as: list[str]) -> bool:
    if not recognized:
        return False
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
    r = norm(recognized)
    return any(norm(k) and (norm(k) in r or r in norm(k)) for k in known_as) if r else False
