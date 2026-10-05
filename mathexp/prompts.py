"""Prompt templates, menus, seeds, JSON schemas and response parsing."""

from __future__ import annotations

import json
import re
import tomllib
from functools import lru_cache

from . import PKG

PROMPTS = PKG / "prompts"
STEPS = ("predict", "weave_menu", "weave_free", "mutate")


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
NUM01 = {"type": "number", "minimum": 0, "maximum": 1}
SCHEMAS = {
    "predict": {
        "type": "object",
        "properties": {
            "predictions": {"type": "array", "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "point": {"type": "number"},
                               "lo": {"type": "number"}, "hi": {"type": "number"}},
                "required": ["n", "point", "lo", "hi"]}},
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


def parse_predict(text: str, ns: list[int]) -> dict:
    obj = extract_json(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("predictions"), list):
        raise ParseError("predict: missing predictions list")
    preds = {}
    for p in obj["predictions"]:
        try:
            n = int(p["n"])
            point, lo, hi = float(p["point"]), float(p["lo"]), float(p["hi"])
        except (KeyError, TypeError, ValueError):
            raise ParseError(f"predict: bad item {p!r}") from None
        if n in ns:
            if lo > hi:
                lo, hi = hi, lo
            preds[n] = {"point": point, "lo": lo, "hi": hi}
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


def recognition_matches(recognized: str | None, known_as: list[str]) -> bool:
    if not recognized:
        return False
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
    r = norm(recognized)
    return any(norm(k) and (norm(k) in r or r in norm(k)) for k in known_as) if r else False
