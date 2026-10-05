"""Model backends behind one interface:

    backend.complete(system, prompt, json_schema=None, context=None) -> (text, meta)

``context`` is ignored by real backends; the mock backend reads it (step,
family, spec, ns and, for the weave step only, the computed truth table) so it
can produce every status deterministically.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import REPO

CLAUDE_CWD = REPO / "scratch" / "claude-cwd"
CLAUDE_TIMEOUT = 300
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(RuntimeError):
    pass


class Backend:
    name = "base"
    model = ""

    def complete(self, system: str, prompt: str, json_schema: dict | None = None,
                 context: dict | None = None) -> tuple[str, dict]:
        raise NotImplementedError


# ---- claude -p ---------------------------------------------------------------
class ClaudeBackend(Backend):
    """``claude -p`` in an empty working directory with a replacement system prompt,
    ``--safe-mode`` and no tools: no CLAUDE.md, memory or tools reach the model.

    ``--json-schema`` works in print mode (checked 2026-10-04 with claude 2.1.289):
    the outer JSON then carries a ``structured_output`` object as well as the
    ``result`` text; we prefer ``structured_output`` and fall back to ``result``.
    """

    name = "claude"

    def __init__(self, model: str = "sonnet", cwd: Path = CLAUDE_CWD, timeout: int = CLAUDE_TIMEOUT):
        self.model = model
        self.cwd = Path(cwd)
        self.timeout = timeout
        exe = shutil.which("claude")
        if not exe:
            raise LLMError("claude CLI not found on PATH")
        self.exe = exe

    def command(self, system: str, json_schema: dict | None) -> list[str]:
        cmd = [self.exe, "-p", "--safe-mode", "--no-session-persistence", "--model", self.model,
               "--tools", "", "--system-prompt", system, "--output-format", "json"]
        if json_schema is not None:
            cmd += ["--json-schema", json.dumps(json_schema, separators=(",", ":"))]
        return cmd

    def complete(self, system, prompt, json_schema=None, context=None):
        self.cwd.mkdir(parents=True, exist_ok=True)
        stray = [p.name for p in self.cwd.iterdir()]
        if stray:
            raise LLMError(f"claude cwd {self.cwd} must be empty; found {stray}")
        t0 = time.monotonic()
        try:
            proc = subprocess.run(self.command(system, json_schema), input=prompt, capture_output=True,
                                  text=True, encoding="utf-8", cwd=self.cwd, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise LLMError(f"claude -p timed out after {self.timeout}s") from None
        wall = time.monotonic() - t0
        meta = {"backend": self.name, "requested_model": self.model, "wall_s": round(wall, 2),
                "returncode": proc.returncode, "json_schema": json_schema is not None}
        try:
            outer = json.loads(proc.stdout)
        except json.JSONDecodeError:
            meta["stderr"] = proc.stderr[-2000:]
            meta["stdout_head"] = proc.stdout[:2000]
            raise LLMError(f"claude -p returned non-JSON (rc={proc.returncode}): {proc.stderr[-500:]}")
        usage = outer.get("usage") or {}
        mu = outer.get("modelUsage") or {}
        meta.update({
            "model": next(iter(mu), None),
            "model_usage": mu,
            "input_tokens": usage.get("input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_cost_usd_nominal": outer.get("total_cost_usd"),
            "duration_ms": outer.get("duration_ms"),
            "is_error": outer.get("is_error"),
            "num_turns": outer.get("num_turns"),
            "session_id": outer.get("session_id"),
        })
        if outer.get("is_error"):
            raise LLMError(f"claude -p error: {str(outer.get('result'))[:500]}")
        if outer.get("structured_output") is not None:
            return json.dumps(outer["structured_output"]), meta
        return outer.get("result") or "", meta


# ---- OpenRouter ----------------------------------------------------------------
def _read_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


class OpenRouterBackend(Backend):
    name = "openrouter"

    def __init__(self, model: str, allow_paid: bool = False, timeout: int = 300):
        if not model.endswith(":free") and not allow_paid:
            raise LLMError(f"model {model!r} does not end in ':free'; pass --allow-paid to use it")
        self.model = model
        self.timeout = timeout
        self.key = _read_key()
        if not self.key:
            raise LLMError("OPENROUTER_API_KEY not set (env or repo .env)")

    def complete(self, system, prompt, json_schema=None, context=None):
        body = {"model": self.model, "messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": prompt}]}
        req = urllib.request.Request(OPENROUTER_URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            raise LLMError(f"openrouter request failed: {e}") from None
        usage = data.get("usage") or {}
        meta = {"backend": self.name, "requested_model": self.model, "model": data.get("model"),
                "wall_s": round(time.monotonic() - t0, 2),
                "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                "total_cost_usd_nominal": usage.get("cost")}
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"openrouter: unexpected response {str(data)[:500]}") from None
        return text, meta


# ---- mock ----------------------------------------------------------------------
class MockBackend(Backend):
    """Deterministic, seeded test double. Valid JSON for every step.

    - predict: alternates a distribution with 0.9 on the exact value (unsurprising,
      about 0.15 bits) with one that puts 0.9 on a 4x overestimate (surprising: the
      true value gets only a share of the 0.1 residual, so >= 4 bits once the
      trivial bound exceeds 1); the exact guess peeks at the brute force, which is
      legitimate for a test double and never touches the store.
    - weave: five bounds -- valid and tight (a constant at the largest computed
      value: a win unless f equals the trivial bound), valid and loose (no win),
      violated (-1), unparseable, and a second loose one -- rotated by one per
      call, so k = 3 still covers every status within three calls. In adaptive *narrow*
      mode the single bound alternates loose (no win -> the loop escalates) and
      tight (the prior held).
    - mutate: random single-field mutations; the first call with m >= 2 also
      returns one two-field mutation so refusal (c) is exercised.
    """

    name = "mock"

    def __init__(self, seed: int = 0, model: str = "mock-1"):
        self.model = model
        self.rng = random.Random(seed)
        self.predict_calls = 0
        self.narrow_calls = 0
        self.weave_calls = 0
        self.bad_mutation_sent = False

    def complete(self, system, prompt, json_schema=None, context=None):
        if not context or "step" not in context:
            raise LLMError("mock backend needs a context with the step")
        step = context["step"]
        meta = {"backend": self.name, "model": self.model, "requested_model": self.model,
                "input_tokens": len(prompt) // 4, "output_tokens": 0, "wall_s": 0.0,
                "total_cost_usd_nominal": 0.0}
        out = getattr(self, f"_{step}")(context)
        text = out if isinstance(out, str) else json.dumps(out)
        meta["output_tokens"] = len(text) // 4
        return text, meta

    def _predict(self, ctx):
        fam, spec = ctx["family"], ctx["spec"]
        self.predict_calls += 1
        exact = self.predict_calls % 2 == 1
        preds = []
        for n in ctx["ns"]:
            f = fam.truth(spec, n, 10.0)
            f = fam.trivial_bound(spec, n) if f is None else f
            if exact:
                cands = [[f, 0.9], [f + 1, 0.05]]
                other = 0.05
            else:
                cands = [[4 * f + 3, 0.9]]
                other = 0.1
            preds.append({"n": n, "candidates": cands, "other": other})
        named = ctx.get("seed_name")
        return {"predictions": preds,
                "recognized_as": named.replace("_", " ") if named else None,
                "p_recognized": 0.9 if named else 0.1}

    def _weave(self, ctx):
        fam, spec, truth = ctx["family"], ctx["spec"], ctx["truth"]
        known = [v for v in truth.values() if v is not None]
        top = max(known) if known else 1
        prior = ctx.get("prior") or []
        tight = {"bridge_id": prior[0] if prior else "charvec_fp", "lemma_ids": ["dim_independent"],
                 "translation": "mock: constant bound at the largest computed value",
                 "bound_expr": str(top), "proof_sketch": "mock", "p_holds": 0.7}
        loose = {"bridge_id": "incidence_matrix", "lemma_ids": ["gram_rank"],
                 "translation": "mock: the ambient count", "bound_expr": fam.trivial_expr(spec),
                 "proof_sketch": "mock", "p_holds": 0.95}
        if ctx.get("mode") == "narrow":
            self.narrow_calls += 1
            return {"bounds": [loose if self.narrow_calls % 2 == 1 else tight]}
        if ctx.get("mode") == "probe":
            return {"bounds": [self._probe_bound(ctx, top, prior)]}
        if ctx.get("mode") == "wide3":
            return {"bounds": self._wide3_bounds(ctx, top, prior)}
        bounds = [
            {**tight, "bridge_id": "orbit_averaging"},
            loose,
            {"bridge_id": "double_counting", "lemma_ids": ["pigeonhole"],
             "translation": "mock: a bound that fails", "bound_expr": "-1",
             "proof_sketch": "mock", "p_holds": 0.3},
            {"bridge_id": "fourier_indicator", "lemma_ids": ["parseval"],
             "translation": "mock: an unparseable bound", "bound_expr": "n +* 2",
             "proof_sketch": "mock", "p_holds": 0.5},
            {**loose, "bridge_id": "translates", "lemma_ids": ["disjoint_translates"]},
        ]
        # rotate so that a run with k < 5 still produces every status across calls
        r = self.weave_calls % len(bounds)
        self.weave_calls += 1
        bounds = bounds[r:] + bounds[:r]
        return {"bounds": bounds[: max(1, ctx.get("k", 4))]}

    # ---- run 3 ----------------------------------------------------------------
    #: First-probe bounds, rotated per call: (label, g as a function of the largest
    #: computed value). Constants >= max f are consistent; "-1" is violated. Bands:
    #: loose, close (or moderate when f is small), moderate, tight, loose, failed.
    PROBE1 = (("loose", lambda t: 3 * t + 2), ("close", lambda t: t + 1), ("moderate", lambda t: 2 * t),
              ("tight", lambda t: t), ("loose", lambda t: 4 * t + 3), ("failed", lambda t: -1))

    def _probe_bound(self, ctx, top, prior):
        bridge = (ctx.get("allowed") or prior or ["charvec_fp"])[0]
        if ctx.get("weave_call") == 1:
            self.probe1_calls = getattr(self, "probe1_calls", 0) + 1
            _, g = self.PROBE1[(self.probe1_calls - 1) % len(self.PROBE1)]
            expr = str(g(top))
        else:  # the re-probe: a seeded draw around the first probe's range
            expr = str(top + self.rng.choice([0, 1, 2, top, 2 * top + 1]))
        return {"bridge_id": bridge, "lemma_ids": ["dim_independent"], "translation": "mock probe",
                "bound_expr": expr, "proof_sketch": "mock", "p_holds": 0.8}

    def _wide3_bounds(self, ctx, top, prior):
        """Three bounds from the allowed menu at seeded levels; every third call labels
        its first bound with a prior bridge, which the loop must record but not count."""
        self.wide3_calls = getattr(self, "wide3_calls", 0) + 1
        allowed = list(ctx.get("allowed") or ["double_counting"])
        out = []
        for i in range(min(3, ctx.get("k", 3))):
            bridge = allowed[(self.wide3_calls + i) % len(allowed)]
            if i == 0 and self.wide3_calls % 3 == 0 and prior:
                bridge = prior[0]
            g = top + self.rng.choice([0, 2, 2 * top + 2, 5 * top + 4, 4 * top, 3 * top + 1])
            out.append({"bridge_id": bridge, "lemma_ids": ["pigeonhole", "cauchy_schwarz"][: 1 + i % 2],
                        "translation": "mock wide bound", "bound_expr": str(g), "proof_sketch": "mock",
                        "p_holds": 0.7})
        return out

    def _method(self, ctx):
        from .method import BANDS

        w = [self.rng.random() ** 2 + 0.01 for _ in BANDS]
        s = sum(w)
        out = {"bridge_id": ctx["menu"][0] if ctx.get("choose") else None,
               "bands": {b: round(x / s, 4) for b, x in zip(BANDS, w)}}
        return out

    def _explain(self, ctx):
        self.explain_calls = getattr(self, "explain_calls", 0) + 1
        words = ["mock:", "the", "extremal", "objects", "are", "blocks", "of", "a", "design."]
        if self.explain_calls % 4 == 0:  # an over-long note, to exercise the 80-word cut
            words = words * 12
        return {"explanation": " ".join(words)}

    def _ping(self, ctx):
        return {"ok": True}

    def _mutate(self, ctx):
        fam, spec, m = ctx["family"], ctx["spec"], ctx["m"]
        out = []
        effects = ["larger", "equal", "smaller"]
        if m >= 2 and not self.bad_mutation_sent:
            self.bad_mutation_sent = True
            # change two fields at once: refusal (c) must reject this
            bad = dict(spec)
            if fam.name == "setfam":
                bad["p"] = 7
                bad["int_min"] = 1 if spec.get("int_min") is None else None
            else:
                bad["group"] = "cyclic" if spec.get("group") != "cyclic" else "int"
                bad["equation"] = "sidon" if spec.get("equation") != "sidon" else "3ap"
            out.append({"spec": {k: v for k, v in bad.items() if k != "ns"},
                        "what_if": "mock: two fields at once", "predicted_effect": "larger",
                        "p_effect": 0.5})
        while len(out) < m:
            child = fam.mutate_random(spec, self.rng)
            out.append({"spec": {k: v for k, v in child.items() if k != "ns"},
                        "what_if": "mock: random single-field mutation",
                        "predicted_effect": self.rng.choice(effects),
                        "p_effect": round(self.rng.uniform(0.4, 0.9), 2)})
        return {"mutations": out}


def make_backend(name: str, model: str | None = None, seed: int = 0, allow_paid: bool = False) -> Backend:
    if name == "claude":
        return ClaudeBackend(model or "sonnet")
    if name == "openrouter":
        if not model:
            raise LLMError("openrouter needs --model (a ':free' id)")
        return OpenRouterBackend(model, allow_paid=allow_paid)
    if name == "mock":
        return MockBackend(seed=seed)
    raise LLMError(f"unknown backend {name!r}")
