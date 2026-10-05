# mathexp: lemma-level weaving

An LLM loop that tries to bound small extremal-combinatorics problems by *weaving*:
translate the problem through a bridge (a set becomes its 0/1 vector over F_p), apply a
lemma that holds on the other side (independent vectors number at most the dimension),
and translate the conclusion back as an upper bound g(n). Every prediction is logged
before the brute-force answer is computed, so surprise and calibration can be scored.
The design, definitions and arms are in [PROTOCOL.md](PROTOCOL.md); the code follows it.

Per problem: **predict** a distribution over f(n) (scored in bits) -> **compute** f(n) exactly -> **weave**
bounds with P(holds) -> **check** them -> propose one-field **what-if** mutations -> tally
which (bridge, lemma) pairs win. Breadth (arm A4) adapts to surprise: an unsurprising
problem gets one bound from its prior bridges and one mutation, escalating to the wide
mode if that bound does not win; a surprising one gets five bounds (at least three
outside the prior) and five mutations.

## Quickstart

```sh
uv sync                                    # stdlib only; pytest in the dev group
uv run pytest                              # all tests, about 2 s
uv run python -m mathexp truth --seed oddtown
uv run python -m mathexp show-prompt --step weave_menu --seed cap_set --mode wide
uv run python -m mathexp scan                                       # run 2: scan/*.jsonl, scan/selected.jsonl
uv run python -m mathexp run --backend mock --max-calls 9            # -> scratch/store-mock/
uv run python -m mathexp run --backend mock --seeds-from scan/selected.jsonl --predict-only --notes on
uv run python -m mathexp report --store scratch/store-mock
uv run python -m mathexp run --backend claude --model sonnet --seeds oddtown --max-calls 4 --store scratch/store-smoke
```

`run` flags: `--backend claude|openrouter|mock`, `--model`, `--policy surprise|random`,
`--feedback on|off` (A2), `--menu on|off` (A3), `--breadth fixed|adaptive` (A4, default
adaptive), `--bridges-per-problem` / `--mutations-per-problem` (fixed breadth only,
default 3), `--max-calls`, `--max-problems`, `--seeds all|a,b`, `--store DIR`,
`--run-id auto`, `--seed INT`, `--time-limit SECONDS` (per (problem, n), default 20).
Run 2 adds `--seeds-from scan/selected.jsonl [--top N]` (provenance `scan`, in a seeded
random order of the selection), `--predict-only` (arm A5: one predict call per problem, no
weave or mutation) and `--notes on|off` (arm A5: earlier problems of the same family and
their true f in the predict prompt, most recent first, at most 12).

`scan` flags: `--family setfam|addfree` (default both), `--max-len L` (default setfam 8,
addfree 12), `--top K` (40), `--quota Q` (20 per family), `--time-limit SECONDS` (5 per
(spec, n)). The scan resumes from `cache/scan_truth.jsonl`; golf tables are cached in
`cache/golf/`.

## Layout

```
PROTOCOL.md            the design (definitions fixed before any data)
mathexp/
  __main__.py          CLI: run, report, scan, truth, show-prompt
  guard.py             idle priority + single-threaded numeric libraries
  loop.py              the loop, surprise trigger, prior bridges, A4 breadth
  problems/base.py     Family interface, problem_id, one-field mutation check
  problems/setfam.py   family A: set families on [n] (max clique)
  problems/addfree.py  family B: equation-free subsets of [n], Z_n, F_p^k (backtracking)
  problems/clique.py   bitset branch-and-bound max clique (Tomita-style colouring bound)
  truth.py             brute-force cache: cache/truth.json (gitignored)
  golf.py              golf length G(seq): bottom-up enumeration, cached tables, jump J
  scan.py              run 2 scan: enumerate specs, brute-force, golf, Occam surprise S, score S/L, select top K (S > 0)
  checking.py          weave statuses: violated / consistent / insufficient / error, tight_at, win
  safe_eval.py         AST-allowlist evaluator for bound expressions (no eval/exec)
  store.py             append-only JSONL ledgers + refusals (a) and (b)
  calibration.py       per domain x kind x level x model; runs + change-point tests
  report.py            calibration, surprises, statuses, reuse, A4, what-if tree
  llm.py               claude / openrouter / mock backends
  prompts.py           templates, menus, seeds, JSON schemas, parsing
  prompts/*.md         system, predict, weave_menu, weave_free, mutate (versioned)
  menus/bridges.toml   10 bridges
  menus/lemmas.toml    19 lemmas (unverified entries carry unsure = true)
  seeds.toml           9 named seeds with known_as names
scan/                  run 2 scan output: <family>.jsonl, summary.json, selected.jsonl
tests/                 pytest
store/                 the real store (empty; the mock backend may not write here)
```

A store directory holds `predictions.jsonl` (claims with confidence, then outcome rows
joined by id), `problems.jsonl`, `weaves.jsonl` and `calls.jsonl` (full prompt, raw
response, parse status, model id, tokens, nominal cost).

## Backends

- **claude** (default): `claude -p --safe-mode --no-session-persistence --model <m>
  --tools "" --system-prompt <s> --output-format json --json-schema <schema>`, prompt on
  stdin, run in the empty directory `scratch/claude-cwd/`. No CLAUDE.md, memory or tools
  reach the model. `--json-schema` works in print mode; the structured object is read from
  `structured_output`. Timeout 300 s; one retry on a failed or unparseable call.
  `total_cost_usd` is recorded as nominal (billed to the subscription).
- **openrouter**: urllib POST to the chat-completions endpoint; key from
  `OPENROUTER_API_KEY` or a gitignored `.env` in this repo. The model must end in `:free`
  unless `--allow-paid` is given.
- **mock**: deterministic test double that exercises every path (each weave status, a
  rejected two-field mutation, surprising and unsurprising problems, escalation). Its
  default store is `scratch/store-mock/`; it refuses to write into `store/`.

## Rules enforced in code

- (a) truth for (problem, n) is refused until that n's value prediction (a distribution;
  an interval in run-1 stores) is on disk;
- (b) the mock backend refuses the repo's `store/`;
- (c) a mutation that changes anything but exactly one spec field (ns aside), or is
  invalid, is rejected and logged in `problems.jsonl`; a valid one whose brute force yields
  fewer than 4 values on its ns is logged as `refused_uncheckable` and not queued;
- model-written bounds are evaluated only through the AST allowlist.

## Compute rules

This machine's CPU is shared with the owner's long jobs. The CLI sets idle priority
(Windows `SetPriorityClass` IDLE; POSIX `nice 19`) and `OMP/MKL/OPENBLAS_NUM_THREADS=1`
at start. Everything is single-process and sequential: one brute force at a time, one LLM
call at a time, with a per-(problem, n) time limit and a cache. Do not run several
brute-force or test processes in parallel. There is nothing to put on the GPU. Full runs
need no local data, so they can go to a one-shot cloud routine instead.
