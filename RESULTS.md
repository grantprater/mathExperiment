# Results

## Run 2: A5 notes, and A4 on scanned problems (2026-10-05)

**Setup:**
- claude-sonnet-5-5; seed 2; problems from `scan/selected.jsonl` (K = 40, 20 per family);
- stores `store/run2-a5-{notes,nonotes}` and `store/run2-a4-{fixed,adaptive}`, each with
  `report.txt` and `run.log`;
- the paired and within-family numbers below were recomputed by hand from the stores.

**The scan surfaces lynchpins.** Extracting the extremal families by brute force (done
ad hoc; the extraction code is not in the repo, and run 3 adds it with a unit test):
- "pairwise intersections of size 1 or 3" at n = 7 is the Fano plane (its 7 lines, which
  cover every pair exactly once) plus the whole set;
- "intersections of size 0 or 2" at n = 8 is S(3,4,8) (14 blocks, which cover every triple
  exactly once) plus the empty set.

Both statements have length 3, and the scan was told about neither object. The top scorer
(sizes ≡ 1 mod 5 with intersections at least 2) is a threshold artifact: nothing can
happen until n = 7.

**The model already knows the lynchpins.** At the break it gave 0.6 to 15 (S(3,4,8)) and
0.5 to 8 (Fano). It was less sure of the ordinary small cases before the break.

**A5, notes, is null.**
- The paired difference (notes − no notes) is −0.05 bits per problem (median +0.003).
- Notes helped on 19 problems and hurt on 21; sign test p = 0.88.
- The notes reached 38 of 40 prompts.
- Caveat: the scan picks problems with distinct sequences, so the notes described
  unrelated problems, and they carried values only, not explanations. A fair test of
  compression progress needs related problems (a parent and its mutants, or problems
  sharing an extremal object) and needs the explanation in the notes.

**Side result: cheap surprise against the model's surprise** (no-notes arm). The pooled
Spearman(S, bits) is +0.57 (n = 40), but much of that is the difference between families.

| Within family | Spearman(S, bits) | Spearman(S/L, bits) |
|---|---|---|
| setfam (n = 20) | +0.54 | +0.05 |
| addfree (n = 20) | +0.10 | −0.12 |

Dividing by statement length adds nothing yet.

**A4 with the bits trigger.** About a quarter of the problems were surprising: 9 of 34
(fixed) and 8 of 33 (adaptive).

| | fixed | adaptive |
|---|---|---|
| problems | 34 | 33 |
| bounds | 54 | 57 |
| violated | 0 | 0 |
| tight at the largest n | 5 | 6 |
| median best-per-problem g/trivial | 0.267 | 0.267 |
| wins per 100 calls | 51 | 49 |
| prior held on narrow problems | n/a | 24/25 = 96% (1 escalation) |

- **"Assume the prior" is supported.** On unsurprising problems the prior bridge won 24 of
  25 times, and per-problem bound quality matched the fixed arm.
- **"Widen on surprise" is not supported here.** In the fixed arm, wins from bridges
  outside the prior came on 4 of 9 surprising problems and 15 of 25 unsurprising ones
  (Fisher p = 0.46).
  - A likely reason: the surprise was about values, while the prior governs methods. Being
    unable to predict f(n) (often an irregular, number-theoretic sequence) does not mean
    the usual bound technique fails.
  - Reticle's version works because its surprise and its prior live in the same channel.
    The next test should trigger widening on method surprise, such as the prior's bound
    being loose or failing, rather than on value surprise.
- **Bounds are safe but loose.** Across run 2, 0 of 111 bounds were violated, and only
  about 10% were tight at the largest n.
- **Cross-family reuse fired only for generic tools** (double counting with pigeonhole, or
  with Cauchy–Schwarz). Reuse as a measure of importance rewards weak, general lemmas.
  Weighting reuse by tightness is the obvious next step.

## Run 2 scan (2026-10-05, after A4 run 1)

Command: `python -m mathexp scan` (defaults), one process at idle priority. Outputs:
`scan/setfam.jsonl`, `scan/addfree.jsonl`, `scan/summary.json`, `scan/selected.jsonl`.

The scan score is Occam surprise S / L (PROTOCOL.md "Run 2"). It replaced the golf jump
J / L, which is kept as a reference column. The first scan's J-based selection was
discarded before any run-2 model call.

**Golf size cap 10.** It was chosen by measuring table builds on the setfam ns 3..8,
single-threaded:

| cap | entries | build | peak memory |
|---|---|---|---|
| 9 | 368,679 | 3 s | 134 MB |
| 10 | 1,746,704 | 14 to 27 s | 403 MB |
| 11 | 8,708,681 | 92 s | 1.84 GB, over the 1.5 GB limit |

The tables built at cap 10 were:

| ns | entries | build |
|---|---|---|
| 3..8 (setfam) | 1,746,704 | 22 s |
| 4..30 (int/cyclic) | 3,061,949 | 61 s |
| 1..6 (F_2^k) | 1,410,403 | 13 s |

The Occam walk loads each table's (entries × ns) int32 matrix with numpy. For ns 4..30
that peaks at 1.42 GB, so no chunking was needed.

**Caveat, by construction:** an expression that is invalid at any n of a table is not in
the table. An expression that fits a prefix but overflows or divides by zero at a later n
therefore cannot be a prefix predictor. This holds for both G and S.

**Scan ns and time limit:**
- setfam ns 3..8, because brute force at n = 8 averages 0.2 s per spec;
- int/cyclic ns 4..30;
- F_2^k for k = 1..6, and F_3^k for k = 1..4;
- 5 s per (spec, n), stopping at a spec's first timeout.

**Counts:**

| | setfam (L ≤ 8) | addfree (L ≤ 12, the whole space) |
|---|---|---|
| specs enumerated | 2209 | 1062 |
| refused (fewer than 4 values) | 0 | 66 (all of F_3^k) |
| dropped: no interaction (f = trivial everywhere) | 139 | 1 |
| duplicate truth sequences | 1790 | 726 |
| kept | 280 | 269 |
| prefix never compresses | 220 | 93 |
| S > 0 (eligible) | 39 | 174 |
| selected | 20 | 20 |

- Both families filled their quota of 20, so K = 40.
- The first brute-force pass took 1197 s of wall time: setfam 205 s, addfree 992 s,
  mostly the timeouts on F_2^6 and F_3^4.
- A rerun from the caches, including the Occam walk, takes about 103 s.

**Top scorers:**
- setfam:
  - `|A ∩ B| ≡ 1 (mod 5)` with `|A ∩ B| ≥ 2` gives 1, 1, 1, 1, 2, 8 (S = 15.0, L = 6);
  - `|A ∩ B| ∈ {1, 3}` gives 3, 4, 5, 6, 8, 9: `n` predicts 7 at n = 7.
- addfree:
  - Sidon sets in Z_n (S = 12.9, L = 4, three breaks);
  - several two-equation int/cyclic specs with 4 to 6 breaks each.

**The low end of the selection is setfam.** Its specs have one break at n = 8, and some
witnesses are close to the guard, for example `ilog2((n ** isqrt(n)))` at G = 5 against
Λ = 20.6 bits.

## A4 run 1: fixed vs adaptive breadth (2026-10-05)

**Setup:**
- model claude-sonnet-5-5; policy surprise; feedback off; menu on; seed 1;
- 100 calls per arm;
- stores in `store/a4-fixed` and `store/a4-adaptive`, each with `report.txt` and `run.log`.

**Headline: the principle was not tested.** No problem in either arm crossed the surprise
trigger. The largest error at any n was a factor of 1.4 (max |log2| = 0.49, against a
trigger of 1.0).
- Sonnet's point predictions were almost always exact, including on variants, and its
  80% intervals covered 98–100% of values.
- The fixed arm's split of non-prior wins was 14/34 on unsurprising problems against
  0/0 on surprising ones, so there is no comparison to make.

| | fixed | adaptive |
|---|---|---|
| problems (named / variant) | 34 (9 / 25) | 30 (9 / 21) |
| weaves | 59 | 66 |
| violated | 0 | 0 |
| insufficient (fewer than 3 values computed) | 2 | 37 |
| wins | 52 | 20 |
| wins tight at the largest n | 22 | 13 |
| wins per 100 calls | 52 | 20 |
| prior held on narrow problems | n/a | 19/30 (11 escalations) |

**The gap in wins per 100 calls is mostly an artifact, not a verdict on A4:**
1. On the cap set, the narrow weave's prior (the slice-rank bound 3·2.756ⁿ) failed. It
   beats the trivial 3ⁿ only from n ≥ 13, beyond brute-force reach.
2. The failure escalated to m = 5 mutations.
3. Those led the adaptive arm into F₅ⁿ and F₇ⁿ problems, where brute force reaches only
   1–2 values of n.
4. 37 of its 66 weaves could not be checked.

**Other observations:**
- **Zero violated bounds in 125 weaves.** P(holds) averaged 0.95–0.97 against an actual
  1.00, so the model proposes safe bounds and is slightly underconfident about them.
- **The most reused idea is Frankl–Wilson.** The multilinear polynomial bridge with the
  diagonal criterion and the dimension count won on 14 problems in the fixed arm, 11 of
  them variants. The model reapplies the polynomial method across mutated intersection
  conditions.
- **No (bridge, lemma) pair won in both families, so cross-family importance never
  fired.** The menus are largely family-specific. The one bridge that spans both
  families, the polynomial method (Frankl–Wilson and slice rank), cannot win on cap sets
  at checkable n.
- **The checker's horizon is biased against asymptotic ideas.** A small-n checker
  favours simple bounds over deep ones: the deep idea looks bad within the range that
  can be verified.
- **Mutations still drift back to known problems:**
  - some "what if"s restate the same problem in another syntax (3ap written as the
    generic x + y = 2z);
  - some walk to other seeds (sum-free ↔ 3-AP-free ↔ Sidon);
  - 5 of 23 pilot-1 mutations were duplicates.

**Fixes before run 2:**
1. **Refuse uncheckable problems.** Do not queue a mutation unless brute force reaches at
   least 4 values of n within the time limit, or extend its ns until it does.
2. **Make surprise possible.** Small cases have to stop being predictable. Options:
   - larger n (set families at n = 8, integers up to 40);
   - problems with two simultaneous conditions;
   - generic equations with unusual coefficients;
   - a third family.
3. **Report tight-at-n_max and g/trivial as the headline**, not wins, because the win rule
   is lenient (2·3ⁿ⁻¹ < 3ⁿ counts as a win).

## Pilot 1

See `store/pilot-1/README.md`. Both runs stopped at 10 problems because of a
`--max-problems` default.
