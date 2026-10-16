# Results

## Run 3: method surprise, cluster notes, tight reuse (2026-10-05)

**Setup:**
- claude-sonnet-5-5 answered every call (the stored label says Haiku; see "Data
  integrity"); seed 0; code `run3-build` at 5bd05a2; two one-shot cloud routines,
  `run3 --phase a5b` and `run3 --phase a4b`;
- stores `runs/run3-a5b/a5b` and `runs/run3-a4b/a4b`. Each run folder has the cloud
  report (`a5b_report.txt`, `a4b_report.txt`) and `report_final.txt`, regenerated locally
  from the stores with the model-label fix. The two differ only in the model line;
- A5b: 12 clusters of 5 (60 problems, 48 children), 228 calls, all parsed, no parent
  replaced. A4b: 80 completed problems (39 seeds, 41 mutants), 480 calls, all parsed,
  no retries;
- every number below the reports' own (noise floors, leave-one-out, strata splits) was
  computed from the stores afterwards. Those are diagnostics, not pre-registered tests.

**Headline:**
- **A4b passed its pre-registered test, but only just.** The band-stratified AUC(M) is
  0.717, with a 95% interval of [0.519, 0.912]. That rests on 49 problems with 7
  positives (the plan expected about 65 with about 22). The result is fragile:
  - dropping any one of 5 of the 7 positives moves the interval's lower end below 0.5;
  - stratifying by seed/mutant as well as band gives 0.603;
  - 6 of the 7 positives are a bound improved by exactly 1;
  - in the policy replay, widening on M does no better than widening at random.

  This is weak support that has not been replicated, not evidence that the principle
  pays.
- **A5b: explanations help, in aggregate, where there is room to improve.**
  - N2 − N1 is negative on 33 of 44 untied children (sign test p = 0.0013, Wilcoxon
    p = 0.0009).
  - The whole effect is in addfree: 19 of 24 children are negative, with a median of
    −0.38 bits.
  - Setfam shows nothing (median −0.005). The model already predicts those children at
    about 0.07 bits.
  - A single child's change sits inside the noise between identical prompts. The effect
    shows only across children.
- **Reuse:** neither raw nor tight importance on run 2 predicts bound quality on A4b's
  wide weaves (Spearman −0.029 and −0.084, n = 16). The test does not favour switching
  the prior to tight importance.
- **Forecasts:** 3 of 6 came true, and the Brier score is 0.353, worse than 0.25 for
  saying 0.5 every time. Both primary tests came true, at stated probabilities of 0.20
  and 0.25.

### A4b: widen on method surprise

| population | count |
|---|---|
| completed | 80 (39 seeds, 41 mutants) |
| P empty, bridge chosen by the method call | 2 |
| prior failed (probe 1 has no win) | 25 (6 seeds, 19 mutants; 24 setfam) |
| probe 1 tight | 4 |
| **primary** | **49, with Y = 1 on 7** |

The plan expected about 65 primary problems with about 22 positives. Two things cut
that down: the probe failed to win on 25 problems, and Y was rare. The primary
population's observed bands were:

| band | n | Y = 1 |
|---|---|---|
| close | 15 | 1 |
| moderate | 10 | 0 |
| loose | 24 | 6 |

So 108 of the 122 within-band pairs are in the loose band. Within a band, M is
−log2 P(that band). The test therefore asks whether the model gave "loose" less
probability on the problems where the wide weave then did better.

| measure (primary, n = 49) | value |
|---|---|
| **band-stratified AUC(M)** | **0.717, 95% CI [0.519, 0.912]** (1999 usable samples) |
| stratified AUC of M0 / V / R / Random | 0.500 / 0.602 / 0.504 / 0.443 |
| logistic Y ~ band + log(g − f) + M: coefficient of M | +0.104, 95% CI [−1.220, 3.944] (80 of 2000 samples dropped) |
| unstratified AUC(M), AUC(M0); paired difference | 0.517, 0.269; +0.248 [0.016, 0.449] |
| unstratified AUC(R) / AUC(V) / AUC(Random) | 0.699 / 0.609 / 0.459 |

**How robust the pass is** (diagnostics; the pre-registered number is the seed-0 one
above):
- **Bootstrap seed.** Seeds 1 to 5 give lower ends of 0.505 to 0.523, so Monte Carlo
  noise does not flip the result.
- **Leave one positive out.** The point estimate ranges from 0.678 to 0.773. The
  interval's lower end falls below 0.5 in 5 of the 7 cases (0.473 to 0.496) and stays
  above it in 2 (0.510 and 0.568).
- **Confounds.** Stratifying by family and band gives 0.682; by seed/mutant and band,
  0.603; by all three, 0.643. Within seeds alone it is 0.567 (29 problems, 6
  positives). Within mutants it is 1.000, on a single positive.
  - The model forecast "loose" confidently for mutants: their mean M in the loose band
    was 0.88, against 1.27 for seeds.
  - Mutants gave 1 positive among 7 loose problems; seeds gave 5 among 17.
  - Part of the pooled AUC is this difference between strata.
- **Size of the gains.** In 6 of the 7 positives, the wide bound is exactly 1 below the
  better probe: 15 → 14, 16 → 15 twice, 29 → 28, 82 → 81, and 9 → 8 (= f, tight). One
  gain is large: 37 → 12, with f = 10. No Y depends on flooring g. Seven weaves had a
  non-integer g(n_max), and none of them decides a Y.
- **Other secondary measures.** The unstratified AUC(M) of 0.517 and the logistic
  interval do not support the principle either. The AUC(M) − AUC(M0) interval excludes
  0 only because M0 is worse than chance (0.269): rare bands get a high M0 and almost
  never have Y.

**M ≥ 3 bits fired on 3 of 49 primary problems (6%).** All three were in the close band
with Y = 0. The rarity comes from accurate forecasts, not a faulty instrument:
- the modal band was right on 50 of 80 problems, at a mean stated confidence of 0.49.
  That is underconfident; the Brier score on the top-band rows is 0.202;
- every forecast put less than 1/8 on some band, but the observed band had P ≤ 1/8 only
  6 times in 80;
- mean M on completed problems was 1.43 bits, against 2.32 for a uniform forecast.

**Direction.** On the primary problems, the band matched the forecast median on 29,
was worse (disappointment) on 6, and was better (pleasant) on 14.
- AUC(M if disappointment, else 0) = 0.597 (stratified 0.586).
- AUC(M if pleasant, else 0) = 0.333 (stratified 0.459).

What signal there is comes from disappointment, on 6 problems.

**Policy replay** (80 completed: 25 prior-failed problems are always widened, and each
signal ranks the other 55):

| policy | widened | calls | wins | tight | wins/100 | tight/100 |
|---|---|---|---|---|---|---|
| never (prior failed only) | 25 | 345 | 118 | 13 | 34.2 | 3.8 |
| always | 80 | 400 | 195 | 17 | 48.8 | 4.2 |
| M q=25% | 39 | 439 | 144 | 15 | 32.8 | 3.4 |
| M0 q=25% | 39 | 359 | 137 | 16 | 38.2 | 4.5 |
| V q=25% | 39 | 359 | 134 | 13 | 37.3 | 3.6 |
| R q=25% | 39 | 359 | 133 | 13 | 37.0 | 3.6 |
| Random q=25% | 39 | 359 | 140 | 13 | 39.0 | 3.6 |
| M q=50% | 53 | 453 | 160 | 15 | 35.3 | 3.3 |
| M0 q=50% | 53 | 373 | 165 | 17 | 44.2 | 4.6 |
| V q=50% | 53 | 373 | 149 | 14 | 39.9 | 3.8 |
| R q=50% | 53 | 373 | 149 | 13 | 39.9 | 3.5 |
| Random q=50% | 53 | 373 | 157 | 15 | 42.1 | 4.0 |

- **Acting on M does not pay.** Even before its method calls are charged, M's choice of
  problems to widen matches Random's:

  | q | M wins, tight | Random wins, tight |
  |---|---|---|
  | 25% | 144, 15 | 140, 13 |
  | 50% | 160, 15 | 157, 15 |

  Once its 80 method calls are charged, M has the fewest wins per 100 calls of any
  policy.
- **"Always" has the most wins per 100 calls (48.8), but partly by construction.** A
  wide weave returns up to 3 bounds per call, and wins count bounds. On tight bounds per
  100 calls, M0 is best (4.5 and 4.6), then always (4.2).
- **The prior-failed rule did the useful work.** On prior-failed problems the wide
  weave beat both probes on 12 of 25, mostly by large factors (101 → 22, 37 → 8,
  256 → 93). In the primary population it did so on 7 of 49, mostly by 1.

**Replication of run 2.** A non-prior win came on 14 of 17 value-surprising problems
and on 43 of 63 others (Fisher p = 0.368). Run 2 had 4 of 9 against 15 of 25
(p = 0.46). Both runs are null, in opposite directions: value surprise does not mark
where other methods win.

**Labels and checks:**
- no probe bound fell outside P (0 of 160);
- no wide bound had an excluded label (0 of 125), and none used an off-menu id;
- 0 of 285 weaves were violated. 5 were errors: conditional expressions outside the
  grammar.

### A5b: notes on related problems, with explanations

**Primary.** Over 48 children, N2 − N1 has a mean of −0.226 bits and a median of −0.024.
11 children went up, 33 down and 4 tied. Sign test p = 0.00126; Wilcoxon W+ = 217.5,
p = 0.000867 (n = 44).

**The noise floor.** A parent's N1 and N2 prompts are byte-identical (12 of 12), so any
difference in a parent's bits is sampling noise.

| | parents: median abs(N2 − N1), max | children: median N2 − N1 | children: up / down / tied | sign test p | children's N1 bits (median) |
|---|---|---|---|---|---|
| all | 0.029, 1.385 | −0.024 | 11 / 33 / 4 | 0.0013 | 0.54 |
| setfam | 0.013, 0.069 | −0.005 | 7 / 14 / 3 | 0.19 | 0.069 |
| addfree | 0.730, 1.385 | −0.379 | 4 / 19 / 1 | 0.0026 | 2.16 |

- **One child's change cannot be read.** In addfree, the median child change (0.38
  bits) is about half the median noise between identical prompts (0.73). Over all
  parents, the noise includes two swings of more than a bit (−1.113 and +1.385).
- **The aggregate is real.** The noise is symmetric: parents went 6 up and 5 down,
  while children went 11 up and 33 down. Counting only children whose change exceeds
  the parents' median noise (0.029 bits), 22 of 28 went down (sign test p = 0.0037).
- **The effect is practically meaningful in addfree only.**
  - In the median addfree child, −0.38 bits per value means about 30% more probability
    on the true value.
  - The top candidate was right on 435 of 792 child values in N0, 455 in N1 and 521 in
    N2. By that count, N2 beat N1 on 14 children and lost on 3.
  - The overall median of −0.024 is small because half the children are setfam
    problems, which the model already predicts at the floor.

**Overlap split** (low overlap means transfer; high overlap means lookup):

| | children | N1 − N0 | N2 − N0 | N2 − N1 |
|---|---|---|---|---|
| low overlap | 26 | median +0.007, 14 up / 11 down, p = 0.69 | −0.034, 8 / 16, p = 0.15 | **−0.025, 5 / 18, p = 0.011** |
| high overlap | 22 | −0.015, 7 / 13, p = 0.26 | −0.104, 5 / 16, p = 0.027 | −0.019, 6 / 15, p = 0.078 |

On low-overlap children, the related problem's values alone did nothing; the values
with explanations helped. That is the transfer the protocol was looking for, on small
n and one run.

**Position** (mean bits; position 1 is the parent):

| | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| N0 | 1.080 | 1.151 | 1.860 | 1.398 | 1.692 |
| N1 | 1.388 | 1.216 | 1.625 | 1.133 | 1.035 |
| N2 | 1.465 | 0.938 | 1.384 | 0.902 | 0.881 |

N2 is lowest at every child position. These are means, though, and a few multi-bit
swings dominate them. The parents' mean moves by 0.31 to 0.39 bits from N0 to N1 or N2,
whose prompts differ from N0's by one line. Read the table as description only.

**What the explanations say.** Three explanations and one N2 child's notes were read in
full. They state structure:
- "a common core plus disjoint blocks";
- "the odd residues, which is sum-free because odd + odd is even";
- "a Sidon set ... f grows roughly like √n".

They also restate their own problem's thresholds. None forecasts another problem:
- a search of all 48 for forecast words ("predict", "variant", "child", "will be" and
  similar) found none;
- no explanation cites an n beyond its own problem's range.

Two caveats:
- **The witness travels too.** The explain call also saw the extremal witness at n_max,
  and some explanations pass it on (for example {1, 3, 7, 15} at n = 30). So N2 − N1
  measures the explanation and the witness together.
- **Cuts.** 14 of the 48 explanations were cut at 80 words, often mid-sentence. The
  median length was 77 words, and the longest was 114 before the cut.

### Reuse weighted by tightness

- **Tight importance has nothing to separate.** No pair is tight in more than one
  family, here or in run 2 (where 4 pairs reach one family). So tight importance reduces
  to its tie-break.
- **Top 3 on A4b's wide weaves:**
  1. double_counting + pigeonhole (tie-break 0.591);
  2. orbit_averaging + pigeonhole (0.571);
  3. graph_spectrum + pigeonhole (0.494).
- **Out-of-sample test.** 16 pairs have at least 3 counted, consistent bounds on the
  wide weaves.
  - 7 of them were never used in run 2, so they score 0 on both measures.
  - Across the 16, the two measures correlate at 0.86.
  - Spearman against mean f/g is −0.029 for raw importance and −0.084 for tight.
  - Neither measure predicts, and with these pairs the test cannot tell them apart.
- **Generic lemmas still shape the prior.** Raw importance ranks a bridge by the number
  of families it won in, so double_counting led both families' rankings.
  - It was in P for 63 of 80 completed problems.
  - It was in P for 19 of the 25 problems whose probe failed to win.
  - 36 problems got exactly P = {double_counting, graph_spectrum}.

### Strata

- **Named against variant.** Run 3 has no named problems: every seed is a scanned spec
  (provenance `scan`), and every other problem is a variant. The nearest proxy is the
  model's own recognition at P ≥ 0.5:
  - A4b: 2 completed problems, both primary, both with Y = 0;
  - A5b (N0): 1 parent and 4 children, whose N2 − N1 was 0, 0, −0.065 and −0.018.

  That is too few to split on. The names it recognised show mutants drifting back to
  known problems: sum-free sets, 3-AP-free sets, Sidon sets and Fisher's inequality.
- **Seeds against mutants (A4b):**

| | seeds | mutants |
|---|---|---|
| completed | 39 (19 setfam, 20 addfree) | 41 (all setfam) |
| prior failed | 6 | 19 |
| primary (Y = 1) | 29 (6) | 20 (1) |
| stratified AUC(M) | 0.567 | 1.000 (one positive) |
| Y = 1 among completed | 9 | 11 |
| mean value bits | 1.70 | 0.35 |

**The mutant stratum has no addfree problem, because of a bug.** All 60 addfree
mutations were rejected for changing two fields:
- the int/cyclic seeds store `p = null`;
- the model's mutants set `p` to 2 or 3;
- `validate_mutation` counts `p` as a change.

Seed against mutant is therefore confounded with family.

### Pre-registered forecasts, scored

| claim | P | outcome | number |
|---|---|---|---|
| A4b: the band-stratified AUC(M) interval excludes 0.5, above 0.5 | 0.20 | **true** (just) | 0.717 [0.519, 0.912] |
| A4b: the unstratified AUC(R) > AUC(M) on the primary population | 0.60 | **true** | 0.699 against 0.517 |
| A4b: M ≥ 3 bits on 20–60% of the primary population | 0.50 | false | 3 of 49 (6%) |
| A5b: N2 − N1 < 0 with sign-test p < 0.05 (all 48 children) | 0.25 | **true** | 33 of 44 down, p = 0.0013 |
| A5b: the median N2 − N1 < 0 on low-overlap children | 0.55 | **true** | −0.025 (18 of 23 down) |
| Reuse: no double_counting pair in the top 3 by tight importance on A4b's wide weaves | 0.55 | false | double_counting + pigeonhole ranks first |

The Brier score over the six is 0.353. The forecasts were too pessimistic about both
primary tests and too optimistic about how often M would fire.

### Data integrity

- **The model label.** The `model` field on every call row says
  `claude-haiku-4-5-20251001`, but Sonnet answered.
  - All 708 call rows (228 A5b, 480 A4b) requested `claude-sonnet-5-5`.
  - Each row's `meta.model_usage` has exactly two entries: Haiku with 12–34 output
    tokens (an auxiliary call) and claude-sonnet-5-5 with 175–13,155.
  - Sonnet has the most output tokens on all 708, so it is the answering model on every
    row.
  - `llm.py` took the first key. It now uses the requested model if `modelUsage`
    contains it, and otherwise the model with the most output tokens (tested).
    `report3` derives the model from `meta.model_usage` for stored rows.
  - The append-only stores are unchanged, so the `by` field on prediction and weave rows
    still says Haiku. Each store uses one label throughout, so A4b's prior lookup, which
    filters weaves by `by`, saw all of its own weaves.
  - The generic `report.txt` inside each store labels its calibration domains with the
    Haiku id; read that as Sonnet. The cloud probe call also logged Haiku. Its abort
    check passed because it tests membership in `modelUsage`.
  - Run 2's rows have only Sonnet in `modelUsage`, so run 2 is unaffected.
- **The truth cache was not committed.** The `.gitignore` pattern `cache/` matched it,
  and the witness sidecar `truth.witness.jsonl` sat beside it. The truth values are in
  the problem rows, and the reports read them from there. The witnesses survive only
  inside A5b's explain prompts, or by recomputation.
- **Deviations that touched results:**
  - **g is floored.** 7 weaves had a non-integer g(n_max); none decides a Y.
  - **P is an ordered fill.** 78 of 80 completed problems had |P| = 2.
  - **Explanations are cut at 80 words.** 14 of 48 were cut.
  - **Parents in N1 and N2 saw a "no notes yet" line**, so their prompt differs from
    N0 by one line and N1 = N2. That is what supplied the noise floor above.
  - **Mutants queue behind the seeds**, and the addfree mutation bug above left 41
    mutants, all setfam.

### What it means for run 4

- **A4b needs a replication before anything builds on M.** Use a fresh population,
  pre-register strata for family and seed/mutant, and plan for enough positives. The
  rule that did measurable work here was the cheap one: widen when the prior fails.
- **Lessons on pre-registered definitions** (none was changed here):
  - Y counts any strict improvement, and 6 of the 7 primary positives were
    improvements by 1. A minimum margin would measure widening that matters.
  - The primary population excludes prior-failed problems, which is where widening paid.
    The power estimate assumed few such failures, but 31% of probes failed to win.
  - Stratifying by band alone leaves the family and stratum confounds in place.
  - Tight importance cannot rank anything while no pair is tight in two families.
  - "Wins per 100 calls" counts bounds, which flatters the k = 3 wide weave.
- **Fix the addfree mutation validator** (`p` for int/cyclic groups), or A4b-style runs
  keep a setfam-only mutant stratum.
- **Reuse: the protocol's switch to tight importance is not triggered,** because the
  test favours neither measure. Raw importance still filled P with double_counting, and
  the probes then failed on 25 problems. A prior that down-weights generic pairs is
  worth its own pre-registered test.
- **A5b: test the explanation and the witness separately, on addfree-like problems.**
  Setfam children are at the floor.

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
