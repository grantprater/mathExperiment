# Protocol: lemma-level weaving

## Question

Can an LLM loop do lemma-level weaving on problems the literature does not cover? Weaving
here means: translate the problem into another field's language, apply a lemma that holds
there, and translate the conclusion back as a bound. Two follow-up questions:
- Does prediction error ("surprise") point at where the structure changes?
- Does reuse across unrelated problems point at the ideas that matter?

The worked example behind the design is Oddtown → Eventown → Frankl–Wilson → Borsuk:
- a dictionary (a set becomes its 0/1 vector over F₂) plus one borrowed lemma (linear
  independence bounds the count) proves the Oddtown bound;
- one "what if" (odd → even) turns a linear answer into an exponential one;
- the generalised lemma later won on problems it was never built for.

The loop turns the owner's account of intuition into operations:

| Intuition | Operation |
|---|---|
| change the level of abstraction | bridges (dictionaries between fields) |
| symmetry | translations must respect the problem's symmetries; tight cases are often symmetric |
| surprise | predict small cases before computing them; large misses are leads |
| what if | mutate one hypothesis at a time |
| an idea proclaims its importance | count wins on problems it was not built for |

## One iteration (one problem)

1. **Render** the problem neutrally: plain set or group language, no story, no famous names.
2. **Predict** f(n) for each small n as a point estimate with an 80% interval. Also record
   whether the model recognises the problem as a named result.
3. **Compute** f(n) exactly by brute force (cached), then score the predictions.
4. **Weave.** Choose k bridges. For each one:
   - translate the hypotheses;
   - pick a lemma whose hypotheses now hold;
   - translate its conclusion back into an upper bound g(n), written as an expression;
   - before checking, state P(g holds) as a probability.
5. **Check** g against f on every computed n.
6. **What if.** Propose mutated problems that each change one hypothesis, each with a
   predicted effect, and queue them.
7. **Reuse.** Tally wins for each (bridge, lemma) pair across problems and families.

## Definitions (fixed before any data; never tuned on results)

- **f(n)**: the exact maximum, found by brute force. A timeout makes f(n) unknown, and
  that value is excluded from scoring.
- **Surprise** of a problem: the mean over n of |log2((pred + 1) / (f + 1))|.
  **Interval miss**: f lies outside [lo, hi].
- **Weave status**:
  - `violated` if f(n) > g(n) for some computed n;
  - `consistent` if g(n) ≥ f(n) for every computed n, with at least 3 values computed;
  - `insufficient` if fewer than 3 values were computed;
  - `error` if g failed to parse or evaluate;
  - `tight_at`: the values of n where g(n) = f(n).

  Consistent is not proved; proof sketches are stored unverified.
- **Trivial bound**:
  - family A: the number of subsets that meet the size condition;
  - family B: the size of the ambient set or group.
- **Win**: a consistent weave with g(n_max) < trivial(n_max). The ratio
  g(n_max) / trivial(n_max) is reported as a continuous measure.
- **Importance** of a (bridge, lemma) pair: the number of distinct families in which it
  has a win, with ties broken by the number of problems in which it has a win.
- **Provenance**:
  - `named`: a textbook instance listed in the seed file;
  - `variant`: anything produced by mutation.

  All metrics are reported separately for named and variant problems, because named
  results are memorised.
- **Surprising**: a problem is surprising if, for some n,
  |log2((pred + 1) / (f + 1))| ≥ 1, that is, some small case was off by a factor of 2
  or more. This threshold is a constant in code, not a flag.
- **Prior bridges** of a problem, in order of preference:
  1. the bridges behind its parent's wins;
  2. failing that, the family's top 2 bridges by importance so far;
  3. failing that, the model's own choice.
- **Calibration domain**: (family, prediction kind, model). The metrics are:
  - 80% interval coverage;
  - median |log2 ratio| for value predictions;
  - Brier score for probability predictions, such as P(bound holds) and predicted
    mutation effects.

## Prediction store (design borrowed from ~/reticle)

Reticle's ledger (`~/reticle-store/notes/predictions.jsonl`) taught four things:
- **One flat row per claim, with a numeric confidence on every row.** Reticle's
  pre-registered rows have prose outcomes and no confidence, so they cannot be scored.
- **Record that the prediction came before the look, and enforce it in code.** Reticle's
  conversion audit: "every convention that was merely written down got violated again...
  every one encoded as code that refuses held."
- **Calibration is a gap.** Per domain, compute accuracy minus mean stated confidence.
  Add a runs test and a Bonferroni-corrected change-point test.
- **Split by level** (value / mechanism / ontology). Reticle's most expensive claims were
  also its most overconfident.

This experiment adds:
- a Brier score;
- outcomes joined to predictions by id, not by free text;
- arm A2, which feeds calibration back into the prompt automatically. Reticle only
  reports calibration.

Every prediction is a binary claim with a confidence. The kinds are:

| kind | claim | confidence | level |
|---|---|---|---|
| `interval` | f(n) ∈ [lo, hi] (the point estimate is kept for log error) | 0.8 | value |
| `bound_holds` | g(n) ≥ f(n) for every computed n | the model's P | mechanism |
| `mutation_effect` | at the largest n both problems computed, the child's f is larger than, equal to, or smaller than the parent's | the model's P | mechanism |
| `recognition` | this problem is the named result X | the model's P | ontology |

Recognition is scored only for named seeds; for variants it is scored `couldnt-tell`.

Prediction rows are appended with `outcome: null`. An outcome row
`{kind: "outcome", ref: <id>, outcome: right | wrong | couldnt-tell, observed}` is
appended after the computation. The loop refuses to compute f for a problem until its
interval predictions have been written.

## Arms

Arms run only after the scaffold is verified, and each comparison is matched on the
number of LLM calls.

- **A1, mutation policy**: surprise-first versus random. Measures: wins per 100 calls,
  and the number of distinct winning (bridge, lemma) pairs.
- **A2, calibration feedback**: the predict prompt either shows the family's running
  calibration or does not. Measures: interval coverage and |log error| on later
  predictions.
- **A3, bridge menu**: weaving from the bridge and lemma menus versus a free-form "find
  an upper bound" prompt. Measures: wins and the violated rate.

- **A4, breadth: "assume the prior, widen the search on surprise".** This is a principle
  from reticle.
  - *Fixed* breadth: k = 3 bounds from the full menu and m = 3 mutations for every
    problem.
  - *Adaptive* breadth:
    - an unsurprising problem gets k = 1 bound from its prior bridges and m = 1 mutation;
    - a problem that is surprising, or whose narrow weave produced no win (the prior
      failed), gets k = 5 bounds from the full menu, at least 3 of them outside the
      prior, and m = 5 mutations.

  Measures:
  - wins per 100 calls;
  - how often the prior held on unsurprising problems (if it almost always holds,
    narrowing costs nothing);
  - the principle's own claim: wins from bridges outside the prior are concentrated on
    surprising problems.

  The principle's claim is tested in the fixed arm, because there every problem gets the
  full menu. That makes P(non-prior win | surprising) versus
  P(non-prior win | unsurprising) an unbiased comparison. The adaptive arm then tests
  whether acting on the principle saves calls.

## Run 2: surprise per unit of description (fixed before run 2's data)

Run 1 found no surprising problems: at n ≤ 7, Sonnet recalled or worked out every answer
(RESULTS.md). Run 2 looks for problems that are short to state but whose answers resist
short description. The owner's target is "maximum surprise for minimum entropy", and the
principle behind it is interestingness = compression progress (Schmidhuber).

- **Statement length L(spec)** is the number of atoms in the canonical spec, excluding
  `ns`. Each condition name counts 1, and each number in it counts 1. Example: Oddtown
  `p=2, size_mod=[1], int_mod=[0]` has 6 atoms. A statement may combine two simple
  conditions. In family B that means an optional second equation.
- **Golf length G(seq)** is the node count of the shortest expression in the golf grammar
  that reproduces a sequence exactly.
  - The grammar: terminals `n, 1, 2, 3`; binary `+ - * // % **`, `comb`, `binom_sum`,
    `min`, `max`; unary `isqrt`, `ilog2`.
  - The search is bottom-up enumeration up to a size cap, merging expressions with equal
    value vectors.
  - A sequence with no expression under the cap gets G = cap + 1.
- **Golf jump** J = G(f at every computed n) − G(f at every n but the largest). Kept as
  a reference column only; the scan score below replaces it.
- **Occam surprise S, the scan's model-free surprise** (revised after the first scan;
  see below):
  - Walk through the sequence in prequential order. For each k ≥ 4, take the
    minimal-size expressions that match f at the first k values of n. Together they
    predict the (k+1)-th value.
  - A prediction counts only if the prefix genuinely **compresses**:
    b · G_k < Λ_k, where
    - b = log2(16) = 4 bits per node (16 grammar symbols: 4 terminals, 10 binary, 2 unary);
    - Λ_k = Σ_{i<k} log2(trivial(n_i) + 1) is the cost of stating those k values
      literally.
  - A **break** at k means the prefix compresses and no minimal expression predicts
    f at the next value. The break costs log2(trivial(n) + 1) bits for that next n.
  - S is the sum of break costs over the sequence.
- **Scan score** = S / L. Every spec up to a length cap is enumerated, brute-forced and
  golfed.
  - A spec with fewer than 4 computed values is refused.
  - A spec with **no interaction** is dropped: f(n) = trivial(n) at every computed n,
    meaning the conditions never constrain each other.
  - Specs with identical f sequences keep only the shortest statement.
  - Only specs with S > 0 are eligible. The top K, with a per-family quota, become run 2's
    problems. If fewer qualify, K shrinks and the report says so; zero-score fillers are
    not used.
- **Why S replaced J** (decided after the first scan and before any run-2 model call):
  - On family B's 27-value sequences almost nothing golfs under the cap, so J was 0 for
    267 of 270 specs. The prequential walk finds a break wherever it happens, not only at
    the last n.
  - Size 7–8 witnesses fitted 5 values by coincidence. The compression guard (an
    expression must be shorter than the values it explains) removes them.
  - Pure counting specs had f = trivial and are not extremal problems at all.
- **Model surprise in bits.** For each n, the model gives a distribution over f(n): up to
  5 candidate values with probabilities, plus residual mass. Any value it did not list
  gets an equal share of the residual over the remaining values in [0, trivial(n)].
  - Surprise at n = −log2 P(f(n)), capped at 30 bits.
  - A problem's surprise is the mean over n; its surprise_max is the max over n.
  - A top-1 claim "f(n) = top candidate" at confidence P(top) is scored as a binary row,
    so it fits the reticle-style gap and Brier calibration.
- **Surprising** (replacing run 1's trigger): some n has surprise ≥ 4 bits, meaning the
  true value had probability at most 1/16. This is a constant in code.

**Arms for run 2:**
- **A5, notes and compression progress.** Both conditions take the same top-K list in the
  same seeded random order, with prediction only (no weave, no mutation). One condition
  shows notes in the predict prompt: the statements and true sequences of the problems
  already processed in the same family, most recent first, up to a fixed cap. The other
  shows none.
  - Primary measure: the paired difference in bits per problem, notes minus no-notes.
  - Secondary measure: bits as a function of position in the order.
  - Compression progress means the notes arm's bits fall faster.
- **A4 again**, with the bits trigger. The seeds are the top-scanned problems, and every
  mutation must be checkable (at least 4 computed values) or it is refused.
- **Side result.** Does cheap surprise predict model surprise? Report the Spearman
  correlation between scan score (and J) and the model's mean bits, using the no-notes
  arm of A5.

**Headline metrics for weaves from run 2 on:** bounds tight at the largest computed n,
and g(n_max) / trivial(n_max). The "win" count stays as a secondary column.

## Run 3: method surprise, related notes, tight reuse (fixed before run 3's data)

Run 2 (RESULTS.md) left three open threads:
- "Widen on surprise" failed with a value trigger. Surprise about f says little about
  whether the usual method will work.
- A5 was null, but its notes described unrelated problems and carried no explanation.
- Reuse rewarded weak generic lemmas, because a win is anything below trivial.

Facts about run 2's code that shape the fixes:
- The breadth decision came before any weave (loop.py, step 3), so it could only react
  to value surprise.
- The fixed arm never ran a dedicated prior weave. Only 8 of its 34 problems had both a
  prior bound and an alternative, so its widening comparison rested on 8 problems.
- A parent's prior was every bridge it won with, with no cap. The lemma menu was never
  restricted, and `in_prior` trusted the model's own bridge label.

An independent review of the first draft found that its primary test mostly measured
room to improve and probe luck. The design below includes the fixes. Run 3 excludes run
2's 40 scanned specs from every arm.

### A4b: widen on method surprise (full-information design)

Every problem gets both the prior's bounds and the alternatives' bounds. Any trigger
policy can then be replayed on the same problems, which pairs the comparison and makes
a live adaptive arm unnecessary. The replay holds P at its full-information values.

**Bridge groups.** Bridges that give the same bound under different names count as one:
- {charvec_fp, multilinear_poly, incidence_matrix} (linear-algebra dimension bounds);
- {fourier_indicator, slice_rank};
- every other bridge is its own group.

**Steps for one problem:**
1. Predict f as in run 2 (a distribution per n), then compute the truth.
2. Fix the prior set P, at most 2 bridges:
   - the parent's winning bridges, ranked by their best g(n_max);
   - then the family's ranking (run 2's rule);
   - if P is empty, the method call shows the full menu and forecasts for the bridge it
     would choose, and that bridge becomes P. These problems are excluded from the
     primary test.
3. **Method prediction**: a separate, isolated call. It shows the statement, the true
   values and P's menu entries. It asks only for a distribution over the bands below for
   the probe's best bound, with no derivation. The loop refuses to run the probe until
   this row is stored.
4. **Probe** and **re-probe**: two independent weaves, each with k = 1 and the bridge menu
   restricted to P. As in every weave, the prompt does not show the truth. The
   re-probe controls for luck: if one probe draw is poor, any second draw beats it.
5. **Wide**: a weave with k = 3, with the bridge menu restricted to bridges outside every
   group that contains a P bridge. It does not show the probes' bounds. A wide bound
   whose bridge label falls in an excluded group is recorded but not counted. This runs
   for every problem.
6. Mutate as in the fixed arm (m = 3, refuse uncheckable mutants), with
   `--policy random`. The default orders mutants by the parent's value surprise
   (loop.py:561), which would select the population on V. The mutants feed the queue but
   take no part in the replay.

A problem is **completed** when its probe, re-probe and wide weave are all stored.

**Bands** for r = g(n_max) / f(n_max) of the first probe's best consistent bound:
- tight: r = 1;
- close: 1 < r ≤ 1.25;
- moderate: 1.25 < r ≤ 2;
- loose: r > 2;
- failed: no consistent bound (violated, error or insufficient).

The cut points come from run 2's fixed arm: the quartiles of the best prior-bridge r per
problem, over 21 problems, are 1.14, 1.33 and 2.0. Run 3's problems are new, so this is
not fitting on the evaluation set. The new prediction kind `method_band` sits at the
mechanism level. Its top band is also scored as a binary row for gap and Brier
calibration.

**Signals**, each computed before the wide weave's outcome is known:
- **M, method surprise**: −log2 P(observed band). The binary trigger is M ≥ 3 bits (the
  observed band had probability at most 1/8), a constant in code. A uniform forecast
  scores 2.32 bits on every outcome.
- **M0, the forecast-free comparator**: −log2 of the observed band's leave-one-out
  frequency in the primary population.
- **V, value surprise**: run 2's trigger (surprise_max over n).
- **R, looseness**: the first probe's r. It costs no model call.
- **Random**: a seeded draw.

**Outcome Y** (per problem): the wide weave produced a counted, consistent bound whose
g(n_max) is strictly below the better of the two probes' best g(n_max).

**Primary population**: completed problems whose first probe has a win and is not tight
at n_max, with a non-empty P from step 2. A tight probe cannot be beaten. A probe with
no win is "prior failed" and always widens (as in run 2). Such problems are reported but
are not in the primary test.

**Why the test is stratified.** In run 2 the median f(n_max) was 7, so a "close" probe
usually means g = f + 1, and Y there needs the wide weave to be exactly tight. Y will
therefore cluster in the loose band. Without stratifying, any forecast that is
miscalibrated across bands would score an AUC above 0.5 from room to improve alone.

**Measures:**
- **Primary**: the band-stratified AUC of M for Y. Only (Y = 1, Y = 0) pairs in the same
  observed band are compared. It comes with a 2000-sample bootstrap 95% interval,
  resampled within bands and seeded. The principle is supported if the interval excludes
  0.5 from above.
- **Secondary**:
  - logistic Y ~ band (categorical) + log(g − f) + M, giving the coefficient of M with a
    bootstrap interval;
  - the unstratified AUC(M) − AUC(M0), paired;
  - the unstratified AUC(R) and AUC(V).
- **Direction** (secondary; the principle as stated is two-sided): split M into
  disappointment (the observed band is worse than the forecast's median band) and
  pleasant surprise, and report AUC for each.
- **Policy replay**: each signal ranks problems and widens the top q, for q = 25% and
  50%, plus the "prior failed" problems. Report counted wins and tight bounds per 100
  calls. Every policy pays for predict, both probes and mutate. A policy pays for the
  wide weave only where it widens. M also pays for its method call on every problem.
- **Replication of run 2**: P(non-prior win | value-surprising) against
  P(non-prior win | not), now with every problem having both.

**Size and stopping:**
- 80 completed problems (`--max-problems 80`). That is 6 calls each (predict, method,
  probe, re-probe, wide, mutate). Every retry counts against the budget
  (loop.py:172-176), so the safety cap is `--max-calls 600`.
- Seeds are the 19 remaining eligible setfam specs and the next 20 addfree specs by scan
  rank, then mutants. The 12 A5b parents are among these seeds (calls are stateless, so
  this is harmless). Seeds and mutants are reported as strata.

**Power:**
- Run 2 suggests about 15% of probes are tight, and about a third of problems have Y.
  That third rests on 3 of 8 problems, and the close band will add almost no positives.
- So roughly 65 problems enter the primary test, with about 22 positives.
- For an unstratified AUC, the interval is about ±0.15. A simulation at that size puts
  power at about 55% for a true AUC of 0.65 and about 80% for 0.70. The stratified test
  has less power still.
- A null below 0.70 is therefore uninformative, and RESULTS.md will say so rather than
  report it as evidence against the principle.

### A5b: notes on related problems, with explanations

**Clusters** are built before any model call, by `python -m mathexp clusters`:
- A parent is one of the top 6 eligible specs per family by scan score, excluding run
  2's 40.
- Children come from `mutate_random` (one field changed, seeded), drawn until 4 qualify
  or 50 draws pass. A child qualifies if:
  - it is checkable (at least 4 values);
  - it has an interaction (f is not trivial);
  - its f sequence differs from the parent's and its siblings';
  - it is not one of run 2's 40 specs or in another cluster.
- A parent with fewer than 4 qualifying children is replaced by the next eligible spec,
  and the report lists the replacements. A dry run against cached truths (seed 0, 80
  draws) gave 2 to 9 qualifying children per setfam parent and 7 to 11 per addfree
  parent.
- That gives 12 clusters of 5, 60 problems in all. The order is seeded: clusters in
  random order, and within each cluster the parent first, then the children.
- Brute force keeps an **extremal witness** at n_max: the family or set achieving f.
  - setfam: `max_clique` returns the vertices, and setfam.py:149 currently discards them.
  - addfree: `max_free_set` returns a count only, so it must return the argmax.
  - The truth cache stores values only, so the witness is stored in the cache too.
  - A unit test re-derives the Fano plane (n = 7, intersections in {1, 3}) and S(3,4,8)
    (n = 8, intersections in {0, 2}). This commits the extraction that RESULTS.md
    quotes, which was done ad hoc in run 2 and is not in the repo.

**Conditions**, all prediction-only, with the same order:
- N0: no notes.
- N1: notes on the earlier problems in the **same cluster**, giving statement and true
  sequence.
- N2: N1 plus an explanation for each earlier problem.
  - After a problem's truth is computed, a separate call sees its statement, its true
    values and the witness at n_max.
  - It writes at most 80 words stating only the rule or structure behind the values,
    with no forecasts for other problems. That text is the note.
  - The model compresses its own observations; nothing is hand-written.
  - The explain call is skipped for each cluster's last child, whose note is never
    shown.

**Overlap** of a child: the maximum, over the problems shown in its notes, of the share
of common n where the two f values are equal. Mutants can change ns (addfree.py:329-330),
so only common n count. Low overlap means below 0.5.

**Measures**, on the 48 children (the parents have no notes in any condition):
- **Primary**: paired bits per child, N2 − N1, with a sign test (ties dropped) and a
  Wilcoxon test. N2 contains N1, so this isolates what the explanation adds. N2 − N0
  could be won by copying values alone.
- **Secondary**:
  - N1 − N0 and N2 − N0 on low-overlap children, which is transfer rather than copying;
  - the same on high-overlap children, which is lookup;
  - bits as a function of the child's position (2 to 5) in each condition.

The calls total 228: 60 for N0, 60 for N1, and for N2, 60 predict plus 48 explain. With
48 pairs and no ties, a sign test at p < 0.05 needs 32 of 48 children to improve (31
gives p = 0.06).

### Reuse weighted by tightness

- A **pair** is (bridge, sorted lemma_ids).
- **Tight importance** of a pair is the number of distinct families in which it has a
  bound tight at n_max. Ties are broken by its mean best f(n_max) / g(n_max) over the
  problems where it was used, with a minimum of 3 problems. A sum would reward frequent
  use and bring back the generic-lemma bias. Raw wins stay as a secondary column.
- **Out-of-sample test:**
  - rank the pairs by each measure on run 2's two A4 stores;
  - score each pair on A4b's **wide weaves only**, by its mean f(n_max) / g(n_max) over
    its counted, consistent bounds, for pairs with at least 3;
  - report Spearman for raw and for tight importance.

  Probe uses are forced by P, which ranks by raw wins, so only the wide weave's free
  choices are scored. The test is descriptive, because the number of pairs will be
  small.
- Switching the prior to tight importance is run 4's step, if this test favours it.

### Pre-registered forecasts

These were written with the protocol, before any run-3 data. They are scored in
RESULTS.md.

| claim | P |
|---|---|
| A4b: the band-stratified AUC(M) interval excludes 0.5, above 0.5 | 0.20 |
| A4b: the unstratified AUC(R) > AUC(M) on the primary population | 0.60 |
| A4b: M ≥ 3 bits on 20–60% of the primary population | 0.50 |
| A5b: N2 − N1 < 0 with sign-test p < 0.05 (all 48 children) | 0.25 |
| A5b: the median N2 − N1 < 0 on low-overlap children | 0.55 |
| Reuse: on A4b's wide weaves, no double_counting pair is in the top 3 by tight importance | 0.55 |

### Build and run order

1. Code, each piece with a mock-backend test:
   - witness extraction and the witness cache;
   - the `clusters` command;
   - the `method_band` prompt, parse and outcome;
   - breadth mode `full`: P capped at 2, probe and re-probe restricted to P, the wide
     weave excluding P's groups, and the excluded-label check;
   - notes scope `cluster`, with optional explanations;
   - report sections for A4b (stratified AUC, bootstrap, replay), A5b (overlap split),
     and tight importance.
2. Build the clusters and the A4b seed list in a cloud routine, since brute force is
   CPU-heavy.
3. Run A5b (228 calls), then A4b (about 480 calls), both in one-shot cloud routines on
   Sonnet 5.5. Results are pushed to a new branch.
4. Write up in RESULTS.md: named against variant, seeds against mutants, and the
   forecasts scored.

The graph family stays deferred. 154 addfree specs remain eligible, so the scan has not
run thin.

## Known limits

- Consistency at small n is weak evidence for an asymptotic bound, and the proof sketches
  are unverified. The next stage would add an LLM judge for the sketches, and Lean for
  the statements that survive.
- General theorems already cover many variants; modular Frankl–Wilson covers much of
  family A. The test is whether the loop derives correct bounds on unseen instances, not
  whether the results are new.
- The menus are hand-written and bias which weaves the loop finds. A3 measures how much.
- `claude -p` runs in an isolated working directory with `--safe-mode` and a replacement
  system prompt, so no CLAUDE.md, memory, or tools reach the model. Every row records the
  model id.

## Compute

Brute force runs single-threaded at idle priority, with a time limit and a cache. LLM
calls run one at a time. Full runs need no local data, so they can go to a one-shot
cloud routine (a 4-core Linux VM with an authenticated `claude` CLI).
