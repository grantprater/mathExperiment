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
