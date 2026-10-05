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
- **Calibration domain**: (family, prediction kind, model). The metrics are:
  - 80% interval coverage;
  - median |log2 ratio| for value predictions;
  - Brier score for probability predictions, such as P(bound holds) and predicted
    mutation effects.

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
