# Results

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
