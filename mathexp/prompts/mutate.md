version: mutate-v1
Step: mutate

Current problem:
{{problem}}

Its spec:
{{spec}}

Computed values f(n) for it: {{truth}}

{{schema}}

Propose {{m}} "what if" variants of this problem. Each variant must change exactly ONE field of the spec (ns aside) and must be a valid spec. Prefer changes that you expect to alter the structure of the answer, not just its size.
For each, say in "what_if" what changes, in "predicted_effect" whether the variant's f will be "larger", "equal" or "smaller" than this problem's f at the largest n both problems compute, and in "p_effect" your probability that this prediction is right.

Answer with JSON only, in this shape ({{m}} items):
{"mutations": [{"spec": <full spec object without ns>, "what_if": <one sentence>, "predicted_effect": "larger" | "equal" | "smaller", "p_effect": <number in [0, 1]>}, ...]}
