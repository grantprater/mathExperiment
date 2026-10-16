version: weave_free-v1
Step: weave

Problem:
{{problem}}

Find upper bounds g(n) on f(n). Use any method you like; for each bound name the method in a short "bridge_id" (a few words, snake_case) and the key results used in "lemma_ids".

{{breadth}}

Write each bound as an expression "bound_expr" in n{{params}}, using numbers, + - * / // % ** and the functions comb(a, b), factorial, floor, ceil, sqrt, log, log2, exp, min, max and binom_sum(n, s) = sum_{i=0}^{s} comb(n, i). No other names.
For each bound give "p_holds": your probability, stated before any checking, that g(n) >= f(n) holds for every n in {{ns}}.
{{feedback}}
Answer with JSON only, in this shape (at most {{k}} bounds):
{"bounds": [{"bridge_id": <method>, "lemma_ids": [<result>, ...], "translation": <how the problem is reformulated>, "bound_expr": <expression>, "proof_sketch": <short argument>, "p_holds": <number in [0, 1]>}, ...]}
