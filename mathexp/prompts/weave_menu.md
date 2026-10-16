version: weave_menu-v1
Step: weave

Problem:
{{problem}}

Find upper bounds g(n) on f(n) by weaving: choose a bridge from the menu below (a dictionary that translates the problem into another field's language), translate the hypotheses through it, apply one or more lemmas from the menu whose hypotheses now hold, and translate the conclusion back into an upper bound on f(n). Respect the problem's symmetries.

{{breadth}}

Bridges:
{{bridges}}

Lemmas:
{{lemmas}}

Write each bound as an expression "bound_expr" in n{{params}}, using numbers, + - * / // % ** and the functions comb(a, b), factorial, floor, ceil, sqrt, log, log2, exp, min, max and binom_sum(n, s) = sum_{i=0}^{s} comb(n, i). No other names.
For each bound give "p_holds": your probability, stated before any checking, that g(n) >= f(n) holds for every n in {{ns}}.
{{feedback}}
Answer with JSON only, in this shape (at most {{k}} bounds):
{"bounds": [{"bridge_id": <bridge id>, "lemma_ids": [<lemma id>, ...], "translation": <how the hypotheses translate>, "bound_expr": <expression>, "proof_sketch": <short argument>, "p_holds": <number in [0, 1]>}, ...]}
