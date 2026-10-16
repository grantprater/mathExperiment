version: predict-v2
Step: predict

Problem:
{{problem}}
{{notes}}
Nothing has been computed yet for this problem. Predict f(n) for each of these values of n: {{ns}}.
For each n give your probability distribution over f(n): up to 5 "candidates", each a pair [value, probability] with an integer value, and "other", the probability that f(n) is none of your candidates. The probabilities for one n should sum to 1. Mass in "other" is spread evenly over every value f(n) could take that you did not list. You are scored by the probability you put on the true value, so spread your mass honestly when unsure.

Also say whether you recognise this problem as a known named result: put its name in "recognized_as" (null if none), and in "p_recognized" your probability that this identification is correct.
{{feedback}}
Answer with JSON only, in this shape:
{"predictions": [{"n": <int>, "candidates": [[<int>, <number in [0, 1]>], ...], "other": <number in [0, 1]>}, ...], "recognized_as": <string or null>, "p_recognized": <number in [0, 1]>}
