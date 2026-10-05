version: predict-v1
Step: predict

Problem:
{{problem}}

Nothing has been computed yet. Predict f(n) for each of these values of n: {{ns}}.
For each n give a point estimate "point" and an 80% interval ["lo", "hi"]: integers such that you expect f(n) to lie in [lo, hi] four times out of five.

Also say whether you recognise this problem as a known named result: put its name in "recognized_as" (null if none), and in "p_recognized" your probability that this identification is correct.
{{feedback}}
Answer with JSON only, in this shape:
{"predictions": [{"n": <int>, "point": <int>, "lo": <int>, "hi": <int>}, ...], "recognized_as": <string or null>, "p_recognized": <number in [0, 1]>}
