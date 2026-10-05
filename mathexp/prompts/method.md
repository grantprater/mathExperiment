version: method-v1
Step: forecast the method

Problem:
{{problem}}

Its true values, computed by brute force: {{truth}}

A solver who has NOT seen these values will be asked for one upper bound g(n) on f(n), derived through one of the bridges below (a dictionary that translates the problem into another field's language) and lemmas from a fixed menu. The bound is then checked against the true values.
{{choose}}
Bridges:
{{bridges}}

Forecast how good the solver's bound will be at n = {{n_max}}, where f({{n_max}}) = {{f_max}}. Let r = floor(g({{n_max}})) / f({{n_max}}). The bands are:
- "tight": r = 1 (the bound is exact at n = {{n_max}});
- "close": 1 < r <= 1.25;
- "moderate": 1.25 < r <= 2;
- "loose": r > 2;
- "failed": no valid bound (g(n) < f(n) at some computed n, or g does not evaluate).
Give your probability for each band; the five should sum to 1. Do not derive a bound; give only the forecast.

Answer with JSON only, in this shape:
{"bridge_id": {{bridge_shape}}, "bands": {"tight": <p>, "close": <p>, "moderate": <p>, "loose": <p>, "failed": <p>}}
