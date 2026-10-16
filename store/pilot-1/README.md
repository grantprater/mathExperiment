# Pilot 1 (2026-10-04)

Arm A4 (fixed vs adaptive breadth), claude-sonnet-5-5, policy surprise, feedback off, menu on, seed 1.

Both runs stopped at 10 problems, not at the 100-call budget, because `--max-problems` defaulted to 10
(fixed at 30 calls, adaptive at 31). Nine of the ten problems in each arm are named seeds, so this is
not a budget-matched A4 result. It is kept because it shows the contamination baseline: Sonnet predicted
every seed's small cases exactly or almost exactly (surprise 0 except Salem–Spencer, max 0.15).
