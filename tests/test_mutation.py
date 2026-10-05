import random

from mathexp.problems import get_family, validate_mutation

A = get_family("setfam")
B = get_family("addfree")
ODD = {"p": 2, "size_mod": [1], "int_mod": [0], "ns": [3, 4, 5]}


def test_single_field_change_accepted():
    assert validate_mutation(A, ODD, {**ODD, "size_mod": [0]}) == []
    assert validate_mutation(A, ODD, {**ODD, "p": 3}) == []
    # ns changes do not count
    assert validate_mutation(A, ODD, {**ODD, "size_mod": [0], "ns": [3]}) == []


def test_two_field_change_rejected():
    errs = validate_mutation(A, ODD, {**ODD, "size_mod": [0], "int_mod": [1]})
    assert errs and "exactly one" in errs[0]


def test_no_change_rejected():
    assert validate_mutation(A, ODD, dict(ODD))
    # reordering a residue list is not a change
    two = {"p": 3, "size_mod": [0, 2], "ns": [3]}
    assert validate_mutation(A, two, {**two, "size_mod": [2, 0]})


def test_invalid_specs_rejected():
    assert validate_mutation(A, ODD, {**ODD, "p": 4})
    assert validate_mutation(A, ODD, {**ODD, "size_mod": [5]})  # residue out of range
    assert validate_mutation(A, ODD, {**ODD, "size_mod": [0, 1]})  # every residue: use null
    assert validate_mutation(A, ODD, {**ODD, "colour": "red"})
    assert validate_mutation(A, {**ODD, "size_mod": [0]}, {**ODD, "size_mod": [0], "size_abs": [3]})  # empty
    cap = {"group": "fpk", "p": 3, "equation": "3ap", "ns": [1, 2]}
    assert validate_mutation(B, cap, {**cap, "equation": {"a": 1, "b": 0, "c": 2}})
    assert validate_mutation(B, cap, {**cap, "equation": "4ap"})
    assert validate_mutation(B, cap, {**cap, "equation": {"a": 1, "b": 1, "c": 1}}) == []


def test_random_mutations_are_single_field_and_valid():
    rng = random.Random(1)
    seeds = [(A, ODD), (A, {"p": 3, "size_abs": [3], "int_mod": [0], "ns": [3, 4]}),
             (B, {"group": "fpk", "p": 3, "equation": "3ap", "ns": [1, 2]}),
             (B, {"group": "int", "p": 3, "equation": "sidon", "ns": [4, 5]})]
    for fam, spec in seeds:
        for _ in range(40):
            child = fam.mutate_random(spec, rng)
            assert validate_mutation(fam, spec, child) == [], (spec, child)
            assert fam.problem_id(child) != fam.problem_id(spec)
