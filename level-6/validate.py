"""Helpers expected by campaign.py: parse inputs/outputs and replay programs with the level 5 simulator."""
import opt


def read_cases(source):
    return opt.parse(open(source).read())


def read_programs(output, n):
    progs = opt.read_progs(str(output), n)
    if any(p is None for p in progs):
        raise ValueError(f"{output}: expected {n} programs")
    return progs


def simulate(r, program, pattern):
    return opt.l5.run(r, pattern, program)


def validate(source, output):
    cases = read_cases(source)
    progs = read_programs(output, len(cases))
    bad = [i for i, ((r, p, c), pr) in enumerate(zip(cases, progs)) if simulate(r, pr, p) < c]
    if bad:
        raise ValueError(f"{output}: invalid programs at cases {bad[:10]}")
    return {"cases": len(cases), "instruction_lines": sum(map(len, progs))}
