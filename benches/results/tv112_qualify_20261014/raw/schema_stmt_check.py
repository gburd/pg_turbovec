#!/usr/bin/env python3
"""Statement-level schema comparison: strip comments, split on ';' at statement
end, normalise whitespace, compare as multisets (pgrx's emission order is not
stable, so ordered diff is noisy; line-sorting alone could hide a clause moving
between statements)."""
import re, sys, collections
def stmts(path):
    t = open(path).read()
    t = re.sub(r'/\*.*?\*/', '', t, flags=re.S)
    t = re.sub(r'--[^\n]*', '', t)
    out = [' '.join(x.split()) for x in t.split(';')]
    return collections.Counter(x for x in out if x)
a, b = stmts(sys.argv[1]), stmts(sys.argv[2])
only_a, only_b = a - b, b - a
print(f"statements: {sum(a.values())} vs {sum(b.values())}; only in first: {sum(only_a.values())}; only in second: {sum(only_b.values())}")
for s in list(only_a)[:5]: print("  - ", s[:160])
for s in list(only_b)[:5]: print("  + ", s[:160])
sys.exit(1 if only_a or only_b else 0)
