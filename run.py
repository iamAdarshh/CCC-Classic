import sys, glob, os, time, importlib.util

level = sys.argv[1]
d = f"level-{level}"
spec = importlib.util.spec_from_file_location("solve", os.path.join(d, "solve.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

for inp in sorted(glob.glob(os.path.join(d, "in_*.txt"))):
    is_ex = "example" in inp
    if ("--example" in sys.argv) != is_ex:
        continue
    out = inp.replace(os.sep + "in_", os.sep + "out_")
    t = time.time()
    res = mod.solve(open(inp).read())
    dt = time.time() - t
    if is_ex:
        exp = open(out).read()
        print(inp, "OK" if res == exp else "MISMATCH", f"{dt:.3f}s")
        if res != exp:
            open(out + ".mine", "w").write(res)
    else:
        open(out, "w").write(res)
        print(inp, "->", out, f"{dt:.3f}s")
