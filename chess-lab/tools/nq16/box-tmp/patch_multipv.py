s = open("/extnvme/active/matches/eval_worker.py").read()

# MultiPV on at engine setup
a = """        self.send("setoption name Threads value 1")"""
b = """        self.send("setoption name Threads value 1")
        self.send("setoption name MultiPV value 5")"""
assert s.count(a) == 1
s = s.replace(a, b)

# parse 5 pv lines per position
a = """            m = re.search(r"score (cp|mate) (-?\\d+)", last) if last else None
            mpv = re.search(r" pv (.+?)\\s*$", last) if last else None
            pout.append(mpv.group(1) if mpv else None)
            if not m:
                out.append(None)
                continue"""
b = """            m = re.search(r"score (cp|mate) (-?\\d+)", last) if last else None
            mpv = re.search(r" pv (.+?)\\s*$", last) if last else None
            pout.append(mpv.group(1) if mpv else None)
            if not m:
                out.append(None)
                continue
            del m, mpv"""
assert s.count(a) == 1
s = s.replace(a, b)

# eval_fens must collect multipv lines: switch loop to gather per-multipv last lines
a = """            self.send("position fen " + fen)
            self.send(f"go depth {depth}")
            last = ""
            while True:
                l = self.p.stdout.readline()
                if not l:
                    break
                if l.startswith("info") and " score " in l and " pv " in l:
                    last = l
                if l.startswith("bestmove"):
                    break"""
b = """            self.send("position fen " + fen)
            self.send(f"go depth {depth}")
            lines = {}
            while True:
                l = self.p.stdout.readline()
                if not l:
                    break
                if l.startswith("info") and " score " in l and " pv " in l:
                    mm = re.search(r" multipv (\\d+) ", l)
                    k = int(mm.group(1)) if mm else 1
                    lines[k] = l
                if l.startswith("bestmove"):
                    break
            last = lines.get(1, "")"""
assert s.count(a) == 1
s = s.replace(a, b)

# store top-5 as compact strings "cp|pv"
a = """            pout.append(mpv.group(1) if mpv else None)"""
b = """            top5 = []
            for k in range(1, 6):
                lk = lines.get(k)
                if not lk:
                    continue
                mk = re.search(r"score (cp|mate) (-?\\d+)", lk)
                pk = re.search(r" pv (.+?)\\s*$", lk)
                if mk and pk:
                    vk = int(mk.group(2))
                    if mk.group(1) == "mate":
                        vk = (10000 + min(abs(vk), 900)) if vk > 0 else -(10000 + min(abs(vk), 900))
                    top5.append(f"{vk}|{pk.group(1)}")
            pout[-1] = " ;; ".join(top5) if top5 else None"""
assert s.count(a) == 1
s = s.replace(a, b)

open("/extnvme/active/matches/eval_worker.py", "w").write(s)
import py_compile
py_compile.compile("/extnvme/active/matches/eval_worker.py", doraise=True)
print("MultiPV-5 patched, compiles")
