import json, argparse, sys
from pathlib import Path

def iter_json_file(p):
    s = p.read_text(encoding="utf-8")
    s_strip = s.lstrip()
    if s_strip.startswith("["):
        for obj in json.loads(s):
            yield obj
    else:
        for line in s.splitlines():
            line=line.strip()
            if not line: continue
            yield json.loads(line)

def main(files, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    idx = 0
    with out.open("w", encoding="utf-8") as fw:
        for f in files:
            p = Path(f)
            if not p.exists():
                print("Missing:", f, file=sys.stderr); continue
            for obj in iter_json_file(p):
                # normalize minimal fields are kept downstream; keep full object here
                obj.setdefault("id", f"elife_{idx}")
                fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
                idx += 1
    print("Wrote", out, "records:", idx)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="eLife JSON/JSONL files to merge")
    ap.add_argument("--out", default="data/elife_all.jsonl")
    args = ap.parse_args()
    main(args.files, args.out)