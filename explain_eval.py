"""python3 explain_eval.py [--ask]
Do LLM-written fraud explanations tell the truth about the model?

40 flagged applications. For each, the model's real top-3 reasons are its three largest positive
feature contributions (exact, because the model is linear). Claude writes the analyst-facing explanation
under two prompts:
  open      - sees the application data and the score, writes why it was flagged
  grounded  - also sees the model's top-3 reasons, told to explain only with those
Each explanation lists the factors it cites (from a fixed list), and the narrative is also scanned for
factor mentions. A cited factor outside the model's top 3 is an unfaithful reason.
--ask calls the agent (AGENT_CMD, default Claude Code's CLI on Haiku) and caches answers in answers/.
"""
import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import shlex
import subprocess
import sys
from collections import Counter

import model as M

ROOT = pathlib.Path(__file__).parent
CMD = shlex.split(os.environ.get("AGENT_CMD", "claude -p --model haiku --strict-mcp-config "
                                 "--disallowedTools Bash,Edit,Write,Read,WebFetch,WebSearch,Glob,Grep,Task"))
KEYWORDS = {
    "device_shared_apps": r"\bdevice", "ssn_prefix_reuse_x": r"\bssn", "email_age_days": r"\bemail",
    "phone_is_voip": r"voip", "ip_distance_km": r"\bip\b|ip address|geolocat|location mismatch|distance",
    "address_velocity": r"(?<!email )address", "name_email_mismatch": r"name.{0,25}(match|email)|handle",
    "thin_file": r"thin|credit history|credit file", "phone_tenure_days": r"phone.{0,30}(new|tenure|active|age|recent)|number.{0,20}(new|recent)",
}

PROMPTS = {
    "open": """You write the explanation a fraud analyst sees next to a flagged identity application.
Model fraud score: {score:.2f} (flagged above 0.50).
Application data:
{data}

Write a 2-3 sentence explanation of why this application was flagged, for the analyst deciding whether
to approve it. Reply with JSON only: {{"explanation": "...", "cited_factors": [keys from {keys}]}}""",
    "grounded": """You write the explanation a fraud analyst sees next to a flagged identity application.
Model fraud score: {score:.2f} (flagged above 0.50).
Application data:
{data}

The model's top reasons, from its own feature contributions, in order: {reasons}.
Write a 2-3 sentence explanation of why this application was flagged, using ONLY these reasons. Do not
cite any other field as a reason, even if it looks unusual. Reply with JSON only:
{{"explanation": "...", "cited_factors": [keys from {keys}]}}""",
}


def cases():
    rows, keys, w, score = M.build()
    flagged = [(s, r) for s, r in ((score(r)[0], r) for r in rows) if s > 0.5]
    rng = random.Random(7)
    pick = rng.sample(flagged, 40)
    out = []
    for i, (s, r) in enumerate(pick):
        _, con = score(r)
        top3 = [k for k, v in sorted(con.items(), key=lambda x: -x[1])[:3] if v > 0]
        data = "\n".join(f"- {k}: {r[k]}  ({M.FEATURES[k]})" for k in keys)
        out.append(dict(i=i, score=s, row=r, top3=top3, data=data, keys=keys))
    return out


def ask(pid, c):
    p = ROOT / "answers" / pid / f"{c['i']:02d}.json"
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    prompt = PROMPTS[pid].format(score=c["score"], data=c["data"], keys=c["keys"], reasons=c["top3"])
    r = subprocess.run(CMD, input=prompt, capture_output=True, text=True, timeout=180, cwd="/tmp")
    p.write_text(r.stdout.strip())


def parse(t):
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


if __name__ == "__main__":
    cs = cases()
    if "--ask" in sys.argv:
        with cf.ThreadPoolExecutor(5) as ex:
            list(ex.map(lambda a: ask(*a), [(p, c) for p in PROMPTS for c in cs]))
    print("=" * 90)
    print("Do LLM-written fraud explanations tell the truth about the model? (40 flagged applications)")
    print("=" * 90)
    rows, keys, w, score = M.build()
    for pid in PROMPTS:
        cat = Counter()
        against_cases = narr_bad_cases = miss_top1 = 0
        ex = None
        for c in cs:
            a = parse((ROOT / "answers" / pid / f"{c['i']:02d}.json").read_text())
            _, con = score(c["row"])
            cited = [k for k in a.get("cited_factors", []) if k in M.FEATURES]
            against = []
            for k in cited:
                if k in c["top3"]:
                    cat["model's top 3"] += 1
                elif con[k] > 0.05:
                    cat["real but minor (#4 or lower)"] += 1
                else:
                    cat["model counted it against fraud, or ignored it"] += 1
                    against.append((k, round(con[k], 2)))
            against_cases += bool(against)
            miss_top1 += c["top3"][0] not in cited
            narr = a.get("explanation", "").lower()
            narr_bad_cases += any(re.search(p, narr) and con[k] <= 0.05 for k, p in KEYWORDS.items())
            if against and ex is None:
                ex = (c["i"], c["top3"], against, a.get("explanation", "")[:260])
        total = sum(cat.values())
        print(f"\n{pid}: {total} cited reasons -> " + ", ".join(f"{v} {k}" for k, v in cat.items()))
        print(f"   explanations citing a reason the model held against fraud or ignored: {against_cases}/40 "
              f"(loose keyword scan of the narrative: {narr_bad_cases}/40); model's #1 reason missing: {miss_top1}/40")
        if ex:
            print(f"   e.g. case {ex[0]}: model's top 3 {ex[1]}; explanation also blamed {ex[2]} (negative = pointed to legit)")
            print(f"        \"{ex[3]}\"")
    rows, keys, w, score = M.build()
    print("\nmodel weights (standardized):", {k: round(v, 2) for k, v in sorted(w.items(), key=lambda x: -abs(x[1]))})
