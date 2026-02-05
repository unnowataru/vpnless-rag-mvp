#!/usr/bin/env python3
# Minimal Lexical RAG (no extra deps): chunks.jsonl -> char-3gram overlap -> TopK -> bedrock converse (via aws cli)
import argparse
import json
import re
import subprocess
from collections import Counter

MODEL_ID = "google.gemma-3-4b-it"
REGION = "ap-northeast-1"
PROFILE = "rag"

JP_RUN = re.compile(r"[0-9A-Za-z一-龠々〆ヵヶぁ-ゔァ-ヴー]+", re.UNICODE)

FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
MONTH_RE = re.compile(r"([0-9０-９]+)\s*(?:ヶ|か)?月")

def extract_months(text: str):
    t = text.translate(FULLWIDTH_DIGITS)
    out = []
    for m in MONTH_RE.finditer(t):
        try:
            out.append(int(m.group(1)))
        except Exception:
            pass
    return out

def sanitize(s: str) -> str:
    s = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", s)
    s = re.sub(r"\b\d{2,4}-\d{2,4}-\d{3,4}\b", "[TEL]", s)
    return s

def normalize(s: str) -> str:
    return "".join(JP_RUN.findall(s.lower()))

def ngrams(s: str, n: int = 3):
    if not s:
        return []
    if len(s) <= n:
        return [s]
    return [s[i:i+n] for i in range(len(s) - n + 1)]

def score_overlap(qc: Counter, cc: Counter) -> int:
    return sum(min(v, cc.get(k, 0)) for k, v in qc.items())

def call_bedrock(question: str, evidence: str, max_tokens: int) -> str:
    payload = {
        "messages": [
            {"role": "user", "content": [{"text":
                "You must answer ONLY using the Evidence blocks.\n"
                "Output in Japanese.\n"
                "Rules:\n"
                "1) Cite evidence by block number like [1], [2]. Every factual claim MUST have a citation.\n"
                "2) If the question is ambiguous in scope (e.g., depends on document/employee type), DO NOT say 'Evidence is insufficient.'\n"
                "   Instead: ask ONE clarification question, and show the candidate answers with citations in one line.\n"
                "3) Say exactly 'Evidence is insufficient.' ONLY when the Evidence does not contain an explicit answer.\n"
                "4) Do not quote the evidence verbatim. Summarize.\n\n"
                f"Question:\n{question}\n\nEvidence:\n{evidence}"

            }]}
        ],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0},
    }
    tmp = "/tmp/converse_rag.json"
    with open(tmp, "w", encoding="utf-8") as w:
        json.dump(payload, w, ensure_ascii=False)

    cmd = [
        "aws", "bedrock-runtime", "converse",
        "--region", REGION,
        "--profile", PROFILE,
        "--model-id", MODEL_ID,
        "--cli-input-json", f"file://{tmp}",
        "--no-cli-pager",
        "--query", "output.message.content[0].text",
        "--output", "text",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return r.stdout.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="path to chunks.jsonl")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-context-chars", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("question", nargs="*")
    args = ap.parse_args()

    q = " ".join(args.question).strip() or input("Q> ").strip()
    qc = Counter(ngrams(normalize(q), 3))

    scored = []
    chunks = []
    with open(args.chunks, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            o = json.loads(line)
            txt = o.get("text", "")
            cc = Counter(ngrams(normalize(txt), 3))
            s = score_overlap(qc, cc)
            chunks.append(o)
            if s > 0:
                scored.append((s, i))

    scored.sort(reverse=True)
    top = scored[: args.topk]

    ev_parts = []
    for rank, (s, idx) in enumerate(top, start=1):
        o = chunks[idx]
        snippet = sanitize(o.get("text", "").replace("\n", " "))[:1200]
        ev_parts.append(
            f"[{rank}] score={s} doc={o.get('doc')} page={o.get('page')} chunk={o.get('chunk')}\n{snippet}\n"
        )

    evidence = ("\n".join(ev_parts))[: args.max_context_chars]

    print("=== TOPK EVIDENCE ===")
    print(evidence if evidence else "(no hits)")
    print("=== BEDROCK ANSWER ===")
    print(call_bedrock(q, evidence, args.max_tokens) if evidence else "Evidence is insufficient.")

if __name__ == "__main__":
    main()
