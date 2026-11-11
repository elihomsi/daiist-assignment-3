"""
Assignment 3: evaluation (`main.py evaluate`).

Stages (each one saves its result in eval/, so a rerun only does what is missing):
  1. Retrieval: hit rate at k=1, 3, 5 for every index design (a question is a
     hit when ALL its expected (source, item) pairs are in the top k passages).
  2. Agent: runs agent.py on every question (add_document is always denied here)
     and saves answers, tool trace, latency and tokens.
  3. Code checks: citations are well formed and come from retrieved passages,
     unanswerable questions are refused, answerable ones are not.
  4. LLM judge: grades correct and supported with the rubric in eval/rubric.md,
     using the retrieved passages as evidence.
  5. Agreement between the judge and your own grades in eval/human_labels.csv.
  6. Cost and latency per question.

Grade your 20+ answers BEFORE you look at eval/judge.jsonl or eval/results.csv.
Use eval/answers_for_grading.csv, which has no judge verdicts.
"""

import csv
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "eval"

KS = (1, 3, 5)
PAUSE_SECONDS = 4  # between agent calls, to stay under free-tier rate limits
# Price of the agent model in USD per 1 million tokens (look it up and fill it in; 0 = cost not computed).
PRICE_IN_PER_M = 0.0
PRICE_OUT_PER_M = 0.0

REFUSAL = re.compile(
    r"do(?:es)? not cover|not covered|cannot (?:find|answer)|can't (?:find|answer)|no information"
    r"|not (?:in|found in) the (?:indexed )?reports|unable to (?:find|answer)|not (?:available|mentioned) in the",
    re.I,
)
CITATION = re.compile(r"\b([a-z0-9]+(?:-[a-z0-9]+)+)\s*,\s*(Item\s+\d+[A-Z]?)", re.I)
HEADER = re.compile(r"^\[\d+\] .*? \| ([a-z0-9]+(?:-[a-z0-9]+)+) \| (Item [^|]+?) \| similarity", re.M)

JUDGE_PROMPT = """You grade an assistant's answer about company annual reports.

RUBRIC (follow it exactly):
{rubric}

QUESTION: {question}
KIND: {kind}
REFERENCE ANSWER (empty means the reports do not answer it, so the right answer is a refusal): {reference}

PASSAGES THE ASSISTANT RETRIEVED (the only evidence):
{evidence}

ASSISTANT ANSWER:
{answer}

Reply with JSON only: {{"correct": 0 or 1, "supported": 0 or 1, "reason": "one short sentence"}}"""


# ---------- small helpers ----------

def item_id(location: str) -> str:
    match = re.match(r"\s*Item\s+(\d+[A-Z]?)", location or "", re.I)
    return match.group(1).upper() if match else (location or "").strip().lower()


def load_questions() -> list[dict]:
    path = EVAL / "questions.csv"
    if not path.exists():
        raise SystemExit("eval/questions.csv not found. Write your questions first.")
    with open(path, newline="", encoding="utf-8") as f:
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]
    rows = [r for r in rows if r.get("id") and r.get("question")]
    if not rows:
        raise SystemExit("eval/questions.csv has no questions yet.")
    return rows


def is_unanswerable(row: dict) -> bool:
    return row["kind"].lower() == "unanswerable"


def expected_pairs(row: dict) -> set[tuple[str, str]]:
    """Several expected sources/locations can be separated by ';' (use it for multi questions)."""
    sources = [s.strip() for s in row["expected_source"].split(";") if s.strip()]
    locations = [loc.strip() for loc in row["expected_location"].split(";")]
    if len(locations) == 1:
        locations = locations * len(sources)
    return {(s, item_id(loc)) for s, loc in zip(sources, locations)}


def read_jsonl(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                out[record["id"]] = record
    return out


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def retrieved_text(record: dict) -> str:
    return "\n\n".join(t["result"] for t in record["tool_trace"] if t["tool"] == "search")


def retrieved_pairs(record: dict) -> set[tuple[str, str]]:
    return {(s, item_id(loc)) for s, loc in HEADER.findall(retrieved_text(record))}


# ---------- stage 1: retrieval ----------

def stage_retrieval(rows: list[dict]) -> None:
    import index

    answerable = [r for r in rows if not is_unanswerable(r)]
    if not answerable:
        print("Retrieval: no answerable questions yet.")
        return
    questions = [r["question"] for r in answerable]
    vectors = []
    for start in range(0, len(questions), 50):
        vectors += index.embed_queries(questions[start:start + 50])

    summary, per_question = [], []
    for design in index.INDEX_DESIGNS:
        collection = index.get_collection(design)
        if collection.count() == 0:
            print(f"Retrieval: design '{design}' has no chunks yet, skipped (run main.py index).")
            continue
        result = collection.query(query_embeddings=vectors, n_results=max(KS))
        hits = {k: [] for k in KS}
        for row, metas in zip(answerable, result["metadatas"]):
            got = [(m["source"], item_id(m["location"])) for m in metas]
            record = {"design": design, "id": row["id"], "kind": row["kind"]}
            for k in KS:
                hit = expected_pairs(row) <= set(got[:k])
                hits[k].append(hit)
                record[f"hit@{k}"] = int(hit)
            per_question.append(record)
        for k in KS:
            summary.append({"design": design, "k": k, "hits": sum(hits[k]), "n": len(hits[k]),
                            "hit_rate": round(sum(hits[k]) / len(hits[k]), 3),
                            "chunks_in_index": collection.count()})
    write_csv(EVAL / "retrieval_results.csv", summary)
    write_csv(EVAL / "retrieval_by_question.csv", per_question)
    print("\nRETRIEVAL HIT RATE")
    for s in summary:
        print(f"  {s['design']:<10} k={s['k']}: {s['hits']}/{s['n']} = {s['hit_rate']:.2f}")


# ---------- stage 2: agent ----------

def stage_agent(rows: list[dict]) -> dict:
    import agent

    path = EVAL / "agent_answers.jsonl"
    done = read_jsonl(path)
    todo = [r for r in rows if r["id"] not in done or done[r["id"]]["question"] != r["question"]]
    if todo:
        print(f"\nAGENT: {len(todo)} questions to answer ({len(rows) - len(todo)} already saved)")
    for number, row in enumerate(todo, start=1):
        error = ""
        for attempt in range(3):
            try:
                result = agent.ask(row["question"], approve=lambda name, args: False)
                break
            except Exception as exc:  # network, quota, model errors
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                time.sleep(20 * (attempt + 1))
        else:
            print(f"  {row['id']}: failed after 3 tries ({error}); run again to retry it")
            continue
        record = {"id": row["id"], "question": row["question"], **result}
        append_jsonl(path, record)
        done[row["id"]] = record
        print(f"  [{number}/{len(todo)}] {row['id']}: {result['seconds']}s")
        time.sleep(PAUSE_SECONDS)
    # drop stale records of questions that no longer exist or changed
    return {r["id"]: done[r["id"]] for r in rows if r["id"] in done and done[r["id"]]["question"] == r["question"]}


# ---------- stage 3: code checks ----------

def code_checks(row: dict, record: dict) -> dict:
    answer = record["answer"]
    retrieved = retrieved_pairs(record)
    cited = [(s, item_id(i)) for s, i in CITATION.findall(answer)]
    valid = [pair for pair in cited if pair in retrieved]
    refused = bool(REFUSAL.search(answer))
    added = any(t["tool"] == "add_document" and "added" in t["result"].lower() for t in record["tool_trace"])
    if is_unanswerable(row):
        passed = refused
    else:
        passed = (not refused) and len(cited) > 0 and len(valid) == len(cited)
    return {
        "citations": len(cited),
        "valid_citations": len(valid),
        "refused": int(refused),
        "searched": int(any(t["tool"] == "search" for t in record["tool_trace"])),
        "document_added": int(added),
        "code_check_passed": int(passed),
        "expected_in_retrieved": "" if is_unanswerable(row) else int(expected_pairs(row) <= retrieved),
    }


# ---------- stage 4: LLM judge ----------

def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0)) if match else {}


def stage_judge(rows: list[dict], answers: dict) -> dict:
    import agent

    rubric_path = EVAL / "rubric.md"
    if not rubric_path.exists():
        raise SystemExit("eval/rubric.md not found. Write your judge rubric first (what counts as correct and as supported).")
    rubric = rubric_path.read_text(encoding="utf-8")
    path = EVAL / "judge.jsonl"
    done = read_jsonl(path)
    model = None
    for row in rows:
        record = answers.get(row["id"])
        if not record:
            continue
        digest = hashlib.sha1((record["answer"] + rubric + row["reference_answer"]).encode()).hexdigest()[:12]
        if row["id"] in done and done[row["id"]].get("digest") == digest:
            continue
        model = model or agent.make_model()
        prompt = JUDGE_PROMPT.format(
            rubric=rubric, question=row["question"], kind=row["kind"], reference=row["reference_answer"] or "(none)",
            evidence=retrieved_text(record)[:12000] or "(nothing retrieved)", answer=record["answer"],
        )
        verdict = {}
        for attempt in range(3):
            try:
                verdict = parse_json(agent.content_text(model.invoke(prompt).content))
                if "correct" in verdict:
                    break
            except Exception:
                time.sleep(20 * (attempt + 1))
        if "correct" not in verdict:
            print(f"  judge failed for {row['id']}; run again to retry it")
            continue
        out = {"id": row["id"], "digest": digest, "correct": int(verdict["correct"]),
               "supported": int(verdict.get("supported", 0)), "reason": str(verdict.get("reason", ""))}
        append_jsonl(path, out)
        done[row["id"]] = out
        time.sleep(1)
    return done


# ---------- stage 5: agreement with human grades ----------

def to_bit(value: str):
    v = (value or "").strip().lower()
    if v in ("1", "yes", "y", "true", "correct"):
        return 1
    if v in ("0", "no", "n", "false", "incorrect"):
        return 0
    return None


def load_labels() -> dict:
    path = EVAL / "human_labels.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        labels = {}
        for row in csv.DictReader(f):
            bits = (to_bit(row.get("correct")), to_bit(row.get("supported")))
            if row.get("id", "").strip() and None not in bits:
                labels[row["id"].strip()] = bits
    return labels


def agreement(labels: dict, judge: dict) -> dict:
    ids = [i for i in labels if i in judge]
    out = {"graded_by_you": len(ids)}
    if not ids:
        return out
    for index_, name in ((0, "correct"), (1, "supported")):
        same = [i for i in ids if labels[i][index_] == judge[i][name]]
        out[f"agreement_{name}"] = round(len(same) / len(ids), 3)
        out[f"disagreements_{name}"] = [i for i in ids if i not in same]
    return out


# ---------- stage 6: cost and latency ----------

def cost_latency(answers: dict) -> dict:
    records = list(answers.values())
    seconds = sorted(r["seconds"] for r in records)
    tokens_in = sum(r["input_tokens"] for r in records)
    tokens_out = sum(r["output_tokens"] for r in records)
    cost = tokens_in / 1e6 * PRICE_IN_PER_M + tokens_out / 1e6 * PRICE_OUT_PER_M
    return {
        "questions": len(records),
        "latency_median_s": round(statistics.median(seconds), 2),
        "latency_p95_s": round(seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))], 2),
        "input_tokens_per_question": round(tokens_in / len(records)),
        "output_tokens_per_question": round(tokens_out / len(records)),
        "cost_per_question_usd": round(cost / len(records), 5) if (PRICE_IN_PER_M or PRICE_OUT_PER_M) else "set PRICE_* in evaluate.py",
    }


# ---------- main ----------

def main() -> None:
    EVAL.mkdir(exist_ok=True)
    rows = load_questions()
    kinds = [r["kind"].lower() for r in rows]
    print(f"{len(rows)} questions ({kinds.count('unanswerable')} unanswerable, {kinds.count('multi')} multi)")

    stage_retrieval(rows)
    answers = stage_agent(rows)
    if not answers:
        raise SystemExit("No agent answers yet.")

    checks = {r["id"]: code_checks(r, answers[r["id"]]) for r in rows if r["id"] in answers}
    write_csv(EVAL / "answers_for_grading.csv", [
        {"id": r["id"], "kind": r["kind"], "question": r["question"], "reference_answer": r["reference_answer"],
         "answer": answers[r["id"]]["answer"], "sources_retrieved": "; ".join(sorted(f"{s} {i}" for s, i in retrieved_pairs(answers[r["id"]])))}
        for r in rows if r["id"] in answers
    ])
    passed = sum(c["code_check_passed"] for c in checks.values())
    print(f"\nCODE CHECKS: {passed}/{len(checks)} passed")
    for r in rows:
        if r["id"] in checks and not checks[r["id"]]["code_check_passed"]:
            print(f"  failed: {r['id']} ({r['kind']})")

    judge = stage_judge(rows, answers)
    labels = load_labels()
    result = agreement(labels, judge)
    print("\nJUDGE vs YOUR GRADES:", {k: v for k, v in result.items()})
    if result["graded_by_you"] < 20:
        print("  You need 20+ own grades in eval/human_labels.csv (columns id,correct,supported,notes). "
              "Grade from eval/answers_for_grading.csv.")

    table = []
    for r in rows:
        rid = r["id"]
        if rid not in answers:
            continue
        j, h = judge.get(rid, {}), labels.get(rid, ("", ""))
        table.append({"id": rid, "kind": r["kind"], "question": r["question"], "answer": answers[rid]["answer"],
                      **checks[rid], "judge_correct": j.get("correct", ""), "judge_supported": j.get("supported", ""),
                      "judge_reason": j.get("reason", ""), "human_correct": h[0], "human_supported": h[1],
                      "seconds": answers[rid]["seconds"]})
    write_csv(EVAL / "results.csv", table)

    judged = [t for t in table if t["judge_correct"] != ""]
    if judged:
        print(f"\nJUDGE: correct {sum(t['judge_correct'] for t in judged)}/{len(judged)}, "
              f"supported {sum(t['judge_supported'] for t in judged)}/{len(judged)}")
    stats = cost_latency(answers)
    print("\nCOST AND LATENCY:", stats)
    (EVAL / "summary.json").write_text(json.dumps({"agreement": result, "cost_latency": stats}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()