"""
Assignment 3: build the index (`main.py index`).

Download the corpus (annual reports of several companies), extract and clean
the text, chunk it with two designs, embed the chunks with Gemini and store
them in index/ (ChromaDB). Vectors are saved batch by batch, and only chunks
without a saved vector are embedded.

server.py imports chunk_file() and embed_and_store() from here, so nothing
runs on import.

    uv run python index.py            # build the index
    uv run python index.py --stats    # words and chunks per document, for REPORT.md
"""

import hashlib
import json
import re
import sys
import time
import warnings
from pathlib import Path

import chromadb
import requests
from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning
from google import genai
from google.genai import types

from api_key import load_api_key

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
INDEX_DIR = ROOT / "index"
DOCUMENTS_FILE = ROOT / "documents.json"  # committed: pins the exact filings

# The SEC asks for a real contact in the User-Agent. Use your own.
USER_AGENT = "ehomsi.ieu2022@student.ie.edu"

EMBEDDING_MODEL = "gemini-embedding-2"
INDEX_DESIGNS = ["sections"]  # add "fixed" on another day (free tier: about 1,000 embeddings a day)
BATCH = 50  # texts per embedding call
DAILY_CAP = 900  # at most this many new embeddings per run

# Starter corpus: short name -> (company, SEC CIK). Any other company is added
# later with the add-document tool.
COMPANIES = {
    "apple": ("Apple", "0000320193"),
    "nvidia": ("NVIDIA", "0001045810"),
    "tesla": ("Tesla", "0001318605"),
}

# The parts of each annual report that we index, by form. A 10-K is a US
# annual report, a 20-F is the same for foreign companies (such as Nubank).
ITEM_TITLES = {
    "10-K": {"1": "Business", "1A": "Risk Factors", "7": "Management's Discussion and Analysis"},
    "20-F": {"3": "Key Information", "5": "Operating and Financial Review and Prospects"},
}
KEEP_ITEMS = {form: set(titles) for form, titles in ITEM_TITLES.items()}


# ---------- the corpus ----------

def known_documents() -> dict:
    """Pinned documents (source name -> company, form, date, url). No network."""
    return json.loads(DOCUMENTS_FILE.read_text()) if DOCUMENTS_FILE.exists() else {}


def find_annual_report(short: str) -> dict:
    company, cik = COMPANIES[short]
    recent = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    ).json()["filings"]["recent"]
    rows = zip(recent["form"], recent["filingDate"], recent["accessionNumber"], recent["primaryDocument"])
    for form, date, accession, document in rows:
        if form in ("10-K", "20-F"):
            return {
                "company": company,
                "form": form,
                "date": date,
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{document}",
            }
    raise RuntimeError(f"No 10-K or 20-F found for {company}")


def load_documents() -> dict:
    """The first time, ask the SEC for each company's latest annual report and
    save documents.json. Commit that file, so everybody indexes the same filings."""
    if DOCUMENTS_FILE.exists():
        return known_documents()
    documents = {}
    for short in COMPANIES:
        found = find_annual_report(short)
        documents[f"{short}-{found['form'].replace('-', '').lower()}-{found['date'][:4]}"] = found
        time.sleep(0.3)
    DOCUMENTS_FILE.write_text(json.dumps(documents, indent=2))
    return documents


def download_corpus() -> list[Path]:
    """Download each document once into corpus/ (git-ignored)."""
    CORPUS_DIR.mkdir(exist_ok=True)
    paths = []
    for name, doc in load_documents().items():
        path = CORPUS_DIR / f"{name}.htm"
        if not path.exists():
            response = requests.get(doc["url"], headers={"User-Agent": USER_AGENT}, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
            print(f"downloaded {name}", file=sys.stderr)
            time.sleep(0.3)  # the SEC allows at most 10 requests a second
        paths.append(path)
    return paths


# ---------- extract and clean ----------

BLOCKS = ["p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"]


def html_to_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all("ix:header"):  # hidden XBRL data, not content
        tag.decompose()
    # Keep each table row on one line, so a number stays next to its label.
    for row in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        row.replace_with(NavigableString(" | ".join(c for c in cells if c) + "\n"))
    for tag in soup.find_all(BLOCKS):
        tag.insert_after(NavigableString("\n"))
    return clean_lines(soup.get_text(" ").split("\n"))


def clean_lines(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        line = " ".join(line.split())  # collapse spaces, tabs, non-breaking spaces
        if not line or line.isdigit() or line.lower() == "table of contents":
            continue  # blanks, page numbers, repeated page header
        out.append(line)
    return out


ITEM = re.compile(r"^item\s+(\d+[a-j]?)\b", re.I)
SUB = re.compile(r"^([A-J])\.\s*\|?\s*([A-Z][^|]{2,70})$")  # 20-F: "D. Risk Factors" or "D. | Risk Factors"
RISKS = re.compile(r"^risks?\s+(relating|related)\s+to\b", re.I)
PART = re.compile(r"^part\s+[ivx]+$", re.I)
NUMBERED = re.compile(r"^(\d{1,2})\.\s+([A-Z][A-Z ,&()/\-]{3,})$")  # "6. INCOME AND RELATED EXPENSES"


def detect_form(lines: list[str]) -> str | None:
    head = " ".join(lines[:150]).upper()
    if "FORM 20-F" in head:
        return "20-F"
    if "FORM 10-K" in head:
        return "10-K"
    return None


def is_item_line(line: str) -> bool:
    return bool(ITEM.match(line)) and len(line.split()) <= 20


def is_caps_heading(line: str) -> bool:
    words = line.split()
    return (
        "|" not in line
        and 2 <= len(words) <= 10
        and line.upper() == line
        and re.search(r"[A-Z]{4}", line) is not None
        and not line[0].isdigit()
    )


def extract_sections(path, min_words: int = 50) -> list[dict]:
    """Split a document into sections: {location, text}.

    A 10-K or 20-F is split on its 'Item N' headings and only the items in
    KEEP_ITEMS are kept. Any other document is split on short all-caps lines
    and numbered headings. Tiny sections (table of contents rows) are dropped."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in (".htm", ".html"):
        lines = html_to_lines(raw)
    elif path.suffix.lower() in (".txt", ".md"):
        lines = clean_lines(raw.split("\n"))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    form = detect_form(lines)
    structured = form is not None and sum(is_item_line(line) for line in lines) >= 5
    keep = KEEP_ITEMS.get(form, set())
    sections, item, buffer = [], None, []
    heading = "Document start"

    def add():
        text = " ".join(buffer)
        if len(text.split()) < min_words:
            return
        if structured and item not in keep:
            return
        sections.append({"location": heading[:90], "text": text})

    for line in lines:
        if PART.match(line):
            continue  # "PART I": not a useful section
        if structured:
            if is_item_line(line):
                add()
                item = ITEM.match(line).group(1).upper()
                title = ITEM_TITLES[form].get(item, "")
                heading, buffer = f"Item {item}. {title}".strip(". "), []
                continue
            sub = SUB.match(line)
            if form == "20-F" and sub and item and len(line.split()) <= 8:
                add()
                heading, buffer = f"Item {item}.{sub.group(1)}. {sub.group(2)}", []
                continue
            if item and RISKS.match(line) and len(line.split()) <= 15:
                add()
                heading, buffer = f"Item {item}. {line}", []
                continue
            buffer.append(line)
        else:
            num = NUMBERED.match(line)
            if num:
                add()
                heading, buffer = f"{num.group(1)}. {num.group(2).strip(' ,').capitalize()}", []
            elif is_caps_heading(line):
                add()
                heading, buffer = line, []
            else:
                buffer.append(line)
    add()
    return sections


# ---------- chunk ----------

def make_chunk(source: str, company: str, location: str, text: str, design: str) -> dict:
    """The id is a hash of the content, so re-running never duplicates a chunk."""
    digest = hashlib.sha1(f"{design}|{source}|{location}|{text}".encode()).hexdigest()[:16]
    return {
        "id": digest,
        "source": source,
        "company": company,
        "location": location,
        "design": design,
        "text": text,
    }


def chunk_fixed(source: str, company: str, sections: list[dict], size: int = 150, overlap: int = 30) -> list[dict]:
    """Design A: windows of `size` words over the whole document. Location is the
    section the window starts in."""
    words, locations = [], []
    for s in sections:
        w = s["text"].split()
        words += w
        locations += [s["location"]] * len(w)
    if not words:
        return []
    chunks = []
    for start in range(0, max(len(words) - overlap, 1), size - overlap):
        end = min(start + size, len(words))
        chunks.append(make_chunk(source, company, locations[start], " ".join(words[start:end]), "fixed"))
    return chunks


def chunk_by_section(
    source: str, company: str, sections: list[dict], max_words: int = 300, overlap: int = 30
) -> list[dict]:
    """Design B: never cross a section boundary. Long sections are cut into pieces."""
    chunks = []
    for s in sections:
        words = s["text"].split()
        for start in range(0, max(len(words) - overlap, 1), max_words - overlap):
            piece = " ".join(words[start : start + max_words])
            chunks.append(make_chunk(source, company, s["location"], piece, "sections"))
    return chunks


def chunk_file(path, design: str = "sections", company: str | None = None) -> list[dict]:
    """Extract, clean and chunk one file. server.py's add-document tool uses this.
    The source is the file name without its extension. The company comes from
    documents.json, from `company`, or else from the file name."""
    path = Path(path)
    source = path.stem
    company = company or known_documents().get(source, {}).get("company", source)
    sections = extract_sections(path)
    if design == "fixed":
        return chunk_fixed(source, company, sections)
    if design == "sections":
        return chunk_by_section(source, company, sections)
    raise ValueError(f"Unknown design: {design}")


# ---------- embed and store ----------

_client = None


def get_client() -> genai.Client:
    """Created on first use. On a rate limit (429) or server error it waits a
    minute and retries, up to 5 attempts."""
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=load_api_key(),
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=5,
                    initial_delay=60,
                    max_delay=60,
                    exp_base=1,
                    jitter=0,
                    http_status_codes=[429, 500, 502, 503, 504],
                )
            ),
        )
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed already-prefixed texts in one call. One Content per text."""
    result = get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[types.Content(parts=[types.Part(text=t)]) for t in texts],
    )
    return [e.values for e in result.embeddings]


def embed_documents(chunks: list[dict]) -> list[list[float]]:
    return embed_texts([f"title: {c['company']}, {c['location']} | text: {c['text']}" for c in chunks])


def embed_queries(questions: list[str]) -> list[list[float]]:
    return embed_texts([f"task: search result | query: {q}" for q in questions])


def get_collection(design: str):
    """The ChromaDB collection for one chunking design, saved in index/."""
    INDEX_DIR.mkdir(exist_ok=True)
    db = chromadb.PersistentClient(path=str(INDEX_DIR))
    return db.get_or_create_collection(
        name=f"chunks_{design}",
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,  # we always pass our own vectors
    )


def embed_and_store(chunks: list[dict], design: str, cap: int = DAILY_CAP) -> dict:
    """Embed the chunks that have no saved vector yet and store them, one batch
    at a time. A chunk already in the index is never embedded again."""
    unique = list({c["id"]: c for c in chunks}.values())
    collection = get_collection(design)
    known = set(collection.get(ids=[c["id"] for c in unique])["ids"]) if unique else set()
    pending = [c for c in unique if c["id"] not in known]
    todo = pending[:cap]
    done = 0
    try:
        for start in range(0, len(todo), BATCH):
            batch = todo[start : start + BATCH]
            collection.add(
                ids=[c["id"] for c in batch],
                embeddings=embed_documents(batch),
                documents=[c["text"] for c in batch],
                metadatas=[
                    {
                        "source": c["source"],
                        "company": c["company"],
                        "location": c["location"],
                        "design": c["design"],
                    }
                    for c in batch
                ],
            )
            done += len(batch)  # saved before the next call, so a crash loses nothing
            print(f"  {design}: embedded {done}/{len(todo)}", file=sys.stderr)
    except Exception as error:  # daily limit, network: stop, a rerun resumes
        print(f"Stopped after {done} chunks: {error}", file=sys.stderr)
    return {"already_indexed": len(known), "embedded": done, "still_missing": len(pending) - done}


# ---------- commands ----------

def build_index() -> None:
    paths = download_corpus()
    for design in INDEX_DESIGNS:
        chunks = [c for p in paths for c in chunk_file(p, design)]
        result = embed_and_store(chunks, design)
        print(f"{design}: {len(chunks)} chunks, {result}")


def report() -> None:
    """Numbers for the corpus table in REPORT.md."""
    paths = download_corpus()
    documents = known_documents()
    total_words = 0
    for path in paths:
        doc = documents.get(path.stem, {})
        sections = extract_sections(path)
        words = sum(len(s["text"].split()) for s in sections)
        total_words += words
        print(f"{path.stem}: {doc.get('company')}, {doc.get('form')}, filed {doc.get('date')}, "
              f"{words:,} words kept, {len(sections)} sections")
        print(f"      {doc.get('url')}")
        for s in sections:
            print(f"      {len(s['text'].split()):>7,} words  {s['location']}")
    print(f"TOTAL KEPT: {total_words:,} words" + ("   WARNING: below 50,000" if total_words < 50_000 else ""))
    for design in ("fixed", "sections"):
        chunks = [c for p in paths for c in chunk_file(p, design)]
        print(f"design {design}: {len(chunks)} chunks")


def main() -> None:
    if "--stats" in sys.argv:
        report()
    else:
        build_index()


if __name__ == "__main__":
    main()