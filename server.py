"""
Assignment 3: MCP server with two tools, `search` and `add_document`.

Your editor starts it (see .vscode/mcp.json and .mcp.json). It loads the saved
index from index/ and never re-embeds the corpus: `search` embeds only the
question, and `add_document` embeds only the new document's chunks.

Only MCP messages may go to stdout, so anything printed here goes to stderr.
"""

from pathlib import Path

try:
    from mcp.server.mcpserver import MCPServer as Server  # mcp 2.x
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server

import index

DESIGN = "sections"  # the chunking design the server searches (the one chosen in evaluate.py)
MAX_RESULTS = 10

server = Server("corpus")


def indexed_companies(collection) -> list[str]:
    metadatas = collection.get(include=["metadatas"])["metadatas"] or []
    return sorted({m["company"] for m in metadatas})


@server.tool()
def search(query: str, company: str | None = None, k: int = 5) -> str:
    """Search the indexed company annual reports (Business, Risk Factors and
    Management's Discussion and Analysis of each report) and return the most
    relevant passages with citations.

    Use it for any question about a company's business, financial performance,
    risks or outlook. Pass `company` (for example "Apple", "NVIDIA", "Tesla") to
    search only that company's report; leave it empty to search all of them.
    `k` is the number of passages to return (1 to 10, default 5).

    Each passage starts with a citation like
    [1] NVIDIA | nvidia-10k-2026 | Item 1A. Risk Factors | similarity 0.71
    Cite it as (source, location). If no passage answers the question, the
    answer is not in the indexed reports: say so, do not guess. If the company
    asked about is not indexed, the tool says which companies are."""
    query = (query or "").strip()
    if not query:
        return "Empty query. Ask a question about a company's annual report."
    k = max(1, min(int(k), MAX_RESULTS))
    collection = index.get_collection(DESIGN)
    if collection.count() == 0:
        return "The index is empty. Run `uv run python main.py index` first."

    where = None
    if company:
        companies = indexed_companies(collection)
        match = next((c for c in companies if c.lower() == company.strip().lower()), None)
        if match is None:
            return f"No company named {company!r} is indexed. Indexed companies: {', '.join(companies)}."
        where = {"company": match}

    vector = index.embed_queries([query])[0]
    result = collection.query(query_embeddings=[vector], n_results=k, where=where)
    hits = zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    blocks = []
    for n, (text, meta, distance) in enumerate(hits, start=1):
        header = f"[{n}] {meta['company']} | {meta['source']} | {meta['location']} | similarity {1 - distance:.2f}"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks) if blocks else "No passages found."


@server.tool()
def add_document(path: str, company: str | None = None) -> str:
    """Add one document to the index so that `search` can find it. Use it only
    when the user asks to add a company's annual report (a 10-K or 20-F saved as
    .htm or .html) or a plain text (.txt, .md) file.

    `path` is the path of the file on this computer. `company` is the company's
    name as it should appear in citations and in the `company` filter (for
    example "Microsoft"); if empty, the file name is used.

    The file goes through the same pipeline as the starter corpus: extract,
    clean, chunk, embed, store. If the same document is already indexed
    (identical text), nothing is embedded. If a document with the same file name
    is indexed but its text changed, its old passages are replaced by the new
    ones. Embedding costs quota (free tier: about 1,000 texts a day)."""
    file = Path(path).expanduser()
    if not file.is_file():
        return f"File not found: {path}"
    if file.suffix.lower() not in (".htm", ".html", ".txt", ".md"):
        return f"Unsupported file type {file.suffix!r}. Use .htm, .html, .txt or .md."

    chunks = index.chunk_file(file, DESIGN, company=company)
    if not chunks:
        return "No indexable text found in that file. Nothing was added."

    collection = index.get_collection(DESIGN)
    source = chunks[0]["source"]
    old_ids = set(collection.get(where={"source": source})["ids"])
    new_ids = {c["id"] for c in chunks}
    stale = old_ids - new_ids
    if stale:
        collection.delete(ids=list(stale))  # a changed document replaces its old version

    result = index.embed_and_store(chunks, DESIGN)

    if old_ids and not stale and result["embedded"] == 0:
        status = "already indexed, nothing changed and nothing embedded"
    elif old_ids:
        status = f"updated: {len(stale)} old passages replaced"
    else:
        status = "added"
    message = (
        f"{source} ({chunks[0]['company']}): {status}. {len(new_ids)} passages, "
        f"{result['embedded']} newly embedded."
    )
    if result["still_missing"]:
        message += f" Stopped early (quota or network): {result['still_missing']} passages are still missing, run it again later."
    return message


if __name__ == "__main__":
    server.run()