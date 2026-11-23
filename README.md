# SQL Table Query

Natural-language querying over relational database tables. Rows are pulled from
MySQL, turned into searchable documents, and indexed for hybrid retrieval, so a
question in plain English returns the matching records with a written summary
rather than a result grid.

SQL is precise but demands that you already know the schema. The gap this closes
is for someone who knows what they want but not which columns hold it — the
query goes in as a sentence and the retrieval layer decides where to look.

---

## How it works

```
MySQL  ──►  fetch rows  ──►  compose a document per row
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
              sentence-transformer              BM25 (rank_bm25)
              embeddings → FAISS                sparse index
                      └───────────────┬───────────────┘
                                      ▼
                          hybrid retrieval → ranked rows
                                      ▼
                              written summary
```

**Both retrieval halves are needed.** Dense embeddings match intent, so a
question phrased differently from the stored text still lands. BM25 matches
exact tokens — codes, identifiers, proper nouns — which embeddings tend to
blur. A query naming a specific identifier needs the sparse side; a vague
descriptive query needs the dense side.

Each row is flattened into a text document combining its descriptive fields, so
one embedding represents the whole record rather than any single column. The
indexer joins a primary table to a secondary one and handles rows where the
join key is a delimited list rather than a single value.

Indexing reports its own statistics as it runs — rows fetched, duplicates
skipped, documents written — so a partial or skewed index is visible rather
than silent.

---

## Usage

```bash
pip install -r requirements.txt

# 1. Point at your database and tables
export DB_HOST=localhost DB_USER=... DB_PASSWORD=... DB_NAME=...
export ITEMS_TABLE=items CATALOGS_TABLE=catalogs KEY_COLUMN=record_id

# 2. Build the index
python train.py

# 3. Serve the query interface
python app.py
```

Connection details and table names come from the environment. The schema in
`SCHEMA` is **illustrative** — a primary table of records joined to a secondary
catalogue table by a shared key. Point the variables at your own tables and
adjust the selected columns in `train.py` to match.

Neither the index nor any database export is included in this repository.

## Stack

Python · MySQL · SQLAlchemy · PyMySQL · sentence-transformers · FAISS ·
rank_bm25 · NLTK · Flask · PyTorch
