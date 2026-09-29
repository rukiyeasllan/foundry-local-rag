import sqlite3
from pathlib import Path

import numpy as np


DB_PATH = Path("data/rag.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    return conn


def initialize_database():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                UNIQUE(source, chunk_index)
            )
            """
        )

        conn.commit()


def insert_document(source, chunk_index, content, embedding):
    vector = np.asarray(embedding, dtype=np.float32)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO documents
            (
                source,
                chunk_index,
                content,
                embedding
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                source,
                chunk_index,
                content,
                vector.tobytes(),
            ),
        )

        conn.commit()


def get_all_documents():
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                source,
                chunk_index,
                content,
                embedding
            FROM documents
            ORDER BY id
            """
        ).fetchall()

    documents = []

    for row in rows:
        documents.append(
            {
                "id": row["id"],
                "source": row["source"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "embedding": np.frombuffer(
                    row["embedding"],
                    dtype=np.float32
                ).copy(),
            }
        )

    return documents


def count_documents():
    with get_connection() as conn:
        result = conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()

    return result[0]


if __name__ == "__main__":
    initialize_database()

    print("SQLite veritabani basariyla olusturuldu.")
    print("Konum:", DB_PATH.resolve())
    print("Kayit sayisi:", count_documents())
