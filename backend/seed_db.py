"""
One-time database seeding script.
Run once before starting the server:
    python seed_db.py
"""

import sys
import os


sys.path.insert(0, os.path.dirname(__file__))

from database import init_db, upsert_scheme
from vector_store import add_scheme_to_vector_store
from schemes_data import SCHEMES


def seed():
    print("=" * 50)
    print("  Government Scheme DB Seeder")
    print("=" * 50)

    
    print("\n[1/3] Initialising SQLite database...")
    init_db()

    
    print(f"\n[2/3] Inserting {len(SCHEMES)} schemes into SQLite...")
    for i, scheme in enumerate(SCHEMES, 1):
        upsert_scheme(scheme)
        print(f"  ✅ [{i:02d}/{len(SCHEMES)}] {scheme['name_en']}")

    
    print(f"\n[3/3] Adding schemes to ChromaDB vector store...")
    print("  (First run downloads ~80MB model — please wait)\n")
    success_count = 0
    for i, scheme in enumerate(SCHEMES, 1):
        ok = add_scheme_to_vector_store(scheme)
        status = "✅" if ok else "⚠️ "
        print(f"  {status} [{i:02d}/{len(SCHEMES)}] {scheme['id']}")
        if ok:
            success_count += 1

    print("\n" + "=" * 50)
    print(f"✅ Seeded {len(SCHEMES)} schemes into SQLite")
    print(f"{'✅' if success_count > 0 else '⚠️ '} {success_count}/{len(SCHEMES)} schemes indexed in ChromaDB")
    if success_count == 0:
        print("   (ChromaDB indexing failed — eligibility checking will still work via SQL)")
    print("=" * 50)
    print("\nYou can now start the server:")
    print("  uvicorn main:app --reload --port 8000\n")


if __name__ == "__main__":
    seed()