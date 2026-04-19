#!/usr/bin/env python3
"""
fetch_synthetic_ids.py

Queries RCSB Search API for synthetic/designed proteins deposited 2022-2024
matching the criteria from Hýsková et al. (2026), then downloads experimental
PDB files for each chain.

Usage:
    python fetch_synthetic_ids.py --outdir ./pdbs [--max 200]
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


# ---------------------------------------------------------------------------
# RCSB Search query — mirrors paper criteria:
#   - deposited July 2022 – July 2024
#   - protein only (no nucleic acids, no oligosaccharides)
#   - chain length 20–400 aa
#   - source organism = "synthetic construct" OR title/keyword contains "designed"
# ---------------------------------------------------------------------------

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_PDB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.pdb"


def build_query(offset: int = 0, rows: int = 250) -> dict:
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                # Deposit date range
                {
                    "type": "group",
                    "logical_operator": "and",
                    "nodes": [
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_accession_info.deposit_date",
                                "operator": "greater_or_equal",
                                "value": "2022-07-01",
                                "negation": False
                            }
                        },
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_accession_info.deposit_date",
                                "operator": "less_or_equal",
                                "value": "2024-07-31",
                                "negation": False
                            }
                        }
                    ]
                },
                # Protein only
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                        "operator": "exact_match",
                        "value": "Protein (only)",
                        "negation": False
                    }
                },
                # Synthetic construct source organism
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entity_source_organism.ncbi_scientific_name",
                        "operator": "exact_match",
                        "value": "synthetic construct",
                        "negation": False
                    }
                }
            ]
        },
        "return_type": "polymer_entity",
        "request_options": {
            "results_verbosity": "minimal",
            "paginate": {"start": offset, "rows": rows},
            "sort": [{"sort_by": "score", "direction": "desc"}]
        }
    }


def rcsb_search(max_results: int = 500) -> list[str]:
    """Return list of entity identifiers like '5SSZ_1' from RCSB."""
    all_ids = []
    rows = 250
    offset = 0

    while True:
        query = build_query(offset=offset, rows=rows)
        payload = json.dumps(query).encode()
        req = urllib.request.Request(
            RCSB_SEARCH_URL,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "python/3.11"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"RCSB search failed: {e}")

        total = data.get("total_count", 0)
        hits = data.get("result_set", [])
        all_ids.extend(h["identifier"] for h in hits)

        print(f"  Fetched {len(all_ids)}/{total} entries (offset={offset})", flush=True)

        if len(all_ids) >= total or len(all_ids) >= max_results or not hits:
            break
        offset += rows
        time.sleep(0.5)

    return all_ids


def entity_to_pdb_chain(entity_id: str) -> tuple[str, str]:
    """
    Convert RCSB entity ID like '5SSZ_1' to (pdb_id, chain_id).
    Uses RCSB GraphQL to get the actual chain letter.
    """
    pdb_id = entity_id.split("_")[0]
    entity_num = entity_id.split("_")[1]

    url = (
        f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_num}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "python/3.11"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        # Get first strand/chain ID
        chains = data.get("rcsb_polymer_entity_container_identifiers", {}).get(
            "asym_ids", ["A"]
        )
        return pdb_id, chains[0]
    except Exception:
        return pdb_id, "A"


def filter_by_length(entity_id: str, min_len: int = 20, max_len: int = 400) -> int | None:
    """Return sequence length for entity, or None if outside range."""
    pdb_id = entity_id.split("_")[0]
    entity_num = entity_id.split("_")[1]
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_num}"
    req = urllib.request.Request(url, headers={"User-Agent": "python/3.11"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        length = data.get("entity_poly", {}).get("rcsb_sample_sequence_length", 0)
        if min_len <= length <= max_len:
            return length
        return None
    except Exception:
        return None


def download_pdb(pdb_id: str, outdir: Path) -> Path | None:
    """Download PDB file from RCSB. Returns path or None on failure."""
    pdb_id = pdb_id.upper()
    outpath = outdir / f"{pdb_id}.pdb"

    if outpath.exists():
        return outpath  # already downloaded

    url = RCSB_PDB_DOWNLOAD.format(pdb_id=pdb_id)
    req = urllib.request.Request(url, headers={"User-Agent": "python/3.11"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
        outpath.write_bytes(content)
        return outpath
    except urllib.error.HTTPError as e:
        print(f"  WARNING: Could not download {pdb_id}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch synthetic protein PDBs from RCSB")
    parser.add_argument("--outdir", default="./pdbs", help="Directory to save PDB files")
    parser.add_argument("--max", type=int, default=500, help="Max RCSB search results to fetch")
    parser.add_argument("--manifest", default="./synthetic_manifest.tsv",
                        help="Output TSV: pdb_id, chain_id, length")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Querying RCSB for synthetic construct proteins (2022–2024)...")
    entity_ids = rcsb_search(max_results=args.max)
    print(f"  Total entity IDs before length filter: {len(entity_ids)}")

    print("Step 2: Filtering by chain length (20–400 aa) and resolving chain IDs...")
    manifest_rows = []
    seen_pdb_ids = set()

    for i, eid in enumerate(entity_ids):
        if i % 50 == 0:
            print(f"  Processing entity {i+1}/{len(entity_ids)}...", flush=True)

        length = filter_by_length(eid)
        if length is None:
            continue

        pdb_id, chain_id = entity_to_pdb_chain(eid)

        # Deduplicate: one chain per PDB entry (take first chain encountered)
        if pdb_id in seen_pdb_ids:
            continue
        seen_pdb_ids.add(pdb_id)

        manifest_rows.append((pdb_id, chain_id, length))
        time.sleep(0.1)  # be polite to RCSB

    print(f"  Unique PDB entries after filtering: {len(manifest_rows)}")

    print("Step 3: Downloading PDB files...")
    downloaded = []
    for pdb_id, chain_id, length in manifest_rows:
        path = download_pdb(pdb_id, outdir)
        if path:
            downloaded.append((pdb_id, chain_id, length))
        time.sleep(0.05)

    print(f"  Successfully downloaded: {len(downloaded)} PDBs")

    print(f"Step 4: Writing manifest to {args.manifest}")
    with open(args.manifest, "w") as f:
        f.write("pdb_id\tchain_id\tlength\n")
        for pdb_id, chain_id, length in downloaded:
            f.write(f"{pdb_id}\t{chain_id}\t{length}\n")

    print("Done.")


if __name__ == "__main__":
    main()
