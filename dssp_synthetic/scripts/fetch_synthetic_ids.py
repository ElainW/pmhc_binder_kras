#!/usr/bin/env python3
"""
fetch_synthetic_ids.py

Reads the 102 exact PDB IDs from synthetic_ids.txt (obtained from the
HuggingFace dataset hyskova-anna/proteins, label='synthetic'), resolves
the chain ID and sequence length for each via the RCSB REST API, downloads
experimental PDB files, and writes a manifest TSV.

Usage:
    python fetch_synthetic_ids.py \
        --ids synthetic_ids.txt \
        --outdir ./pdbs \
        --manifest ./synthetic_manifest.tsv
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def read_ids(ids_file: str) -> list[str]:
    """Read PDB IDs from a plain text file, one per line."""
    ids = []
    with open(ids_file) as f:
        for line in f:
            pdb_id = line.strip().upper()
            if pdb_id:
                ids.append(pdb_id)
    return ids


def resolve_chain_and_length(pdb_id: str) -> tuple[str, int]:
    """
    Query RCSB REST API for the first polymer entity of pdb_id.
    Returns (chain_id, sequence_length).
    Falls back to ("A", 0) on any error.
    """
    # Entity 1 is almost always the target chain for single-chain designs;
    # for multi-chain entries we take the first entity's first asym_id.
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
    req = urllib.request.Request(url, headers={"User-Agent": "python/3.11"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        chain_id = data.get(
            "rcsb_polymer_entity_container_identifiers", {}
        ).get("asym_ids", ["A"])[0]
        length = data.get("entity_poly", {}).get("rcsb_sample_sequence_length", 0)
        return chain_id, length
    except Exception as e:
        print(f"  WARNING: Could not resolve chain/length for {pdb_id}: {e}",
              file=sys.stderr)
        return "A", 0


def mmcif_to_pdb(cif_path: Path, pdb_path: Path) -> bool:
    """
    Convert mmCIF to PDB using BioPython MMCIFParser + PDBIO.
    Returns True on success. Newer RCSB entries are often mmCIF-only.
    """
    try:
        from Bio.PDB import MMCIFParser, PDBIO
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_path.stem, str(cif_path))
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_path))
        return pdb_path.exists() and pdb_path.stat().st_size > 0
    except Exception as e:
        print(f"  WARNING: mmCIF->PDB conversion failed for {cif_path.name}: {e}",
              file=sys.stderr)
        return False


def download_pdb(pdb_id: str, outdir: Path) -> Path | None:
    """
    Download experimental structure from RCSB.
    Strategy:
      1. Try .pdb format directly (most entries).
      2. On 404, fall back to .cif (mmCIF) and convert to PDB via BioPython.
    Returns path to .pdb file or None on complete failure.
    """
    pdb_id = pdb_id.upper()
    outpath = outdir / f"{pdb_id}.pdb"

    if outpath.exists() and outpath.stat().st_size > 0:
        return outpath  # already downloaded

    # Attempt 1: PDB format
    url_pdb = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    req = urllib.request.Request(url_pdb, headers={"User-Agent": "python/3.11"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
        if len(content) > 100:
            outpath.write_bytes(content)
            return outpath
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  WARNING: {pdb_id} PDB download error {e.code}", file=sys.stderr)
            return None
        # 404 -> fall through to mmCIF

    # Attempt 2: mmCIF format -> convert to PDB
    url_cif = f"https://files.rcsb.org/download/{pdb_id}.cif"
    cif_path = outdir / f"{pdb_id}.cif"
    req2 = urllib.request.Request(url_cif, headers={"User-Agent": "python/3.11"})
    try:
        with urllib.request.urlopen(req2, timeout=30) as resp:
            content = resp.read()
        cif_path.write_bytes(content)
    except urllib.error.HTTPError as e:
        print(f"  WARNING: Could not download {pdb_id} as PDB or mmCIF: {e}",
              file=sys.stderr)
        return None

    success = mmcif_to_pdb(cif_path, outpath)
    cif_path.unlink(missing_ok=True)
    if success:
        print(f"  INFO: {pdb_id} downloaded as mmCIF and converted to PDB", flush=True)
        return outpath
    else:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download experimental PDBs for the 102 synthetic proteins"
    )
    parser.add_argument(
        "--ids", default="./synthetic_ids.txt",
        help="Plain text file with one PDB ID per line (from HuggingFace dataset)"
    )
    parser.add_argument("--outdir", default="./pdbs", help="Directory to save PDB files")
    parser.add_argument(
        "--manifest", default="./synthetic_manifest.tsv",
        help="Output TSV: pdb_id, chain_id, length"
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: read IDs
    print(f"Step 1: Reading PDB IDs from {args.ids}...")
    pdb_ids = read_ids(args.ids)
    print(f"  {len(pdb_ids)} IDs loaded")

    # Step 2: resolve chain ID and length from RCSB
    print("Step 2: Resolving chain IDs and sequence lengths from RCSB REST API...")
    manifest_rows = []
    for i, pdb_id in enumerate(pdb_ids):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(pdb_ids)}]", flush=True)
        chain_id, length = resolve_chain_and_length(pdb_id)
        manifest_rows.append((pdb_id, chain_id, length))
        time.sleep(0.1)  # be polite to RCSB

    # Step 3: download PDB files
    print("Step 3: Downloading PDB files from RCSB...")
    downloaded = []
    for pdb_id, chain_id, length in manifest_rows:
        path = download_pdb(pdb_id, outdir)
        if path:
            downloaded.append((pdb_id, chain_id, length))
        else:
            print(f"  SKIP: {pdb_id} could not be downloaded", file=sys.stderr)
        time.sleep(0.05)

    print(f"  Successfully downloaded: {len(downloaded)}/{len(pdb_ids)} PDBs")

    # Step 4: write manifest
    print(f"Step 4: Writing manifest to {args.manifest}...")
    with open(args.manifest, "w") as f:
        f.write("pdb_id\tchain_id\tlength\n")
        for pdb_id, chain_id, length in downloaded:
            f.write(f"{pdb_id}\t{chain_id}\t{length}\n")

    print("Done.")


if __name__ == "__main__":
    main()