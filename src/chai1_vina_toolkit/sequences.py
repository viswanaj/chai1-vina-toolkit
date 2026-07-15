"""
sequences.py — generic protein sequence resolution shared by every module in
this toolkit.

No project owns a sequence database here: callers resolve a target sequence
in one of three ways, tried in this order:

  1. Pass the sequence directly (--sequence).
  2. Point at a local FASTA file (--fasta) containing exactly one record.
  3. Pass a UniProt accession (--uniprot) and let this module fetch it from
     the public UniProt REST API.

This keeps the toolkit fully decoupled from any particular project's data —
it never assumes a local database, only a sequence string it either already
has or can fetch from a public resource.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{}.json"


def read_fasta_single(path: Path) -> str:
    """Return the sequence from a FASTA file containing exactly one record."""
    text = Path(path).read_text()
    records = [r for r in text.split(">") if r.strip()]
    if len(records) != 1:
        sys.exit(
            f"ERROR: expected exactly one FASTA record in {path}, found {len(records)}"
        )
    lines = records[0].splitlines()
    return "".join(l.strip() for l in lines[1:])


def fetch_uniprot_sequence(accession: str) -> str:
    resp = requests.get(UNIPROT_API.format(accession), timeout=30)
    if resp.status_code != 200:
        sys.exit(f"ERROR: UniProt API returned {resp.status_code} for {accession}")
    data = resp.json()
    seq = data.get("sequence", {}).get("value", "")
    if not seq:
        sys.exit(f"ERROR: no sequence found for {accession} in UniProt")
    return seq


def fetch_uniprot_gene_name(accession: str) -> str | None:
    """Best-effort gene name lookup, used only as a display label."""
    try:
        resp = requests.get(UNIPROT_API.format(accession), timeout=30)
        if resp.status_code != 200:
            return None
        genes = resp.json().get("genes", [])
        if genes and "geneName" in genes[0]:
            return genes[0]["geneName"].get("value")
    except Exception:
        pass
    return None


def resolve_sequence(
    sequence: str | None = None,
    fasta: str | None = None,
    uniprot: str | None = None,
) -> str:
    """Resolve a target sequence from whichever of the three inputs is given,
    in priority order: explicit sequence > FASTA file > UniProt accession."""
    if sequence:
        return sequence
    if fasta:
        return read_fasta_single(Path(fasta))
    if uniprot:
        print(f"  Fetching sequence for {uniprot} from UniProt REST API…")
        seq = fetch_uniprot_sequence(uniprot)
        print(f"  Fetched ({len(seq)} aa)")
        return seq
    sys.exit("ERROR: no sequence source given (pass --sequence, --fasta, or --uniprot)")


def add_sequence_args(parser, required: bool = True) -> None:
    """Attach the standard --sequence/--fasta/--uniprot mutually-informative
    args to an argparse parser. At least one must be supplied at runtime;
    this helper doesn't enforce mutual exclusivity so callers can layer their
    own defaults on top."""
    parser.add_argument("--sequence", default=None, help="Target sequence, given directly")
    parser.add_argument("--fasta", default=None, help="Path to a single-record FASTA file")
    parser.add_argument(
        "--uniprot", default=None,
        help="UniProt accession to fetch the sequence for" + (" (required)" if required else ""),
    )
