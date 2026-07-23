"""
pseudo_id_pipeline.py

Reusable pipeline for generating pseudo-IDs (embedding + LSH + org-type
abbreviation) for any dataset that contains organisation names.

Usage (CLI):
    python pseudo_id_pipeline.py \
        --input path/to/dataset.csv \
        --output path/to/dataset_with_ids.csv \
        --name-col "Name" \
        --abbrev-col "Abbreviated name" \
        --country-col "Country" \
        --header-row 1

Usage (as a library, e.g. from a notebook):
    from pseudo_id_pipeline import PseudoIdPipeline

    pipeline = PseudoIdPipeline()
    df = pipeline.run(df, name_col="Name", abbrev_col="Abbreviated name")

Only --input, --output, and --name-col are required. Everything else
(abbrev/country columns, header row, model, batch size) is optional and
dataset-specific, so the same script drops into any new dataset without
edits — just different CLI flags.
"""

import argparse
import json
import re
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import stopwordsiso as stopwords
import torch
from flashtext import KeywordProcessor
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LANGS = ["en", "fr", "it", "es"]
DEFAULT_N_BITS = 16
DEFAULT_LSH_SEED = 42

COUNTRIES_URL = "https://raw.githubusercontent.com/mledoze/countries/master/countries.json"

ORG_TYPE_MAP = {
    "international": "intl", "internazionale": "intl", "internationale": "intl",
    "organization": "org", "organisation": "org", "organizzazione": "org",
    "committee": "cttee", "comite": "cttee", "comitato": "cttee",
    "commission": "comm", "commissione": "comm",
    "federation": "fed", "federazione": "fed",
    "association": "assoc", "associazione": "assoc",
    "programme": "prog", "program": "prog", "programma": "prog",
    "foundation": "fdn", "fondazione": "fdn", "fondation": "fdn",
}


def strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")


def clean_text(s: str) -> str:
    """Lowercase, strip accents, keep only alphanumerics/spaces, collapse whitespace."""
    s = strip_accents(str(s).lower())
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def truncate_at_slash(name) -> str:
    """If the org name contains '/', keep only the part before the first
    one (e.g. 'World Food Programme / WFP' -> 'World Food Programme')."""
    if pd.isna(name):
        return name
    return str(name).split("/", 1)[0].strip()


def remove_parentheticals(name) -> str:
    """Drop any text inside parentheses, wherever it appears
    (e.g. 'World Food Programme (WFP)' -> 'World Food Programme',
    'Org (Sub) Europe' -> 'Org Europe')."""
    if pd.isna(name):
        return name
    cleaned = re.sub(r"\([^)]*\)", " ", str(name))
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_org_name(name) -> str:
    """Apply all pre-processing used before embedding/normalising/abbreviating
    an org name: drop parenthetical asides, then truncate at the first '/'."""
    if pd.isna(name):
        return name
    return truncate_at_slash(remove_parentheticals(name))


class PseudoIdPipeline:
    """
    Encapsulates everything that used to be module-level globals, so the
    pipeline can be instantiated once (loading the embedding model + country
    lookup a single time) and then reused across as many datasets/columns
    as you like, without re-downloading or re-encoding anything by accident.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        langs=None,
        n_bits: int = DEFAULT_N_BITS,
        lsh_seed: int = DEFAULT_LSH_SEED,
        countries_cache_path: str = "countries.json",
    ):
        self.langs = langs or DEFAULT_LANGS
        self.n_bits = n_bits
        self.countries_cache_path = Path(countries_cache_path)

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=device)

        self.stopwords = {strip_accents(w) for w in stopwords.stopwords(self.langs)}

        self._rng = np.random.default_rng(seed=lsh_seed)
        self._hyperplanes = None  # lazily sized to the embedding dimension

        self.country_matcher, self.country_phrase_to_code = self._build_country_matcher()

    # ---------------- country lookup ----------------
    # Note: country is still extracted (kept as an `extracted_country` column,
    # useful metadata) but is no longer folded into the pseudo_id itself.

    def _load_countries_data(self):
        if self.countries_cache_path.exists():
            with open(self.countries_cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        with urllib.request.urlopen(COUNTRIES_URL) as resp:
            data = json.load(resp)
        with open(self.countries_cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data

    def _build_country_matcher(self):
        countries_data = self._load_countries_data()
        phrase_to_code = {}

        for country in countries_data:
            code = country["cca3"]
            phrases = {country["name"]["common"], country["name"]["official"]}

            for lang3 in ["fra", "ita", "spa"]:
                t = country.get("translations", {}).get(lang3)
                if t:
                    phrases.add(t["common"])
                    phrases.add(t["official"])

            phrases.update(country.get("altSpellings", []))

            for phrase in phrases:
                clean = clean_text(phrase)
                if clean and len(clean) > 2:  # skip very short/ambiguous tokens
                    phrase_to_code[clean] = code

        # Single trie built once — O(text length) lookup per row, regardless
        # of how many hundreds of country-name phrases are registered
        matcher = KeywordProcessor(case_sensitive=False)
        for phrase, code in phrase_to_code.items():
            matcher.add_keyword(phrase, code)

        return matcher, phrase_to_code

    def extract_country_from_name(self, name: str):
        """ISO alpha-3 code of a country name found inside org name text, or
        None. Flags actual country names only — no demonyms/adjectival forms."""
        if pd.isna(name):
            return None
        text = clean_text(name)
        matches = self.country_matcher.extract_keywords(text)
        return matches[0] if matches else None

    # ---------------- name normalisation ----------------

    def normalize_name(self, name: str) -> str:
        tokens = clean_text(name).split()
        tokens = [ORG_TYPE_MAP.get(t, t) for t in tokens]
        tokens = [t for t in tokens if t not in self.stopwords]
        return " ".join(tokens)

    @staticmethod
    def abbreviate(normalized: str, max_words: int = None) -> str:
        """Abbreviate every word in the (already normalized) name by default
        — pass max_words to cap it instead."""
        tokens = normalized.split()
        if max_words is not None:
            tokens = tokens[:max_words]
        return "".join(t[:3] for t in tokens).upper()

    # ---------------- LSH ----------------

    def _lsh_code(self, embedding) -> str:
        embedding = np.asarray(embedding, dtype=float)
        if self._hyperplanes is None:
            self._hyperplanes = self._rng.normal(size=(self.n_bits, len(embedding)))
        bits = (self._hyperplanes @ embedding) > 0
        return "".join("1" if b else "0" for b in bits)

    def _safe_lsh_code(self, embedding):
        if embedding is None or (isinstance(embedding, float) and pd.isna(embedding)):
            return None
        return self._lsh_code(embedding)

    # ---------------- main entry point ----------------

    def run(
        self,
        df: pd.DataFrame,
        name_col: str,
        abbrev_col: str = None,
        country_col: str = None,
        batch_size: int = 256,
        keep_embeddings: bool = True,
    ) -> pd.DataFrame:
        """
        Adds `embeddings`, `lsh_code`, `extracted_country`, `pseudo_id`, and
        `pseudo_id_created_at` columns to a copy of df.

        name_col      : required — column with the organisation name. Any
                          text inside parentheses is dropped, then any text
                          after a '/' is dropped (e.g. "World Food Programme
                          (WFP) / Rome office" is treated as "World Food
                          Programme").
        abbrev_col     : optional — pre-existing abbreviation to use as the ID
                          prefix instead of auto-abbreviating name_col.
        country_col    : optional — pre-existing country column; falls back
                          to country names detected inside name_col. Kept as
                          the `extracted_country` column but no longer used
                          to build the pseudo_id.
        """
        df = df.copy()

        # Work on a slash-truncated version of the name for everything
        # downstream (embeddings, normalisation, country extraction, id).
        clean_name_col = "_clean_name"
        df[clean_name_col] = df[name_col].apply(clean_org_name)
        df[clean_name_col] = df[clean_name_col].apply(lambda x: str(x) if pd.notna(x) else x)

        unique_names = [str(n) for n in df[clean_name_col].dropna().unique()]
        embeddings = self.model.encode(
            unique_names,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        name_to_embedding = dict(zip(unique_names, embeddings))
        df["embeddings"] = df[clean_name_col].map(name_to_embedding)

        df["lsh_code"] = df["embeddings"].apply(self._safe_lsh_code)
        df["extracted_country"] = df[clean_name_col].apply(self.extract_country_from_name)

        def resolve_prefix(row):
            if abbrev_col and pd.notna(row.get(abbrev_col)):
                return re.sub(r"\s+", "", str(row[abbrev_col]).strip()).upper()
            return self.abbreviate(self.normalize_name(row[clean_name_col]))

        def build_id(row):
            if pd.isna(row[clean_name_col]) or row["lsh_code"] is None:
                return None
            prefix = resolve_prefix(row)
            return f"{prefix}-{row['lsh_code']}"

        df["pseudo_id"] = df.apply(build_id, axis=1)

        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        df["pseudo_id_created_at"] = df["pseudo_id"].apply(lambda x: created_at if x is not None else None)

        df = df.drop(columns=[clean_name_col])

        if not keep_embeddings:
            df = df.drop(columns=["embeddings"])

        return df


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-IDs for a dataset of organisation names.")
    parser.add_argument("--input", required=True, help="Path to input CSV.")
    parser.add_argument("--output", required=True, help="Path to write output CSV.")
    parser.add_argument("--name-col", required=True, help="Column containing the organisation name.")
    parser.add_argument("--abbrev-col", default=None, help="Optional column with a pre-existing abbreviation.")
    parser.add_argument("--country-col", default=None, help="Optional column with a pre-existing country.")
    parser.add_argument("--header-row", type=int, default=0, help="0-indexed header row for pd.read_csv.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--drop-embeddings",
        action="store_true",
        help="Don't keep the raw embedding vectors in the output CSV (they bloat file size).",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Subset of columns to keep in the output. Defaults to all input columns plus the new ones.",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input, header=args.header_row, low_memory=False)

    pipeline = PseudoIdPipeline(model_name=args.model)
    result = pipeline.run(
        data,
        name_col=args.name_col,
        abbrev_col=args.abbrev_col,
        country_col=args.country_col,
        batch_size=args.batch_size,
        keep_embeddings=not args.drop_embeddings,
    )

    if args.columns:
        result = result[args.columns]

    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
