"""
app.py — no-code interface for the pseudo-ID pipeline.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Then open the URL Streamlit prints (usually http://localhost:8501).

Safety / traceability notes for anyone deploying this to other users:
- Nothing the user uploads is written to disk or sent anywhere. It's held
  in memory for the browser session only and discarded on refresh/close.
- The only network call is a one-time, cached fetch of the public
  ISO country-name reference list (mledoze/countries) — never the user's data.
- The pipeline only ever ADDS columns; original columns are never edited or
  removed, and the app always keeps a pinned view of the original data next
  to the generated pseudo_id so users can check the mapping themselves.
"""

import io

import pandas as pd
import streamlit as st

from pseudo_id_pipeline import PseudoIdPipeline

st.set_page_config(page_title="Org Name → Pseudo-ID", layout="wide")

st.title("Organisation Name → Pseudo-ID")
st.caption(
    "Upload a messy dataset of organisation names and get back the same "
    "file with a stable `pseudo_id` column added, ready to join against "
    "other datasets. Nothing you upload is stored — it only lives in this "
    "browser session."
)


@st.cache_resource(show_spinner="Loading the matching model (only happens once)…")
def get_pipeline():
    return PseudoIdPipeline()


# ---------------- 1. Upload ----------------

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to get started.")
    st.stop()

try:
    original_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Couldn't read that file as a CSV: {e}")
    st.stop()

if original_df.empty:
    st.error("That file has no rows.")
    st.stop()

st.success(f"Loaded {len(original_df):,} rows, {len(original_df.columns)} columns.")

with st.expander("View original upload (unchanged, always available)", expanded=False):
    st.dataframe(original_df, use_container_width=True)

# ---------------- 2. Column mapping ----------------

st.subheader("Which column has the organisation name?")
columns = list(original_df.columns)
name_col = st.selectbox("Organisation name column (required)", columns)

col_a, col_b = st.columns(2)
with col_a:
    abbrev_col = st.selectbox(
        "Existing abbreviation column (optional)",
        ["— none —"] + columns,
        help="If your data already has a short code/acronym, pick it here. "
        "Otherwise one is generated automatically from the name.",
    )
    abbrev_col = None if abbrev_col == "— none —" else abbrev_col

with col_b:
    country_col = st.selectbox(
        "Existing country column (optional)",
        ["— none —"] + columns,
        help="If your data already has a country, pick it here. Otherwise "
        "the app tries to detect a country name inside the org name text.",
    )
    country_col = None if country_col == "— none —" else country_col

# ---------------- 3. Run ----------------

keep_embeddings = st.checkbox(
    "Keep the raw embedding vectors in the output",
    value=False,
    help="Off by default — embeddings are ~384 numbers per row and mainly "
    "useful for debugging or building your own similarity search on top. "
    "Turning this on will make the downloaded CSV much larger.",
)

run = st.button("Generate pseudo-IDs", type="primary")

if run:
    pipeline = get_pipeline()
    with st.spinner("Matching names and generating IDs…"):
        try:
            result_df = pipeline.run(
                original_df,
                name_col=name_col,
                abbrev_col=abbrev_col,
                country_col=country_col,
                keep_embeddings=keep_embeddings,
            )
        except Exception as e:
            st.error(f"Something went wrong while processing: {e}")
            st.stop()

    st.session_state["result_df"] = result_df
    st.session_state["name_col"] = name_col

# ---------------- 4. Results, with the original always alongside ----------------

if "result_df" in st.session_state:
    result_df = st.session_state["result_df"]
    name_col_used = st.session_state["name_col"]

    st.subheader("Result")
    st.caption(
        f"Original column **{name_col_used}** is kept as-is and pinned next "
        "to the generated `pseudo_id` so you can always check the mapping."
    )

    preview_cols = [name_col_used, "pseudo_id", "extracted_country", "pseudo_id_created_at"]
    preview_cols = [c for c in preview_cols if c in result_df.columns]
    other_cols = [c for c in result_df.columns if c not in preview_cols and c != "embeddings"]

    st.dataframe(result_df[preview_cols + other_cols], use_container_width=True)
    if "embeddings" in result_df.columns:
        st.caption("The `embeddings` column is included in the download but hidden from this preview for readability.")

    n_missing = result_df["pseudo_id"].isna().sum()
    if n_missing:
        st.warning(f"{n_missing} row(s) had a blank name and got no pseudo_id.")

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full CSV (original columns + pseudo_id)",
        data=csv_bytes,
        file_name=f"{uploaded.name.rsplit('.', 1)[0]}_with_pseudo_ids.csv",
        mime="text/csv",
    )

    with st.expander("Look up a single row by original name"):
        pick = st.selectbox(
            "Pick a row to inspect",
            options=result_df.index,
            format_func=lambda i: str(result_df.loc[i, name_col_used]),
        )
        st.write(result_df.loc[[pick]])
