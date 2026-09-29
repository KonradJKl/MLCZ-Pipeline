"""
Dashboard for a trained checkpoint: stitched LCZ maps of each city, test scores and single patches.

Run from the repo root with: streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modules import city_maps
from modules.config import LOGS_DIR, LMDB_PATH, PARQUET_PATH

load_model = st.cache_resource(max_entries=1, show_spinner="Loading checkpoint")(city_maps.load_model)


@st.cache_data(max_entries=3, show_spinner=False)
def predict_city(_model, ckpt: str, city: str):
    """City maps of a checkpoint, the leading underscore keeps the model out of the cache key (ckpt, city)"""
    return city_maps.predict_city(_model, LMDB_PATH, PARQUET_PATH, city)


def figure(images: dict, cols: int, height: int):
    """Images in a grid of cols columns with linked zoom"""
    titles = list(images)
    fig = px.imshow(np.stack(list(images.values())), facet_col=0, facet_col_wrap=cols, binary_string=True)
    # Plotly lists the facets bottom row first, so the titles are matched by the index in "facet_col=i"
    fig.for_each_annotation(lambda a: a.update(text=titles[int(a.text.split("=")[1])]))
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0))
    return fig


def legend(label: np.ndarray, pred: np.ndarray):
    """Legend of the classes that appear in the ground truth or the prediction and of the error colors"""
    st.html(city_maps.legend_html(np.union1d(label, pred)))
    st.caption("Errors: grey is correct, red is wrong, black has no label.")


def sidebar(checkpoints: list, meta: pd.DataFrame):
    """Checkpoint, city and view selection, returns (checkpoint path, model, city, view)"""
    # Lightning saves to <project>/<run id>/checkpoints/, run id and file name are enough and fit the sidebar
    ckpt = st.sidebar.selectbox("Checkpoint", checkpoints, format_func=lambda p: f"{p.parents[1].name}/{p.name}")
    model, args = load_model(str(ckpt))
    st.sidebar.caption(f"{args.arch_name}, pretrained {args.pretrained}, dropout {args.dropout}, epochs {args.epochs}, "
                       f"trained on `{args.metadata_parquet_path}`. Running on {city_maps.DEVICE}.")

    # Smallest city first, so the first prediction is quick
    counts = meta["city"].value_counts(ascending=True)
    city = st.sidebar.selectbox("City", counts.index, format_func=lambda c: f"{c} ({counts[c]} patches)")
    # A radio instead of tabs, tabs would render both views on every rerun
    view = st.sidebar.radio("View", ["City map", "Patch"])
    return ckpt, model, city, view


def city_view(result: dict):
    confusions = result["confusion"]
    # A city_based split can leave a city without test patches, then the next split is scored
    split = next(s for s in ["test", "validation", "train"] if s in confusions)
    scores = city_maps.split_table(confusions)
    left, right = st.columns(2)
    left.metric(f"{split.capitalize()} accuracy", f"{scores.at[split, 'Accuracy']:.1%}")
    right.metric(f"{split.capitalize()} mean IoU", f"{scores.at[split, 'Mean IoU']:.3f}")

    st.plotly_chart(figure(city_maps.panels(result), cols=2, height=900))
    st.caption("Maps: averaged predictions over all splits. Metrics: each patch's own prediction; label 0 ignored.")
    legend(result["label"], result["pred"])

    left, right = st.columns([2, 3])
    left.caption("Per split")
    left.dataframe(scores.style.format(precision=3))
    right.caption(f"Per class, {split} patches")
    right.dataframe(city_maps.class_table(confusions[split]).style.format(precision=3), hide_index=True)


def patch_view(result: dict, meta: pd.DataFrame):
    """One patch of the city as a crop of the stitched maps"""
    left, middle, right = st.columns(3)
    split = left.selectbox("Split", ["All"] + [s for s in ["train", "validation", "test"] if s in set(meta["split"])])
    if split != "All":
        meta = meta[meta["split"] == split]
    dominant = middle.selectbox("Dominant LCZ", sorted(meta["dominant_label"].unique()),
                                format_func=lambda k: city_maps.LABELS[k])
    meta = meta[meta["dominant_label"] == dominant].sort_values("label_diversity", ascending=False)
    patch = right.selectbox("Patch", meta.index,
                            format_func=lambda i: f"{meta.at[i, 'patch_id']} ({meta.at[i, 'label_diversity']} classes)")

    r, c = meta.at[patch, "row_offset"], meta.at[patch, "col_offset"]
    window = np.s_[r:r + city_maps.PATCH, c:c + city_maps.PATCH]
    st.plotly_chart(figure(city_maps.panels(result, window), cols=4, height=330))
    label, pred = result["label"][window], result["pred"][window]
    # Never empty, convert_data.py skips patches without labelled pixels
    st.metric("Pixel accuracy", f"{(pred[label > 0] == label[label > 0]).mean():.1%}")
    st.caption("Crop of the stitched map, so the prediction also contains the overlapping patches")
    legend(label, pred)


def main():
    st.set_page_config(page_title="MLCZ city maps", layout="wide")
    if not (LMDB_PATH.exists() and PARQUET_PATH.exists()):
        st.error(f"No converted data in {LMDB_PATH.parent}, run `python main.py --convert` first")
        st.stop()
    checkpoints = city_maps.find_checkpoints(LOGS_DIR)
    if not checkpoints:
        st.error(f"No checkpoints in {LOGS_DIR}, run `python main.py --train` or set LOGS_DIR in .env")
        st.stop()

    meta = pd.read_parquet(PARQUET_PATH)
    ckpt, model, city, view = sidebar(checkpoints, meta)
    meta = meta[meta["city"] == city]

    st.title(f"Local Climate Zones of {city}")
    with st.spinner(f"Predicting all {len(meta)} patches of {city} on {city_maps.DEVICE}, cached afterwards"):
        result = predict_city(model, str(ckpt), city)
    if view == "City map":
        city_view(result)
    else:
        patch_view(result, meta)


if __name__ == "__main__":
    main()
