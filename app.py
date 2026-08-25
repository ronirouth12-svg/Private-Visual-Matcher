import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity

DB_DIR = Path("people_db")
DB_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Private Visual Matcher", page_icon="🔎")
st.title("🔎 Private Visual Matcher")
st.caption("Matches an uploaded photo against your own authorized image database.")

def embedding(image: Image.Image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (128, 128))
    # Simple visual fingerprint: normalized grayscale pixels.
    # This is NOT a face-identification system.
    vec = gray.astype(np.float32).reshape(-1)
    vec = (vec - vec.mean()) / (vec.std() + 1e-6)
    return vec

def database_images():
    return [p for p in DB_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

with st.sidebar:
    st.header("Authorized database")
    uploaded = st.file_uploader(
        "Add an image", type=["jpg", "jpeg", "png", "webp"]
    )
    label = st.text_input("Label for this image", placeholder="e.g. Person A")
    if st.button("Add to database"):
        if uploaded and label.strip():
            safe = "".join(c for c in label.strip() if c.isalnum() or c in "_- ")
            safe = safe.replace(" ", "_")
            out = DB_DIR / f"{safe}_{len(database_images())}.png"
            Image.open(uploaded).convert("RGB").save(out)
            st.success(f"Added: {safe}")
        else:
            st.warning("Choose an image and enter a label.")

st.subheader("1. Upload a photo to compare")
query_file = st.file_uploader(
    "Comparison image", type=["jpg", "jpeg", "png", "webp"], key="query"
)

threshold = st.slider("Match threshold", 0.50, 0.99, 0.85, 0.01)

if query_file:
    query = Image.open(query_file).convert("RGB")
    st.image(query, caption="Comparison image", width=350)

    records = database_images()
    if not records:
        st.info("Your database is empty. Add authorized images from the sidebar.")
    else:
        q = embedding(query)
        results = []
        for path in records:
            try:
                score = float(cosine_similarity(
                    q.reshape(1, -1),
                    embedding(Image.open(path).convert("RGB")).reshape(1, -1)
                )[0][0])
                results.append((score, path))
            except Exception:
                pass

        results.sort(reverse=True, key=lambda x: x[0])
        best_score, best_path = results[0]

        st.subheader("Result")
        if best_score >= threshold:
            label = best_path.stem.rsplit("_", 1)[0].replace("_", " ")
            st.success(f"Possible visual match: **{label}**")
        else:
            st.warning("No sufficiently similar image found.")

        st.metric("Similarity", f"{best_score:.1%}")
        st.caption("This prototype compares overall image appearance; it does not identify people by face.")
        st.subheader("Top matches")
        for score, path in results[:5]:
            st.write(f"**{path.stem}** — {score:.1%}")
