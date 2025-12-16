import os
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import requests

from config import Config
from src.data.datasets import LABELS_BLOODMNIST_FULL


API_URL = "http://127.0.0.1:8000/predict"


def call_api_predict(image: Image.Image, model_type: str):
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("image.png", buf, "image/png")}
    params = {"model_type": model_type}

    resp = requests.post(API_URL, files=files, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    st.set_page_config(page_title="Klasyfikacja krwinek – BloodMNIST (CNN)", layout="wide")
    st.title("Klasyfikacja krwinek – BloodMNIST (CNN)")

    cfg = Config()

    st.sidebar.header("Ustawienia modelu")
    model_type = st.sidebar.selectbox(
        "Typ modelu",
        options=["simple_cnn", "deep_cnn"],
        format_func=lambda x: "Simple CNN" if x == "simple_cnn" else "Deep CNN",
    )

    if model_type == "simple_cnn":
        default_ckpt = os.path.join(cfg.output_dir, "simple_cnn_adam_aug.pt")
    else:
        default_ckpt = os.path.join(cfg.output_dir, "deep_cnn_adam_noaug.pt")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("1. Wybierz obraz komórki")
        uploaded_file = st.file_uploader(
            "Wgraj obraz (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
        )

        with st.expander("Info o klasach BloodMNIST", expanded=False):
            for idx, name in LABELS_BLOODMNIST_FULL.items():
                st.write(f"{idx}: {name}")

        image_to_show = None
        if uploaded_file is not None:
            image_to_show = Image.open(uploaded_file).convert("RGB")
            st.image(image_to_show, caption="Załadowany obraz")
        else:
            st.warning("Nie załadowano obrazu. Wgraj plik, aby uruchomić predykcję.")

    with col_right:
        st.subheader("2. Predykcja modelu")

        if st.button("Uruchom predykcję", type="primary"):
            if image_to_show is None:
                st.error("Najpierw wgraj obraz (PNG/JPG).")
            else:
                try:
                    result = call_api_predict(image_to_show, model_type)
                except requests.exceptions.RequestException as e:
                    st.error(f"Nie udało się skontaktować z API: {e}")
                    st.info("Czy na pewno uruchomiłeś: uvicorn api:app --reload ?")
                else:
                    pred_idx = int(result["predicted_class_idx"])
                    probs = np.array(result["probabilities"], dtype=float)

                    class_names = [LABELS_BLOODMNIST_FULL[str(i)] for i in range(len(LABELS_BLOODMNIST_FULL))]
                    pred_name = class_names[pred_idx]
                    max_prob = float(probs[pred_idx])

                    if max_prob < 0.6:
                        st.warning(
                            f"Model jest NIEPEWNY: najwyższe prawdopodobieństwo "
                            f"to tylko {max_prob*100:.1f}%. "
                            "Wynik ma charakter wyłącznie poglądowy."
                        )
                    else:
                        st.success(
                            f"Przewidywana klasa: {pred_idx} – {pred_name} "
                            f"({max_prob*100:.1f}% pewności)"
                        )

                    st.markdown("### Rozkład prawdopodobieństw")
                    df = pd.DataFrame({
                        "Klasa": class_names,
                        "Prawdopodobieństwo": probs,
                    })
                    df["Prawdopodobieństwo [%]"] = (df["Prawdopodobieństwo"] * 100).round(2)
                    df = df.sort_values("Prawdopodobieństwo", ascending=False).reset_index(drop=True)

                    st.dataframe(df[["Klasa", "Prawdopodobieństwo [%]"]], use_container_width=True)
                    st.bar_chart(df.set_index("Klasa")["Prawdopodobieństwo"])


if __name__ == "__main__":
    main()