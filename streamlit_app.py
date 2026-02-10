# streamlit_app.py
import time
import threading
from io import BytesIO

import streamlit as st
import requests
import uvicorn

from cbir_api.main import app as fastapi_app

API_URL = "http://127.0.0.1:8001"  # FastAPI interne


@st.cache_resource
def start_api():
    """Démarre FastAPI une seule fois dans le même conteneur Streamlit."""
    def run():
        uvicorn.run(
            fastapi_app,
            host="127.0.0.1",
            port=8001,
            log_level="warning",
        )

    t = threading.Thread(target=run, daemon=True)
    t.start()

    # attendre que l’API réponde
    for _ in range(40):
        try:
            r = requests.get(f"{API_URL}/health", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(0.25)
    return False


def get_image_bytes(image_path: str) -> BytesIO:
    """
    image_path: ex "/media/gallery/user_1_1.jpg"
    On fetch côté serveur Streamlit, puis on affiche en bytes (OK sur mobile).
    """
    r = requests.get(f"{API_URL}{image_path}", timeout=20)
    r.raise_for_status()
    return BytesIO(r.content)


# ----------------------------
# Boot
# ----------------------------
st.set_page_config(page_title="CBIR Demo", layout="wide")

if not start_api():
    st.error("FastAPI ne démarre pas (import/requirements).")
    st.stop()


# ----------------------------
# Helpers API
# ----------------------------
def fetch_gallery():
    try:
        r = requests.get(f"{API_URL}/gallery", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Impossible de récupérer la galerie depuis l’API. Détail: {e}")
        return []


def search_image(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        r = requests.post(f"{API_URL}/gallery/search", files=files, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur pendant la recherche CBIR. Détail: {e}")
        return None


# ----------------------------
# Navigation (simple)
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"


def go(page_name: str):
    st.session_state.page = page_name


# ----------------------------
# HOME PAGE
# ----------------------------
if st.session_state.page == "home":
    st.title("CBIR Demo — Recherche d’images similaires")

    st.markdown(
        """
Bienvenue dans cette démonstration.

**Objectif :** comparer la photo que vous uploadez avec celles déjà enregistrées dans la base de données,
puis afficher un **classement des images les plus similaires** (Top 1, Top 2, ...), basé sur des distances
calculées à partir d’un embedding.

**Parcours recommandé :**
1. **Voir la base de données** (galerie d’images enregistrées)
2. **Procéder à la démo** (uploader une image requête et voir les résultats)
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.button("📁 Voir la base de données", use_container_width=True, on_click=go, args=("gallery",))
    with col2:
        st.button("🚀 Procéder à la démo", use_container_width=True, on_click=go, args=("demo",))

    st.divider()
    st.caption("✅ Sur Streamlit Cloud, l’API tourne en interne (localhost) et Streamlit l’appelle côté serveur.")


# ----------------------------
# GALLERY PAGE
# ----------------------------
elif st.session_state.page == "gallery":
    st.title("Base de données — Galerie d’images")

    top = st.columns([1, 1, 6])
    with top[0]:
        st.button("⬅️ Retour", on_click=go, args=("home",), use_container_width=True)
    with top[1]:
        if st.button("🔄 Rafraîchir", use_container_width=True):
            st.rerun()

    items = fetch_gallery()

    if not items:
        st.info("Aucune image enregistrée pour le moment.")
    else:
        st.write(f"Images enregistrées : **{len(items)}**")

        cols = st.columns(4)
        for i, img in enumerate(items):
            c = cols[i % 4]
            with c:
                try:
                    st.image(get_image_bytes(img["image_url"]), use_container_width=True)
                except Exception as e:
                    st.warning(f"Image non chargeable: {e}")
                st.caption(f"**{img['name']}**")
                if img.get("description"):
                    st.caption(img["description"])

    st.divider()
    st.button("🚀 Procéder à la démo", on_click=go, args=("demo",), use_container_width=True)


# ----------------------------
# DEMO PAGE
# ----------------------------
elif st.session_state.page == "demo":
    st.title("Démo — Uploader une image et trouver les plus similaires")

    top = st.columns([1, 7])
    with top[0]:
        st.button("⬅️ Retour", on_click=go, args=("home",), use_container_width=True)

    st.markdown(
        """
1) Uploade une image requête  
2) Clique sur **Lancer la recherche**  
3) Observe le **classement** et les **distances** (euclidienne, manhattan, chebyshev, canberra)
        """
    )

    uploaded = st.file_uploader("Choisir une image (jpg/png/jpeg)", type=["jpg", "jpeg", "png"])

    if uploaded:
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Image requête")
            st.image(uploaded, use_container_width=True)

        with right:
            st.subheader("Actions")
            if st.button("🔎 Lancer la recherche", use_container_width=True):
                res = search_image(uploaded)
                if not res:
                    st.stop()

                st.success("Recherche terminée.")

                st.divider()
                st.subheader("Résultats (triés par distance euclidienne)")
                results = res.get("results", [])

                if not results:
                    st.info("Aucun résultat. Ta galerie est peut-être vide.")
                else:
                    for rank, item in enumerate(results, start=1):
                        cols = st.columns([1, 2])
                        with cols[0]:
                            try:
                                st.image(get_image_bytes(item["image_path"]), use_container_width=True)
                            except Exception as e:
                                st.warning(f"Image non chargeable: {e}")
                        with cols[1]:
                            st.markdown(f"### Top {rank} — {item.get('name','(sans nom)')}")
                            if item.get("description"):
                                st.write(item["description"])

                            d = item.get("distances", {})
                            st.write({
                                "euclidean": d.get("euclidean"),
                                "manhattan": d.get("manhattan"),
                                "chebyshev": d.get("chebyshev"),
                                "canberra": d.get("canberra"),
                            })
                        st.divider()
