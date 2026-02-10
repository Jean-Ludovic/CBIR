import streamlit as st
import requests

API_URL = "http://127.0.0.1:8001"  # FastAPI

st.set_page_config(page_title="CBIR Demo", layout="wide")

# ----------------------------
# Helpers
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
    st.caption("Astuce : assure-toi que l’API FastAPI tourne sur http://127.0.0.1:8001")

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

        # Affichage en grille
        cols = st.columns(4)
        for i, img in enumerate(items):
            c = cols[i % 4]
            with c:
                st.image(f"{API_URL}{img['image_url']}", use_container_width=True)
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
                            st.image(f"{API_URL}{item['image_path']}", use_container_width=True)
                        with cols[1]:
                            st.markdown(f"### Top {rank} — {item.get('name','(sans nom)')}")
                            if item.get("description"):
                                st.write(item["description"])

                            d = item.get("distances", {})
                            st.write(
                                {
                                    "euclidean": d.get("euclidean"),
                                    "manhattan": d.get("manhattan"),
                                    "chebyshev": d.get("chebyshev"),
                                    "canberra": d.get("canberra"),
                                }
                            )
                        st.divider()

    st.caption("Si tu vois une erreur, vérifie que l’API FastAPI est bien lancée sur le port 8001.")
