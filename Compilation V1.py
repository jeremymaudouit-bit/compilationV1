import streamlit as st

st.set_page_config(page_title="Plateforme Biomécanique Pro", layout="wide")
st.title("🏃🦴 Plateforme Biomécanique Pro")
st.subheader("Choisissez le type d'analyse")

# Définition des applications
apps = [
    {
        "nom": "Analyse de la marche / course vidéo latérale",
        "url": "https://marchewebv2-ghtjzszkohdbcm4mdvstcf.streamlit.app/",
        "logo": "https://img.icons8.com/color/96/000000/running.png"
    },
    {
        "nom": "Analyse du dos via scan 3D",
        "url": "https://scandosv1-rxlsw3f33usvhgj7hrrbmb.streamlit.app/",
        "logo": "https://img.icons8.com/3d-fluency/96/businessman.png"
    },
    {
        "nom": "Analyse de la posture de dos",
        "url": "https://posturewebv3-85bhxf23ejequnmrhnqypl.streamlit.app/",
        "logo": "https://img.icons8.com/color/96/standing-man.png"
    },
    {
        "nom": "Analyse de la marche vidéo de face",
        "url": "https://marchewebfront-cfpfke9twwrktggz3ervr2.streamlit.app/",
        "logo": "https://img.icons8.com/color/96/000000/front-view.png"
    },
]

# Affichage en grille 2 colonnes
cols = st.columns(2)
for idx, app in enumerate(apps):
    col = cols[idx % 2]
    with col:
        st.markdown(
            f"""
            <div style="
                border:2px solid #4CAF50; 
                border-radius:15px; 
                padding:20px; 
                text-align:center; 
                background-color:#f9f9f9; 
                margin-bottom:20px;
                transition: transform 0.2s;
            "
            onmouseover="this.style.transform='scale(1.05)'" 
            onmouseout="this.style.transform='scale(1)'">
                <img src="{app['logo']}" width="80px" style="margin-bottom:15px;"><br>
                <h3 style="color:#333; font-weight:bold;">{app['nom']}</h3>
                <a href="{app['url']}" target="_blank">
                    <button style="
                        background-color:#4CAF50;
                        color:white;
                        padding:10px 20px;
                        border:none;
                        border-radius:10px;
                        cursor:pointer;
                        font-size:16px;
                        margin-top:10px;
                    ">Ouvrir</button>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )










