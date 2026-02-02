# =========================================================
# BIOMECH AI SUITE – APP STREAMLIT UNIQUE
# PARTIE 1/4 : STRUCTURE & MENU
# =========================================================

import streamlit as st

# ⚠️ UNE SEULE FOIS DANS TOUT LE FICHIER
st.set_page_config(
    page_title="Biomech AI Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Biomech AI Suite")
st.subheader("Plateforme unifiée d’analyses biomécaniques IA")

with st.sidebar:
    st.header("🔍 Choix de l’analyse")
    APP_MODE = st.radio(
        "Sélectionnez l’analyse souhaitée :",
        [
            "🏃 GaitScan – Analyse Frontale",
            "🏃 GaitScan – Analyse Cinématique",
            "🦴 SpineScan Pro 3D",
            "🧍 Analyse Posturale (Photo)"
        ]
    )

st.divider()

# =========================================================
# ROUTEUR (les fonctions sont définies plus bas)
# =========================================================
if APP_MODE == "🏃 GaitScan – Analyse Frontale":
    run_gaitscan_frontal()

elif APP_MODE == "🏃 GaitScan – Analyse Cinématique":
    run_gaitscan_cinematique()

elif APP_MODE == "🦴 SpineScan Pro 3D":
    run_spinescan_3d()

elif APP_MODE == "🧍 Analyse Posturale (Photo)":
    run_posture_photo()
# =========================================================
# PARTIE 2/4 – GAITSCAN FRONTAL COMPLET
# =========================================================

def run_gaitscan_frontal():

    import streamlit as st
    import tensorflow as tf
    import tensorflow_hub as hub
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import tempfile, os
    from scipy.ndimage import gaussian_filter1d
    from datetime import datetime

    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage, Table, TableStyle
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors

    st.header("🏃 GaitScan Pro – Analyse Frontale")
    st.caption("Valgus / varus – abduction / adduction – bassin – tronc")

    # -----------------------------------------------------
    # MODELE MOVENET
    # -----------------------------------------------------
    @st.cache_resource
    def load_movenet():
        return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

    movenet = load_movenet()

    KEYPOINTS = {
        "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14,
        "left_ankle": 15, "right_ankle": 16,
        "left_shoulder": 5, "right_shoulder": 6
    }

    def detect_pose(frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
        outputs = movenet.signatures["serving_default"](tf.cast(img, tf.int32))
        return outputs["output_0"][0, 0, :, :].numpy()

    def angle(a, b, c):
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    # -----------------------------------------------------
    # UI
    # -----------------------------------------------------
    video = st.file_uploader("Vidéo frontale (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    smooth = st.slider("Lissage temporel", 0, 10, 2)

    if not video:
        st.info("Veuillez importer une vidéo frontale.")
        return

    if not st.button("⚙️ Lancer l’analyse"):
        return

    # -----------------------------------------------------
    # TRAITEMENT VIDEO
    # -----------------------------------------------------
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(video.read())

    cap = cv2.VideoCapture(tmp.name)

    pelvis_angles = []
    knee_L, knee_R = [], []
    trunk_angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        kp = detect_pose(frame)

        LH = kp[KEYPOINTS["left_hip"], :2]
        RH = kp[KEYPOINTS["right_hip"], :2]
        LK = kp[KEYPOINTS["left_knee"], :2]
        RK = kp[KEYPOINTS["right_knee"], :2]
        LA = kp[KEYPOINTS["left_ankle"], :2]
        RA = kp[KEYPOINTS["right_ankle"], :2]
        LS = kp[KEYPOINTS["left_shoulder"], :2]
        RS = kp[KEYPOINTS["right_shoulder"], :2]

        pelvis = np.degrees(np.arctan2(RH[1] - LH[1], RH[0] - LH[0]))
        trunk = np.degrees(np.arctan2(RS[1] - LS[1], RS[0] - LS[0]))

        knee_left = angle(LH, LK, LA)
        knee_right = angle(RH, RK, RA)

        pelvis_angles.append(pelvis)
        trunk_angles.append(trunk)
        knee_L.append(knee_left)
        knee_R.append(knee_right)

    cap.release()
    os.unlink(tmp.name)

    # -----------------------------------------------------
    # LISSAGE
    # -----------------------------------------------------
    pelvis_angles = gaussian_filter1d(pelvis_angles, sigma=smooth)
    trunk_angles = gaussian_filter1d(trunk_angles, sigma=smooth)
    knee_L = gaussian_filter1d(knee_L, sigma=smooth)
    knee_R = gaussian_filter1d(knee_R, sigma=smooth)

    # -----------------------------------------------------
    # AFFICHAGE
    # -----------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.plot(pelvis_angles, label="Bassin")
        ax.plot(trunk_angles, label="Tronc")
        ax.set_title("Angles frontaux (°)")
        ax.legend()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(knee_L, label="Genou gauche")
        ax.plot(knee_R, label="Genou droit")
        ax.set_title("Valgus / Varus genoux (°)")
        ax.legend()
        st.pyplot(fig)

    # -----------------------------------------------------
    # EXPORT PDF
    # -----------------------------------------------------
    st.divider()
    st.subheader("📄 Export PDF")

    if st.button("📥 Générer le rapport PDF"):

        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>GaitScan – Analyse Frontale</b>", styles["Title"]))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph(
            f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.5 * cm))

        table_data = [
            ["Mesure", "Moyenne (°)"],
            ["Bascule bassin", f"{np.mean(pelvis_angles):.2f}"],
            ["Inclinaison tronc", f"{np.mean(trunk_angles):.2f}"],
            ["Genou gauche", f"{np.mean(knee_L):.2f}"],
            ["Genou droit", f"{np.mean(knee_R):.2f}"],
        ]

        table = Table(table_data, colWidths=[7 * cm, 4 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))

        story.append(table)
        doc.build(story)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Télécharger le PDF",
                f,
                file_name="gaitscan_frontal.pdf",
                mime="application/pdf"
            )
# =========================================================
# PARTIE 3/4 – GAITSCAN CINÉMATIQUE COMPLET
# =========================================================

def run_gaitscan_cinematique():

    import streamlit as st
    import tensorflow as tf
    import tensorflow_hub as hub
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import tempfile, os
    from scipy.ndimage import gaussian_filter1d
    from datetime import datetime

    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors

    st.header("🏃 GaitScan – Analyse Cinématique (Sagittale)")
    st.caption("Flexion / extension hanche – genou – cheville")

    # -----------------------------------------------------
    # MOVENET
    # -----------------------------------------------------
    @st.cache_resource
    def load_movenet():
        return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

    movenet = load_movenet()

    KP = {
        "hip": 12,
        "knee": 14,
        "ankle": 16,
        "shoulder": 6
    }

    def detect_pose(frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
        outputs = movenet.signatures["serving_default"](tf.cast(img, tf.int32))
        return outputs["output_0"][0, 0, :, :].numpy()

    def angle(a, b, c):
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    # -----------------------------------------------------
    # UI
    # -----------------------------------------------------
    video = st.file_uploader("Vidéo sagittale (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    smooth = st.slider("Lissage temporel", 0, 10, 2)

    if not video:
        st.info("Veuillez importer une vidéo sagittale.")
        return

    if not st.button("⚙️ Lancer l’analyse cinématique"):
        return

    # -----------------------------------------------------
    # VIDEO PROCESSING
    # -----------------------------------------------------
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(video.read())

    cap = cv2.VideoCapture(tmp.name)

    hip_angles = []
    knee_angles = []
    ankle_angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        kp = detect_pose(frame)

        hip = kp[KP["hip"], :2]
        knee = kp[KP["knee"], :2]
        ankle = kp[KP["ankle"], :2]
        shoulder = kp[KP["shoulder"], :2]

        hip_angle = angle(shoulder, hip, knee)
        knee_angle = angle(hip, knee, ankle)

        foot_ref = ankle + np.array([1.0, 0.0])
        ankle_angle = angle(knee, ankle, foot_ref)

        hip_angles.append(hip_angle)
        knee_angles.append(knee_angle)
        ankle_angles.append(ankle_angle)

    cap.release()
    os.unlink(tmp.name)

    # -----------------------------------------------------
    # LISSAGE
    # -----------------------------------------------------
    hip_angles = gaussian_filter1d(hip_angles, sigma=smooth)
    knee_angles = gaussian_filter1d(knee_angles, sigma=smooth)
    ankle_angles = gaussian_filter1d(ankle_angles, sigma=smooth)

    # -----------------------------------------------------
    # AFFICHAGE
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hip_angles, label="Hanche")
    ax.plot(knee_angles, label="Genou")
    ax.plot(ankle_angles, label="Cheville")
    ax.set_title("Cinématique sagittale (°)")
    ax.legend()
    st.pyplot(fig)

    # -----------------------------------------------------
    # EXPORT PDF
    # -----------------------------------------------------
    st.divider()
    st.subheader("📄 Export PDF")

    if st.button("📥 Générer le rapport PDF cinématique"):

        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>GaitScan – Analyse Cinématique</b>", styles["Title"]))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph(
            f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.5 * cm))

        table_data = [
            ["Articulation", "Angle moyen (°)"],
            ["Hanche", f"{np.mean(hip_angles):.2f}"],
            ["Genou", f"{np.mean(knee_angles):.2f}"],
            ["Cheville", f"{np.mean(ankle_angles):.2f}"],
        ]

        table = Table(table_data, colWidths=[7 * cm, 4 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))

        story.append(table)
        doc.build(story)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Télécharger le PDF",
                f,
                file_name="gaitscan_cinematique.pdf",
                mime="application/pdf"
            )
# =========================================================
# PARTIE 4/4 – SPINESCAN 3D + POSTURE PHOTO
# =========================================================

def run_spinescan_3d():

    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    import tempfile, os
    from datetime import datetime

    from plyfile import PlyData
    from scipy.signal import savgol_filter

    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors

    st.header("🦴 SpineScan Pro 3D")
    st.caption("Analyse rachidienne 3D – axe – inclinaisons")

    ply_file = st.file_uploader("Scan 3D du rachis (.ply)", type=["ply"])

    if not ply_file:
        st.info("Veuillez importer un fichier .PLY")
        return

    if not st.button("⚙️ Lancer l’analyse 3D"):
        return

    # -----------------------------------------------------
    # LECTURE PLY
    # -----------------------------------------------------
    ply = PlyData.read(ply_file)
    x = np.array(ply["vertex"]["x"])
    y = np.array(ply["vertex"]["y"])
    z = np.array(ply["vertex"]["z"])

    # -----------------------------------------------------
    # AXE RACHIDIEN (approximation)
    # -----------------------------------------------------
    z_sorted_idx = np.argsort(z)
    x, y, z = x[z_sorted_idx], y[z_sorted_idx], z[z_sorted_idx]

    x_smooth = savgol_filter(x, 31, 3)
    y_smooth = savgol_filter(y, 31, 3)

    angles_front = np.degrees(np.arctan2(np.gradient(x_smooth), np.gradient(z)))
    angles_sag = np.degrees(np.arctan2(np.gradient(y_smooth), np.gradient(z)))

    # -----------------------------------------------------
    # AFFICHAGE
    # -----------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.plot(z, angles_front)
        ax.set_title("Inclinaison frontale rachidienne (°)")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(z, angles_sag)
        ax.set_title("Inclinaison sagittale rachidienne (°)")
        st.pyplot(fig)

    # -----------------------------------------------------
    # EXPORT PDF
    # -----------------------------------------------------
    st.divider()
    st.subheader("📄 Export PDF")

    if st.button("📥 Générer le rapport SpineScan"):

        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>SpineScan Pro 3D</b>", styles["Title"]))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph(
            f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.5 * cm))

        table_data = [
            ["Mesure", "Valeur moyenne (°)"],
            ["Inclinaison frontale", f"{np.mean(np.abs(angles_front)):.2f}"],
            ["Inclinaison sagittale", f"{np.mean(np.abs(angles_sag)):.2f}"],
        ]

        table = Table(table_data, colWidths=[7 * cm, 4 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))

        story.append(table)
        doc.build(story)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Télécharger le PDF",
                f,
                file_name="spinescan_3d.pdf",
                mime="application/pdf"
            )


# =========================================================
# ANALYSE POSTURALE PHOTO
# =========================================================

def run_posture_photo():

    import streamlit as st
    import tensorflow as tf
    import tensorflow_hub as hub
    import numpy as np
    import tempfile
    from datetime import datetime
    from PIL import Image

    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors

    st.header("🧍 Analyse Posturale – Photo Frontale")
    st.caption("Épaules – bassin – genoux")

    @st.cache_resource
    def load_movenet():
        return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

    movenet = load_movenet()

    def detect_pose(img):
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
        outputs = movenet.signatures["serving_default"](tf.cast(img, tf.int32))
        return outputs["output_0"][0, 0, :, :].numpy()

    photo = st.camera_input("Photo frontale debout")

    if not photo:
        st.info("Prenez une photo frontale.")
        return

    if not st.button("⚙️ Lancer l’analyse posturale"):
        return

    image = np.array(Image.open(photo).convert("RGB"))
    kp = detect_pose(image)

    LS, RS = kp[5][:2], kp[6][:2]
    LH, RH = kp[11][:2], kp[12][:2]
    LK, RK = kp[13][:2], kp[14][:2]

    shoulder_angle = np.degrees(np.arctan2(RS[1] - LS[1], RS[0] - LS[0]))
    pelvis_angle = np.degrees(np.arctan2(RH[1] - LH[1], RH[0] - LH[0]))
    knee_diff = np.linalg.norm(LK - RK)

    st.success("Analyse terminée")

    st.write(f"🔹 Inclinaison épaules : **{shoulder_angle:.2f}°**")
    st.write(f"🔹 Inclinaison bassin : **{pelvis_angle:.2f}°**")
    st.write(f"🔹 Asymétrie genoux (px) : **{knee_diff:.1f}**")

    # -----------------------------------------------------
    # EXPORT PDF
    # -----------------------------------------------------
    st.divider()
    st.subheader("📄 Export PDF")

    if st.button("📥 Générer le rapport postural"):

        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>Analyse Posturale – Photo</b>", styles["Title"]))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph(
            f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.5 * cm))

        table_data = [
            ["Mesure", "Valeur"],
            ["Inclinaison épaules (°)", f"{shoulder_angle:.2f}"],
            ["Inclinaison bassin (°)", f"{pelvis_angle:.2f}"],
            ["Asymétrie genoux (px)", f"{knee_diff:.1f}"],
        ]

        table = Table(table_data, colWidths=[7 * cm, 4 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))

        story.append(table)
        doc.build(story)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Télécharger le PDF",
                f,
                file_name="analyse_posturale.pdf",
                mime="application/pdf"
            )
