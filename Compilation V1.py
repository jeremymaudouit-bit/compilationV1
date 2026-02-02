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
# PARTIE 2/4 – GAITSCAN FRONTAL COMPLET
# =========================================================

#st.set_page_config(page_title="GaitScan Pro - Frontal", layout="wide")
st.title("🏃 GaitScan Pro - Analyse Frontale")
st.subheader("Abduction/adduction et posture frontale")

# ==============================
# CHARGEMENT MOVE NET
# ==============================
@st.cache_resource
def load_movenet():
    return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

movenet = load_movenet()

def detect_pose(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
    input_img = tf.cast(img, dtype=tf.int32)
    outputs = movenet.signatures['serving_default'](input_img)
    keypoints = outputs['output_0'].numpy()
    return keypoints[0,0,:,:]

# ==============================
# ARTICULATIONS
# ==============================
JOINTS_IDX = {
    "Hanche G": 11, "Genou G": 13, "Cheville G": 15,
    "Hanche D": 12, "Genou D": 14, "Cheville D": 16,
    "Epaule G": 5, "Epaule D": 6
}

def angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

# ==============================
# TRAITEMENT VIDEO / CAMERA (FRONTAL)
# ==============================
def process_video_frontal(video_file, frame_skip=2):
    cap = cv2.VideoCapture(video_file)
    results = {joint: [] for joint in ["Hanche G","Genou G","Cheville G","Hanche D","Genou D","Cheville D","Tronc","Pelvis"]}
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % frame_skip == 0:
            kp = detect_pose(frame)
            # Hanche abduction/adduction
            results["Hanche G"].append(angle(kp[JOINTS_IDX["Epaule G"],:2], kp[JOINTS_IDX["Hanche G"],:2], kp[JOINTS_IDX["Genou G"],:2]))
            results["Hanche D"].append(angle(kp[JOINTS_IDX["Epaule D"],:2], kp[JOINTS_IDX["Hanche D"],:2], kp[JOINTS_IDX["Genou D"],:2]))
            # Genou valgus/varus
            results["Genou G"].append(angle(kp[JOINTS_IDX["Hanche G"],:2], kp[JOINTS_IDX["Genou G"],:2], kp[JOINTS_IDX["Cheville G"],:2]))
            results["Genou D"].append(angle(kp[JOINTS_IDX["Hanche D"],:2], kp[JOINTS_IDX["Genou D"],:2], kp[JOINTS_IDX["Cheville D"],:2]))
            # Cheville inversion/éversion
            results["Cheville G"].append(angle(kp[JOINTS_IDX["Genou G"],:2], kp[JOINTS_IDX["Cheville G"],:2], kp[JOINTS_IDX["Cheville G"],:2]+np.array([1,0])))
            results["Cheville D"].append(angle(kp[JOINTS_IDX["Genou D"],:2], kp[JOINTS_IDX["Cheville D"],:2], kp[JOINTS_IDX["Cheville D"],:2]+np.array([1,0])))
            # Tronc (inclinaison latérale)
            results["Tronc"].append(angle(kp[JOINTS_IDX["Epaule G"],:2], (kp[JOINTS_IDX["Hanche G"],:2]+kp[JOINTS_IDX["Hanche D"],:2])/2, kp[JOINTS_IDX["Epaule D"],:2]))
            # Pelvis rotation
            pelvis_angle = np.degrees(np.arctan2(kp[JOINTS_IDX["Hanche D"],1]-kp[JOINTS_IDX["Hanche G"],1],
                                                kp[JOINTS_IDX["Hanche D"],0]-kp[JOINTS_IDX["Hanche G"],0]))
            results["Pelvis"].append(pelvis_angle)
        frame_idx +=1
    cap.release()
    return results

# ==============================
# MODÈLE NORMAL FRONTAL (courbes plus lisses)
# ==============================
def normal_hip_frontal(length=100, sigma=5):
    cycle_percent = np.array([0, 15, 45, 60, 75, 100])
    angles = np.array([-5, -5, -10, 0, 5, 0])
    x = np.linspace(0, 100, length)
    curve = np.interp(x, cycle_percent, angles)
    return gaussian_filter1d(curve, sigma=sigma)

def normal_knee_frontal(length=100, sigma=5):
    cycle_percent = np.array([0, 60, 100])
    angles = np.array([-3.5, -3.5, 0])
    x = np.linspace(0, 100, length)
    curve = np.interp(x, cycle_percent, angles)
    return gaussian_filter1d(curve, sigma=sigma)

def normal_ankle_frontal(length=100, sigma=5):
    cycle_percent = np.array([0, 5, 15, 45, 60, 80, 100])
    angles = np.array([3, -5, -5, 0, 5, 2, 0])
    x = np.linspace(0, 100, length)
    curve = np.interp(x, cycle_percent, angles)
    return gaussian_filter1d(curve, sigma=sigma)

def normal_pelvis(length=100, sigma=5):
    t = np.linspace(0, 1, length)
    curve = 5*np.sin(2*np.pi*t)
    return gaussian_filter1d(curve, sigma=sigma)

# ==============================
# EXPORT PDF
# ==============================
def export_pdf(patient_info, joint_images, summary_table):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "rapport_analyse_frontal.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>Bilan Analyse Frontale</b>", styles['Title']),
        Paragraph(f"Patient : {patient_info['nom']} {patient_info['prenom']}", styles['Normal']),
        Paragraph(f"Date : {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']),
        Spacer(1,1*cm)
    ]
    for joint, img_path in joint_images.items():
        story.append(Paragraph(f"<b>{joint}</b>", styles['Heading2']))
        story.append(PDFImage(img_path, width=15*cm, height=6*cm))
        story.append(Spacer(1,0.5*cm))
    story.append(Paragraph("<b>Résumé des angles (°)</b>", styles['Heading2']))
    table_data = [["Articulation", "Min", "Moyenne", "Max"]] + summary_table
    table = Table(table_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(table)
    doc.build(story)
    return path

# ==============================
# INTERFACE
# ==============================
with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    st.subheader("📹 Source")
    video_file = st.file_uploader("Charger une vidéo", type=["mp4","mov","avi"])
    live_cam = st.checkbox("Ou utiliser la caméra live")
    st.subheader("⚙️ Paramètres")
    smoothing = st.slider("Lissage des courbes", 0, 10, 2)
    show_normal = st.checkbox("Afficher modèle normal à côté", value=True)

# ==============================
# ANALYSE
# ==============================
video_ready = False
if live_cam:
    cam_file = st.camera_input("🎥 Caméra")
    if cam_file:
        video_file = cam_file
        video_ready = True
elif video_file:
    video_ready = True

if video_ready and st.button("⚙️ Lancer l'analyse"):
    with st.spinner("Analyse en cours..."):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        results = process_video_frontal(tfile.name, frame_skip=2)
        os.unlink(tfile.name)

        joint_imgs = {}
        summary_table = []

        articulation_pairs = [("Hanche G","Hanche D"), ("Genou G","Genou D"), ("Cheville G","Cheville D")]
        normal_funcs = [normal_hip_frontal, normal_knee_frontal, normal_ankle_frontal]

        for (joint_pair, normal_func) in zip(articulation_pairs, normal_funcs):
            col1, col2 = st.columns(2)

            # Colonne 1 : réel
            fig, ax = plt.subplots(figsize=(6,4))
            for joint, color in zip(joint_pair, ['red','blue']):
                angles_smooth = gaussian_filter1d(results[joint], sigma=smoothing)
                ax.plot(angles_smooth, lw=2, color=color, label=joint)
                summary_table.append([joint, f"{np.min(results[joint]):.1f}", f"{np.mean(results[joint]):.1f}", f"{np.max(results[joint]):.1f}"])
            ax.set_title(f"{joint_pair[0].split()[0]} : Réel")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Angle (°)")
            ax.legend()
            col1.pyplot(fig)
            img_path = os.path.join(tempfile.gettempdir(), f"{joint_pair[0]}_reel.png")
            fig.savefig(img_path, bbox_inches='tight')
            plt.close(fig)
            joint_imgs[f"{joint_pair[0]} & {joint_pair[1]} Réel"] = img_path

            # Colonne 2 : modèle normal
            if show_normal:
                fig2, ax2 = plt.subplots(figsize=(6,4))
                length = len(results[joint_pair[0]])
                normal_curve = normal_func(length, sigma=smoothing*2)  # courbes normales plus lisses
                ax2.plot(normal_curve, lw=2, color='green', label="Modèle normal")
                ax2.set_title(f"{joint_pair[0].split()[0]} : Modèle normal")
                ax2.set_xlabel("Frame")
                ax2.set_ylabel("Angle (°)")
                ax2.legend()
                col2.pyplot(fig2)
                img_path2 = os.path.join(tempfile.gettempdir(), f"{joint_pair[0]}_normal.png")
                fig2.savefig(img_path2, bbox_inches='tight')
                plt.close(fig2)
                joint_imgs[f"{joint_pair[0]} & {joint_pair[1]} Normal"] = img_path2

        # Pelvis
        angles_smooth = gaussian_filter1d(results["Pelvis"], sigma=smoothing)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(angles_smooth, lw=2, color='purple', label="Pelvis réel")
        if show_normal:
            normal_curve = normal_pelvis(len(angles_smooth), sigma=smoothing*2)
            ax.plot(normal_curve, lw=2, color='green', linestyle='--', label="Pelvis modèle")
        ax.set_title("Bascule Pelvis")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (°)")
        ax.legend()
        st.pyplot(fig)
        img_path = os.path.join(tempfile.gettempdir(), "Pelvis.png")
        fig.savefig(img_path, bbox_inches='tight')
        plt.close(fig)
        joint_imgs["Pelvis"] = img_path
        summary_table.append(["Pelvis", f"{np.min(results['Pelvis']):.1f}", f"{np.mean(results['Pelvis']):.1f}", f"{np.max(results['Pelvis']):.1f}"])

        # Tronc
        angles_smooth = gaussian_filter1d(results["Tronc"], sigma=smoothing)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(angles_smooth, lw=2, color='green', label="Tronc")
        ax.set_title("Tronc (inclinaison latérale)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (°)")
        ax.legend()
        st.pyplot(fig)
        img_path = os.path.join(tempfile.gettempdir(), f"Tronc.png")
        fig.savefig(img_path, bbox_inches='tight')
        plt.close(fig)
        joint_imgs["Tronc"] = img_path
        summary_table.append(["Tronc", f"{np.min(results['Tronc']):.1f}", f"{np.mean(results['Tronc']):.1f}", f"{np.max(results['Tronc']):.1f}"])

        # Export PDF
        pdf_path = export_pdf({"nom": nom, "prenom": prenom}, joint_imgs, summary_table)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Télécharger le rapport PDF", f, f"Analyse_Frontale_{nom}.pdf")

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

# =========================================================
# ROUTEUR FINAL (APRES TOUTES LES DEFINITIONS)
# =========================================================

if APP_MODE == "🏃 GaitScan – Analyse Frontale":
    run_gaitscan_frontal()

elif APP_MODE == "🏃 GaitScan – Analyse Cinématique":
    run_gaitscan_cinematique()

elif APP_MODE == "🦴 SpineScan Pro 3D":
    run_spinescan_3d()

elif APP_MODE == "🧍 Analyse Posturale (Photo)":
    run_posture_photo()


