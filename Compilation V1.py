import streamlit as st
import os
import tempfile
from datetime import datetime

# ============================
# MENU DE SÉLECTION
# ============================
st.set_page_config(page_title="Plateforme Biomécanique Pro", layout="wide")

analyse_type = st.sidebar.selectbox(
    "Choisissez le type d'analyse :",
    ["GaitScan Frontale", "GaitScan Cinématique", "SpineScan 3D", "Analyse Posturale Photo"]
)

# ============================
# SECTION: GaitScan Frontale
# ============================
if analyse_type == "GaitScan Frontale":
    # ===== Copié depuis le premier code =====
    import tensorflow as tf
    import tensorflow_hub as hub
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from scipy.ndimage import gaussian_filter1d

    st.title("🏃 GaitScan Pro - Analyse Frontale")
    st.subheader("Abduction/adduction et posture frontale")

    @st.cache_resource
def load_movenet():
        return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

    movenet = load_movenet()

    # ===== Le reste du code complet de GaitScan Frontale =====
    # Inclut detect_pose, process_video_frontal, normal_*, export_pdf, interface utilisateur

# ============================
# SECTION: GaitScan Cinématique
# ============================
elif analyse_type == "GaitScan Cinématique":
    # ===== Copié depuis le second code =====
    import tensorflow as tf
    import tensorflow_hub as hub
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    import tempfile, os

    st.title("🏃 GaitScan Pro - Analyse Cinématique")
    st.subheader("Flexion/extension des membres et posture du dos")

    @st.cache_resource
def load_movenet():
        return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

    movenet = load_movenet()

    # ===== Le reste du code complet de GaitScan Cinématique =====
    # Inclut detect_pose, process_video, normal_*, export_pdf, interface utilisateur

# ============================
# SECTION: SpineScan 3D
# ============================
elif analyse_type == "SpineScan 3D":
    # ===== Copié depuis le troisième code =====
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    from plyfile import PlyData
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.pagesizes import A4
    import tempfile, os

    st.title("🦴 SpineScan Pro 3D")

    # ===== Le reste du code complet de SpineScan 3D =====
    # Inclut load_ply_numpy, compute_cobb_angle, compute_sagittal_arrows, export_pdf_pro, interface utilisateur

# ============================
# SECTION: Analyse Posturale Photo
# ============================
elif analyse_type == "Analyse Posturale Photo":
    # ===== Copié depuis le quatrième code =====
    import numpy as np
    import cv2
    from PIL import Image
    import math
    from fpdf import FPDF
    import tensorflow as tf
    import tensorflow_hub as hub
    import os

    st.title("🧍 Analyseur Postural Pro")

    # ===== Le reste du code complet de Analyse Posturale Photo =====
    # Inclut preprocess, calculate_angle, tibia_vertical_angle, generate_pdf, interface utilisateur
