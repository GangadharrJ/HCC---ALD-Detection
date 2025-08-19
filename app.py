import streamlit as st
import cv2
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image

# Detection function
def detect_hcc_ald(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    flat = blur.reshape(-1, 1)
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(flat)
    pca_image = pca_result.reshape(blur.shape)
    pca_image_norm = cv2.normalize(pca_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    edges = cv2.Canny(pca_image_norm, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    hcc_regions = ald_regions = tumor_regions = nontumor_regions = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = pca_image_norm[y:y + h, x:x + w]
            mean_val = np.mean(roi)

            label, color = "", (255, 255, 255)
            if mean_val > 160 and area > 600:
                label, color = "Tumor", (255, 0, 255)
                tumor_regions += 1
            elif mean_val > 140:
                label, color = "HCC", (0, 0, 255)
                hcc_regions += 1
            elif 100 < mean_val <= 140:
                label, color = "ALD", (0, 255, 0)
                ald_regions += 1
            else:
                label, color = "Non-Tumor", (0, 255, 255)
                nontumor_regions += 1

            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return result, (hcc_regions, ald_regions, tumor_regions, nontumor_regions)

# Streamlit UI
st.title("🧪 HCC & ALD Detection in Liver CT Images")

uploaded_file = st.file_uploader("Upload CT Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    result_img, (hcc, ald, tumor, nontumor) = detect_hcc_ald(image)

    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Processed Result")

    st.subheader("📋 Detection Report")
    if tumor:
        st.error("🟣 Tumor detected! Consult an oncologist urgently.")
    if hcc:
        st.warning("🔴 HCC regions detected. Visit a hepatologist.")
    if ald:
        st.info("🟢 ALD signs detected. Stop alcohol immediately.")
    if not (tumor or hcc or ald):
        st.success("✅ No Tumor, HCC, or ALD detected.")
