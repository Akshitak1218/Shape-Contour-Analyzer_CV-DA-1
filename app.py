import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Shape & Contour Analyzer",
    page_icon="📐",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS (STAND-OUT UI)
# --------------------------------------------------
st.markdown("""
<style>
.metric-box {
    background-color: #0f172a;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
.title-text {
    font-size: 40px;
    font-weight: bold;
    color: #22d3ee;
}
.subtitle {
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown('<p class="title-text">📐 Shape & Contour Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Interactive geometric shape detection & feature extraction dashboard</p>', unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("⚙️ Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

canny_low = st.sidebar.slider("Canny Edge - Low Threshold", 50, 150, 80)
canny_high = st.sidebar.slider("Canny Edge - High Threshold", 100, 300, 200)

epsilon_factor = st.sidebar.slider("Contour Approximation Accuracy", 1, 10, 2)
min_area = st.sidebar.slider("Minimum Object Area", 100, 5000, 300)

show_contours = st.sidebar.checkbox("Show Contours", True)
show_labels = st.sidebar.checkbox("Show Shape Labels", True)

# --------------------------------------------------
# IMAGE PROCESSING FUNCTION
# --------------------------------------------------
def analyze_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    data = []
    shape_count = {"Triangle":0, "Rectangle":0, "Circle":0, "Polygon":0}

    obj_id = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_factor * peri / 100, True)
        vertices = len(approx)

        if vertices == 3:
            shape = "Triangle"
            color = (255, 0, 0)
        elif vertices == 4:
            shape = "Rectangle"
            color = (0, 255, 0)
        else:
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity > 0.8:
                shape = "Circle"
                color = (0, 0, 255)
            else:
                shape = "Polygon"
                color = (255, 255, 0)

        shape_count[shape] += 1

        if show_contours:
            cv2.drawContours(output, [cnt], -1, color, 2)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        if show_labels:
            cv2.putText(
                output,
                f"{shape}",
                (cx - 30, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        data.append([obj_id, shape, round(area,2), round(peri,2), (cx, cy)])
        obj_id += 1

    df = pd.DataFrame(
        data,
        columns=["Object ID", "Shape", "Area (px²)", "Perimeter (px)", "Centroid"]
    )

    return edges, output, df, shape_count

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    edges, result_img, df, shape_count = analyze_image(image)

    # --------------------------------------------------
    # IMAGE DISPLAY
    # --------------------------------------------------
    st.subheader("Image Processing Pipeline")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        st.image(edges, caption="Edge Detection", use_column_width=True)

    with col3:
        st.image(result_img, caption="Detected Shapes", use_column_width=True)

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    st.subheader(" Detection Summary")

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Total Objects", len(df))
    m2.metric("Triangles", shape_count["Triangle"])
    m3.metric("Rectangles", shape_count["Rectangle"])
    m4.metric("Circles", shape_count["Circle"])
    m5.metric("Polygons", shape_count["Polygon"])

    # --------------------------------------------------
    # DATA TABLE
    # --------------------------------------------------
    st.subheader("📋 Shape Details")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Results as CSV",
        csv,
        "shape_analysis.csv",
        "text/csv"
    )

    # --------------------------------------------------
    # VISUAL ANALYTICS
    # --------------------------------------------------
    st.subheader("📈 Visual Analytics")

    fig1 = px.bar(
        x=list(shape_count.keys()),
        y=list(shape_count.values()),
        labels={"x": "Shape Type", "y": "Count"},
        title="Shape Distribution"
    )

    fig2 = px.pie(
        names=list(shape_count.keys()),
        values=list(shape_count.values()),
        title="Shape Percentage"
    )

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info(" Upload an image from the sidebar to begin analysis.")

