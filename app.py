import streamlit as st
import cv2, tempfile, time
import numpy as np
from PIL import Image
from detector    import GuardianDetector
from log_writer  import DetectionLogger
from claude_layer import summarise

st.set_page_config(page_title="Project Guardian", layout="wide", page_icon="🛡️")

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Project Guardian")
    st.caption("Maritime Domain Awareness")
    st.divider()
    model_path = st.text_input("Model path", value="models/guardian.pt")
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.95, 0.4, 0.05)
    use_claude  = st.toggle("Enable threat narrative (Claude API)", value=False)
    st.divider()
    mode = st.radio("Input mode", ["Image", "Video", "Qualifier Video"])

# ── Load model (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_model(path, conf):
    return GuardianDetector(path, conf)

try:
    detector = load_model(model_path, conf_thresh)
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

# ── Shared helpers ────────────────────────────────────────────────────────
def threat_badge(level: str):
    colour = {"HIGH PRIORITY": "red", "PRIORITY": "orange",
              "SMALL CRAFT": "yellow", "CIVILIAN": "green"}.get(level, "gray")
    return f'<span style="background:{colour};color:#000;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600">{level}</span>'

def render_detections(detections: list[dict]):
    if not detections:
        st.info("No vessels detected.")
        return
    for d in detections:
        col1, col2, col3 = st.columns([3, 2, 1])
        col1.write(d["class_name"].replace("_", " ").title())
        col2.markdown(threat_badge(d["threat_level"]), unsafe_allow_html=True)
        col3.write(f"{d['confidence']:.0%}")

# ═══════════════════════════════════════════════════════════════════════════
# MODE: IMAGE
# ═══════════════════════════════════════════════════════════════════════════
if mode == "Image":
    st.header("Image detection")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = np.array(Image.open(uploaded).convert("RGB"))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        detections = detector.predict(frame_bgr)
        annotated  = detector.annotate(frame_bgr, detections)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col_img, col_det = st.columns([3, 2])
        with col_img:
            st.image(annotated_rgb, use_container_width=True, caption="Detection result")
        with col_det:
            st.subheader("Detections")
            render_detections(detections)

            if use_claude and detections:
                with st.spinner("Generating threat narrative…"):
                    narrative = summarise(detections)
                if narrative:
                    st.info(f"**Threat assessment:** {narrative}")

        # Download log
        logger = DetectionLogger("outputs/single_image_log.csv")
        logger.log(frame_id=0, detections=detections)
        logger.close()
        with open("outputs/single_image_log.csv") as f:
            st.download_button("⬇ Download detection log", f, "detection_log.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════
# MODE: VIDEO
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "Video":
    st.header("Video detection")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())

        cap    = cv2.VideoCapture(tfile.name)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        skip   = max(1, int(fps // 5))   # process ~5 frames/sec equivalent

        st_frame  = st.empty()
        st_det    = st.empty()
        progress  = st.progress(0)
        logger    = DetectionLogger("outputs/video_log.csv")

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % skip == 0:
                detections = detector.predict(frame)
                annotated  = detector.annotate(frame, detections)
                logger.log(frame_id, detections)

                st_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)
                with st_det.container():
                    render_detections(detections)

                progress.progress(min(frame_id / max(total, 1), 1.0))
            frame_id += 1

        cap.release()
        logger.close()
        st.success(f"Processed {frame_id} frames.")
        with open("outputs/video_log.csv") as f:
            st.download_button("⬇ Download detection log", f, "video_log.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════
# MODE: QUALIFIER VIDEO (fixed path for submission script)
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "Qualifier Video":
    st.header("Qualifier video run")
    qualifier_path = st.text_input("Video path", value="data/qualifier_clip.mp4")
    if st.button("▶ Run and generate log"):
        cap    = cv2.VideoCapture(qualifier_path)
        logger = DetectionLogger("outputs/qualifier_detection_log.csv")
        bar    = st.progress(0)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fid    = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.predict(frame)
            logger.log(fid, detections)
            bar.progress(min(fid / max(total, 1), 1.0))
            fid += 1

        cap.release()
        logger.close()
        st.success(f"✅ Log saved → outputs/qualifier_detection_log.csv  ({fid} frames)")

        with open("outputs/qualifier_detection_log.csv") as f:
            st.download_button("⬇ Download qualifier log", f,
                               "qualifier_detection_log.csv", "text/csv")