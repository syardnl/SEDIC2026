import streamlit as st
import cv2, tempfile
import numpy as np
from PIL import Image
from detector   import GuardianDetector
from log_writer import DetectionLogger

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Guardian — MDA",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark navy + cyan maritime theme) ──────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #050d1a;
    color: #c9d6e3;
    font-family: 'Courier New', monospace;
}
[data-testid="stSidebar"] {
    background-color: #070f1f;
    border-right: 1px solid #0d3a5c;
}
.guardian-header {
    background: linear-gradient(90deg, #020b18 0%, #051e3e 50%, #020b18 100%);
    border: 1px solid #0a4a7a;
    border-radius: 8px;
    padding: 18px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.guardian-header h1 {
    color: #00d4ff;
    font-size: 1.8rem;
    margin: 0;
    letter-spacing: 3px;
    text-transform: uppercase;
    text-shadow: 0 0 20px #00d4ff88;
}
.guardian-header p {
    color: #5b8db8;
    margin: 2px 0 0 0;
    font-size: 0.75rem;
    letter-spacing: 2px;
}
h2, h3 {
    color: #00d4ff !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid #0d3a5c;
    padding-bottom: 6px;
}
.metric-card {
    background: #070f1f;
    border: 1px solid #0d3a5c;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
    text-shadow: 0 0 12px #00d4ff66;
}
.metric-card .label {
    font-size: 0.7rem;
    color: #5b8db8;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}
.det-row {
    background: #070f1f;
    border: 1px solid #0d3a5c;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.det-row:hover { border-color: #00d4ff55; }
.det-name { color: #c9d6e3; font-size: 0.9rem; letter-spacing: 1px; }
.det-conf { color: #00d4ff; font-weight: 700; font-size: 0.9rem; }
.military-alert {
    background: linear-gradient(135deg, #1a0000, #2d0000);
    border: 2px solid #ff2222;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    animation: pulse-border 1.5s infinite;
    text-align: center;
}
.military-alert .alert-title {
    color: #ff4444;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-shadow: 0 0 10px #ff000088;
}
.military-alert .alert-body {
    color: #ffaaaa;
    font-size: 0.85rem;
    margin-top: 6px;
    letter-spacing: 1px;
}
@keyframes pulse-border {
    0%   { box-shadow: 0 0 0px #ff2222; }
    50%  { box-shadow: 0 0 16px #ff222288; }
    100% { box-shadow: 0 0 0px #ff2222; }
}
.local-alert {
    background: linear-gradient(135deg, #1a0800, #2d1400);
    border: 2px solid #ff8800;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    text-align: center;
}
.local-alert .alert-title {
    color: #ff9900;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 3px;
}
.local-alert .alert-body {
    color: #ffcc88;
    font-size: 0.85rem;
    margin-top: 6px;
    letter-spacing: 1px;
}
[data-testid="stImage"] img {
    border: 1px solid #0d3a5c;
    border-radius: 8px;
}
.stButton > button {
    background: #051e3e;
    color: #00d4ff;
    border: 1px solid #00d4ff55;
    border-radius: 6px;
    letter-spacing: 2px;
    font-family: 'Courier New', monospace;
}
.stButton > button:hover {
    background: #0a3a6e;
    border-color: #00d4ff;
    color: #ffffff;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #003d66, #00d4ff) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label {
    color: #5b8db8 !important;
    letter-spacing: 1px;
    font-size: 0.8rem;
}
hr { border-color: #0d3a5c !important; }
.no-det {
    color: #5b8db8;
    font-size: 0.85rem;
    letter-spacing: 1px;
    padding: 12px;
    text-align: center;
    border: 1px dashed #0d3a5c;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="guardian-header">
    <div style="font-size:2rem">🛡️</div>
    <div>
        <h1>Project Guardian</h1>
        <p>MARITIME DOMAIN AWARENESS SYSTEM &nbsp;|&nbsp; SEDIC 2026</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ SYSTEM CONFIG")
    st.divider()
    model_path  = st.text_input("Model path", value="models/guardian.pt")
    conf_thresh = st.slider("Confidence threshold", 0.01, 0.95, 0.25, 0.01)
    show_debug  = st.toggle("Show diagnostics", value=False)
    st.divider()
    st.markdown("### 📡 INPUT MODE")
    mode = st.radio("", ["Image", "Video", "Qualifier Video"], label_visibility="collapsed")
    st.divider()
    st.markdown(
        '<div style="color:#00ff88;font-size:0.7rem;letter-spacing:1px;">'
        '● SYSTEM ONLINE<br>'
        '<span style="color:#5b8db8">● MODEL LOADED<br>● LOG READY</span></div>',
        unsafe_allow_html=True
    )

# ── Load model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path, conf):
    return GuardianDetector(path, conf)

try:
    detector = load_model(model_path, conf_thresh)
except Exception as e:
    st.error(f"⚠ Model failed to load: {e}")
    st.stop()

# ── Helpers ───────────────────────────────────────────────────────────────
BADGE_STYLES = {
    "HIGH PRIORITY": "background:#ff2222;color:#fff",
    "PRIORITY":      "background:#ff8800;color:#000",
    "MONITOR":       "background:#cc8800;color:#000",
    "SMALL CRAFT":   "background:#ccaa00;color:#000",
    "CIVILIAN":      "background:#00aa55;color:#fff",
}

def threat_badge(level: str) -> str:
    style = BADGE_STYLES.get(level, "background:#333;color:#aaa")
    return f'<span class="badge" style="{style}">{level}</span>'

def military_alert(detections: list):
    foreign = [d for d in detections if d["class_name"] == "foreign_military_ship"]
    local   = [d for d in detections if d["class_name"] == "local_military_ship"]
    if foreign:
        conf = foreign[0]["confidence"]
        st.markdown(f"""
        <div class="military-alert">
            <div class="alert-title">⚠ HIGH PRIORITY ALERT ⚠</div>
            <div class="alert-body">
                FOREIGN MILITARY VESSEL DETECTED<br>
                Confidence: {conf:.0%} &nbsp;|&nbsp; Threat Level: HIGH PRIORITY<br>
                Immediate action required — notify duty officer
            </div>
        </div>
        """, unsafe_allow_html=True)
    if local:
        conf = local[0]["confidence"]
        st.markdown(f"""
        <div class="local-alert">
            <div class="alert-title">⚡ PRIORITY ALERT</div>
            <div class="alert-body">
                LOCAL MILITARY VESSEL DETECTED<br>
                Confidence: {conf:.0%} &nbsp;|&nbsp; Threat Level: PRIORITY<br>
                Log and monitor — report to command
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_detections(detections: list):
    if not detections:
        st.markdown('<div class="no-det">— NO VESSELS DETECTED —</div>', unsafe_allow_html=True)
        return
    for d in detections:
        name  = d["class_name"].replace("_", " ").upper()
        badge = threat_badge(d["threat_level"])
        conf  = f"{d['confidence']:.0%}"
        st.markdown(f"""
        <div class="det-row">
            <span class="det-name">⬡ {name}</span>
            {badge}
            <span class="det-conf">{conf}</span>
        </div>
        """, unsafe_allow_html=True)

def render_metrics(detections: list):
    total    = len(detections)
    military = sum(1 for d in detections if "military" in d["class_name"])
    civilian = sum(1 for d in detections if d.get("threat_level") == "CIVILIAN")
    threat   = sum(1 for d in detections if d.get("threat_level") in ("HIGH PRIORITY", "PRIORITY"))
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, total,    "TOTAL VESSELS"),
        (c2, military, "MILITARY"),
        (c3, civilian, "CIVILIAN"),
        (c4, threat,   "THREATS"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

def render_diagnostics(frame, detections):
    if not show_debug:
        return
    with st.expander("🔬 Model Diagnostics"):
        st.write("**Model:**", model_path)
        st.write("**Threshold:**", conf_thresh)
        st.write("**Image shape:**", frame.shape)
        st.write("**Classes:**", list(detector.model.names.values()))
        if detections:
            st.dataframe(
                [{"class": d["class_name"], "conf": d["confidence"],
                  "threat": d["threat_level"], "bbox": d["bbox"]}
                 for d in detections], use_container_width=True)
        else:
            probe = detector.predict(frame, conf=0.01)
            st.warning(f"No detections at {conf_thresh:.2f}. Probe found {len(probe)} candidate(s).")
            if probe:
                st.dataframe(
                    [{"class": d["class_name"], "conf": d["confidence"]} for d in probe],
                    use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODE: IMAGE
# ═══════════════════════════════════════════════════════════════════════════
if mode == "Image":
    st.markdown("## 📷 IMAGE DETECTION")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")
    if uploaded:
        img       = np.array(Image.open(uploaded).convert("RGB"))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        detections    = detector.predict(frame_bgr)
        annotated     = detector.annotate(frame_bgr, detections)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        military_alert(detections)
        render_metrics(detections)

        col_img, col_det = st.columns([3, 2])
        with col_img:
            st.image(annotated_rgb, use_container_width=True)
        with col_det:
            st.markdown("### 🎯 DETECTIONS")
            render_detections(detections)
            render_diagnostics(frame_bgr, detections)

        logger = DetectionLogger("outputs/single_image_log.csv")
        logger.log(frame_id=0, detections=detections)
        logger.close()
        st.divider()
        with open("outputs/single_image_log.csv") as f:
            st.download_button("⬇ DOWNLOAD DETECTION LOG", f, "detection_log.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════
# MODE: VIDEO
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "Video":
    st.markdown("## 🎥 VIDEO DETECTION")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"],
                                label_visibility="collapsed")
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())

        cap   = cv2.VideoCapture(tfile.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        skip  = max(1, int(fps // 5))

        col_img, col_det = st.columns([3, 2])
        st_frame  = col_img.empty()
        st_alert  = col_det.empty()
        st_det    = col_det.empty()
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
                               use_container_width=True)

                alert_html = ""
                if any(d["class_name"] == "foreign_military_ship" for d in detections):
                    alert_html = '<div class="military-alert"><div class="alert-title">⚠ FOREIGN MILITARY DETECTED ⚠</div></div>'
                elif any(d["class_name"] == "local_military_ship" for d in detections):
                    alert_html = '<div class="local-alert"><div class="alert-title">⚡ LOCAL MILITARY DETECTED</div></div>'

                with st_alert.container():
                    if alert_html:
                        st.markdown(alert_html, unsafe_allow_html=True)
                with st_det.container():
                    render_detections(detections)

                progress.progress(min(frame_id / max(total, 1), 1.0))
            frame_id += 1

        cap.release()
        logger.close()
        st.success(f"✅ Processed {frame_id} frames.")
        with open("outputs/video_log.csv") as f:
            st.download_button("⬇ DOWNLOAD DETECTION LOG", f, "video_log.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════
# MODE: QUALIFIER VIDEO
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "Qualifier Video":
    st.markdown("## 📋 QUALIFIER VIDEO RUN")
    st.markdown(
        '<div style="color:#5b8db8;font-size:0.8rem;letter-spacing:1px;margin-bottom:12px">'
        'Processes every frame and generates the official submission log.</div>',
        unsafe_allow_html=True
    )
    qualifier_path = st.text_input("Video path", value="data/qualifier_clip.mp4")

    if st.button("▶ RUN AND GENERATE LOG"):
        cap = cv2.VideoCapture(qualifier_path)
        if not cap.isOpened():
            st.error(f"⚠ Cannot open video: {qualifier_path}")
            st.stop()

        logger    = DetectionLogger("outputs/qualifier_detection_log.csv")
        bar       = st.progress(0)
        total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st_frame  = st.empty()
        st_info   = st.empty()
        fid       = 0
        mil_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.predict(frame)
            annotated  = detector.annotate(frame, detections)
            logger.log(fid, detections)

            if any("military" in d["class_name"] for d in detections):
                mil_count += 1

            st_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                           use_container_width=True)
            st_info.markdown(
                f'<div style="color:#5b8db8;font-size:0.75rem;letter-spacing:1px">'
                f'Frame {fid} / {total} &nbsp;|&nbsp; '
                f'Military detections: <span style="color:#ff4444">{mil_count}</span></div>',
                unsafe_allow_html=True
            )
            bar.progress(min(fid / max(total, 1), 1.0))
            fid += 1

        cap.release()
        logger.close()

        st.markdown(f"""
        <div style="background:#070f1f;border:1px solid #00d4ff44;border-radius:8px;
                    padding:16px;margin-top:12px">
            <div style="color:#00d4ff;letter-spacing:2px;font-size:0.9rem">✅ LOG GENERATED</div>
            <div style="color:#5b8db8;font-size:0.8rem;margin-top:6px">
                Frames processed: {fid} &nbsp;|&nbsp;
                Military detections: <span style="color:#ff4444">{mil_count}</span> frames<br>
                Output: outputs/qualifier_detection_log.csv
            </div>
        </div>
        """, unsafe_allow_html=True)

        with open("outputs/qualifier_detection_log.csv") as f:
            st.download_button("⬇ DOWNLOAD QUALIFIER LOG", f,
                               "qualifier_detection_log.csv", "text/csv")