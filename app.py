import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
import streamlit as st
from torchvision import transforms

from train import FaceIDNet, MEAN, STD

st.set_page_config(page_title="FaceID Pro", page_icon="◈", layout="wide")
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "best_model.pt"
META_PATH = ARTIFACTS / "metadata.json"

st.markdown("""
<style>
.stApp{background:#07090d}.block-container{max-width:1400px;padding-top:1.4rem}
[data-testid="stSidebar"]{background:#0b0e14}
.hero{padding:30px;border:1px solid #202938;border-radius:24px;background:linear-gradient(135deg,#101722,#0b1018);margin-bottom:20px}
.brand{font-size:13px;font-weight:800;letter-spacing:.18em;color:#9ca9ba}
.hero h1{font-size:42px;line-height:1.05;margin:10px 0}.hero p{color:#aab4c3;max-width:760px}
.card{background:#0c1119;border:1px solid #202938;border-radius:18px;padding:18px;min-height:105px}
.label{font-size:11px;color:#7f8b9d;text-transform:uppercase;letter-spacing:.12em}.value{font-size:23px;font-weight:800;margin-top:7px}
.result{background:#0d1412;border:1px solid #24533c;border-radius:20px;padding:22px}
.row{display:flex;justify-content:space-between;padding:10px 12px;border:1px solid #202938;border-radius:10px;margin:6px 0}
.small{font-size:12px;color:#7f8b9d}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, [], "Train the model first. artifacts/ is missing."
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    classes = meta.get("classes", [])
    if len(classes) < 2:
        return None, [], "metadata.json contains fewer than two identities."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceIDNet(len(classes)).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except Exception as exc:
        return None, [], f"Checkpoint error: {exc}"
    model.eval()
    return model, classes, None


model, classes, error = load()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with st.sidebar:
    st.markdown("## ◈ FaceID Pro")
    st.caption("LFW-based identity classification prototype")
    threshold = st.slider("Unknown threshold", 0.50, 0.99, 0.80, 0.01)
    topk = st.selectbox("Top-K candidates", [3, 5, 10], index=1)
    st.divider()
    st.write(f"**Runtime:** {'CUDA GPU' if device.type == 'cuda' else 'CPU'}")
    st.write(f"**Identities:** {len(classes)}")
    st.write("**Backbone:** ResNet-50")
    st.divider()
    st.caption("Research/educational prototype. Do not treat softmax confidence as calibrated biometric probability.")

st.markdown("""
<div class="hero">
<div class="brand">FACEID PRO · LFW BENCHMARK</div>
<h1>Identity intelligence,<br>with an auditable pipeline.</h1>
<p>ResNet-50 transfer learning, controlled LFW subset, ranked identity candidates, and explicit unknown rejection.</p>
</div>
""", unsafe_allow_html=True)

if error:
    st.warning(error)

cols = st.columns(4)
for col, label, value in zip(
    cols,
    ["Dataset", "Identities", "Architecture", "Runtime"],
    ["LFW", str(len(classes)), "ResNet-50", "CUDA" if device.type == "cuda" else "CPU"],
):
    with col:
        st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

left, right = st.columns([1.05, .95], gap="large")
with left:
    st.markdown("#### Input")
    uploaded = st.file_uploader("Upload a face image", type=["jpg","jpeg","png","webp"])
    image = None
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)
        except (UnidentifiedImageError, OSError):
            st.error("Unreadable image.")

with right:
    st.markdown("#### Result")
    if model is None:
        st.markdown('<div class="result"><div class="small">STATUS</div><h2>Model not trained</h2><p class="small">Run prepare_lfw.py and train.py first.</p></div>', unsafe_allow_html=True)
    elif image is None:
        st.markdown('<div class="result"><div class="small">STATUS</div><h2>Ready for input</h2><p class="small">Upload an image to run the classifier.</p></div>', unsafe_allow_html=True)
    else:
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)
        ])
        with torch.inference_mode():
            logits = model(tf(image).unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)[0]
            k = min(topk, len(classes))
            values, indices = torch.topk(probs, k)
        ranked = [(classes[int(i)], float(v)) for v, i in zip(values, indices)]
        name, score = ranked[0]

        if score >= threshold:
            st.markdown(f'<div class="result"><div class="small">PRIMARY MATCH</div><h2>{name}</h2><b>Model confidence: {score*100:.2f}%</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result"><div class="small">DECISION</div><h2>Unknown / low confidence</h2><b>No candidate passed the configured threshold.</b></div>', unsafe_allow_html=True)

        st.markdown("##### Ranked candidates")
        for name, score in ranked:
            st.markdown(f'<div class="row"><span>{name}</span><b>{score*100:.2f}%</b></div>', unsafe_allow_html=True)

st.divider()
st.markdown("#### Evaluation artifacts")
st.caption("Training produces a best checkpoint, metadata, confusion matrix and per-class classification report. For a true biometric verification study, use LFW's prescribed pair-based evaluation rather than only closed-set classification.")
