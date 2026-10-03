from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parent
required = [
    "app.py", "train.py", "prepare_lfw.py", "smoke_test.py",
    "requirements.txt", "README.md", "PROJECT_REPORT.md"
]
for f in required:
    assert (ROOT / f).exists(), f"Missing: {f}"
for f in ["app.py", "train.py", "prepare_lfw.py"]:
    ast.parse((ROOT / f).read_text(encoding="utf-8"))
    print(f"[OK] syntax: {f}")
req = (ROOT / "requirements.txt").read_text().lower()
for p in ["torch", "torchvision", "scikit-learn", "numpy", "pillow", "streamlit"]:
    assert p in req, f"Missing dependency: {p}"
print("[OK] dependency list")
print("[OK] FaceID Pro final smoke test passed")
