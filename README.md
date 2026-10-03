# FaceID Pro — Final LFW Edition

## Project summary

FaceID Pro is a controlled face-identity classification prototype built with PyTorch, ResNet-50 transfer learning and Streamlit.

The final submission uses the **Labeled Faces in the Wild (LFW)** dataset. LFW was specifically created for face recognition in unconstrained conditions; the standard tasks include face verification and face identification. The official ecosystem documents both identity labels and pair-based verification protocols.

**Important:** the project is a research/educational prototype. It is not a production biometric security system.

## Dataset

LFW contains face images of public figures collected from the web. It is a recognized face-recognition benchmark and is documented by UMass and scikit-learn. The project does not redistribute the dataset inside this ZIP.

Official source:
http://vis-www.cs.umass.edu/lfw/

A current scikit-learn interface can download/access LFW through `fetch_lfw_people`. Some current torchvision versions no longer support automatic LFW downloading, so this project intentionally uses scikit-learn for dataset acquisition.

## One-command dataset preparation

```bash
python prepare_lfw.py
```

Default configuration:
- at least 20 images per identity
- 20 identities maximum
- at most 30 images retained per identity

This creates:

```text
dataset/
├── identity_1/
├── identity_2/
└── ...
```

## Train

```bash
python train.py --data dataset --epochs 18 --batch 16
```

The first run may download pretrained ResNet-50 weights.

Generated:

```text
artifacts/
├── best_model.pt
├── metadata.json
├── classification_report.txt
└── confusion_matrix.npy
```

## Run the application

```bash
streamlit run app.py
```

## Verify the package

```bash
python smoke_test.py
```

## Architecture

```text
LFW Dataset
     ↓
Identity-folder preparation
     ↓
Stratified train/validation split
     ↓
Image augmentation + normalization
     ↓
ResNet-50 pretrained backbone
     ↓
BN + Dropout + Linear identity head
     ↓
Softmax Top-K ranking
     ↓
Unknown threshold
     ↓
Streamlit dashboard
```

## Engineering improvements

- reproducible seed
- dataset validation
- stratified split
- transfer learning warm-up
- lower learning rate for backbone fine-tuning
- AdamW optimizer
- cosine learning-rate schedule
- label smoothing
- gradient clipping
- early stopping
- best-checkpoint persistence
- confusion matrix
- classification report
- CPU/CUDA support
- robust checkpoint errors
- path resolution independent of launch directory

## What not to claim

Do not describe the displayed softmax score as a calibrated probability of identity.

Do not claim that closed-set classification accuracy alone proves biometric verification security.

For a stronger research extension, implement an embedding-based verifier and report FAR/FRR, ROC/DET curves and threshold-specific verification results using LFW's pair protocol.

## Privacy and ethics

The dataset contains images of public figures obtained from the web. Follow the dataset's terms and do not deploy the prototype for unauthorized identification, surveillance, or access-control decisions.
