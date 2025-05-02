# Worker-Safety-PPE-Detection

A YOLOv8-based computer vision project that detects workers and verifies the presence of various Personal Protective Equipment (PPE) such as helmets, gloves, vests, and more. The goal is to automate safety compliance monitoring through image-based detection.

---

## Problem Statement

Ensuring on-site worker safety through automated detection of individuals and their protective gear. The model identifies the presence (or absence) of PPE like hard hats, gloves, boots, and safety harnesses across diverse scenarios using computer vision.

---

## Project Structure



Ensuring on-site worker safety through automated detection of individuals and their protective gear. The model identifies the presence or absence of PPE like hard hats, gloves, boots, and safety harnesses across diverse scenarios using computer vision.

---

## Project Structure
```text
Worker-Safety-PPE-Detection/
├── datasets/
│   ├── converted/              # Converted YOLO-format dataset
│   └── datasets/               # Original Pascal VOC annotations and images
├── weights/
│   └── best.pt                 # Trained YOLOv8 model
├── modules/
│   ├── inference.py            # Inference script to detect persons and PPE
│   └── output_inference/       # Result images after running inference
├── training.ipynb              # Model training script (Colab-compatible)
└── interact.py                 # GUI-based image filter and viewer
```

---

## Model and Training

**Model Used**: YOLOv8n from Ultralytics  
**Training Done On**: Google Colab  

**Detection Classes**:
- person  
- hard-hat  
- gloves  
- mask  
- glasses  
- boots  
- vest  
- ppe-suit  
- ear-protector  
- safety-harness  

---

## Optional GUI

Use `interact.py` to filter and view predictions with a user-friendly interface.
