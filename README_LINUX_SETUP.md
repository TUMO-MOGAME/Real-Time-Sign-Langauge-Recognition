# Linux Setup Guide

This project works best on **Linux with Python 3.10**.

Why Python 3.10?
- It is a safe choice for this repo's pinned versions of **TensorFlow 2.15.0** and **MediaPipe 0.10.9**.
- It helps avoid package compatibility problems that can happen with newer Python versions.

## 1) Check Python 3.10

Run:

```bash
python3.10 --version
```

You should see something like:

```bash
Python 3.10.x
```

If `python3.10` is missing on Ubuntu/Debian, install it first:

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip libgl1
```

> `libgl1` is important for OpenCV on Linux.

## 2) Go into the project folder

```bash
cd Real-Time-Sign-Langauge-Recognition
```

## 3) Create a virtual environment

```bash
python3.10 -m venv .venv
```

## 4) Activate the virtual environment

```bash
source .venv/bin/activate
```

After activation, check that the project is using the correct Python:

```bash
which python
python --version
```

Expected result:
- `which python` should point to `.venv/bin/python`
- `python --version` should show **Python 3.10.x**

## 5) Upgrade pip tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 6) Install project dependencies

```bash
pip install -r requirements.txt
```

## 7) Quick verification

Run:

```bash
python -c "import tensorflow as tf; import mediapipe as mp; print('tensorflow', tf.__version__); print('mediapipe', mp.__version__)"
```

If that works, your main dependencies are installed correctly.

## 8) Optional project checks

Run the OpenCV app:

```bash
python cv2main.py
```

Or run the Streamlit app:

```bash
streamlit run st_app.py
```

## 9) For your custom training pipeline

Dry-run the custom gloss training setup:

```bash
python scripts/train_custom_gloss.py --dry-run --max-sequences-per-sign 3 --batch-size 4
```

Start full training later with:

```bash
python scripts/train_custom_gloss.py
```

## Notes

- Always activate `.venv` before running the project.
- If you open a new terminal, run this again:

```bash
source .venv/bin/activate
```

- If `pip install -r requirements.txt` fails, do **not** mix random package versions. Fix the environment first and keep Python at **3.10**.