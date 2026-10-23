# Smart Garbage Classifier (software prototype)

This project trains an image classifier for six TrashNet classes:
cardboard, glass, metal, paper and trash.

## 1. Dataset layout

Put the extracted dataset inside this project. The script automatically searches for a folder containing the six category subfolders.

A supported layout is:

```text
Smart-Garbage-Classifier/
├── dataset/
│   └── dataset-resized/
│       ├── cardboard/
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       └── trash/
├── models/
├── train.py
├── predict.py
├── app.py
└── requirements.txt
```

The dataset can be nested one folder deeper; `train.py` searches under `dataset`.

## 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, try Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

If TensorFlow installation fails, copy the complete error message. Python/TensorFlow compatibility depends on the installed Python version.

## 3. Train

From the project root:

```powershell
python train.py
```

The first run downloads pretrained MobileNetV2 weights, so an internet connection is needed. Training can take a while. The model and class list are saved under `models/`.

## 4. Predict one image

Use an image path that exists on your computer:

```powershell
python predict.py "C:\path\to\your\image.jpg"
```

You can also select an image from one of the dataset folders.

## 5. Launch the upload interface

```powershell
streamlit run app.py
```

A local browser page opens. Upload a JPG, JPEG, PNG or WEBP image.

## Important notes

- No camera is needed for this software prototype.
- The dataset is small and imbalanced; validation results may not reflect performance on real-world photos.
- The model predicts one dominant class for one image. It does not yet detect multiple objects in a scene or physically sort waste.
- Keep the SSD connected and use the same project folder when running the scripts.
- For Raspberry Pi deployment, install compatible dependencies on the Pi and test performance there. Training on your laptop first is recommended.
