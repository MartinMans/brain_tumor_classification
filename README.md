# 🧠 Brain Tumor Classification (CMPS 261 - Spring 2025)

This project aims to develop a deep learning model to classify brain MRI scans as either containing a tumor or not. It was completed as part of the course **CMPS 261 - Machine Learning** at the American University of Beirut (AUB).

We approach the task as a **binary classification problem** using paired MRI images and label files. Each `.jpg` image is accompanied by a `.txt` label file:
- The first value in the label indicates tumor presence (`1` = tumor, `0` = no tumor).
- Additional values define a bounding box, which we ignore for this project (as per instructor's note).

---

## 📁 Project Structure


```
brain_tumor_classification/
├── data/
│   ├── images/                # MRI image files (.jpg)
│   └── labels/                # Corresponding label files (.txt)
├── Notebook.ipynb             # Main Jupyter notebook containing full ML pipeline
├── brain_tumor_model.keras    # Saved trained model in Keras format
├── .gitignore                 # Git ignore rules (e.g. .h5 files, cache folders)
└── README.md                  # Project overview, structure, and documentation
```


---

## 🔄 Workflow & Thought Process

### 1. 🧹 Data Understanding & Cleaning
- Analyzed the format of label files: tumor class + bounding box values.
- Removed image-label mismatches (893 images vs. 878 labels initially).
- Ignored bounding box values for negative samples.
- Final dataset: **878 valid image-label pairs**.
- Class distribution was relatively balanced, with 419 no-tumor samples and 459 tumor samples.

### 2. ⚙️ TensorFlow Data Pipeline
- Built an efficient `tf.data.Dataset` pipeline from a generalized generator function.
- Images are resized to **224×224**, normalized, and batched.
- Applied shuffling, batching, prefetching, and `.repeat()` for multi-epoch training.

### 3. 🧠 Model Architecture (Transfer Learning)
- Used **MobileNetV2** as a pretrained base (weights frozen).
- Added a custom classification head:
  - `GlobalAveragePooling2D`
  - Dense layer → Dropout → Sigmoid output for binary prediction.

### 4. 🏋️ Training, Evaluation & Reproducibility
- Split data into **80% training / 20% validation** using `random_state=1`.
- Fixed all random seeds (`random`, `numpy`, `tensorflow`, and `PYTHONHASHSEED`) for full reproducibility.
- Trained model with proper `steps_per_epoch` and dataset repetition.
- Evaluated on accuracy, precision, and recall using both Keras and custom `sklearn` logic.
- Achieved **~90% accuracy** on the validation set.

### 5. 🔍 Inference & Threshold Tuning
- Interactive inference section with label predictions, confidence scores, and visual feedback (`[✓]` or `[X]`).
- Visualized performance metrics across various classification thresholds (0.0 → 1.0).
- Settled on a **custom threshold of 0.4** to balance precision, recall, and accuracy (~90% each).
- Notebook inference and visualizations use this threshold consistently.

---

## 📌 Reproducibility

To ensure consistent results on every run, all random seeds are fixed and the notebook runs deterministically.

---

## 👥 Collaborators

- Martin Mansour  
- Alex Berdkan
