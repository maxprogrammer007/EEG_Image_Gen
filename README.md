
# 🧠 Brain-Computer Interface (BCI) with GAN and CNN Classifier

This project explores a Brain-Computer Interface pipeline using deep learning. It includes a **GAN** for image generation from EEG signals and a **CNN classifier** to identify visual categories from EEG data. The entire pipeline supports preprocessing, visualization, and model training.

---

## 📁 Project Structure

```
├── activations/              # Activation maps from CNN layers
├── checkpoints/              # Saved model weights
├── comparison/               # Comparison results (real vs. generated)
├── data_preprocessing/       # EEG preprocessing scripts
├── Dataset/                  # EEG + image dataset loaders
├── generated_images/         # Output from GAN
├── models/
│   ├── CNNClassifier.py      # CNN architecture
│   ├── GAN.py                # Generator & Discriminator
│   └── weights/              # Model weights (optional)
├── plots/
│   └── .png                  # Visualization of training metrics
├── main.py                   # Main script for training GAN
├── train_classifier.py       # CNN classifier training
├── utils.py                  # Helper functions
└── .gitignore
```

---

## 🧪 Features

- EEG data preprocessing for clean input signals
- GAN model to generate visual patterns from brain signals
- CNN classifier for EEG signal classification
- Visualization of training metrics, activations, and predictions
- Evaluation metrics: Accuracy, F1, Confusion Matrix, etc.

---

## 🚀 Getting Started

### 1. Clone this repo:
```bash
git clone https://github.com/Maxprogrammer007/EEG_Image_Gen.git
cd EEG_Image_Generation
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Prepare data
- Place EEG data and images inside `Dataset/`
- Use preprocessing scripts in `data_preprocessing/` if needed

---

## 🧠 Training the Models

### Train the GAN:
```bash
python main.py
```

### Train the CNN Classifier:
```bash
python train_classifier.py
```

---

## 📊 Outputs

- **Generated Images**: `generated_images/`
- **Training Curves & Metrics**: `plots/`
- **Activations**: `activations/`
- **Model Checkpoints**: `checkpoints/`

---

## 📌 TODOs

- [ ] Add multi-subject support
- [ ] Extend to real-time inference
- [ ] Improve GAN stability with StyleGAN

---
