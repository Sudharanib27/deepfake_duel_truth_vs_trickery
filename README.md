# Deepfake Duel: Truth vs Trickery

A deep learning-based solution to detect real vs fake images and classify them into `human_faces`, `animals`, or `vehicles`. Built using a multi-task model trained on the ArtiFact_240K dataset.

---

## Project Highlights

- Multi-task model: jointly predicts real/fake label and image class
- EfficientNetV2 backbone: pretrained on ImageNet for strong feature extraction
- Class-weighted loss: handles class imbalance for better generalization
- Test-Time Augmentation (TTA): boosts prediction stability on unseen manipulations
- Grad-CAM explainability: visualizes model focus regions for each prediction

---

## Dataset Structure

The dataset used is ArtiFact_240K, organized as follows:

```
/train/real/[class]/image.jpg  
/train/fake/[class]/image.jpg  
/validation/real/[class]/image.jpg  
/validation/fake/[class]/image.jpg  
/test/image.jpg
```

---

## Model Architecture

```
Input Image
     │
EfficientNetV2 (Pretrained Backbone)
     │
 ┌───┴────────────┐
 │                │
Real/Fake Head    Class Head
 (Sigmoid)         (Softmax)
```

- Binary output head uses `BCEWithLogitsLoss`
- Class output head uses `CrossEntropyLoss` with inverse frequency class weights

---

## Setup & Run

### 1. Clone and Install

```bash
git clone https://github.com/your-username/deepfake-duel.git
cd deepfake-duel
pip install -r requirements.txt
```

### 2. Train the Model

If you're using Colab, download the notebook as a script:

```bash
File > Download > Download .py
```

Then run:

```bash
python deepfake_duel_truth_vs_trickery.py
```

### 3. Run Inference

Once the model is trained and saved:

```bash
python deepfake_duel.py --inference --model_path best_model.pt --test_dir /path/to/test --tta
```

The predictions will be saved to `test_tta_confidence.csv`.

---

## Evaluation

| Metric        | Validation Set |
|---------------|----------------|
| Real/Fake Acc | 97.01%            |
| Class Acc     | 99.94%           	|

---

## Explainability

Grad-CAM visualizations highlight the image regions used by the model for decision-making. Sample output:

```
saved in gradcam_outputs/sample_10_real_fake.png
```

---

## Output Files

- `deepfake_duel.py` – full pipeline for training, inference, Grad-CAM
- `best_model.pt` – saved weights
- `training_log.csv` – metrics per epoch
- `test.csv` – final predictions with confidence
- `gradcam_outputs/` – Grad-CAM heatmaps for selected samples

---

## Author

- Sudharani Bannengala – Model Design and Implementation

