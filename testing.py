"""
AER850
Project 3
Step 5

Alina Saleem
501129840

This python script:
    -loads trained model (.keras)
    -loads the class mapping 
    -runs predictions on every image in test subfolder
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from matplotlib import patheffects as pe


#same as training
IMG_SIZE = (500, 500)
CHANNELS = 3
TEST_DIR = Path(r"C:\Users\alina\Documents\GitHub\AER850_Project2\Project 2 Data\Data\test")


#figuring out which model to load
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to the trained .keras model (e.g., models/variant_model.keras)")
args = parser.parse_args()

#loading model and label mapping
model = tf.keras.models.load_model(args.model)
script_dir = Path(__file__).resolve().parent
class_map_path = script_dir / "tables"  / "class_indices.json"

#creating output folders
figs_dir = Path("figures"); figs_dir.mkdir(parents=True, exist_ok=True)
tables_dir = Path("tables"); tables_dir.mkdir(parents=True, exist_ok=True)


with open(class_map_path) as f:
    #for ex. {"crack":0, "missing-head":1, "paint-off":2}
    class_indices = json.load(f)           
    #invert to {0:"crack",...}
idx2label = {v: k for k, v in class_indices.items()} 
    #index ordered label names 
label_list = [k for k, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]  


#preprocessing a single image path (resizing, normalize)
def preprocess_image(path: Path):
    """Load image → resize (500x500) → convert to array → scale to [0,1] → add batch dim."""
    #PIL image
    img = load_img(path, target_size=IMG_SIZE, color_mode="rgb") 
    #float32 normalized
    arr = img_to_array(img) / 255.0                 
    #(1,500,500,3)
    arr = np.expand_dims(arr, axis=0)                           
    return img, arr


#walking the entire test set and collecting (gt_label, img_path)
items = []
for cls_dir in sorted([d for d in TEST_DIR.iterdir() if d.is_dir()]):
    for p in cls_dir.rglob("*"):
        if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".gif",".tif",".tiff"]:
            items.append((cls_dir.name, p))


#predicting every test image and saving as CSV row per image
rows, y_true, y_pred = [], [], []

for gt_label, img_path in items:
    #batch shape: (1,500,500,3)
    pil_img, batch = preprocess_image(img_path)       
    #softmax vector
    probs = model.predict(batch, verbose=0)[0]      
    #index of highest probability
    pred_idx = int(np.argmax(probs))         
    #map index to human label        
    pred_label = idx2label[pred_idx]        
    #prob for the predicted class        
    pred_prob  = float(probs[pred_idx])               
    rows.append({
        "file": str(img_path),
        "gt_label": gt_label,
        "predicted_label": pred_label,
        "predicted_prob": pred_prob,
        "probs_vector": ";".join([f"{i}:{p:.4f}" for i, p in enumerate(probs)])
    })

    y_true.append(gt_label)
    y_pred.append(pred_label)


#saving per image predictions table
pd.DataFrame(rows).to_csv(tables_dir / "predictions_test_all.csv", index=False)



#test metrics (precision, recall, f1, accuracy) + confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

#convertng labels to indices using same order as label_list
true_idx = [label_list.index(lbl) for lbl in y_true]
pred_idx = [label_list.index(lbl) for lbl in y_pred]

#confusion matrix and figure
cm = confusion_matrix(true_idx, pred_idx, labels=range(len(label_list)))
pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in label_list],
    columns=[f"pred_{c}" for c in label_list]
).to_csv(tables_dir / "confusion_matrix_test.csv", index=True)

plt.figure()
im = plt.imshow(cm, aspect="auto")
plt.title("Figure: Confusion Matrix — Test")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(ticks=range(len(label_list)), labels=label_list, rotation=45, ha="right")
plt.yticks(ticks=range(len(label_list)), labels=label_list)
plt.colorbar(im, fraction=0.046, pad=0.04)
for i in range(len(label_list)):
    for j in range(len(label_list)):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.savefig(figs_dir / "Figure_confusion_matrix_test.png", dpi=200, bbox_inches="tight")
plt.close()

#final full test set classification report as table ()
rep = classification_report(y_true, y_pred, labels=label_list, output_dict=True, zero_division=0)
pd.DataFrame(rep).transpose().to_csv(tables_dir / "metrics_test.csv")

print("DONE. See tables/ and figures/ for Test results.")

#list of (true label, path_to_test_image) pairs
required_samples = [
    ("crack",        TEST_DIR / "crack"        / "test_crack.jpg"),
    ("missing-head", TEST_DIR / "missing-head" / "test_missinghead.jpg"),
    ("paint-off",    TEST_DIR / "paint-off"    / "test_paintoff.jpg"),
]

#storing prediction results for the three classes above in table
fig3_rows = []

#create horizontal row of 3 image display panels
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes = np.atleast_1d(axes)

#loop for each image and its display axis
for ax, (gt_label, img_path) in zip(axes, required_samples):
    ax.axis("off")

#wanring for image file not existing if in-case-of
    if not img_path.exists():
        ax.set_title(f"{gt_label}\n[MISSING: {img_path.name}]", fontsize=9)
        continue

#load the image to match model input format
    pil_img, batch = preprocess_image(img_path)
    #run model to get class probs
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = idx2label[pred_idx]
    pred_prob = float(probs[pred_idx])

#finding class with highest prob score
    fig3_rows.append({
        "file": str(img_path),
        "gt_label": gt_label,
        "predicted_label": pred_label,
        "predicted_prob": pred_prob,
        "probs_vector": ";".join([f"{i}:{p:.4f}" for i, p in enumerate(probs)])
    })

#actual test image
    ax.imshow(pil_img)
#creating prob test overlay
    lines = [f"{lbl.replace('-', ' ').title()}: {probs[i]*100:.1f}%" for i, lbl in enumerate(label_list)]
    overlay_text = "\n".join(lines)

    ax.text(
        0.5, 0.80, overlay_text,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=16, color="green",
        path_effects=[pe.withStroke(linewidth=3, foreground="black")]  # makes text readable
    )

    ax.set_title(
        f"GT: {gt_label.replace('-', ' ').title()} | Pred: {pred_label.replace('-', ' ').title()} ({pred_prob:.1%})",
        fontsize=10
    )
#adding title and saving figure to disk
fig.suptitle("Figure 3: Model Testing Examples", y=1.05, fontsize=12)
plt.tight_layout()
plt.savefig(figs_dir / "Figure3_model_testing.png", dpi=200, bbox_inches="tight")
plt.close()

#save prediction results for 3 images in csv table
pd.DataFrame(fig3_rows).to_csv(tables_dir / "predictions_required_three.csv", index=False)
