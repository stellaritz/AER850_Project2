# -*- coding: utf-8 -*-
"""
AER850
Project 2
Steps 1-4

Alina Saleem
501129840

This python script:
    -loads dataset
    -builds two CNNs (baseline and variant)
    -Trains both CNNs and saves labelled accuracy/loss figures
    -evaluates on validation set, writes tables (CSV) for performance metrics
    -confusion matric figure + CSV
    -one-line comparison of CSV of overall validation accuracy
"""

#imports
# dave small JSON (history, class indices)
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#metrics table
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping


#DL framework
import tensorflow as tf
# Force eager mode (safe for TF 2.x; fixes ".numpy()" errors from graph mode)
print("TF version:", tf.__version__)
print("Eager before:", tf.executing_eagerly())
tf.config.run_functions_eagerly(True)
print("Eager after:", tf.executing_eagerly())
#keras model definition APIs
from tensorflow.keras import layers, models
#directory loaders and augmentation 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Step 1 Data Processing

#required image (width, height)
IMG_SIZE = (500,500)
#RGB images
CHANNELS = 3
# given batch size 
BATCH_SIZE = 32
#to ensure reproducibility
SEED = 42
 

# establishing the train and validation data directory
DATASET_DIR=Path(r"C:\Users\alina\Documents\GitHub\AER850_Project2\Project 2 Data\Data")
TRAIN_DIR= DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

#assigning images to folders
train_datagen = ImageDataGenerator(
    #normalizing pixels to [0,1]
    rescale = 1./255,
    #light shear augmentation
    shear_range=0.1,
    #zoom augmentation
    zoom_range=0.1,
    #random horizontal flips
    horizontal_flip=True
    )

#validation/test generators, normalized
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


#directory iterations, inferring to labels from subfolder names
train_gen = train_datagen.flow_from_directory(
    #where training images are from
    directory=str(TRAIN_DIR),
    #resize every image to 500x500
    target_size=IMG_SIZE,
    #number of imagers per batch
    batch_size=BATCH_SIZE,
    #one-hot labels for 3 classes
    class_mode="categorical",
    #shuffle train batches
    shuffle=True,
    seed=SEED
    )

val_gen= val_datagen.flow_from_directory(
    directory=str(VAL_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    #validation order fixed
    shuffle=False
    )

test_gen=test_datagen.flow_from_directory(
    directory=str(TEST_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    #test order is fixed
    shuffle=False
    )

#saving class index mapping {"crack":0, "missing-head":1, "paint-off":2}

tables_dir = Path("tables"); tables_dir.mkdir(parents=True, exist_ok=True)
with open(tables_dir / "class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)
    
# step 2 netural network architecture design

#using ReLU for hidden layers and softmax for output (multi-class)

HIDDEN_ACT="relu"

def build_baseline_model(input_shape=(500,500,3), num_classes=3):
    """Baseline CNN: 
        -3conv blocks with 3x3 kernals (32,64,128)
        -maxpool after each block to downsample
        -flatten to dense (128, ReLU) to dropout (0.3) to dense (3,softmax)
        """
    m = models.Sequential(name="baseline_cnn")
    #block 1
    m.add(layers.Conv2D(32, (3,3), activation=HIDDEN_ACT, padding="same", input_shape=input_shape))
    m.add(layers.MaxPooling2D((2,2)))
    #block 2
    m.add(layers.Conv2D(64, (3,3), activation=HIDDEN_ACT, padding="same"))
    m.add(layers.MaxPooling2D((2,2)))
    #block 3
    m.add(layers.Conv2D(128, (3,3), activation=HIDDEN_ACT, padding="same"))
    m.add(layers.MaxPooling2D((2,2)))
    #classifier head
    m.add(layers.Flatten())
    m.add(layers.Dense(128, activation=HIDDEN_ACT))
    m.add(layers.Dropout(0.3))
    m.add(layers.Dense(num_classes, activation="softmax"))  # Softmax for 3 classes
    return m


def build_variant_model(input_shape=(500,500,3), num_classes=3):
    """Variant CNN: 
        -4 conv blocks (64,128,128,256)
        -more capacity than baseline
        -dropout(0.5) before softmax to reduce everything
        """
    m = models.Sequential(name="variant_cnn")
    #block 1
    m.add(layers.Conv2D(64, (3,3), activation=HIDDEN_ACT, padding="same", input_shape=input_shape))
    m.add(layers.MaxPooling2D((2,2)))
    #block 2
    m.add(layers.Conv2D(128, (3,3), activation=HIDDEN_ACT, padding="same"))
    m.add(layers.MaxPooling2D((2,2)))
    #block 3
    m.add(layers.Conv2D(128, (3,3), activation=HIDDEN_ACT, padding="same"))
    m.add(layers.MaxPooling2D((2,2)))
    #block 4
    m.add(layers.Conv2D(256, (3,3), activation=HIDDEN_ACT, padding="same"))
    m.add(layers.MaxPooling2D((2,2)))
    #classifier head
    m.add(layers.Flatten())
    m.add(layers.Dense(256, activation=HIDDEN_ACT))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(num_classes, activation="softmax"))
    return m
    
 
# step 3 hyperparameters and training

#multi-class one hot classification
LOSS = "categorical_crossentropy"

#adam optimizer
OPT=tf.keras.optimizers.Adam(1e-3)

#metric to track during training/validation
METRICS = ["accuracy"]


#starting small to avoid overfitting
EPOCHS_BASELINE = 12
EPOCHS_VARIANT  = 12

def make_optimizer():
    return tf.keras.optimizers.Adam(1e-3)


def train_and_eval(model, model_name, epochs, train_gen, val_gen):
   #attach loss/optimizer/metrics
    model.compile(optimizer=make_optimizer(), loss=LOSS, metrics=METRICS)
   
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
            )
        ]
    #training the model with validation monitoring
    history = model.fit(
      train_gen,
      validation_data=val_gen,
      epochs=epochs,
      callbacks=callbacks,
    )

    #Prepare output directories
    models_dir = Path("models");  models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir   = Path("logs");    logs_dir.mkdir(parents=True, exist_ok=True)
    figs_dir   = Path("figures"); figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path("tables");  tables_dir.mkdir(parents=True, exist_ok=True)

    # saving the trained model (.keras) and the raw history (JSON)
    model.save(models_dir / f"{model_name}.keras")
    with open(logs_dir / f"history_{model_name}.json", "w") as f:
        json.dump(history.history, f, indent=2)

    #plotting accuracy curve
    plt.figure()
    plt.plot(history.history["accuracy"],     label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title(f"Figure: training vs validation accuracy — {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(figs_dir / f"Figure_training_accuracy_{model_name}.png", dpi=200, bbox_inches="tight")
    plt.close()

    #plotting loss curve
    plt.figure()
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"Figure: Training vs Validation Loss — {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(figs_dir / f"Figure_training_loss_{model_name}.png", dpi=200, bbox_inches="tight")
    plt.close()


# step 4 
    #validation matrix (performance metrics like precision, recall, f1, accuracy) and confusion matrix 
    #running predictions over the entire validation set 
    #ensuring same order
    val_gen.reset()        
    #softmax probabilities                                
    val_probs = model.predict(val_gen)     
    #class index with argmax              
    val_pred  = np.argmax(val_probs, axis=1)    
    #true class indices        
    val_true  = val_gen.classes               
    #ex. ['crack', 'missing-head', 'paint-off']           
    class_names = list(val_gen.class_indices.keys())      

    #classification report as CSV
    report = classification_report(
        val_true, val_pred,
        target_names=class_names,
        #dict to convert to dataframe
        output_dict=True,      
        #avoiding NaN if class has no predictions
        zero_division=0       
    )
    
    pd.DataFrame(report).transpose().to_csv(tables_dir / f"metrics_{model_name}.csv")

    #confusion matrix (CSV)
    cm = confusion_matrix(val_true, val_pred, labels=range(len(class_names)))
    pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names]
    ).to_csv(tables_dir / f"confusion_matrix_{model_name}.csv")
    
    #labelled plot
    plt.figure()
    im = plt.imshow(cm, aspect="auto")
    plt.title(f"Figure: confusion matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45, ha="right")
    plt.yticks(ticks=range(len(class_names)), labels=class_names)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    #writing counts in each cell
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.savefig(figs_dir / f"Figure_confusion_matrix_{model_name}.png", dpi=200, bbox_inches="tight")
    plt.close()


    #building both models, then training/evaluating and writing comparison table
if __name__ == "__main__":
    
    #building models with correct input shape and # of classes
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], CHANNELS)
    n_classes = train_gen.num_classes

    baseline = build_baseline_model(input_shape=input_shape, num_classes=n_classes)
    variant  = build_variant_model (input_shape=input_shape, num_classes=n_classes)

    #training and eval
    train_and_eval(baseline, "baseline_model", EPOCHS_BASELINE, train_gen, val_gen)
    train_and_eval(variant,  "variant_model",  EPOCHS_VARIANT,  train_gen, val_gen)

    #small comparison table of validation accuracy overall taken from the CSVs
    def overall_accuracy_from_csv(path: Path) -> float:
        df = pd.read_csv(path, index_col=0)
        if "accuracy" in df.index:
            return float(df.loc["accuracy", "precision"])
        return float(df.loc["macro avg", "recall"])

    tables_dir = Path("tables")
    acc_base = overall_accuracy_from_csv(tables_dir / "metrics_baseline_model.csv")
    acc_var  = overall_accuracy_from_csv(tables_dir / "metrics_variant_model.csv")

    pd.DataFrame({
        "model": ["baseline_model", "variant_model"],
        "val_overall_accuracy": [acc_base, acc_var]
    }).to_csv(tables_dir / "metrics_comparison.csv", index=False)

    print("DONE. See models/, figures/, tables/, logs/")