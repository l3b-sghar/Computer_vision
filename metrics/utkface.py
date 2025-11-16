import os
import zipfile
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from transformers import pipeline


# ==========================================================
# 1. UTKFACE EXTRACTION + DATASET PREPARATION
# ==========================================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

UTK_ZIP = os.path.join(SCRIPT_DIR, "archive.zip")          # downloaded from Kaggle
UTK_FOLDER = os.path.join(SCRIPT_DIR, "UTKFace")           # extracted folder name
DATASET_ROOT = os.path.join(SCRIPT_DIR, "dataset", "classification")

AGE_SPLITS = {
    "young": lambda age: age <= 20,
    "adult": lambda age: 21 <= age <= 50,
    "old":   lambda age: age > 50,
}


def prepare_utk_dataset():
    print(f"Checking for UTKFace zip at: {UTK_ZIP}")
    
    if not os.path.exists(UTK_ZIP):
        print(f"ERROR: {UTK_ZIP} not found!")
        print("Please download the UTKFace dataset from Kaggle and place it in the metrics folder.")
        return False
    
    print("Extracting UTKFace zip...")
    with zipfile.ZipFile(UTK_ZIP, 'r') as z:
        z.extractall(SCRIPT_DIR)

    # Create folders
    gender_folders = ["male", "female"]
    age_folders = ["young", "adult", "old"]

    for g in gender_folders:
        os.makedirs(os.path.join(DATASET_ROOT, "gender", g), exist_ok=True)
    for a in age_folders:
        os.makedirs(os.path.join(DATASET_ROOT, "age", a), exist_ok=True)

    print("Sorting UTKFace images into dataset structure...")

    for img_name in os.listdir(UTK_FOLDER):
        if not img_name.endswith(".jpg"):
            continue

        try:
            # filename format: age_gender_ethnicity_.jpg
            parts = img_name.split("_")
            age = int(parts[0])
            gender = int(parts[1])  # 0=female, 1=male
        except:
            continue

        src_path = os.path.join(UTK_FOLDER, img_name)

        # ---- Gender
        gender_label = "male" if gender == 1 else "female"
        dst_gender = os.path.join(DATASET_ROOT, "gender", gender_label, img_name)
        shutil.copy(src_path, dst_gender)

        # ---- Age (young/adult/old)
        for cls, rule in AGE_SPLITS.items():
            if rule(age):
                dst_age = os.path.join(DATASET_ROOT, "age", cls, img_name)
                shutil.copy(src_path, dst_age)
                break

    print("Dataset preparation complete!")
    return True


# ==========================================================
# 2. EVALUATION FUNCTION (HF PIPELINE)
# ==========================================================

def evaluate_hf_model(model_id, dataset_path, sample_fraction=0.25):
    print(f"\n\n===== Evaluating: {model_id} =====")
    print(f"Dataset path: {dataset_path}")
    print(f"Sample fraction: {sample_fraction*100:.0f}%")

    clf = pipeline("image-classification", model=model_id)

    true_labels = []
    pred_labels = []

    classes = sorted(os.listdir(dataset_path))
    print(f"Classes found: {classes}")

    total_images = 0
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith(".jpg")]
        total_images += len(images)
    
    print(f"Total images available: {total_images}")
    print(f"Images to evaluate: {int(total_images * sample_fraction)}")
    
    processed = 0
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        print(f"\nProcessing class: {cls}")
        
        all_images = [f for f in os.listdir(cls_path) if f.endswith(".jpg")]
        # Sample a fraction of images
        sample_size = max(1, int(len(all_images) * sample_fraction))
        sampled_images = all_images[:sample_size]
        
        for img_file in sampled_images:
            img = Image.open(os.path.join(cls_path, img_file))

            pred = clf(img)[0]["label"].lower()

            true_labels.append(cls)
            pred_labels.append(pred)
            
            processed += 1
            if processed % 50 == 0:
                print(f"  Processed {processed} images...")

    # ---- Metrics
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n{'='*50}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, zero_division=0))

    # ---- Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(f"Confusion Matrix: {model_id}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"{model_id.replace('/', '_')}_confusion_matrix.png"))
    plt.show()

    return acc


# ==========================================================
# 3. MAIN
# ==========================================================

if __name__ == "__main__":
    print("="*70)
    print("UTKFace Dataset Evaluation for Age & Gender Models")
    print("="*70)
    
    # Step 1 — prepare dataset (skip if already exists)
    gender_path = os.path.join(DATASET_ROOT, "gender")
    age_path = os.path.join(DATASET_ROOT, "age")
    
    if not os.path.exists(gender_path) or not os.path.exists(age_path):
        print("\nDataset not found. Preparing dataset...")
        if not prepare_utk_dataset():
            print("Dataset preparation failed. Exiting...")
            exit(1)
    else:
        print("\nDataset already exists. Skipping extraction...")

    # Step 2 — gender evaluation
    if os.path.exists(gender_path):
        gender_acc = evaluate_hf_model(
            "rizvandwiki/gender-classification",
            gender_path
        )
    else:
        print("Gender dataset not found. Skipping gender evaluation.")
        gender_acc = None

    # Step 3 — age evaluation
    if os.path.exists(age_path):
        age_acc = evaluate_hf_model(
            "nateraw/vit-age-classifier",
            age_path
        )
    else:
        print("Age dataset not found. Skipping age evaluation.")
        age_acc = None
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    if gender_acc is not None:
        print(f"Gender Classification Accuracy: {gender_acc:.4f} ({gender_acc*100:.2f}%)")
    if age_acc is not None:
        print(f"Age Classification Accuracy: {age_acc:.4f} ({age_acc*100:.2f}%)")
    print("="*70)
