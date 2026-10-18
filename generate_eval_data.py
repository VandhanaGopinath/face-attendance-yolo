import os
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

# 1. Download LFW dataset (filtering for people who have at least 10 photos)
print("Downloading sample face dataset (LFW)...")
lfw = fetch_lfw_people(min_faces_per_person=10, color=True, resize=1.0)

# 2. Pick the first 3 distinct people and save 5 photos for each
output_base = os.path.join("data", "eval")
os.makedirs(output_base, exist_ok=True)

unique_Ids = np.unique(lfw.target)[:3]

for idx, person_id in enumerate(unique_Ids):
    # Clean name for folder (e.g., data/eval/Person_1_Name)
    raw_name = lfw.target_names[person_id].replace(" ", "_")
    person_dir = os.path.join(output_base, raw_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Get first 5 images of this person
    person_indices = np.where(lfw.target == person_id)[0][:5]
    
    for i, img_idx in enumerate(person_indices):
        img_rgb = lfw.images[img_idx]
        # Convert float [0,1] or [0,255] RGB to uint8 BGR for OpenCV
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        else:
            img_rgb = img_rgb.astype(np.uint8)
            
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Pad/upscale slightly so YOLO face detector has plenty of context
        img_bgr = cv2.copyMakeBorder(
            img_bgr, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        save_path = os.path.join(person_dir, f"photo_{i+1}.jpg")
        cv2.imwrite(save_path, img_bgr)
        
    print(f"Saved 5 photos to: {person_dir}")

print("\nDone! You can now run: python evaluate.py")