import cv2 as cv
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
import time
import shutil  # for moving files
import tkinter as tk
from tkinter import filedialog

# ==== CONFIG ====
API_KEY = "Z2ZjbWt3Zjk2dTRrdjg2NTc1b21mOmM0a3Y4MzdiMFplTkJBM3NTMlZ4ZEtsNVRhV2dGajdC"
layout_image_path = 'lc001.png'  # Plan view image
num_images_to_process = 3  # Number of latest images to process each cycle
processed_directory = "C:/Users/DELL/Desktop/DDP/homography-computation-master/processed"  # Fixed path to move processed images

zones_manual = [
    (39, 44, 115, 196),
    (39, 198, 112, 357),
    (116, 45, 237, 195),
    (114, 197, 241, 351),
    (239, 46, 310, 193),
    (239, 197, 304, 346)
]

# Homography dictionary
image_point_data = {
    'LC302_1.png': {
        'src_points': [[170, 269], [313, 284], [470, 273], [657, 256], [98, 688], [468, 690], [1238, 690], [1021, 284], [1234, 167], [1121, 82], [1052, 49], [424, 44], [617, 96], [1034, 51], [917, 27], [981, 41], [1035, 63], [922, 43], [1270, 254], [1206, 241], [1120, 227]],
        'dst_points': [[115, 44], [125, 96], [123, 132], [123, 179], [45, 83], [51, 138], [47, 227], [122, 240], [179, 421], [265, 417], [304, 414], [239, 56], [207, 169], [252, 324], [248, 266], [252, 293], [229, 317], [229, 268], [121, 329], [120, 300], [122, 273]]
    },
    'LC302_2.png': {
        'src_points': [[312, 491], [237, 438], [394, 353], [322, 317], [478, 241], [403, 218], [466, 152], [33, 278], [57, 224], [109, 177], [22, 202], [739, 464], [864, 326], [518, 97], [558, 59], [592, 31], [933, 27], [158, 138], [199, 101], [246, 68], [284, 43]],
        'dst_points': [[257, 338], [230, 340], [252, 308], [229, 310], [246, 279], [227, 280], [228, 253], [122, 313], [110, 283], [110, 253], [51, 255], [296, 308], [301, 280], [226, 224], [226, 196], [225, 169], [305, 115], [110, 228], [110, 197], [110, 169], [110, 142]]
    },
    'LC302_4.png': {
        'src_points': [[638, 375], [428, 446], [228, 509], [88, 559], [341, 563], [581, 493], [874, 431], [884, 322], [1133, 363], [1106, 277], [1322, 241], [1479, 217], [156, 279], [175, 254], [208, 207], [343, 185], [526, 75], [610, 50],[698,131]],
        'dst_points': [[227, 194], [227, 225], [229, 251], [228, 281], [247, 222], [245, 195], [244, 169], [226, 169], [243, 140], [225, 139],[224, 113], [223, 84], [120, 345], [101, 346], [57, 348], [110, 285], [54, 228], [53, 198], [54, 169]]
    }
}
#[1378, 311] -> , [246,114] removed

# ==== FUNCTIONS ====
def detect_persons(image_path, api_key):
    url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
    with open(image_path, "rb") as image_file:
        files = {"image": ("image.jpg", image_file, "image/jpeg")}
        data = {"prompts": "person", "model": "agentic"}
        headers = {"Authorization": f"Basic {api_key}"}
        try:
            response = requests.post(url, files=files, data=data, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result['data'][0]
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return []

def get_homography_matrix(src_pts, dst_pts):
    src_np = np.array(src_pts).reshape(-1, 1, 2)
    dst_np = np.array(dst_pts).reshape(-1, 1, 2)
    H, _ = cv.findHomography(src_np, dst_np, cv.RANSAC, 5.0)
    return H

def get_transformed_centers(detections, H):
    centers = []
    for d in detections:
        box = d['bounding_box']
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        centers.append([cx, cy])
    centers = np.array(centers, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv.perspectiveTransform(centers, H)
    return transformed.reshape(-1, 2)

def merge_detections(all_points, eps):
    clustering = DBSCAN(eps=eps, min_samples=1).fit(all_points)
    labels = clustering.labels_
    merged = []
    for label in np.unique(labels):
        points = all_points[labels == label]
        merged_center = points.mean(axis=0)
        merged.append(merged_center)
    return np.array(merged)

def count_people_in_manual_zones(points, zones):
    counts = [0] * len(zones)
    for x, y in points:
        for i, (x1, y1, x2, y2) in enumerate(zones):
            if x1 <= x <= x2 and y1 <= y <= y2:
                counts[i] += 1
                break
    return counts

def show_manual_zones_with_counts(image, points, zones, counts):
    img = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(zones):
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv.putText(img, f"Zone {i+1}: {counts[i]}", (x1 + 5, y1 + 25),
                     cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.scatter(points[:, 0], points[:, 1], c='red', s=50)
    plt.title("Zones with Merged Student Count")
    plt.axis('off')
    plt.show()

def get_latest_images(directory, num_images):
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:num_images]

def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide tkinter main window
    folder_selected = filedialog.askdirectory(title="Select Image Directory")
    return folder_selected

# ==== MAIN PROCESSING LOOP ====
def main():
    # open folder selection dialog
    image_directory = select_folder()
    if not image_directory:
        print("No folder selected. Exiting...")
        return

    os.makedirs(processed_directory, exist_ok=True)

    while True:
        print(f"[{time.ctime()}] Scanning for {num_images_to_process} latest images in '{image_directory}'...")
        latest_image_paths = get_latest_images(image_directory, num_images_to_process)
        
        if not latest_image_paths:
            print("No images found. Waiting for new images...")
        else:
            all_transformed_centers = []
            
            for image_path in latest_image_paths:
                image_filename = os.path.basename(image_path)
                name, _ = os.path.splitext(image_filename)
                parts = name.split('_')
                if len(parts) >= 2:
                    cam_number = parts[1]
                    lookup_key = f"LC302_{cam_number}.png"
                else:
                    print(f"Warning: Could not parse filename '{image_filename}'. Skipping...")
                    continue

                
                print(f"\nProcessing {image_filename} (mapped to {lookup_key})...")
                
                if lookup_key in image_point_data:
                    src_points = image_point_data[lookup_key]['src_points']
                    dst_points = image_point_data[lookup_key]['dst_points']

                    try:
                        detections = detect_persons(image_path, API_KEY)
                        H = get_homography_matrix(src_points, dst_points)
                        transformed = get_transformed_centers(detections, H)
                        all_transformed_centers.append(transformed)
                        print(f"Found {len(detections)} persons in {image_filename}.")
                    except Exception as e:
                        print(f"Error processing {image_filename}: {e}")
                else:
                    print(f"Warning: No homography points found for '{lookup_key}'. Skipping...")

            if all_transformed_centers:
                all_transformed_centers = np.vstack(all_transformed_centers)
                print(f"\nTotal raw detections across all images: {len(all_transformed_centers)}")

                unique_students = merge_detections(all_transformed_centers, eps=15)
                print(f"Unique students after DBSCAN merging: {len(unique_students)}")

                layout_img = cv.imread(layout_image_path)
                zone_counts = count_people_in_manual_zones(unique_students, zones_manual)
                show_manual_zones_with_counts(layout_img, unique_students, zones_manual, zone_counts)

                print("\nZone-wise Student Counts:")
                for i, count in enumerate(zone_counts, 1):
                    print(f"Zone {i}: {count} students")
            else:
                print("No detections were processed in this cycle.")

            # === Move processed images ===
            for img_path in latest_image_paths:
                dest_path = os.path.join(processed_directory, os.path.basename(img_path))
                shutil.move(img_path, dest_path)
                print(f"Moved {img_path} -> {dest_path}")

        print("\nWaiting for 1 minute before the next run...")
        time.sleep(60)

if __name__ == "__main__":
    main()
