



#this is the testing  code 






import requests
from PIL import Image

source = 'imgs/football.png'

#lc001.png
# src1=[[638, 375], [428, 446], [228, 509], [88, 559], [341, 563], [581, 493], [874, 431], [884, 322], [1133, 363], [1106, 277], [1378, 311], [1322, 241], [1479, 217], [156, 279], [175, 254], [208, 207], [343, 185], [526, 75], [610, 50], [698, 31]]
# dst1=[[227, 194], [227, 225], [229, 251], [228, 281], [247, 222], [245, 195], [244, 169], [226, 169], [243, 140], [225, 139], [247, 115], [224, 113], [223, 84], [120, 345], [101, 346], [57, 348], [110, 285], [54, 228], [53, 198], [54, 169]]

#football.png
src_list=[[344, 521], [694, 539], [519, 471], [117, 183], [419, 191], [534, 193], [648, 197], [240, 15], [457, 30], [541, 43], [626, 32], [863, 15]]
dst_list= [[119, 421], [220, 421], [171, 389], [32, 241], [136, 240], [170, 241], [205, 239], [31, 33], [119, 59], [169, 92], [221, 61], [307, 33]]

#basketball.jpg
# src1=[[398, 371], [71, 663], [901, 441], [1471, 539], [316, 444], [203, 539], [514, 483], [420, 584], [1150, 735], [1453, 661]]
# dst1=[[14, 15], [16, 155], [148, 15], [279, 15], [14, 67], [15, 102], [75, 68], [75, 103], [220, 103], [279, 67]]

#kabaddi.jpg
# src1=[[209, 465], [241, 487], [680, 732], [762, 780], [834, 687], [930, 656], [374, 465], [452, 452], [1221, 571], [1440, 507], [665, 401], [925, 379], [1044, 360]] 
# dst1=[[47, 111], [48, 143], [49, 372], [47, 403], [101, 368], [135, 371], [101, 144], [135, 143], [250, 371], [362, 368], [249, 115], [361, 142], [448, 144]]        
# src_list=[]
# dst_list=[]
zones_manual = [
    (39, 44, 115, 196),
    (39, 198, 112, 357),
    (116, 45, 237, 195),
    (114, 197, 241, 351),
    (239, 46, 310, 193),
    (239, 197, 304, 346)
]  # stores manually drawn zones [(x1, y1, x2, y2), ...]
zone_temp_start = None  # temp for 1st point of each zone

layout= 'imgs/dst.jpg'

def detect_persons(image_path, api_key):
    url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
    
    with open(image_path, "rb") as image_file:
        files = {
            "image": ("image.jpg", image_file, "image/jpeg")
        }
        
        data = {
            "prompts": "person",
            "model": "agentic"
        }
        
        headers = {
            "Authorization": f"Basic {api_key}"
        }
        
        try:
            response = requests.post(url, files=files, data=data, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # Count number of persons from the result
            num_persons = len(result['data'][0])  # Each object in data[0] represents one person
            
            return {
                'total_persons': num_persons,
                'detections': result['data'][0],  # Full detection data if needed
                'raw_response': result  # Complete response if needed
            }
            
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}


# Example usage:
if __name__ == "__main__":
    # Replace these with your actual values
    IMAGE_PATH = source
    API_KEY = "Z2ZjbWt3Zjk2dTRrdjg2NTc1b21mOmM0a3Y4MzdiMFplTkJBM3NTMlZ4ZEtsNVRhV2dGajdC"
    
    result = detect_persons(IMAGE_PATH, API_KEY)
    
    if 'error' in result:
        print("Error:", result['error'])
    else:
        print(f"Number of persons detected: {result['total_persons']}")
        
        # Optional: Print details about each detection
        for i, detection in enumerate(result['detections'], 1):
            print(f"\nPerson {i}:")
            print(f"Confidence Score: {detection['score']}")
            print(f"Bounding Box: {detection['bounding_box']}")

#result



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.stats import gaussian_kde

def get_box_centers(detections):
    """Calculate center points of all bounding boxes"""
    centers = []
    for detection in detections['detections']:  # Access the correct JSON structure
        box = detection['bounding_box']
        center_x = (box[0] + box[2]) / 2  # Average of x1 and x2
        center_y = (box[1] + box[3]) / 2  # Average of y1 and y2
        centers.append([center_x, center_y])
    return np.array(centers)

def create_density_heatmap(image_path, detections):
    # Load image to get dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Get center points
    centers = get_box_centers(detections)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot original image
    plt.imshow(img)
    
    # Create a grid of points
    x_grid = np.linspace(0, img_width, 100)
    y_grid = np.linspace(0, img_height, 100)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Calculate kernel density
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kernel = gaussian_kde(centers.T)
    z = np.reshape(kernel(positions), xx.shape)
    
    # Plot density overlay
    plt.contourf(xx, yy, z, levels=20, cmap='Reds', alpha=0.5)
    
    # Plot center points
    plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=20, alpha=0.6)
    
    plt.title('Occupant Density Heatmap')
    plt.colorbar(label='Density')
    plt.axis('off')  # Hide axes
    plt.show()

# Usage
image_path = IMAGE_PATH
create_density_heatmap(image_path, result)

# Print center coordinates
centers = get_box_centers(result)
print("\nCenter points (x, y):")
for i, center in enumerate(centers, 1):
    print(f"Person {i}: ({center[0]:.1f}, {center[1]:.1f})")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.stats import gaussian_kde

def get_box_centers(detections):
    """Calculate center points of all bounding boxes"""
    centers = []
    for detection in detections['detections']:
        box = detection['bounding_box']
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        centers.append([center_x, center_y])
    return np.array(centers)

def create_multiple_visualizations(image_path, detections):
    # Load image to get dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Get center points
    centers = get_box_centers(detections)
    
    # Create grid for density estimation
    x_grid = np.linspace(0, img_width, 100)
    y_grid = np.linspace(0, img_height, 100)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Calculate kernel density
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kernel = gaussian_kde(centers.T)
    z = np.reshape(kernel(positions), xx.shape)
        
    # # 1. Image with density overlay
    # plt.figure(figsize=(8, 8))
    # plt.imshow(img)
    # density1 = plt.contourf(xx, yy, z, levels=20, cmap='Reds', alpha=0.5)
    # plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=20, alpha=0.6)
    # plt.title('Density Overlay on Image')
    # plt.colorbar(density1, label='Density')
    # plt.axis('off')
    # plt.show()

    # # 2. Density map alone
    # plt.figure(figsize=(8, 8))
    # density2 = plt.contourf(xx, yy, z, levels=20, cmap='Reds')
    # plt.title('Density Heatmap')
    # plt.xlabel('X Position (pixels)')
    # plt.ylabel('Y Position (pixels)')
    # plt.colorbar(density2, label='Density')
    # plt.xlim(0, img_width)
    # plt.ylim(img_height, 0)  # Reverse Y-axis to match image coordinates
    # plt.show()

    # 3. Points plot
    plt.figure(figsize=(8, 8))
    plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=50)
    plt.title('Detection Points')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.xlim(0, img_width)
    plt.ylim(img_height, 0)  # Reverse Y-axis to match image coordinates
    plt.show()

    
    # Print statistics
    print(f"\nTotal detections: {len(centers)}")
    print(f"Image dimensions: {img_width}x{img_height} pixels")
    print(f"Average density: {len(centers)/(img_width*img_height)*10000:.2f} persons per 10000 pixels")

# Usage
image_path = IMAGE_PATH
create_multiple_visualizations(image_path, result)



















import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

drawing = False
src_x, src_y = -1, -1
dst_x, dst_y = -1, -1
# src_list = []
# dst_list = []

# Mouse callback function for selecting points
def select_points_src(event, x, y, flags, param):
    global src_x, src_y, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        src_x, src_y = x, y
        cv.circle(src_copy, (x, y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

def select_points_dst(event, x, y, flags, param):
    global dst_x, dst_y, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        dst_x, dst_y = x, y
        cv.circle(dst_copy, (x, y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        
    # Right-click to select zones
    elif event == cv.EVENT_RBUTTONDOWN:
        global zone_temp_start
        zone_temp_start = (x, y)

    elif event == cv.EVENT_RBUTTONUP:
        if zone_temp_start:
            x1, y1 = zone_temp_start
            x2, y2 = x, y
            zone = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            zones_manual.append(zone)
            cv.rectangle(dst_copy, (zone[0], zone[1]), (zone[2], zone[3]), (255, 0, 0), 2)
            zone_temp_start = None

            # Print the newly added zone
            print(f"Zone {len(zones_manual)} defined: {zone}")


# Function to compute homography
def get_plan_view(src, dst):
    src_pts = np.array(src_list).reshape(-1, 1, 2)
    dst_pts = np.array(dst_list).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    print("Homography Matrix (H):")
    print(H)
    plan_view = cv.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
    return plan_view, H

# Overlay detected points on top-view layout
def plot_transformed_points(dst, centers, H):
    # Transform detected points using homography matrix
    transformed_centers = cv.perspectiveTransform(np.array([centers], dtype=np.float32), H)[0]

    # Plot the transformed points on the layout
    plt.figure(figsize=(8, 8))
    plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
    plt.scatter(transformed_centers[:, 0], transformed_centers[:, 1], c='red', s=50)
    plt.title("Transformed Detection Points on Layout")
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.show()

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
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Plot with matplotlib for color accuracy
    plt.figure(figsize=(10, 8))
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.scatter(points[:, 0], points[:, 1], c='red', s=50)
    plt.title("Custom Zones with People Count")
    plt.show()

# Load images
src = cv.imread(source, -1)
dst = cv.imread(layout, -1)
src_copy = src.copy()
dst_copy = dst.copy()

cv.namedWindow('src')
cv.moveWindow("src", 80, 80)
cv.setMouseCallback('src', select_points_src)

cv.namedWindow('dst')
cv.moveWindow("dst", 780, 80)
cv.setMouseCallback('dst', select_points_dst)

while True:
    cv.imshow('src', src_copy)
    cv.imshow('dst', dst_copy)
    k = cv.waitKey(1) & 0xFF

    if k == ord('s'):
        # src_list=src1.copy()#see line 6
        # dst_list=dst1.copy()#see line 7
        print('Save points')
        cv.circle(src_copy, (src_x, src_y), 5, (0, 255, 0), -1)
        cv.circle(dst_copy, (dst_x, dst_y), 5, (0, 255, 0), -1)
        src_list.append([src_x, src_y])
        dst_list.append([dst_x, dst_y])
        print("Source Points:", src_list)
        print("Destination Points:", dst_list)


    elif k == ord('h') and len(src_list) >= 4:
        print('Creating plan view...')
        plan_view, H = get_plan_view(src, dst)
        # cv.imshow("Plan View", plan_view)

        # Example detected persons' centers
        detected_centers = np.array(centers)

        # Plot transformed points
        plot_transformed_points(dst, detected_centers, H)

        transformed_centers = cv.perspectiveTransform(np.array([centers], dtype=np.float32), H)[0]
        # Only run if zones were drawn
        if zones_manual:
            manual_counts = count_people_in_manual_zones(transformed_centers, zones_manual)
            show_manual_zones_with_counts(dst, transformed_centers, zones_manual, manual_counts)

            print("\nManual Zone Counts:")
            for i, count in enumerate(manual_counts, 1):
                print(f"Zone {i}: {count} people")
        else:
            print("No manual zones defined.")


    elif k == ord('u'):
        if src_list and dst_list:
            print('Undo last point pair')
            src_list.pop()
            dst_list.pop()
            src_copy = src.copy()
            dst_copy = dst.copy()
            for pt in src_list:
                cv.circle(src_copy, tuple(pt), 5, (0, 255, 0), -1)
            for pt in dst_list:
                cv.circle(dst_copy, tuple(pt), 5, (0, 255, 0), -1)
                
    elif k == 27:  # ESC key to exit
        break

cv.destroyAllWindows()
