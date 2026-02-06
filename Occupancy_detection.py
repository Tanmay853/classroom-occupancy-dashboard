import requests
import base64

def detect_persons(image_path, api_key):
    url = "https://api.landing.ai/v1/tools/agentic-object-detection"
    
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
    IMAGE_PATH = "C:/Users\DELL\Downloads\homography-computation-master\homography-computation-master\imgs\class1.jpg"
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
        
    # 1. Image with density overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    density1 = plt.contourf(xx, yy, z, levels=20, cmap='Reds', alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=20, alpha=0.6)
    plt.title('Density Overlay on Image')
    plt.colorbar(density1, label='Density')
    plt.axis('off')
    plt.show()

    # 2. Density map alone
    plt.figure(figsize=(8, 8))
    density2 = plt.contourf(xx, yy, z, levels=20, cmap='Reds')
    plt.title('Density Heatmap')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.colorbar(density2, label='Density')
    plt.xlim(0, img_width)
    plt.ylim(img_height, 0)  # Reverse Y-axis to match image coordinates
    plt.show()

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