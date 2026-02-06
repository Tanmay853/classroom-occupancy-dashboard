import cv2

# Load your image
img = cv2.imread("C:/Users\DELL\Desktop\DDP\homography-computation-master\processed\LC302_L1_2F_CAM_4 clip_2025_4_7_9_24_6.Bmp")

# Check if image loaded correctly
if img is None:
    raise FileNotFoundError("Image not found. Check the path!")

# Coordinates of the point
point = (698, 31)

# Draw a red circle at the point
cv2.circle(img, point, radius=5, color=(0, 0, 255), thickness=-1)

# Show the image in a window
cv2.imshow("Point Marked", img)
cv2.waitKey(0)
cv2.destroyAllWindows()