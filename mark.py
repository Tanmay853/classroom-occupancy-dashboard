import cv2

# Load image
img = cv2.imread("lc001.png")

if img is None:
    raise FileNotFoundError("Image not found!")

# Mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # left mouse button click
        print(f"Clicked at: ({x}, {y})")
        # draw a small circle on the image where you clicked
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", img)

# Display image
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
