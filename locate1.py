import cv2
import time

# Define points
src_points = [[638, 375], [428, 446], [228, 509], [88, 559], [341, 563], [581, 493], 
              [874, 431], [884, 322], [1133, 363], [1106, 277], [1378, 311], [1322, 241], 
              [1479, 217], [156, 279], [175, 254], [208, 207], [343, 185], [526, 75], 
              [610, 50], [698, 31]]

dst_points =  [[227, 194], [227, 225], [229, 251], [228, 281], [247, 222], [245, 195], [244, 169], [226, 169], [243, 140], [225, 139], [246,114],[224, 113], [223, 84], [120, 345], [101, 346], [57, 348], [110, 285], [54, 228], [53, 198], [54, 169]]


# Load two images
img1 = cv2.imread("C:/Users\DELL\Desktop\DDP\homography-computation-master\processed\LC302_L1_2F_CAM_4 clip_2025_4_7_9_24_6.Bmp")
img2 = cv2.imread("lc001.png")

if img1 is None or img2 is None:
    raise FileNotFoundError("One or both images not found. Check the paths!")

for i in range(len(src_points)):
    # Make fresh copies for display
    temp1 = img1.copy()
    temp2 = img2.copy()

    # Mark current points
    cv2.circle(temp1, tuple(src_points[i]), radius=8, color=(0, 0, 255), thickness=-1)
    cv2.putText(temp1, f"P{i+1}", (src_points[i][0]+10, src_points[i][1]+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.circle(temp2, tuple(dst_points[i]), radius=8, color=(255, 0, 0), thickness=-1)
    cv2.putText(temp2, f"P{i+1}", (dst_points[i][0]+10, dst_points[i][1]+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show both images
    cv2.imshow("Image 1 (src)", temp1)
    cv2.imshow("Image 2 (dst)", temp2)

    # Wait 10 seconds or until key is pressed
    if cv2.waitKey(1000) & 0xFF == 27:  # press ESC to exit early
        break

cv2.destroyAllWindows()
