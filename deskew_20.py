import cv2
import numpy as np
import matplotlib.pyplot as plt

def correct_perspective_distortion(image, template):
    """
    Corrects perspective distortion using SIFT feature matching and RANSAC.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_template, des_template = sift.detectAndCompute(gray_template, None)
    kp_image, des_image = sift.detectAndCompute(gray_image, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_template, des_image, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.65 * n.distance]

    if len(good_matches) < 4:
        print("Not enough good matches to compute homography.")
        return image, None, None, None

    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    H_inv = np.linalg.inv(H)
    h_temp, w_temp = template.shape[:2]
    warped_image = cv2.warpPerspective(image, H_inv, (w_temp, h_temp),
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    template_corners = np.float32([[0, 0], [w_temp, 0], [w_temp, h_temp], [0, h_temp]]).reshape(-1, 1, 2)
    projected_corners = cv2.perspectiveTransform(template_corners, H)
    
    matches_img = cv2.drawMatches(template, kp_template, image, kp_image, good_matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return warped_image, H, projected_corners, matches_img

# Load images
template = cv2.imread(r"exp_temp.jpg")  
image = cv2.imread(r"exp_img_2.jpg")  

if template is None or image is None:
    raise ValueError("Template and/or input image could not be loaded. Check file paths.")

# Perform correction
corrected_image, homography, projected_corners, matches_img = correct_perspective_distortion(image, template)

# Draw projected template corners
image_with_corners = image.copy()
if projected_corners is not None:
    projected_corners_int = np.int32(projected_corners)
    cv2.polylines(image_with_corners, [projected_corners_int], True, (0, 255, 0), 3)

# Plot results
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
plt.title("Input Image with Projected Template Corners")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
plt.title("SIFT Feature Matches")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
plt.title("Perspective Corrected Image")
plt.axis("off")

plt.tight_layout()
plt.show()
