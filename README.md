# Perspective Correction using OpenCV (SIFT, FLANN, and RANSAC)

## Introduction
This project corrects **perspective distortion** in images using **SIFT (Scale-Invariant Feature Transform)**, **FLANN-based feature matching**, and **RANSAC (Random Sample Consensus)** to compute a **homography matrix**. The process enables deskewing of images and aligning them to a reference template.

## How It Works
1. **Feature Detection (SIFT):** Extracts key points and descriptors from the input image and a reference template.
2. **Feature Matching (FLANN):** Matches features between the distorted image and the template.
3. **Outlier Removal (Lowe's Ratio Test):** Filters good matches to ensure reliable correspondences.
4. **Homography Estimation (RANSAC):** Computes the transformation needed to align the input image to the template.
5. **Perspective Warp:** Warps the input image using the computed transformation to match the template’s perspective.

## Installation
Ensure you have Python and the required dependencies installed:

```bash
pip install opencv-python numpy matplotlib
```

## Usage
1. Place the template and distorted images in a folder.
2. Update the file paths in `perspective_correction.py`.
3. Run the script:

```bash
python perspective_correction.py
```

## Code Overview
### Import Dependencies
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

### Function to Correct Perspective Distortion
```python
def correct_perspective_distortion(image, template):
    """
    Corrects perspective distortion using SIFT, FLANN, and RANSAC.
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
        return image, None, None

    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    H_inv = np.linalg.inv(H)
    h_temp, w_temp = template.shape[:2]
    warped_image = cv2.warpPerspective(image, H_inv, (w_temp, h_temp), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return warped_image, H, None
```

### Running the Code
```python
template = cv2.imread("path/to/template.jpg")
image = cv2.imread("path/to/input.jpg")
if template is None or image is None:
    raise ValueError("Error loading images. Check paths.")

corrected_image, homography, _ = correct_perspective_distortion(image, template)

plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
plt.title("Corrected Image")
plt.axis("off")
plt.show()
```

## Results
The script will:
- Detect features using **SIFT**.
- Match key points using **FLANN**.
- Compute the **homography matrix** using **RANSAC**.
- Warp the image to correct perspective distortions.

## Applications
- **Document alignment** (e.g., scanned receipts, Aadhar cards, invoices).
- **Perspective correction** in photographs.
- **Augmented reality** applications requiring precise object alignment.

## License
This project is open-source under the MIT License.

---

### Contributing
Feel free to fork the repository, submit pull requests, or report issues!

