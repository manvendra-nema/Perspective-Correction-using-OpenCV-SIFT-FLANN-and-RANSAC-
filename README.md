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


### Running the Code
Change path
```python
template = cv2.imread("path/to/template.jpg")
image = cv2.imread("path/to/input.jpg")
```

## Results
The script will:
- Detect features using **SIFT**.
- Match key points using **FLANN**.
- Compute the **homography matrix** using **RANSAC**.
- Warp the image to correct perspective distortions.


## Visualize
The detected SIFT keypoints used for feature matching and the corrected perspective after applying homography.
![image](https://github.com/user-attachments/assets/cb716479-cb18-4008-9699-f6273fc6c1f1)

![image](https://github.com/user-attachments/assets/dcba563f-81ff-4710-8d25-cac86a368ea0)

The script will:
- Detect features using **SIFT**.
- Match key points using **FLANN**.
- Compute the **homography matrix** using **RANSAC**.
- Warp the image to correct perspective distortions.


## Applications
- **Document alignment** (e.g., scanned receipts, Aadhar cards, invoices).
- **Perspective correction** in photographs.
- **Augmented reality** applications requiring precise object alignment.

## Medium Article

For a detailed explanation, check out the accompanying Medium article (update with actual link).

## License
This project is open-source under the MIT License.

---

### Contributing
Feel free to fork the repository, submit pull requests, or report issues!

