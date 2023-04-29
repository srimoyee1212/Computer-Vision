In our assignment, we implemented Extra Credit Action Item 2. 
We applied the Scale Invariant Feature Transform (SIFT) algorithm from scratch without using cv2.

a. Custom Descriptor - SIFT

Keypoint type = Harris
Descriptor type = Custom
Matcher type = Ratio Test
Threshold (10^x) = -0.6

Average AUC – 1.0

Please look at Yosemite_SIFT_ROC.jpg in report.pdf for ROC curve.

Process for SIFT implementation:
1. The input image is first normalized and converted to grayscale using the OpenCV function cv2.cvtColor().
2. The function creates two NumPy arrays, orientation and desc, to store the orientation and feature descriptor data for each keypoint.
3. The Sobel operator is applied to the grayscale image to obtain the horizontal and vertical gradients, which are used to compute the orientation of each pixel in the image.
4. For each keypoint, the code loops over a 16x16 window around the keypoint and divides it into 4x4 cells. Within each cell, it computes a histogram of gradient orientations (with 8 bins) weighted by the gradient magnitude.
5. The resulting histograms are concatenated to form a 128-dimensional feature descriptor vector for the keypoint.
6. The code then normalizes the descriptor vector using L2 normalization and checks whether the variance of the descriptor vector is below a certain threshold. If the variance is below the threshold, the descriptor vector is set to zero.
7. Finally, the code returns a NumPy array desc containing the feature descriptor vectors for all keypoints.