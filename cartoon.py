import cv2
import numpy as np
import os

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Anime Filter")

# Ensure save directory exists
save_dir = os.path.join(os.getcwd(), "captured_anime_images")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Trackbar callback function (dummy)
def nothing(x):
    pass

# Create trackbars for tuning effect
cv2.createTrackbar("Blur", "Anime Filter", 5, 20, nothing)
cv2.createTrackbar("Edge Threshold", "Anime Filter", 100, 255, nothing)

photo_counter = 1  # Counter for saving images

def anime_filter(frame, blur_value, edge_thresh):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply median blur
    blur_value = max(1, blur_value * 2 + 1)  # Ensure blur is odd
    gray_blur = cv2.medianBlur(gray, blur_value)

    # Detect edges using Laplacian filter (stronger than thresholding)
    edges = cv2.Laplacian(gray_blur, cv2.CV_8U, ksize=5)
    edges = cv2.threshold(edges, edge_thresh, 255, cv2.THRESH_BINARY)[1]

    # Apply bilateral filter to smooth colors while keeping edges sharp
    color = cv2.bilateralFilter(frame, 9, 300, 300)

    # Reduce colors (Color Quantization) for a cel-shaded look
    Z = color.reshape((-1, 3))
    Z = np.float32(Z)
    K = 8  # Number of color clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(color.shape)

    # Combine edges with the filtered image for an anime look
    anime = cv2.bitwise_and(quantized, quantized, mask=edges)

    return anime

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        break

    blur_value = cv2.getTrackbarPos("Blur", "Anime Filter")
    edge_thresh = cv2.getTrackbarPos("Edge Threshold", "Anime Filter")

    anime_frame = anime_filter(frame, blur_value, edge_thresh)

    cv2.imshow("Anime Filter", anime_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Quit the app
        print("Exiting program...")
        break
    elif key == ord("c"):  # Capture and save image
        filename = os.path.join(save_dir, f"anime_photo_{photo_counter}.jpg")
        cv2.imwrite(filename, anime_frame)
        print(f"Photo saved: {filename}")
        photo_counter += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Camera released. Windows closed.")
