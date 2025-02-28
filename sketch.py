import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

# Create a named window
cv2.namedWindow("Sketch Filter")

# Trackbar callback function (does nothing, just needed)
def nothing(x):
    pass

# Create trackbars for Blur and Threshold
cv2.createTrackbar("Blur", "Sketch Filter", 5, 20, nothing)  # Blur kernel size
cv2.createTrackbar("Threshold", "Sketch Filter", 100, 255, nothing)  # Edge detection threshold

photo_counter = 1  # Counter for saving images

def sketch_filter(frame, blur_value, edge_thresh):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur_value = max(1, blur_value * 2 + 1)  # Ensure blur is an odd number
    blurred = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)  # Apply blur

    edges = cv2.Canny(blurred, edge_thresh, edge_thresh * 2)  # Edge detection
    sketch = cv2.bitwise_not(edges)  # Invert for better black & white effect

    return sketch

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Read values from trackbars
    blur_value = cv2.getTrackbarPos("Blur", "Sketch Filter")
    edge_thresh = cv2.getTrackbarPos("Threshold", "Sketch Filter")

    # Apply sketch filter
    sketch = sketch_filter(frame, blur_value, edge_thresh)

    # Show the result
    cv2.imshow("Sketch Filter", sketch)

    # Capture keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Quit the app
        break
    elif key == ord("c"):  # Capture and save image
        filename = f"sketch_photo_{photo_counter}.jpg"
        cv2.imwrite(filename, sketch)
        print(f"Photo saved: {filename}")
        photo_counter += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
