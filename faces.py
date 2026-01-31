# Import required libraries
import cv2              # OpenCV for image processing and webcam access
import dlib             # dlib for face detection
import os               # OS operations like folder creation
import sys              # System-related functions (not used directly here)
import random           # For random lighting augmentation

# Directory to store captured face images
output_dir = './my_faces'

# Size to which each face image will be resized
size = 64

# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to change lighting conditions of the image
# light -> controls brightness
# bias  -> controls contrast
def relight(img, light=1, bias=0):
    w = img.shape[1]    # Image width
    h = img.shape[0]    # Image height
    
    # Iterate over each pixel and color channel
    for i in range(0, w):
        for j in range(0, h):
            for c in range(3):
                # Apply lighting transformation
                tmp = int(img[j, i, c] * light + bias)
                
                # Clip pixel values to valid range [0, 255]
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                
                img[j, i, c] = tmp
    return img

# Initialize dlib's frontal face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Open default webcam (0)
camera = cv2.VideoCapture(0)

# Image counter
index = 1

# Start capturing face images
while True:
    # Capture up to 10,000 face images
    if index <= 10000:
        print('Being processed picture %s' % index)

        # Read frame from webcam
        success, img = camera.read()
        if not success:
            print("Failed to capture image")
            break

        # Convert frame to grayscale for face detection
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        dets = detector(gray_img, 1)

        # Loop through detected faces
        for i, d in enumerate(dets):
            # Get bounding box coordinates safely
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            # Crop face region from the original image
            face = img[x1:y1, x2:y2]

            # Apply random lighting augmentation
            face = relight(
                face,
                random.uniform(0.5, 1.5),
                random.randint(-50, 50)
            )

            # Resize face image to fixed size
            face = cv2.resize(face, (size, size))

            # Display the captured face
            cv2.imshow('image', face)

            # Save the face image to disk
            cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)

            # Increment image counter
            index += 1

        # Press ESC key to exit
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        print('Finished!')
        break

# Release webcam and close windows
camera.release()
cv2.destroyAllWindows()
