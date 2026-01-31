# -*- coding: utf-8 -*-
# Script to extract faces from images and store them as "other faces" dataset

import sys              # System exit handling
import os               # File and directory operations
import cv2              # OpenCV for image processing
import dlib             # dlib for face detection

# Input directory containing images
input_dir = './input_img'

# Output directory to store extracted face images
output_dir = './other_faces'

# Size to resize face images
size = 64

# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize dlib's frontal face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Counter for saved face images
index = 1

# Traverse input directory and its subdirectories
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        # Process only JPG images
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % index)

            # Construct full image path
            img_path = path + '/' + filename

            # Read image from file
            img = cv2.imread(img_path)

            # Convert image to grayscale for face detection
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            dets = detector(gray_img, 1)

            # Loop through detected faces
            for i, d in enumerate(dets):
                # Ensure coordinates are within image boundaries
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                # Crop face region from original image
                # img[y:y+h, x:x+w]
                face = img[x1:y1, x2:y2]

                # Resize face image to fixed size
                face = cv2.resize(face, (size, size))

                # Display extracted face
                cv2.imshow('image', face)

                # Save face image to output directory
                cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)

                # Increment image counter
                index += 1

            # Press ESC key to exit
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)
