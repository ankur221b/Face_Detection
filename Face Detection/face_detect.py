import cv2
import os

curr_path = os.getcwd()

path =  curr_path + "/input_image/pic1.png"

# Read the image

image = cv2.imread(path)

# Create the haar cascade

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.08,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    )

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

out_path = curr_path + '/output_image' #output path

#saving the ouput_image

cv2.imwrite(os.path.join(out_path , 'out.jpg'), image)

