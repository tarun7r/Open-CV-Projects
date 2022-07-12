import cv2

# loading the image In which we want to detect the faces
img = cv2.imread(r'C:\Users\tarun\Desktop\open\mars.png')


# loading the trained data which we can use for the refernce
trained_data = cv2.CascadeClassifier(r'C:\Users\tarun\Desktop\open\face.xml')


# convert the image to grayscaled so that we can get rid of extra data and computation easily
BandW_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



# face cordinates detection and store it in an array
face_coordinates = trained_data.detectMultiScale(BandW_img)


# Drawing the rectangle
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)




# displaying the final image with faces identified with rectangle box
cv2.imshow('Face detector',img)
cv2.waitKey()
