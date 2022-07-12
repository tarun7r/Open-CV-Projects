import cv2


# loading the image In which we want to detect the faces
img = cv2.imread(r'C:\Users\tarun\Desktop\open\smile.jpg')

# loading the trained data which we can use for the refernce
face_trained = cv2.CascadeClassifier(r'C:\Users\tarun\Desktop\open\face.xml')
smile_trained = cv2.CascadeClassifier(r'C:\Users\tarun\Desktop\open\smile.xml')


# convert the image to grayscaled so that we can get rid of extra data and computation easily
BandW_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# face cordinates detection and store it in an array
faces = face_trained.detectMultiScale(BandW_img)



# Drawing the rectangle
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    # extracting the face and comapring it with smile loadset
    the_face = img[y:y+h,x:x+w]
    face_gray = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)
    smiles = smile_trained.detectMultiScale(face_gray,scaleFactor =1.2,minNeighbors=10)
    
    #for(a,b,c,d) in smiles:
        #drawing the smile 
        #cv2.rectangle(the_face,(a,b),(a+c,b+d),(0,255,0),2)
    
    if len(smiles)>0:
        cv2.putText(img,"smiling",(x,y+h+10),fontScale=0.5,fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,color=(0,0,0))



# displaying the final image with faces identified with rectangle box
cv2.imshow('Smile detector',img)
cv2.waitKey()


