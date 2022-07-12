import cv2

# loading the video In which we want to detect the cars
video = cv2.VideoCapture(r'C:\Users\tarun\Desktop\open\Tesla.mp4')


# loading the trained data which we can use for the refernce
trained_data = cv2.CascadeClassifier(r'C:\Users\tarun\Desktop\open\cars.xml')

while True:
    
    #Read the video frame by frame
    (read_stats,frame) = video.read()

    if read_stats:
        # convert the image to grayscaled so that we can get rid of extra data and computation easily
	    BandW_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # Car cordinates detection and store it in an array 
    car_coordinates = trained_data.detectMultiScale(BandW_img)
    
    # Drawing the rectangl
    for (x,y,w,h) in car_coordinates:
	    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    # displaying the final image with faces identified with rectangle box
    cv2.imshow('Face detector',frame)
    key = cv2.waitKey(1)
    
    if key ==81 or key==113:
        break