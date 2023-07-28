import cv2
import cvzone
from statistics import mean
from cvzone.PlotModule import LivePlot
from cvzone.FaceMeshModule import FaceMeshDetector

ratio=[]
total=[]
blink_counter=0
counter=0

cap=cv2.VideoCapture('sample.mp4')
detector=FaceMeshDetector(maxFaces=1)#max number of faces in the video as one

ployY=LivePlot(640,360,[20,50],invert=True)

id_lst=[22,23,24,26,110,157,158,159,160,161,130,243]

while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES)==cap.get(cv2.CAP_PROP_FRAME_COUNT):


        #prop_pos_frame gives current frame we are at
        #frame count gives total no of frames, if both are equal we will set the pos of the frame to zero
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    success, img = cap.read()
    img, faces=detector.findFaceMesh(img,draw=False)
    if faces:
        face=faces[0]
        for id in id_lst:
            cv2.circle(img,face[id],5,(255,0,255),cv2.FILLED)

        leftup=face[159]
        leftdown=face[23]
        left_left=face[130]
        left_right=face[243]


        # one of the problem with this approach is the parameters change when you move your head forward and backwards
        #instead we will consider the ratios of the upper and lower lid instead of distance

        len_vertical,_=detector.findDistance(leftup,leftdown)
        cv2.line(img, leftup, leftdown, (0, 200, 0), 3)


        len_horizontal,_=detector.findDistance(left_left,left_right)
        cv2.line(img,left_left,left_right,(0,200,0),3)

        r=int((len_vertical/len_horizontal)*100)
        if len(ratio)>3:
            ratio.pop(0)
        ratio.append(r)
        total.append(r)
        ratio_avg=sum(ratio)/len(ratio)
        if ratio_avg<mean(total) and counter==0:
            '''This step is to count the occasions when avg falls below 35 and additional counter
            is used to prevent counting multiple frames of the same blink. so it will wait for 10 frames after 
            each blink
            '''
            blink_counter+=1
            counter=1
        if counter!=0:
            counter+=1
            if counter>14:
                counter=0


        cvzone.putTextRect(img,f'Blink count:{blink_counter}',(50,100))
        imgplot=ployY.update(ratio_avg)
        #cv2.imshow("Imageplot",imgplot)
        img=cv2.resize(img,(640,360))
        imgStack=cvzone.stackImages([img,imgplot],2,1)
    else:
        img=cv2.resize(img,(640,360))
        imgStack=cvzone.stackImages([img,img],1,1)




    #img=cv2.resize(img,(640,360))
    cv2.imshow("Image",imgStack)
    cv2.waitKey(25)