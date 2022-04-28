import cv2
import imutils
import matplotlib.pyplot as plt 
import numpy as np

plt.style.use('ggplot')

img = cv2.imread("C:\\users\\USER\\DOcuments\\car_plate3.jpeg",cv2.IMREAD_COLOR)
img = cv2.resize(img, (620, 480))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 13, 15, 15)

edged = cv2.Canny(gray,30,200) # Edge detection

contours = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours,key=cv2.contourArea,reverse=True)[:10]
screenCnt = None

for c in contours:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, 0.018 * peri,True)
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print('No countour is detected')
else:
    detected = 1

if detected == 1:
    f = cv2.drawContours(img, [screenCnt], -1,(255,0,0),5) 

mask = np.zeros(gray.shape,np.uint8)
new_img = cv2.drawContours(mask,[screenCnt],0,255,-1)
new_img = cv2.bitwise_and(img,img,mask=mask)

(x,y) = np.where(mask == 255)
(topx,topy) = (np.min(x),np.min(y))
(bottomx,bottomy) = (np.max(x),np.max(y))
cropped = gray[topx:bottomx+1,topy:bottomy+1]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.grid(False)
cv2.waitKey(0)
