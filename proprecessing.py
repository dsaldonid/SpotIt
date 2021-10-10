import cv2 
import numpy as np

#applying different filters to facilitate edge detection
img = cv2.imread("Images/spotItCard.jpg")
kernel = np.ones((5,5),np.uint8)


imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(img,100,100)
imgDialation = cv2.dilate(imgCanny,kernel,iterations =1)
imgEroded = cv2.erode(imgDialation,kernel,iterations =1)

cv2.imshow("Gray Image",imgGray)
cv2.imshow("Blurred Image",imgBlur)
cv2.imshow("Canny Image",imgCanny)
cv2.imshow("Dilated Image",imgDialation)
cv2.imshow("Eroded Image",imgEroded)


img = cv2.imread("Images/spotItSymbols.jpg")
kernel = np.ones((5,5),np.uint8)


imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(img,25,25)
imgDialation = cv2.dilate(imgCanny,kernel,iterations =1)
imgEroded = cv2.erode(imgDialation,kernel,iterations =1)


cv2.imshow("Gray Image",imgGray)
cv2.imshow("Blurred Image",imgBlur)
cv2.imshow("Canny Image",imgCanny)
cv2.imshow("Dilated Image",imgDialation)
cv2.imshow("Eroded Image",imgEroded)



#drawing edges to contours above a particular area threshold
img = cv2.imread("Images/allSymbols.jpg",0)

image,contours,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

print(hierarchy)
external_contours = np.zeros(image.shape)

for i in range(len(contours)):
    
    #external contours
    if hierarchy[0][i][3] == -1:
        
        cv2.drawContours(external_contours,contours,i,255,-1)

cv2.imshow("Image Contour",external_contours)



#using Canny algorithm to spot external contours of shapes        
imgResize = cv2.resize(img,(600,600))
imgCanny = cv2.Canny(imgResize,100,60)

imgContour = imgResize.copy()
def getContours(img):
    _,contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >25:
            cv2.drawContours(imgContour,cnt,-1,(255,0,0),1)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02 *peri,True)
            objCor = len(approx)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),1)

getContours(imgCanny)
cv2.imshow("Image Contour",imgContour)



#using match feature to detect symbols within image
#low accuracy for transformed images

all_symbols = cv2.imread('Images/allSymbols.jpg',cv2.IMREAD_UNCHANGED)
card = cv2.imread('Images/cards.jpg',cv2.IMREAD_UNCHANGED)
img_rotate_90_clockwise = cv2.rotate(all_symbols, cv2.ROTATE_90_CLOCKWISE)
ice_cube = cv2.imread('Images/ice_cube.jpg',cv2.IMREAD_UNCHANGED)
clock = cv2.imread('Images/clock.jpg',cv2.IMREAD_UNCHANGED)
exclamation = cv2.imread('Images/exclamation.jpg',cv2.IMREAD_UNCHANGED)

# print(all_symbols.shape)
results = cv2.matchTemplate(card,clock,cv2.TM_CCOEFF_NORMED)

min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(results)

print('Best match top left position: %s'% str(max_loc))
print('Best match confidence: %s' % str(max_val))


threshold = 0.35
locations = np.where(results >= threshold)
locations = list(zip(*locations[::-1]))


if locations:
    
    clock_w = clock.shape[1]
    clock_h = clock.shape[0]
    
    for loc in locations:
        top_left = loc
        bottom_right = (top_left[0] + clock_w, top_left[1] + clock_h)
    
        cv2.rectangle(card,top_left,bottom_right, color = (0,255,0), thickness =2,
                     lineType = cv2.LINE_4)
    cv2.imshow("Result",card)
    
    
#using ORB to match features between objects
 
path = 'Images/Classes'
images = []
classNames = []
myList = os.listdir(path)

#ORB (Oriented Fast and Rotated Brief)
orb = cv2.ORB_create(nfeatures=500)

for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)
def findDes(images):
    desList = []
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

desList = findDes(images)


def findID(img,desList):
    kp2,des2 = orb.detectAndCompute(img,None)
    # #create brute force matcher
    bf = cv2.BFMatcher()
    matchList = []
    findVal = -1
    for des in desList:
        matches = bf.knnMatch(des,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append([m])
        matchList.append(len(good))
    
        
        

exclamation = cv2.imread('Images/Classes/exclamation.jpg',0)
card = cv2.imread('Images/ex1Card.jpg',0)
# exclamation = cv2.resize(exclamation,(100,100))


# #des are array of features
kp1, des1 = orb.detectAndCompute(exclamation,None)
kp2, des2 = orb.detectAndCompute(card,None)

# #draw key points
imgKp1 = cv2.drawKeypoints(exclamation,kp1,None)
imgKp2 = cv2.drawKeypoints(card,kp2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
good = []
for m,n in matches:
    if m.distance < .75*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(exclamation,kp1,card,kp2,good,None,flags =2)

cv2.imshow('Kp1',imgKp1)
cv2.imshow('Kp2',imgKp2)
cv2.imshow('exclamation',exclamation)
cv2.imshow('Card',card)
cv2.imshow('Matches',img3)

