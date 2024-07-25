import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL as pic
import pandas as pd
import joblib


from mss import mss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix








#Left click on image, getting the values of areas and find the mean
def Mouse_Event(event, x, y, flags, param):
     if event == cv2.EVENT_LBUTTONDOWN:
         #read colour 3 by 3
        print("coordinate:", x, y)
        c = -2  #adjust for area size
        size = 5 #adjust for area size, odd numbers
        i = -1
        temp = []
        temp2 = []
        col = []
        lumi = []
        lumi_2 = []
        lumi_3 = []
        for i in range(size):
            c = c+i
            temp2.append((y+c, x+1))
            temp2.append((y+c, x))
            temp2.append((y+c, x-1))
        #print(temp2)
        temp3 = np.array(temp2)
        #print(temp3)
        

        

        for i in range(size*3):
            col.append(sct_img.pixel(temp3[i][1], temp3[i][0]))
        

        col = np.array(col)
        #print(col)

        
        #print("Here: ", col[8][2])
        
        

        for j in range(3):
            for i in range(size*3):
                #print((i,j))
                lumi.append((0.299*int(col[i][j])) + (0.587*int(col[i][j])) + (0.114*int(col[i][j]))) #formula1
                lumi_2.append((0.299*(col[i][j]**2) + 0.587*(col[i][j]**2) + 0.114*(col[i][j]**2))**0.5) #formula2
                lumi_3.append(0.2126*int(col[i][j]) + 0.7152*int(col[i][j]) + 0.0722*int(col[i][j])) #formula3


        #print("Lumi_1", lumi)
        #print("Lumi_2", lumi_2)


        mean = sum(lumi)/len(lumi)
        #print("Mean Lumi1:", mean)

        mean_2 = sum(lumi_2)/len(lumi_2)
        #print("Mean Lumi2:", mean_2)

        mean_3 = sum(lumi_3)/len(lumi_3)
        print("Mean Luminance:", mean_3, "\n\n")

        #####Prediction stuff####
            
        
        hazard = switch(choice, mean_3)


        blank = np.zeros((100, 800, 3), np.uint8) #create a blank image
        blank[:,:] = (252, 220, 166)
        cv2.waitKey(1)
        text = "Luminance:" + str(round(mean_3, 2)) + "Prediction: " + str(hazard)
        cond2 = cv2.putText(blank, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("condition", cond2)


####Prediction stuff#######
def switch(choose, lu):
            if choose == "Al":
                saved_model = joblib.load('/Users/kjx/Desktop/Al.ckp')
                prediction = saved_model.predict(np.array(lu).reshape(-1,1))
                if prediction == 1:
                     msg = 'Hazardous'
                else:
                     msg = 'Not Hazardous'
                return msg
            elif choose == "Mg":
                saved_model = joblib.load('/Users/kjx/Desktop/Mg.ckp')
                prediction = saved_model.predict(np.array(lu).reshape(-1,1))
                if prediction == 1:
                     msg = 'Hazardous'
                else:
                     msg = 'Not Hazardous'
                return msg
            elif choose == "Co":
                #saved_model = joblib.load('')
                prediction = 1
                if prediction == 1:
                     msg = 'Hazardous'
                else:
                     msg = 'Not Hazardous'
                return msg
            elif choose == "Fe":
                saved_model = joblib.load('/Users/kjx/Desktop/Fe.ckp')
                prediction = saved_model.predict(np.array(lu).reshape(-1,1))
                if prediction == 1:
                     msg = 'Hazardous'
                else:
                     msg = 'Not Hazardous'
                return msg
            elif choose == "Cu":
                saved_model = joblib.load('/Users/kjx/Desktop/Cu.ckp')
                prediction = saved_model.predict(np.array(lu).reshape(-1,1))
                if prediction == 1:
                     msg = 'Hazardous'
                else:
                     msg = 'Not Hazardous'
                return msg
            else:
                 print('Please key in the correct element >>> "Al", "Mg", "Co", "Fe", "Cu"')
                 




bounding_box = {'top': 0, 'left': 0, 'width': 700, 'height': 1710}

sct = mss()

choice = input("Please key in your element:")
    

while True:
    sct_img = sct.grab(bounding_box)
    cv2.imshow('screen', np.array(sct_img))
    cv2.setMouseCallback('screen', Mouse_Event)

#W key to change input
    if (cv2.waitKey(1) & 0xFF) == ord('w'):
         choice = input("Please key in your element:")

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)  
        break


