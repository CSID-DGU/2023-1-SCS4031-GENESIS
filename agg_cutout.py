import numpy as np
from os import listdir
from random import *
import  cv2
from google.colab.patches import cv2_imshow
#d1_images cutout 하기 
path ='/content/before_aug/images/val/'  #cutout을 추가하고 싶은 이미지가 있는 경로 
for file in listdir(path): #경로에 있는 파일 하나하나 
  cutout = cv2.imread(path+file) #사진 파일 1개 (cutout을 추가하고 싶은 원본사진)
  labelfile = file.replace('.jpg','.txt') 

  label = open('/content/before_aug/labels/val/'+labelfile,'r')
  list=label.readline() #label 첫 줄 읽기 
  for i in range(3):
    f = open('/content/labels/val/'+'cutout'+str(i)+'__'+labelfile,"w") 
    while(list):#바운딩 박스 당 
      #print(str(i) +list)
      f.write(list) #바운딩 박스 정보를 새로운 라벨 파일에 적는다. 
      #print(list)
      list=list.split()
      x_center=float(list[1])
      y_center=float(list[2])
      height=float(list[4])*832
      width=float(list[3])*1664
      x_center_new=x_center*1664
      y_center_new=y_center*832
       #print(x_center_new,y_center_new)
      x1=uniform(x_center_new-width/2,x_center_new+width/4) 
      y1=uniform(y_center_new-height/2,y_center_new+height/4)
      mark=width/2
      x2=x1+mark
      y2=y1+mark
      x1=int(x1)
      x2=int(x2)
      y1=int(y1)
      y2=int(y2)
     #print(x1," ",y1," ",x2," ",y2)       
      cutout = cv2.rectangle(cutout, (x1, y1), (x2,y2), (0, 0, 0), cv2.FILLED)
      list=label.readline() #다음 바운딩 박스를 위해 다음 줄을  읽는다.  
    f.close()
    #cv2_imshow(cutout)
    label = open('/content/before_aug/labels/val/'+labelfile,'r')
    list=label.readline()
    cv2.imwrite('/content/images/val/'+'cutout'+str(i)+'__'+file,cutout)
    cutout = cv2.imread(path+file)
    #라벨 복사