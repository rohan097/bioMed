import os
from PIL import Image


input_path = "/home/rohan/Course Projects/bioMed/Data/Preictal"
output_path = "/home/rohan/Course Projects/bioMed/Data/Preictal_Sampled/"
img_list = os.listdir(input_path)
k = 0
while k != 3200:
    for j in img_list:
        for i in range(4):
            img = Image.open(input_path+"/"+j)
            img.save(output_path+"precictal_%d.png" %k)
            k+=1
