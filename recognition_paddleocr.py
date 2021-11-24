from paddleocr import PaddleOCR, draw_ocr
import numpy as np
# from matplotlib import pyplot as plt
from PIL import Image
import os
# import time

root_train = '../data/train'
root_test = '../data/test'

train_list = os.listdir(root_train)
test_list = os.listdir(root_test)

input_size = 1000

# # pre_resize
# img_temp = Image.open(root_train+'/114.jpg')
# if img_temp.size[0]==5334:
#     img_temp = img_temp.resize((int(5334*0.2),int(2475*0.2)))
#     img_temp.save(root_train+'/114.jpg')
# del img_temp

ocr = PaddleOCR()
file_train = 'train_list_temp.txt'
file_test = 'test_list_temp.txt'
def ocr_list(root,list,file_name):
    f = open(file_name,'w')
    for img_name in list:
        img_path = os.path.join(root,img_name)
        img = Image.open(img_path).convert('RGB')
        w,h = img.size
        if max(w,h) > input_size:
            resize_multi = input_size/max(w,h)
            new_w, new_h = int(w*resize_multi),int(h*resize_multi)
            img = img.resize((new_w,new_h))
            img.save(img_path)
            print(img_path,'is resized.')
        img = np.asarray(img)
        result = ocr.ocr(img)
        # time.sleep(1)
        f.write(img_name+'\t'+str(result)+'\n')
        print(img_path,'finished.')
    f.close()

ocr_list(root_train,train_list,file_train)
ocr_list(root_test,test_list,file_test)