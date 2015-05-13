import sys
sys.path.append('H:/work_git/3rdparty/python-mnist/')
from mnist import MNIST
from scipy.misc import imsave
import numpy as np

import scipy as sp
from PIL import Image, ImageDraw, ImageFont

def prepareDigits():
    lstout=[]
    fontsize=25
    font = ImageFont.truetype("arial.ttf", fontsize)
    for i in range(10):
        sx,sy=font.getsize(str(i))
    #f = ImageFont.load_default()
        txt=Image.new('L', (28,28))
        d = ImageDraw.Draw(txt)
        d.text( ((28-sx)/2, (28-sy)/2), str(i),  font=font, fill=255)
        arr=np.array(txt)
        lstout.append(arr)
    return lstout
        #imsave(str(i)+".png", arr)
        #txt.save(str(i)+".png", "PNG")



digitslst = prepareDigits()

mndata = MNIST('H:/tools/MNIST/')
#mndata.load_training()
ims, labels = mndata.load_testing()
#print len(ims[0])
im0 = np.array(ims[0]).reshape((28,28))

for i, d in enumerate(digitslst):
    mse = ((im0 - d) ** 2).mean(axis=None)
    print i, mse

print labels[0]


#imsave("test1.jpg", im0)
#print labels[0]
