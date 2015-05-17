import sys
sys.path.append('H:/work_git/3rdparty/python-mnist/')
from mnist import MNIST
from scipy.misc import imsave
import numpy as np

import scipy as sp
from PIL import Image, ImageDraw, ImageFont
from scipy.signal.signaltools import correlate2d as c2d
from scipy.ndimage.interpolation import affine_transform
from math import sin, cos, radians

from sklearn.decomposition import RandomizedPCA, PCA, KernelPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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


def simpleKNN(choices, item):
    reslst=np.zeros(10)
    for i, d in enumerate(choices):
        mse = ((item - d) ** 2).mean(axis=None)
        reslst[i]=mse
    pred=np.argmin(reslst)
    return pred

def correlateKNN(choices, item):
    reslst=np.zeros(10)
    for i, d in enumerate(choices):
        c11 = c2d(item, d, mode='full')
        #print c11
        reslst[i]=c11.max()
    #print reslst
    pred=np.argmax(reslst)
    return pred


def afine(img, dx, dy, sx, sy, angle):
    imat=np.identity(2)
    imat[0,0]=imat[0,0]*sx
    imat[1,1]=imat[1,1]*sy
    #print imat
    #anglemat=np.identity(2)
    img =affine_transform(img, imat)
    #return img
    a=radians(angle)
    transform=np.array([[cos(a),-sin(a)],[sin(a),cos(a)]])
    centre=0.5*np.array(img.shape)
    offset=(centre-centre.dot(transform)).dot(np.linalg.inv(transform))
    finoff = -1*offset + [dx,dy]
    return affine_transform(img, transform, finoff)


def testafine(ims):
    im0 = np.array(ims[0]).reshape((28,28))
    for dx in [-5, 0, 5]:
        for dy in [-5, 0, 5]:
            for sx in [0.9, 1.0, 1.1, 1.3]:
                for sy in [0.9, 1.0, 1.1, 1.3]:
                    for angle in [-20, -10, 0, 10, 20]:
                        res=afine(im0, dx,dy, sx, sy,angle)
                        fname = str(dx)+str(dy)+str(sx)+str(sy)+str(angle)+".png"
                        imsave("testres/"+fname, res)


def mseaffine(im, tp):
    #im0 = np.array(ims[0]).reshape((28,28))
    mses=[]
    for dx in range(-5,5,2):
        for dy in range(-5,5,2):
            for sx in [0.9, 1.0, 1.1, 1.3]:
                for sy in [0.9, 1.0, 1.1, 1.3]:
                    for angle in range(-20,20,2):
                        res=afine(im, dx,dy, sx, sy,angle)
                        mse = ((res - tp) ** 2).mean(axis=None)
                        mses.append(mse)
    return np.min(np.array(mses))



def afineKNN(choices, item):
    reslst=np.zeros(10)
    for i, d in enumerate(choices):
        #mse = ((item - d) ** 2).mean(axis=None)
        mse = mseaffine(item, d)
        reslst[i]=mse
    pred=np.argmin(reslst)
    return pred


def recognizePCA(train, trainlab, test, labels, num=None):

    train4pca = np.array(train)

    n_components = 20

    print "fitting pca"
    pca = RandomizedPCA(n_components=n_components).fit(train4pca)
    #pca = PCA(n_components=n_components, whiten=False).fit(train4pca)
    print "fitted pca"

    xtrain = pca.transform(train4pca)

    if num is None:
        num=len(test)

    test4pca = np.array(test)

    xtest = pca.transform(test4pca)

    clf = KNeighborsClassifier()
    #clf = DecisionTreeClassifier()
    #clf=LinearSVC()
    print "fitting knn"
    clf = clf.fit(xtrain, trainlab)
    print "fitted knn"
    y_pred = clf.predict(xtest[:num])
    print "predicted"
    r=0
    w=0
    for i in range(num):
        if y_pred[i] == labels[i]:
            r+=1
        else:
            w+=1
    print "tested ", num, " digits"
    print "correct: ", r, "wrong: ", w, "error rate: ", float(w)*100/(r+w), "%"
    print "got correctly ", float(r)*100/(r+w), "%"

    return pca.components_
    #saveIm




digitslst = prepareDigits()

mndata = MNIST('H:/tools/MNIST/')
trainims, trainlabels = mndata.load_training()
ims, labels = mndata.load_testing()
#print len(ims[0])


def saveIm(ims, pref, n):
    for i in range(n):
        fname=pref + str(i)+".png"
        im0 = np.array(ims[i]).reshape((28,28))
        imsave("testres/"+fname, im0)

def runtest(ims, labels, predictFun, testlen=None):
    r=0
    w=0
    if testlen is None:
        testlen=len(labels)
    for n in range(testlen):
        #print i, mse
        im0 = np.array(ims[n]).reshape((28,28))
        pred = predictFun(digitslst, im0)
        print pred, labels[n]
        if pred==labels[n]:
            r+=1
        else:
            w+=1
    print r, w, float(w)/(r+w)


pcas = recognizePCA(trainims, trainlabels, ims, labels)
saveIm(pcas, "pcas", pcas.shape[0])
#testafine(ims)
#runtest(ims, labels, afineKNN, 10)
#saveIm(ims, "mnist", 10)
#saveIm(digitslst, "digits", 10)

#imsave("test1.jpg", im0)
#print labels[0]
