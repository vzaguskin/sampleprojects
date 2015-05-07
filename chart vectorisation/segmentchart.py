import numpy as np
from scipy.misc import imshow, imsave, imread
from scipy.ndimage import filters, morphology, measurements
from skimage.draw import line, circle

img = imread("laGK6.jpg")

r = img[:,:, 0]
g = img[:,:, 1]
b = img[:,:, 2]

mask = (r.astype(np.float)-np.maximum(g,b) ) > 20

mask2 = morphology.binary_erosion(mask)
mask2 = morphology.binary_erosion(mask2)
mask2 = morphology.binary_erosion(mask2)
mask2 = morphology.binary_erosion(mask2)

mask2 = morphology.binary_dilation(mask2)

label, numvertices = measurements.label(mask2)




mc = measurements.center_of_mass(mask2, label, range(1,numvertices+1) )






arr = range(numvertices)

connections=[]
for i in range( numvertices):
    arr.remove(i)
    for j in arr:
        rr,cc = line(mc[i][0], mc[i][1], mc[j][0], mc[j][1])
        ms = np.sum(mask[rr,cc]).astype(np.float)/len(rr)
        if ms > 0.9:
            connections.append((i,j))
            

print "vertices: ", mc
print "connections: ", connections

mask3 = np.zeros(mask2.shape, dtype=np.uint8)
mask3[:]=255

for p in mc:
    rr, cc = circle(p[0], p[1], 5)
    #mask3[p[0], p[1]]=255
    mask3[rr,cc]=20
    
for cn in connections:
    i, j = cn
    rr,cc = line(mc[i][0], mc[i][1], mc[j][0], mc[j][1])
    mask3[rr,cc]=12

imsave("vectorized.jpg", mask3)
        
    
    
    
    



