import numpy as np
from scipy.misc import imshow, imsave, imread, imresize
from scipy.ndimage import filters, morphology
from skimage import filter
from skimage.draw import line
from scipy import ndimage, spatial
from sklearn import datasets, svm, metrics


def houghLines(img):
    w,h = img.shape
    acc=[]
    for i in range(h):
        rr,cc = line(0, i, w-1, h-i-1)
        acc.append(np.sum(img[rr, cc]))
        #print acc[i]
    mi = np.argmax(acc)
    ret = np.zeros(img.shape, dtype=np.bool)
    rr,cc = line(0, mi, w-1, h-mi-1)
    ret[rr,cc]=True
    return ret
    #print mi, acc[mi]
        

def removeLines(img, n):
    imfft = np.fft.fft2(imggray)
    imffts = np.fft.fftshift(imfft)
    
    mags = np.abs(imffts)
    angles = np.angle(imffts)
    
    visual = np.log(mags)

    #visual2 = (visual - visual.min()) / (visual .max() - visual.min())*255
    
    #print np.mean(visual)
    visual3 = np.abs(visual.astype(np.int16) - np.mean(visual))
    
    ret = houghLines(visual3)
    ret = morphology.binary_dilation(ret )
    ret = morphology.binary_dilation(ret )
    ret = morphology.binary_dilation(ret )
    ret = morphology.binary_dilation(ret )
    ret = morphology.binary_dilation(ret )
    w,h=ret.shape
    ret[w/2-3:w/2+3, h/2-3:h/2+3]=False
    
    
    delta = np.mean(visual[ret]) - np.mean(visual)
    imsave("visual_re" + str(n) + ".jpg", visual)
    
    visual_blured = ndimage.gaussian_filter(visual, sigma=5)
    
    #visual[ret] = np.minimum(visual[ret], np.mean(visual)) # visual[vismask] - delta
    
    visual[ret] =visual_blured[ret]
    imsave("visual_ret" + str(n) + ".jpg", visual)
    
    newmagsshift = np.exp(visual)
    
    newffts = newmagsshift * np.exp(1j*angles)
    
    newfft = np.fft.ifftshift(newffts)
    
    imrev = np.fft.ifft2(newfft)
    
    newim2 =  np.abs(imrev).astype(np.uint8)
    
    #newim2[newim2<20] = 255
    newim2 = np.maximum(newim2, img)
    
    
    
    #newim2 = morphology.grey_closing(newim2, 3 )
    
    
    return newim2

class BBox(object):
    def __init__(self, x1, y1, x2, y2):
        '''
        (x1, y1) is the upper left corner,
        (x2, y2) is the lower right corner,
        with (0, 0) being in the upper left corner.
        '''
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    def taxicab_diagonal(self):
        '''
        Return the taxicab distance from (x1,y1) to (x2,y2)
        '''
        return self.x2 - self.x1 + self.y2 - self.y1
    def overlaps(self, other):
        '''
        Return True iff self and other overlap.
        '''
        return not ((self.x1 > other.x2)
                    or (self.x2 < other.x1)
                    or (self.y1 > other.y2)
                    or (self.y2 < other.y1))
    def __eq__(self, other):
        return (self.x1 == other.x1
                and self.y1 == other.y1
                and self.x2 == other.x2
                and self.y2 == other.y2)

def remove_overlaps(bboxes):
    '''
    Return a set of BBoxes which contain the given BBoxes.
    When two BBoxes overlap, replace both with the minimal BBox that contains both.
    '''
    # list upper left and lower right corners of the Bboxes
    corners = []

    # list upper left corners of the Bboxes
    ulcorners = []

    # dict mapping corners to Bboxes.
    bbox_map = {}

    for bbox in bboxes:
        ul = (bbox.x1, bbox.y1)
        lr = (bbox.x2, bbox.y2)
        bbox_map[ul] = bbox
        bbox_map[lr] = bbox
        ulcorners.append(ul)
        corners.append(ul)
        corners.append(lr)        

    # Use a KDTree so we can find corners that are nearby efficiently.
    tree = spatial.KDTree(corners)
    new_corners = []
    for corner in ulcorners:
        bbox = bbox_map[corner]
        # Find all points which are within a taxicab distance of corner
        indices = tree.query_ball_point(
            corner, bbox_map[corner].taxicab_diagonal(), p = 1)
        for near_corner in tree.data[indices]:
            near_bbox = bbox_map[tuple(near_corner)]
            if bbox != near_bbox and bbox.overlaps(near_bbox):
                # Expand both bboxes.
                # Since we mutate the bbox, all references to this bbox in
                # bbox_map are updated simultaneously.
                bbox.x1 = near_bbox.x1 = min(bbox.x1, near_bbox.x1)
                bbox.y1 = near_bbox.y1 = min(bbox.y1, near_bbox.y1) 
                bbox.x2 = near_bbox.x2 = max(bbox.x2, near_bbox.x2)
                bbox.y2 = near_bbox.y2 = max(bbox.y2, near_bbox.y2) 
    return set(bbox_map.values())

def slice_to_bbox(slices):
    for s in slices:
        dy, dx = s[:2]
        yield BBox(dx.start, dy.start, dx.stop+1, dy.stop+1)

def findDigitCanditates(img):
    label, numfeatures = ndimage.measurements.label(~img)
    data_slices = ndimage.find_objects(label)
    boxes = remove_overlaps(slice_to_bbox(data_slices))
    return boxes

def drawBounds(img, boxes):
    outimg = np.zeros((img.shape[0], img.shape[1], 3))
    outimg[:,:]=[255,255,255]
    outimg[~img]=[0,0,0]
    
    for box in boxes:
        if abs(box.x1-box.x2) < 10 or abs(box.y1-box.y2) < 15:
            continue
        
        rr,cc = line(box.x1, box.y1, box.x1, box.y2)
        try:
            outimg[cc,rr]=[255,0,0]
        except IndexError:
            pass
        rr,cc = line(box.x1, box.y1, box.x2, box.y1)
        try:
            outimg[cc,rr]=[255,0,0]
        except IndexError:
            pass
        rr,cc = line(box.x1, box.y2, box.x2, box.y2)
        try:
            outimg[cc,rr]=[255,0,0]
        except IndexError:
            pass
        rr,cc = line(box.x2, box.y1, box.x2, box.y2)
        try:
            outimg[cc,rr]=[255,0,0]
        except IndexError:
            pass
    return outimg
        

def trainSVM():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    print "data:", data[0].shape
    classifier = svm.SVC(gamma=0.001, C=100.)

    # We learn the digits on the first half of the digits
    classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])
    #print digits.target[:n_samples / 2]
    return classifier
        
def recogniseDigits(img, boxes, nimg, clf):
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    for i, box in enumerate(boxes):
        if abs(box.x1-box.x2) < 10 or abs(box.y1-box.y2) < 15:
            continue
        
        res = img[box.y1:box.y2, box.x1:box.x2]
        resrs = imresize(res, (8,8))
        predicted = clf.predict(255-resrs.reshape((64,)))
        print "pred1", predicted
        #predicted = clf.predict(data[(n_samples / 2)+i])
        #print "pred2", predicted
        
        imsave("result_box_" + str(nimg)+"_"+ str(i) +".jpg", 255-resrs)
        #imsave("result_box_" + str(nimg)+"_"+ str(i) +"_sam"+".jpg", data[(n_samples / 2)+i].reshape((8,8)))
    

if __name__ == '__main__':
    files = ["color2.jpeg", "digitchar.jpeg"]
    
    for i, fname in enumerate(files):
        img = imread(fname)[:,:,:3]
        imggray = np.mean(img, -1)
        res = removeLines(imggray, i)
        imsave("result" + str(i) +".jpg", res)
        
        val = filter.threshold_otsu(res)
        
        imsave("result_otsu" + str(i) +".jpg", res > val)
        imsave("result_otsu_inv" + str(i) +".jpg", res > val)
        boxes = findDigitCanditates(res > val)
        bounds = drawBounds(res > val, boxes)
        imsave("result_box" + str(i) +".jpg", bounds)
        clf = trainSVM()
        recogniseDigits(res > val, boxes, i, clf)
    