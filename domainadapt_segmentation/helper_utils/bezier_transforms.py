
import numpy as np
import random
import matplotlib.pyplot as plt
import monai 
from scipy.special import comb
from monai.transforms import MapTransform,Transform
from monai.utils import convert_to_tensor
from monai.data.meta_obj import get_track_meta


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x, prob=0.5,p1=[0.25,0.25],p2=[0.75,0.75],times_mod=100):
    #if random.random() >= prob:
    #    return x
    points = [[0, 0], p1, p2, [1, 1]]
    #xvals, yvals = bezier_curve(points, nTimes=100000)
    xvals, yvals = bezier_curve(points, nTimes=times_mod)
    xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def mod_entire_image(img,mask,p1,p2):
    f_mask = mask !=0 
    mod_kidney = nonlinear_transformation(img[f_mask],p1=p1,p2=p2,times_mod=50).astype(np.float32)
    print(mod_kidney.max())
    print(mod_kidney.min()) 
    img[f_mask]=  monai.data.MetaTensor(mod_kidney)
    print(img[f_mask].max())
    return img 

class Bezier(Transform):
    def __init__(self,b1=None,b2=None,update_meta=True):
        self.b1 = b1 
        self.b2 = b2 
        self.update_meta =  update_meta
    def __call__(self,img,mask):
        img = convert_to_tensor(img, track_meta=get_track_meta()) 
        new_img = mod_entire_image(img,mask,self.b1,self.b2)
        return new_img

class Bezierd(MapTransform): 
    def __init__(self,keys,update_meta=True,allow_missing_keys=False,b1=[0.25,0.5],b2=[0.2,0.75],label_key=None):
        MapTransform.__init__(self,keys=keys,allow_missing_keys=allow_missing_keys)
        self.label_key = label_key
        self.converter = Bezier(b1=b1,b2=b2,update_meta=update_meta)
    def __call__(self,data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key],mask=d[self.label_key])
        return d 