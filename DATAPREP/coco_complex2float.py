import grup
import numpy as np
import sys


def complex2float(coco_in,tofile=False,postfix='float'):
    try:
        if type(coco_in)==np.ndarray: #generic numpy array
            arr_in=coco_in
        elif type(coco_in)==str: #coco filename
            i0=grup.raster(coco_in)
            arr_in=i0.z
        elif type(coco_in)==grup.raster: #grup raster class
            arr_in=coco_in.z
    except:
        print('ERROR: Cannot understand the data type of \'coco_in\'')
        arr_in=None
    
    if arr_in!=None:
        arr_out=np.zeros((2,arr_in.shape[0],arr_in.shape[1]))
        arr_out[0,:,:]=np.real(arr_in)
        arr_out[1,:,:]=np.imag(arr_in)

        if type(coco_in)==str and tofile:
            i0.z=arr_out
            i0.numbands=2
            i0.write(coco_in.replace('.tif','_{}.tif'.format(postfix)))


if __name__=='__main__':
    complex2float(sys.argv[1],True)
    
    

    