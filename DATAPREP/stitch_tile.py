import glob
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os

path_tile='/Users/seongsu/Desktop/ACCESS/PRED/TRAIN_atrous_32init_drop0.2_customLossR727.dir'
path_stitched='/Users/seongsu/Desktop/ACCESS/PRED/STITCHED'
if not os.path.exists(path_stitched):
    os.makedirs(path_stitched)

#path_tile='/Users/seongsu/Desktop/ACCESS/atrous_32init_drop0.2_customLossR727.dir_ALL'
nx_tile=512
ny_tile=512
flag_gaussian_weight=True
sigma_kernel=0.5
#buid the kernel

if flag_gaussian_weight:
    gx=np.arange(nx_tile)
    gx=(gx-gx[-1]/2.0)/(nx_tile/2)
    gy=np.arange(ny_tile)
    gy=(gy-gy[-1]/2.0)/(ny_tile/2)
    gxx,gyy=np.meshgrid(gx,gy)

    kernel_weight=np.exp(-(gxx**2+gyy**2)/sigma_kernel)
    del(gx,gy,gxx,gyy)
    
else:
    kernel_weight=np.ones((ny_tile,nx_tile))



list_tile=glob.glob('{}/pred*.png'.format(path_tile))

dict_dinsar_tile={}

print('Identifying the tiles and source DInSAR names...')
for tilename in list_tile:
    name_dinsar=tilename.split('/')[-1].split('pred_')[1].split('_x')[0]
    if not name_dinsar in dict_dinsar_tile.keys():
        dict_dinsar_tile[name_dinsar]=[tilename]
    else:
        dict_dinsar_tile[name_dinsar].append(tilename)

list_dinsar_src=list(dict_dinsar_tile.keys())

print('Done!')
for dinsar_to_stitch in list_dinsar_src:
    print(dinsar_to_stitch,'- # tile:',len(dict_dinsar_tile[dinsar_to_stitch]))
    filename_dinsar_stitch='{}/{}.png'.format(path_stitched,dinsar_to_stitch)
    list_tile_to_stitch=dict_dinsar_tile[dinsar_to_stitch]
    numtiles=len(list_tile_to_stitch)
    list_x0=np.array([0]*numtiles,dtype=np.int32)
    list_y0=np.array([0]*numtiles,dtype=np.int32)
    for i,tile_to_stitch in enumerate(list_tile_to_stitch):
        list_x0[i]=int(tile_to_stitch.split('_x')[1].split('_y')[0])
        list_y0[i]=int(tile_to_stitch.split('_y')[1].split('_DIR')[0])
    
    #determine the output tile size
    nx_out=list_x0.max()+nx_tile
    ny_out=list_y0.max()+ny_tile

    arr_sum=np.zeros((ny_out,nx_out))
    arr_weight=np.zeros((ny_out,nx_out))
    
    for i,tile_to_stitch in enumerate(list_tile_to_stitch):
        tile_in=misc.imread(tile_to_stitch)
        #arr_sum[list_y0[i]:list_y0[i]+ny_tile,list_x0[i]:list_x0[i]+nx_tile]=tile_in.astype(np.float)
        #arr_weight[list_y0[i]:list_y0[i]+ny_tile,list_x0[i]:list_x0[i]+nx_tile]=(tile_in>0).astype(np.float)
        arr_sum[list_y0[i]:list_y0[i]+ny_tile,list_x0[i]:list_x0[i]+nx_tile]=arr_sum[list_y0[i]:list_y0[i]+ny_tile,list_x0[i]:list_x0[i]+nx_tile]+tile_in.astype(np.float)*kernel_weight
        arr_weight[list_y0[i]:list_y0[i]+ny_tile,list_x0[i]:list_x0[i]+nx_tile]=arr_weight[list_y0[i]:list_y0[i]+ny_tile,list_x0[i]:list_x0[i]+nx_tile]+kernel_weight
        
    
    arr_out=arr_sum/arr_weight
    arr_out[np.isnan(arr_out)]=0.0
    plt.subplot(1,2,1)
    plt.imshow(arr_out)
    #plt.title(dinsar_to_stitch)
    plt.subplot(1,2,2)
    plt.imshow(arr_out>200)
    #Question:
    # - Is PNG okay for the stitched raw result?
    # - How do we preserve the geocoding information in the raw data?
    misc.imsave(filename_dinsar_stitch,(arr_out/arr_out.max()*255).astype(np.ubyte))
    
    #plt.show()



