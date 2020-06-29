#! /usr/bin/env python3
# GRUP (GDAL Raster Utility for Python)
#
# Depencency: GDAL, Numpy and Scipy
#             matplotlib if using the visualization functionality
#
# Initial code in 08/30/2018 by Seongsu Jeong (UCI)
#
# Change log:
# 12/04/2018 - Supports multiband save
#            - Going bigtiff is optional: see geotiff.go_bigtiff()
#
# 01/25/2019 - Extending the code to work not only for geotiff but also other GDAL-compatible raster
# 02/08/2019 - Support for GDAL Virtual File Systems
#            - Automatically set logscale=True in grup.visualize() unless the user explicitly define that
# 02/11/2019 - Default mode for visualizing complex data is "intensity-scaled phase" (on test)
# 
# Note:
# - Default driver (i.e. file format) for I/O is geotiff.
# - Default action of this code in terminal is viewing mode
# 
#  
from osgeo import gdal
from osgeo import osr #TODO: check if importing this is really necessary - veridct: YES
import numpy as np
import os
import scipy
import sys
import matplotlib

default_raster_options={'GTiff':['COMPRESS=LZW']}

class raster:
    def __init__(self,filename_raster=None,nx=1,ny=1,numbands=1,dtype='uint8'):
        if filename_raster==None:
            self.filename=''
            self.str_driver=None #default driver for i/o is geotiff
            self.numbands=1
            self.x=np.zeros((1,1))
            self.y=np.zeros((1,1))
            self.z=np.zeros((1,1),dtype='uint8')
            self.nx=1
            self.ny=1
            self.nz=1
            self.GeoTransform=(0,0,0,0,0,0)
            self.Projection=''
            #self.option=['COMPRESS=LZW']
            self.option=[]
            self.nodata=None
            self.isArea=True
        else:
            self.option=[]
            self.read(filename_raster)
    
    def go_bigtiff(self,force=False): #format-specifit function
        if self.z.dtype==np.uint8 or self.z.dtype==np.int8:
            bytes_per_pixel=1
        elif self.z.dtype==np.uint16 or self.z.dtype==np.int16:
            bytes_per_pixel=2
        elif self.z.dtype==np.uint32 or self.z.dtype==np.int32:
            bytes_per_pixel=4
        elif self.z.dtype==np.float32:
            bytes_per_pixel=4
        elif self.z.dtype==np.float64:
            bytes_per_pixel=8
        elif self.z.dtype==np.complex64:
            bytes_per_pixel=8
        elif self.z.dtype==np.complex128:
            bytes_per_pixel=16
        
        est_bytes=self.nx*self.ny*self.numbands*bytes_per_pixel
        if est_bytes >= 4*1024*1024*1024: #4GB
            return True
        else:
            if force:
                self.option.append('BIGTIFF=YES')
                return True
            else:
                return False
        
    def read(self,filename_raster):
        self.filename=filename_raster
        if os.path.isfile(self.filename) or 'vsi' in self.filename[:10] or self.filename.startswith('NETCDF'):
            raster_in=gdal.Open(filename_raster)

            if raster_in != None:
                self.str_driver=raster_in.GetDriver().GetDescription()
                if self.str_driver in default_raster_options.keys():
                    self.option+=default_raster_options[self.str_driver]
                
                self.z=raster_in.ReadAsArray()
                self.GeoTransform=raster_in.GetGeoTransform()
                self.Projection=raster_in.GetProjection()
                self.nx=raster_in.RasterXSize
                self.ny=raster_in.RasterYSize
                self.nz=self.nx*self.ny
                band1=raster_in.GetRasterBand(1)
                meta_raster=raster_in.GetMetadata_Dict()
                try:
                    if meta_raster['AREA_OR_POINT']=='Area':
                        self.isArea=True
                        self.x=np.linspace(self.GeoTransform[0]+self.GeoTransform[1]/2,self.GeoTransform[0]+self.nx*self.GeoTransform[1]-self.GeoTransform[1]/2,num=self.nx)
                        self.y=np.linspace(self.GeoTransform[3]+self.GeoTransform[5]/2,self.GeoTransform[3]+self.ny*self.GeoTransform[5]-self.GeoTransform[5]/2,num=self.ny)

                    elif meta_raster['AREA_OR_POINT']=='Point':
                        self.isArea=False
                        self.x=np.linspace(self.GeoTransform[0],self.GeoTransform[0]+self.nx*self.GeoTransform[1],num=self.nx)
                        self.y=np.linspace(self.GeoTransform[3],self.GeoTransform[3]+self.ny*self.GeoTransform[5],num=self.ny)
                except:
                    print('    NOTE: Cannot find geocoding information from the source file')

                self.nodata=band1.GetNoDataValue()
                
                if len(self.z.shape)<=2:
                    self.numbands=1
                else:
                    self.numbands=self.z.shape[0]

                #set if BIGTIFF option is necessary
                if self.str_driver=='GTiff':
                    self.go_bigtiff()
            else:
                print('Not a rgeular GDAL-compatible file. Attempting to load the file as npy file.')
                self.z=np.load(filename_raster)
                if len(self.z.shape)==2:
                    self.numbands=1
                else:
                    self.numbands=self.z.shape[2]

            return 0
        else:
            print('ERROR: File does not exist -',filename_raster)
            return 1
            
    def write(self,filename_raster):
        #define str_drive if it has not set yet
        if self.str_driver==None:
            self.str_driver='GTiff'
            
        if self.filename==None:
            print('Error: Filename is not specified.')
            return 2
        driver=gdal.GetDriverByName(self.str_driver)
        if self.z.dtype=='uint8':
            output=driver.Create(filename_raster,\
                                 self.nx,self.ny,self.numbands,\
                                 gdal.GDT_Byte,
                                 self.option)
        elif self.z.dtype=='uint16':
            output=driver.Create(filename_raster,\
                                 self.nx,self.ny,self.numbands,\
                                 gdal.GDT_UInt16,
                                 self.option)
        elif self.z.dtype=='int16':
            output=driver.Create(filename_raster,\
                                 self.nx,self.ny,self.numbands,\
                                 gdal.GDT_Int16,
                                 self.option)
        elif self.z.dtype=='float32':
            output=driver.Create(filename_raster,\
                                 self.nx,self.ny,self.numbands,\
                                 gdal.GDT_Float32,
                                 self.option)
        elif self.z.dtype=='float64':
            output=driver.Create(filename_raster,\
                                 self.nx,self.ny,self.numbands,\
                                 gdal.GDT_Float64,
                                 self.option)
        elif self.z.dtype=='complex64':
            output=driver.Create(filename_raster,\
                                 self.nx,self.ny,self.numbands,\
                                 gdal.GDT_CFloat32,
                                 self.option)
        elif self.z.dtype=='complex128':
            output=driver.Create(filename_raster,\
                                 self.nx,self.ny,self.numbands,\
                                 gdal.GDT_CFloat64,
                                 self.option)
        
        else:
            print('ERROR: data type not recognized: '+str(self.z.dtype))
            return 1
        
        if not self.isArea and 'tif' in filename_raster[-5:].lower():
            output.SetMetadataItem('AREA_OR_POINT','Point','')

        output.SetGeoTransform(self.GeoTransform)

        #set the projection info
        if self.Projection==None:
            print('Warning: Projection information was not set.')
        output.SetProjection(self.Projection)
    
        #write the array and the file
        if self.numbands==1:
            band_out=output.GetRasterBand(1)
            if self.nodata!=None:
                band_out.SetNoDataValue(self.nodata)
            band_out.WriteArray(self.z)
    
        else: #multiple band raster
            for i in range(self.numbands):
                print('band',i+1)
                #band_out=output.GetRasterBand(i+1)
                if self.nodata!=None:
                    output.GetRasterBand(i+1).SetNoDataValue(self.nodata)
                    output.GetRasterBand(i+1).WriteArray(self.z[i])
                #output.GetRasterBand(i+1).WriteArray(self.z[::i])
                else:
	                output.GetRasterBand(i+1).WriteArray(self.z[i])

        output.FlushCache()
        ouput=None
        
        return 0

    def clone(self,raster_src,clone_z=True):
        try:
            self.x=np.copy(raster_src.x)
            self.y=np.copy(raster_src.y)
            self.GeoTransform=raster_src.GeoTransform
            self.Projection=raster_src.Projection
        except:
            print('NOTE: One or more geocoding information is missing from the source image.')
        
        self.nx=raster_src.nx
        self.ny=raster_src.ny
        self.nz=raster_src.nz
        self.numbands=raster_src.numbands
        self.option=raster_src.option.copy()

        if clone_z:
            self.z=raster_src.z.copy()

    def OverrideGeocodingInfo(self,raster_ref):
        if (self.nx==raster_ref.nx) and (self.ny==raster_ref.ny):
            self.GeoTransform=raster_ref.GeoTransform
            self.Projection=raster_ref.Projection
            return 0

        else:
            print('Error: Dimension of the reference geotiff is different: self:{}by{}, ref: {}by{} '.\
            format(self.nx,self.ny,raster_ref.nx,raster_ref.ny))
            return 1

    def set_area_or_point(self,str_in=None):
        if str_in.upper()=='AREA':
            self.isArea=True
            print('Raster is in AREA MODE')
        elif str_in.upper()=='POINT':
            self.isArea=False
            print('Raster is in POINT MODE')
        elif str_in==None or str_in=='':
            if self.isArea:
                self.isArea=False
                print('Raster is in POINT MODE')
            else:
                self.isArea=True
                print('Raster is in AREA MODE')

    def set_imgarr(self,arr_z,copy_arr=False):
        if copy_arr:
            self.z=np.copy(arr_z)
        else:
            self.z=arr_z
        self.nx=self.z.shape[1]
        self.ny=self.z.shape[0]

    def set_extent(self,rx=None,ry=None,auto_res=True):
        list_gtf=list(self.GeoTransform)

        if rx!=None: #manipulate x axis
            list_gtf[0]=rx[0]
            list_gtf[1]=(rx[1]-rx[0])/self.nx
        
        if ry!=None:
            list_gtf[3]=ry[1]
            list_gtf[5]=-(ry[1]-ry[0])/self.ny

        self.GeoTransform=tuple(list_gtf)

    def set_res(self,xres=None,yres=None):
        list_gtf=list(self.GeoTransform)
        if xres!=None:
            list_gtf[1]=xres
        
        list_gtf[5]=-abs(yres)
    
    def set_proj_by_EPSG(self,num_epsg):
        SR=osr.SpatialReference()
        #SR.ImportFromEPSG('{:04d}'.format(num_epsg))
        SR.ImportFromEPSG(num_epsg)
        self.Projection=SR.ExportToWkt()
        
    def get_grid_x(self):
        if self.isArea:
            xmin=self.GeoTransform[0]+self.GeoTransform[1]/2
            xmax=self.GeoTransform[0]+self.GeoTransform[1]*self.nx-self.GeoTransform[1]/2
        else:
            xmin=self.GeoTransform[0]
            xmax=self.GeoTransform[0]+self.GeoTransform[1]*self.nx

        outvec=np.linspace(xmin,xmax,num=self.nx)
        return outvec

    def get_grid_y(self):
        if self.isArea:
            ymax=self.GeoTransform[3]-self.GeoTransform[5]/2
            ymin=self.GeoTransform[3]+self.GeoTransform[5]*self.ny+self.GeoTransform[1]/2
        else:
            ymax=self.GeoTransform[3]
            ymin=self.GeoTransform[3]+self.GeoTransform[5]*self.ny

        outvec=np.linspace(ymax,ymin,num=self.ny)
        return outvec


#check if the raster file is legit
def is_valid_rasterfile(filename_raster):
    try:
        i0=raster(filename_raster)
        return True
    except:
        print('Invalid raster!')
        return False
    #if hasattr(i0,'z'):
    #    return True
    #else:
    #    return False
        
#Renerage a RGB image from a speed map
def spdplot_uci(raster_or_filename,background=None,log_bg=True,spdmin=1.5,spdmax=3000.,rgbonly=False):
    #TODO: deal with the NaN cases in the input array
    #take care of raster
    if type(raster_or_filename)==str:
        i0=raster(raster_or_filename)
        imgarr=i0.z.copy().astype(np.float32)
        del(i0)
    elif type(raster_or_filename)==raster:
        imgarr=raster_or_filename.z.copy().astype(np.float32)
    elif type(raster_or_filename)==np.ndarray:
        imgarr=raster_or_filename.copy().astype(np.float32)
    else:
        print('ERROR: Cannot recognize the input data')
        return 1

    #take care of background if there is any
    if background!=None:
        if type(background)==str:
            i0=raster(background)
            bgarr=i0.z.copy().astype(np.float32)
            del(i0)
        elif type(background)==raster:
            bgarr=background.z.copy().astype(np.float32)
        elif type(background)==np.ndarray:
            bgarr=background.copy().astype(np.float32)
        else:
            print('ERROR: Cannot recognize the background data')
            return 2

        
        if bgarr.max()>2**16:
            #typically SAR images
            maxi = np.mean(bgarr)+2.*np.std(bgarr) if np.mean(bgarr)+2.*np.std(bgarr) < 30000 else 30000
            mini = np.mean(bgarr)-2.*np.std(bgarr) if np.mean(bgarr)-2.*np.std(bgarr) > 0 else 0
        else:
            #typically optical images 
            maxi=bgarr[bgarr!=0].max()
            mini=bgarr[bgarr!=0].min()
            bgarr=(bgarr-mini)/(maxi-mini)
            bgarr[bgarr>1.0]=1.0
            bgarr[bgarr<0.0]=0.0
            

    else:
        bgarr=np.ones(imgarr.shape,dtype=np.float32)
        bgarr=bgarr*0.7

    


    #refine the input array
    imgarr[np.isnan(imgarr)]=0.0
    mask_zerovel=imgarr==0
    imgarr[imgarr>spdmax]=spdmax
    imgarr[imgarr<spdmin]=spdmin

    bgarr[mask_zerovel]=0.0

    if log_bg:
        h=np.log10(imgarr)
    else:
        h=imgarr
    h=h/np.max(h)
    h[mask_zerovel]=0.0

    s=(0.5+imgarr/125.0)/1.5
    s[s>1.0]=1.0
    s[mask_zerovel]=0.0

    hsv=np.zeros((imgarr.shape[0],imgarr.shape[1],3))
    hsv[:,:,0]=h
    hsv[:,:,1]=s
    hsv[:,:,2]=bgarr

    rgb=matplotlib.colors.hsv_to_rgb(hsv)
    rgb=(rgb*255).astype(np.uint8)
    if not rgbonly: #going to plot the color-coded speed map
        import matplotlib.pyplot as plt
        plt.imshow(rgb)
        plt.show()
    #convert rgb in float 32 into uint8
    
    return rgb
    

def visualize_gamma(filename_raster,filename_par):
    print('grup.visualize_gamma')
    #grab the image dimension and the data type
    nrg=0
    naz=0
    data_type=None
    with open(filename_par,'r') as fin_par:
        lines_par=fin_par.readlines()
        for line_par in lines_par:
            if 'range_samples' in line_par:
                nrg=int(line_par.split(':')[1])
            elif 'azimuth_lines' in line_par:
                naz=int(line_par.split(':')[1])
            elif 'image_format' in line_par:
                data_type=line_par.split(':')[1].replace(' ','').replace('\n','')

    print('range_looks:',naz)
    print('azimuth_looks:',nrg)
    print('image_format:',data_type)

    #load the data
    print('Loading raster')
    if data_type=='FCOMPLEX':
        arr_in=np.fromfile(filename_raster,dtype=np.complex64).byteswap().reshape(naz,nrg)
    else:
        print('ERROR: data load for this data type was not implemented yet.')
        arr_in=np.zeros([1,1]).astype(np.complex64)

    #visualize
    import matplotlib.pyplot as plt
    cm=plt.get_cmap('hsv')
    print('Calculating phase RGB')
    phase_rgb=cm(np.angle(arr_in)/(2*np.pi)+0.5)
    print('Calculating log-scaled power')
    arr_intensity=np.log(np.abs(arr_in)+1)
    del(arr_in)
    print('Blending')
    phase_rgb[:,:,0]=phase_rgb[:,:,0]*arr_intensity/arr_intensity.max()
    phase_rgb[:,:,1]=phase_rgb[:,:,1]*arr_intensity/arr_intensity.max()
    phase_rgb[:,:,2]=phase_rgb[:,:,2]*arr_intensity/arr_intensity.max()
    del(arr_intensity)
    plt.imshow(phase_rgb)
    plt.show()


def complex_to_geotiff(filename_raster):
    i0=raster(filename_raster)
    i1=raster()
    i1.nx=i0.nx
    i1.ny=i0.ny
    i1.numbands=3
    i1.GeoTransform=i0.GeoTransform
    i1.Projection=i0.Projection
    rgbout=visualize(i0,plot=False,rgbout=True)
    rgbout=(rgbout*255).astype(np.uint8)
    i1.z=np.zeros((3,i1.ny,i1.nx),dtype=np.uint8)
    i1.z[0,:,:]=rgbout[:,:,0]
    i1.z[1,:,:]=rgbout[:,:,1]
    i1.z[2,:,:]=rgbout[:,:,2]

    i1.write(filename_raster+'.tif')
    
def pansharpen(filename_panchromatic,filename_r,filename_g,filename_b,plot=False,rgbout=True):
    #Pan-sharpen the multispectral channel images, and generate the pan-sharpened image
    print('grup.pansharpen is not implemented yet - IMPLEMENT ME!!!!')

    

def visualize(raster_or_filename,bands=None,logscale=None,normalize=True,subplot_complex=False,thres=[None,None],plot=True,rgbout=False):
    #Functionality:
    # - Shwow the intensity of a single-band image
    #   - Automatically enhance the image
    # - Show the RGB image of a multiband image
    #   - Choose the first three bands if the bands of interest are not provided
    # - Plot the intensity and phase of a single-band image whose datatype is complex number
    # 
    #Input parameter:
    # - raster_or_filename : Raster object or name of the file name of the raster
    # - bands : Bands to visualize in case the number of bands in the input raster >=3
    #           This can be used to designate each bands to red, green, and blue channel. For example, bands=[2,1,0] will inverse the channel order
    # - logscale : Flag whether log-scale the intensity of the raster data
    #
    import matplotlib.pyplot as plt

    str_colormap_intensity='gray'
    str_colormap_fringe='hsv'
    
    #determine if the input argument is filename or raster object
    if type(raster_or_filename)==raster:
        print('GRUP raster object was given for visualization')
        raster_to_visualize=raster_or_filename
    elif type(raster_or_filename)==str:
        print('Filename was given for visualization')
        if os.path.exists(raster_or_filename):
            raster_to_visualize=raster(raster_or_filename)
        else:
            print('ERROR: File not exists ({})'.format(raster_or_filename))    

    #define the band to visualize if it is not done yet
    if bands==None:
        if raster_to_visualize.numbands==1:
            bands=[0]
        elif raster_to_visualize.numbands==2:
            bands=[0,1,-1] #-1 means no band assignment to the correspondong color channel
        else:
            bands=[0,1,2]

    if logscale==None:
        if str(raster_to_visualize.z.dtype).startswith('complex'):
            print('Note: Complex data type. Log-scaling the magnitude because user did not provide flag \'logscale\'.')
            logscale=True
        else:
            logscale=False

    if len(bands)==1: #plot in grayscale or intensity / fringe
        if 'complex' in str(raster_to_visualize.z.dtype):
            #TODO: consider plotting the phase image whose amplitude is normalized by the magnitude (i.e. single plot rather than a subplot with two panels)
            #plotting complex data 
            print('Plotting complex mumber data')
            #arr_intensity=np.sqrt(raster_to_visualize.z.real*raster_to_visualize.z.real + raster_to_visualize.z.imag*raster_to_visualize.z.imag)
            arr_intensity=np.abs(raster_to_visualize.z)
            if logscale:
                arr_intensity=np.log(arr_intensity+1)
            #arr_phase=np.arctan2(raster_to_visualize.z.real,raster_to_visualize.z.imag)
            arr_phase=np.angle(raster_to_visualize.z)
            
            if subplot_complex:
                if plot:
                    plt.subplot(1,2,1)
                    plt.imshow(arr_intensity,cmap=str_colormap_intensity)
                    plt.colorbar()
                    plt.subplot(1,2,2)
                    plt.imshow(arr_phase,cmap=str_colormap_fringe)
                    plt.colorbar()
                    plt.show()
            else:
                cm=plt.get_cmap('hsv')
                phase_rgb=cm(arr_phase/(2*np.pi)+0.5)
                phase_rgb[:,:,0]=phase_rgb[:,:,0]*arr_intensity/arr_intensity.max()
                phase_rgb[:,:,1]=phase_rgb[:,:,1]*arr_intensity/arr_intensity.max()
                phase_rgb[:,:,2]=phase_rgb[:,:,2]*arr_intensity/arr_intensity.max()
                if plot:
                    plt.imshow(phase_rgb)
                    plt.show()

        else:
            #plotting simple grayscale image
            arr_vis=np.zeros((raster_to_visualize.ny,raster_to_visualize.nx),dtype=raster_to_visualize.z.dtype)
            if raster_to_visualize.numbands==1:
                arr_vis=raster_to_visualize.z
                #thresholding
                if thres[0]!=None:
                    arr_vis[arr_vis<thres[0]]=float('nan')
                if thres[1]!=None:
                    arr_vis[arr_vis>thres[1]]=float('nan')
            else:
                arr_vis=raster_to_visualize.z[bands[0]]
            
            if plot:
                plt.imshow(arr_vis,cmap=str_colormap_intensity)
                plt.colorbar()
                plt.show()


    elif len(bands)==3:
        arr_vis=np.zeros((raster_to_visualize.ny,raster_to_visualize.nx,3),dtype=raster_to_visualize.z.dtype)
        for i in range(3):
            if bands[i]>=0:
                arr_vis[:,:,i]=raster_to_visualize.z[bands[i]]
        
        if plot:
            plt.imshow(arr_vis,cmap=str_colormap_intensity)
            plt.show()

    else:
        print('ERROR : Something gone wrong with the band designation:',bands)

    if rgbout:
        if 'complex' in str(raster_to_visualize.z.dtype):
            return phase_rgb
        else:
            return arr_vis
    

def pct(raster_in,normalize=True,dtype=None):
    #create the mask to detect the valid pixels
    if dtype==None:
        dtype_out='float32'
    else:
        dtype_out=dtype

    mask_valid_px=np.full((raster_in.ny,raster_in.nx),True,dtype=bool)
    for i in range(raster_in.numbands):
        print('masking -',i)
        mask_valid_px=np.logical_and(mask_valid_px,raster_in.z[i]!=0)
        
    numpx=np.sum(mask_valid_px)
    arr_valid_sample=np.zeros((numpx,raster_in.numbands),dtype=dtype_out)
    
    for i in range(raster_in.numbands):
        sample_band=raster_in.z[i][mask_valid_px].astype('float32')
        
        #arr_valid_sample[0:numpx,i]=geotiff_in.z[i][mask_valid_px].reshape(numpx)
        arr_valid_sample[0:numpx,i]=sample_band

    print('Sampling completed')

    vec_mean=np.mean(arr_valid_sample,axis=0)
    for i in range(raster_in.numbands):
        arr_valid_sample[:,i]-=vec_mean[i]
    
    covmat=np.cov(arr_valid_sample,rowvar=False)

    w,v=np.linalg.eig(covmat)

    PC=v[:,w.argmax()]
    #PC=v[:,w.argmin()] #test code - get rid of this or comment this out after use!!
    
    if PC.sum()<0:
        PC=-PC
    print('Principal component: ',PC)

    #transform the bands

    out=raster()
    out.clone(raster_in,clone_z=False)
    out.z=np.zeros((out.ny,out.nx),dtype='float32')
    out.numbands=1

    for i in range(raster_in.numbands):
        raster_oi=raster_in.z[i].astype('float32')-vec_mean[i]*PC[i]
        raster_oi[np.logical_not(mask_valid_px)]=0.0
        out.z+=raster_oi

    if normalize:
        min_z=out.z[mask_valid_px].min()
        max_z=out.z[mask_valid_px].max()

        out.z=(out.z-min_z)/(max_z-min_z)*254.0+1.0
        out.z[out.z>255.0]=255.0
        out.z[out.z<0.0]=1.0
        out.z[np.logical_not(mask_valid_px)]=0.0

    return out


def intensity(raster_in,normalize=False,dtype=None):
    print('PLACEHOLDER')
    #TODO: apply CIE.
    mask_valid_px=np.full((raster_in.ny,raster_in.nx),True,dtype=bool)
    for i in range(raster_in.numbands):
        print('masking -',i)
        mask_valid_px=np.logical_and(mask_valid_px,raster_in.z[i]!=0)
    
    out=raster()
    out.clone(raster_in,clone_z=False)
    out.z=np.zeros((out.ny,out.nx),dtype='float32')
    out.numbands=1
    
    for i in range(raster_in.numbands):
        out.z+=raster_in.z[i].astype('float32')

    out.z[np.logical_not(mask_valid_px)]=0.0

    if normalize:
        minz=out.z[mask_valid_px].min()
        maxz=out.z[mask_valid_px].max()
        print(minz,maxz)
        out.z=(out.z-minz)/(maxz-minz)*254+1.0
    out.z[np.logical_not(mask_valid_px)]=0.0
    out.z[out.z>255.0]=255.0
    out.z[out.z<0.0]=0.0
    return out


def warp_to_reference(str_src,str_ref,str_dst,str_commandlist=None,resampling='cubic'):
    #get the information of the reference image
    raster_ref=gdal.Open(str_ref)
    #self.z=raster.ReadAsArray()
    GeoTransform_ref=raster_ref.GetGeoTransform()
    Projection_ref=raster_ref.GetProjection()
    nx_ref=raster_ref.RasterXSize
    ny_ref=raster_ref.RasterYSize
    
    minx=GeoTransform_ref[0]
    maxx=GeoTransform_ref[0]+GeoTransform_ref[1]*nx_ref
    maxy=GeoTransform_ref[3]
    miny=GeoTransform_ref[3]+GeoTransform_ref[5]*ny_ref
    
    form_command_gdalwarp='gdalwarp -r {RESAMPLE} -co COMPRESS=LZW -co BIGTIFF=YES '
    form_command_gdalwarp+='-t_srs {PROJ} -tr {XR} {YR} -te {xmin} {ymin} {xmax} {ymax} {raster_in} {raster_out}'

    #deal with the projection information
    osr_dst=osr.SpatialReference()
    osr_dst.ImportFromWkt(Projection_ref)
    #try detecting the EPSG
    osr_dst.AutoIdentifyEPSG()
    
    try:
        str_proj=osr_dst.GetAttrValue('AUTHORITY',1)
        str_proj='EPSG:'+str_proj
    except:
        str_proj='"{}"'.format(osr_dst.ExportToProj4())
    
    #print(str_proj)
    str_command=form_command_gdalwarp.format(RESAMPLE=resampling,\
                                             PROJ=str_proj,\
                                             XR=GeoTransform_ref[1],\
                                             YR=-GeoTransform_ref[5],\
                                             xmin=minx,\
                                             ymin=miny,\
                                             xmax=maxx,\
                                             ymax=maxy,\
                                             raster_in=str_src,\
                                             raster_out=str_dst)
    if str_commandlist==None:
        #directly execute the command
        import subprocess
        subprocess.call(str_command,shell=True)

    else:
        #Put the commands to the output file
        if os.path.exists(str_commandlist):
            fout=open(str_commandlist,'a')
        else:
            fout=open(str_commandlist,'w')
        fout.write(str_command+'\n')
        fout.close()
    

if __name__=='__main__':
    #Example usage:
    # grup.py [filename]
    # grup.py [filename] -p [par file] -t [thres_min,thres_max]
    filename_raster=sys.argv[1]
    filename_par=None
    thres=[None,None]
    id_arg=2
    while id_arg<len(sys.argv):
        if sys.argv[id_arg]=='-p': #par file of SLC
            filename_par=sys.argv[id_arg+1]
            id_arg+=2
        elif sys.argv[id_arg]=='-t': #intensity threshold
            seg_thres=sys.argv[id_arg+1].split(',')
            if seg_thres[0]!='':
                thres[0]=float(seg_thres[0])
            if seg_thres[1]!='':
                thres[1]=float(seg_thres[1])
            print('threshold set:',thres)

    if filename_par==None:
        visualize(filename_raster,thres=thres)
    else:
        visualize_gamma(filename_raster,filename_par)

    
#TODO: Define operatior class (__add__, __sub__, etc.)
