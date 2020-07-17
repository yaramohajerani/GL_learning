#!/usr/bin/env python3

import grup
import sys
import glob
import subprocess
import numpy as np

dict_region_epsg={
    'antarctica':3031,
    'greenland':3413
}

class processor:
    def __init__(self,sensor='sentinel1',region='antarctica'):
        self._sensor=sensor
        self._region=region
        self.epsg=dict_region_epsg[region]
        
        self.path_shp=None
        self.path_coco_root=None
        self.path_tile_out_null_incl=None
        self.path_tile_out_null_excl=None #NOTE: Set either of them as 'None' when not in  not want of these.
        
        self.nx_tile=512
        self.ny_tile=512
        self.offset_x=0
        self.ofset_y=0
        self.dirx=True #True: From left to right, False: From right to left
        self.diry=True #True: From upper to bottom, False: From bottom to upper
        self.overwrite=True

        self.format_tile_coco='{PATH}/coco_{SHPNAME}_x{X:03d)}_y{Y:03d}_DIR{DIRX}{DIRY}.tif'
        self.format_tile_delineation='{PATH}/delineation_{SHPNAME}_x{X:03d)}_y{Y:03d}_DIR{DIRX}{DIRY}.tif'

    @property
    def region(self):
         return self._region
    @region.setter
    def region(self,str_region):
        try:
            self.epsg=dict_region_epsg[str_region]
            self._region=str_region
        except:
            print('ERROR: EPSG for \'{}\' was not defined.'.format(str_region))

    


    def chop_raster(imgarr_in,kind,self):
        pass
        #kind: string. Either 'coco' or 'delineation'

def get_shp_list():
    pass
    
        
class sentinel1(processor):

    def find_coco(self,filename_shp):
        pass

    def get_coco(self,shpfilename):
        pass



