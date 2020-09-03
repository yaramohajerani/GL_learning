# GSUP - GDAL Shapefile Utility Package
# Primitive prototype
# Initial code in 01/09/2020 by Seongsu Jeong (UCI)


import os
from osgeo import ogr,osr


class shapefileobj:
    def __init__(self,filename_shp=None):
        #default parameters:
        self._name=None
        self.drv=None
        self.datasrc=None
        self.layer=None
        self.epsg=3031 #PSN
        self.prjobj=None
        self.features=None
        self.nfeatures=0

        #auto-load shapefile
        if filename_shp!=None:
            self.name=filename_shp
        

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,str_filename):
        self._name=str_filename
        self.open()
        #TODO: Automate driver open, datasrc, etc.

    def open(self,filename_shp=None,readonly=True):
        if self.drv==None:
            self.drv=ogr.GetDriverByName('ESRI Shapefile')

        if self._name==None and filename_shp==None:
            print('ERROR: Name of the shapefile was not provided.\nIt has to be defined in the member ''name'' or provided as the input argument.')
            return None

        elif filename_shp!=None:
            self._name=filename_shp
            
        self.datasrc=self.drv.Open(self._name)

        if self.datasrc==None:
            print('Cannot open shapefile:',filename_shp)
        else:
            print('Shapefile loaded:',filename_shp)
            self.layer=self.datasrc.GetLayer()
            self.nfeatures=self.layer.GetFeatureCount()
            print(self.nfeatures,' features found.')




        
    