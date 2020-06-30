#prep_data_csk.py

import glob
from osgeo import ogr,osr,gdal
import datetime
import os
import grup
import numpy as np
import subprocess
import sys


def integrity_check(filename_shp,path_dinsar):
    #
    # coco20160208_20160209-20160311_20160312.flat.topo_off.psfilt.coh.tiff - Corerence
    # coco20160208_20160209-20160311_20160312.flat.topo_off.psfilt.geo.tiff - Phase
    #

    list_coco=glob.glob('{}/coco*.geo.tiff'.format(path_dinsar))
    print(len(list_coco),'DInSAR data found from',path_dinsar)

    #load the source SHP file and iterate
    drv_shp_in=ogr.GetDriverByName('ESRI Shapefile')
    try:
        shp_in=drv_shp_in.Open(filename_shp,0) #readonly
    except:
        print('ERROR: Cannot open source shp file:',filename_shp)
        exit(1)

    lyr_in=shp_in.GetLayer()
    form_coco='coco{M1}_{S1}-{M2}_{S2}.flat.topo_off.psfilt.{KIND}.tiff'
    stack_result=[]
    dict_id_flag={}

    for feat_in in lyr_in:
        str_master1=str(feat_in.GetField('MASTER1'))
        str_master2=str(feat_in.GetField('MASTER2'))
        
        #Try to guess the corresponding coco name
        time_master1=datetime.datetime.strptime(str_master1,'%Y%m%d')
        time_slave1=time_master1+datetime.timedelta(days=1)
        time_master2=datetime.datetime.strptime(str_master2,'%Y%m%d')
        time_slave2=time_master2+datetime.timedelta(days=1)
        str_slave1=time_slave1.strftime('%Y%m%d')
        str_slave2=time_slave2.strftime('%Y%m%d')

        id_dinsar='{}_{}-{}_{}:'.format(str_master1,str_slave1,str_master2,str_slave2)

        filename_coco_coh=form_coco.format(M1=str_master1,S1=str_slave1,M2=str_master2,S2=str_slave2,KIND='coh')
        filename_coco_geo=form_coco.format(M1=str_master1,S1=str_slave1,M2=str_master2,S2=str_slave2,KIND='geo')

        if os.path.exists('{}/{}'.format(path_dinsar,filename_coco_coh)):
            flag_coh=True
        else:
            flag_coh=False

        if os.path.exists('{}/{}'.format(path_dinsar,filename_coco_geo)):
            flag_geo=True
        else:
            flag_geo=False

        dt_master=time_master2-time_master1
        str_remarks=''
        if dt_master.days%16:
            str_remarks+='Check dt between masters ({}days). ;'.format(dt_master.days)

        if not flag_geo:
            str_remarks+=' Cannot find the phase tiff. ;'
        
        if not flag_coh:
            str_remarks+=' Cannot find the coherence tiff. ;'

        if str_remarks=='':
            str_remarks='Integrity check passed. ;'
            dict_id_flag[id_dinsar.replace(':','')]=True
        else:
            dict_id_flag[id_dinsar.replace(':','')]=False
        #if flag_geo!=flag_coh:
        #    print(filename_coco_geo,flag_coh,flag_geo)
        print(id_dinsar,str_remarks)
        stack_result.append(id_dinsar+str_remarks)
        #geom_in=feat_in.GetGeometryRef()
    set_result_append=set(stack_result)
    print('\n\n\n')
    print('\n'.join(list(set_result_append)))
    
    return dict_id_flag
    #parse the 

def rasterize_delineation(filename_shp,
                          id_dinsar,
                          path_dinsar,
                          path_rasterized,
                          epsg_in=3031):
    #NOTE the default values above are for test purpose only. Get rid of them when done with the testing.
    form_command_rasterize='gdal_rasterize -burn {BURN} -ot Byte -tr {RX} {RY} -te {XMIN} {YMIN} {XMAX} {YMAX} -co compress=LZW {SHP_IN} {TIFF_OUT}'
    if not os.path.exists(path_rasterized):
        print('Path for raserized SHP does not exist. Creating.')
        os.makedirs(path_rasterized)

    
    m1=int(id_dinsar.split('-')[0].split('_')[0])
    m2=int(id_dinsar.split('-')[1].split('_')[0])

    #prepare for the output shp
    srs_out=osr.SpatialReference()
    srs_out.ImportFromEPSG(epsg_in) #Antarctica
    
    filename_shp_out='{}/gl_{}.shp'.format(path_rasterized,id_dinsar)
    drv_shp_out=ogr.GetDriverByName('ESRI Shapefile')
    shp_out=drv_shp_out.CreateDataSource(filename_shp_out)
    lyr_out=shp_out.CreateLayer('GL',srs_out,ogr.wkbLineString)
    field_m1=ogr.FieldDefn('MASTER1',ogr.OFTString)
    field_m1.SetWidth(10)
    lyr_out.CreateField(field_m1)
    field_m2=ogr.FieldDefn('MASTER2',ogr.OFTString)
    field_m2.SetWidth(10)
    lyr_out.CreateField(field_m2)
    
    #prepare to load the source shp file
    drv_shp_in=ogr.GetDriverByName('ESRI Shapefile')
    shp_in=drv_shp_in.Open(filename_shp,0)
    lyr_in=shp_in.GetLayer()

    numfeat=0
    for feat_in in lyr_in:
        m1_in=feat_in.GetField('MASTER1')
        m2_in=feat_in.GetField('MASTER2')

        
        if m1_in==m1 and m2_in==m2:
            numfeat+=1
            print('#feat={}'.format(numfeat))
            #retrieve the geometry
            gin=feat_in.GetGeometryRef()
            feat_out=ogr.Feature(lyr_out.GetLayerDefn())
            feat_out.SetField('MASTER1',str(m1))
            feat_out.SetField('MASTER2',str(m2))
            feat_out.SetGeometry(gin)
            lyr_out.CreateFeature(feat_out)
            feat_out=None
    shp_out=None

    #rasterize the extracted shp file
    dn_burn=255
    if numfeat>0:
        filename_corresponding_geo='{}/coco{}.flat.topo_off.psfilt.geo.tiff'.format(path_dinsar,id_dinsar)
        filename_tiff_out='{}/delination_{}.shp.rasterized.tif'.format(path_rasterized,id_dinsar)
        #coco20160208_20160209-20160311_20160312.flat.topo_off.psfilt.geo.tiff
        geo_in=gdal.Open(filename_corresponding_geo)
        gtf=geo_in.GetGeoTransform()

        command_rasterize=form_command_rasterize.format(BURN=dn_burn,
                                                        RX=gtf[1],#dem_gc_par.xposting,
                                                        RY=-gtf[5],#-float(dem_gc_par.yposting),
                                                        XMIN=gtf[0],
                                                        YMIN=gtf[3]+gtf[5]*geo_in.RasterYSize,
                                                        XMAX=gtf[0]+gtf[1]*geo_in.RasterXSize,
                                                        YMAX=gtf[3],
                                                        SHP_IN=filename_shp_out,
                                                        TIFF_OUT=filename_tiff_out)

        subprocess.call(command_rasterize,shell=True)
        return filename_tiff_out

def get_complete_coco(path_dinsar='/Users/seongsu/Desktop/ACCESS/CSK/DInSAR',
                      id_dinsar='20160313_20160314-20160601_20160602'):
    
    #coco file example for CSK
    #coco20160208_20160209-20160311_20160312.flat.topo_off.psfilt.coh.tiff #coherence
    #coco20160208_20160209-20160311_20160312.flat.topo_off.psfilt.geo.tiff #phase [-pi, pi]

    filename_geo='{}/coco{}.flat.topo_off.psfilt.geo.tiff'.format(path_dinsar,id_dinsar)
    filename_coh='{}/coco{}.flat.topo_off.psfilt.coh.tiff'.format(path_dinsar,id_dinsar)
    geoin=grup.raster(filename_geo) #phase
    cohin=grup.raster(filename_coh) #coherence

    coco_complete=grup.raster()
    coco_complete.clone(geoin)
    coco_complete.z=cohin.z*np.cos(geoin.z)+cohin.z*np.sin(geoin.z)*1.0j
    
    return coco_complete

def chop_img(coco_in,delineation_in,path_tile,tilename_prefix,
             nx_tile=512,ny_tile=512,
             offset_x=0,offset_y=0,
             dirx=True,diry=True,
             separate_null=True):
    
    if not os.path.exists(path_tile):
        os.makedirs(path_tile)

    if separate_null:
        path_tile_without_null=path_tile+'_WITHOUT_NULL'
        if not os.path.exists(path_tile_without_null):
            os.makedirs(path_tile_without_null)
        
    coco_tile_out=grup.raster()
    coco_tile_out.nx=nx_tile
    coco_tile_out.ny=ny_tile
    coco_tile_out.set_proj_by_EPSG(3031)
    coco_tile_out.option=['COMPRESS=LZW']
    coco_tile_out.isArea=coco_in.isArea

    delineation_tile_out=grup.raster()
    delineation_tile_out.nx=nx_tile
    delineation_tile_out.ny=ny_tile
    delineation_tile_out.set_proj_by_EPSG(3031)
    delineation_tile_out.option=['COMPRESS=LZW']
    delineation_tile_out.isArea=delineation_in.isArea

    grid_x=np.arange(offset_x,coco_in.nx,nx_tile)
    if not dirx:
        #grid_x=grid_x+raster_coco.shape[1]-grid_x[-1]
        grid_x=np.flipud(np.arange(coco_in.nx-offset_x,0,-nx_tile))
    
    grid_y=np.arange(offset_y,coco_in.ny,ny_tile)
    if not diry:
        #grid_y=grid_y+raster_coco.shape[0]-grid_y[-1]
        grid_y=np.flipud(np.arange(coco_in.ny-offset_y,0,-ny_tile))

    print('nx={},ny={}'.format(coco_in.nx,coco_in.ny))
    print('x grid:',grid_x)
    print('y grid:',grid_y)

    for i in range(len(grid_y)-1):
        for j in range(len(grid_x)-1):
            print('x:{}-{}, y:{}-{}'.format(grid_x[j],grid_x[j+1],grid_y[i],grid_y[i+1]))
            #chop coco
            #chop rasterized delineation
            #if not os.path.exists(path_tile_out):
            #    os.makedirs(path_tile_out)


            filename_coco_out='{PATH_OUT}/coco_{PREFIX}_x{IDX:04d}_y{IDY:04d}_DIR{DIRX:d}{DIRY:d}.tif'.\
                              format(PATH_OUT=path_tile,
                                     PREFIX=tilename_prefix,
                                     IDX=grid_x[j],
                                     IDY=grid_y[i],
                                     DIRX=dirx,
                                     DIRY=diry)
            
            filename_delineation_out='{PATH_OUT}/delineation_{PREFIX}_x{IDX:04d}_y{IDY:04d}_DIR{DIRX:d}{DIRY:d}.tif'.\
                              format(PATH_OUT=path_tile,
                                     PREFIX=tilename_prefix,
                                     IDX=grid_x[j],
                                     IDY=grid_y[i],
                                     DIRX=dirx,
                                     DIRY=diry)
                                    
            
            #Calculate the geotransform information
            #

            coco_tile_out.z=coco_in.z[grid_y[i]:grid_y[i+1],grid_x[j]:grid_x[j+1]]
            delineation_tile_out.z=delineation_in.z[grid_y[i]:grid_y[i+1],grid_x[j]:grid_x[j+1]]

            gtf_list=[coco_in.GeoTransform[0]+grid_x[j]*coco_in.GeoTransform[1],
                      coco_in.GeoTransform[1],
                      0,
                      coco_in.GeoTransform[3]+grid_y[i]*coco_in.GeoTransform[5],
                      0,
                      coco_in.GeoTransform[5]]
            
            coco_tile_out.GeoTransform=tuple(gtf_list)
            delineation_tile_out.GeoTransform=tuple(gtf_list)

            coco_tile_out.write(filename_coco_out)
            delineation_tile_out.write(filename_delineation_out)

            #check if null tile i.e. no delineation in the tile
            if delineation_tile_out.z.max()>0:
                coco_tile_out.write(filename_coco_out.replace(path_tile,path_tile_without_null))
                delineation_tile_out.write(filename_delineation_out.replace(path_tile,path_tile_without_null))


if __name__=='__main__':

    str_usage='''
    prep_data_csk.py -p /Users/username/ACCESS/PSK -s /Users/username/ACCESS/PSK/SHP/shapefile.shp -d /Users/username/ACCESS/PSK/DInSAR
    prep_data_csk.py -e 3031 -p /Users/username/ACCESS/PSK -s /Users/username/ACCESS/PSK/SHP/shapefile.shp -d /Users/username/ACCESS/PSK/DInSAR #define projection by EPSG

    -e: EPSG for projection definition.
    -p: Project directory
    -s: Shapefile name
    -d: DInSAR directory
    '''

    #default parameters
    path_project=''
    filename_shp_src='/Users/seongsu/Desktop/ACCESS/CSK/Pope-smith-Kohler_typo_fixed_20200523/CSK_GL_POPE_SMITH_KOHLER.shp'
    path_dinsar='/Users/seongsu/Desktop/ACCESS/CSK/DInSAR'
    path_tile='/Users/seongsu/Desktop/ACCESS/CSK/TILE'
    path_rasterized='/Users/seongsu/Desktop/ACCESS/CSK/SHP_INDV'
    

    #parse the input parameters
    noi=1
    while noi<len(sys.argv):
        if sys.argv[noi]=='-e':
            EPSG_ARG=int(sys.argv[noi+1])
            noi+=2
        elif sys.argv[noi]=='-p':
            path_project=sys.argv[noi+1]
            path_tile='{}/TILE'.format(path_project)
            path_rasterized='{}/SHP_indv'.format(path_project)
            noi+=2
        elif sys.argv[noi]=='-s':
            filename_shp_src=sys.argv[noi+1]
            noi+=2
        elif sys.argv[noi]=='-d':
            path_dinsar=sys.argv[noi+1]
            noi+=2
        else:
            print('ERROR: Cannot understand the parameter option:',sys.argv[noi])
            exit(1)

    print('Project path:                {}'.format(path_project))
    print('Source shapefile name:       {}'.format(filename_shp_src))
    print('Output tile path:            {}'.format(path_tile))
    print('Rasterized delineation path: {}'.format(path_rasterized))

    #check the data integrity
    dict_integrity=integrity_check(filename_shp_src,path_dinsar)

    for id_dinsar in dict_integrity:
        if dict_integrity[id_dinsar]: #Set of delineaitons which passed the integrity check. Good to preprocess
            filename_rasterized=rasterize_delineation(filename_shp_src,id_dinsar,path_dinsar,path_rasterized,epsg_in=EPSG_ARG)
            coco_in=get_complete_coco(path_dinsar,id_dinsar)
            delineation_in=grup.raster(filename_rasterized)
            delineation_in.isArea=False
            chop_img(coco_in,delineation_in,path_tile,id_dinsar,512,512,0,0,True,True)
            chop_img(coco_in,delineation_in,path_tile,id_dinsar,512,512,0,0,True,False)
            chop_img(coco_in,delineation_in,path_tile,id_dinsar,512,512,0,0,False,True)
            chop_img(coco_in,delineation_in,path_tile,id_dinsar,512,512,0,0,False,False)

            chop_img(coco_in,delineation_in,path_tile,id_dinsar,512,512,256,256,True,True)
            chop_img(coco_in,delineation_in,path_tile,id_dinsar,512,512,256,256,True,False)
            chop_img(coco_in,delineation_in,path_tile,id_dinsar,512,512,256,256,False,True)
            chop_img(coco_in,delineation_in,path_tile,id_dinsar,512,512,256,256,False,False)

            