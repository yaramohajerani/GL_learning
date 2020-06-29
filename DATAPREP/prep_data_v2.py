#forked from prep_data_s1.py
# To generate the version 2 DInSAR dataset i.e. Version 1 + backscatter

import grup
from fparam import geo_param
import glob
import grup
import os
import numpy as np
from scipy import misc
import datetime
import subprocess

        
def chop_img(imgsrc,nx_tile=512,ny_tile=512,dirx=True,diry=True):
    #dirx: left to right when True
    #      right to left when False
    #diry: upper to bottom when True
    #      bottom to upper when False
    
    #determine the grid array
    grid_x=np.arange(0,imgsrc.shape[1],nx_tile)
    if not dirx:
        grid_x=grid_x+imgsrc.shape[1]-grid_x[-1]
    
    grid_y=np.arange(0,imgsrc.shape[0],ny_tile)
    if not diry:
        grid_y=grid_y+imgsrc.shape[0]-grid_y[-1]

    for i in range(len(grid_y)-1):
        for j in range(len(grid_x)-1):
            print('x:{}-{}, y:{}-{}'.format(grid_x[j],grid_x[j+1],grid_y[i],grid_y[i+1]))

    

#Tile setting
nx_tile=512
ny_tile=512
dirx=True #Chopping from left to right when True, other way arounw when False
diry=True #Chopping from upper to bottom when True, other way arounw when False
flag_overwrite_rasterized_shp=True
dn_burn=255

if os.uname().sysname=='Darwin':
    path_project='/Users/seongsu/Desktop/ACCESS'
    path_shp='{}/SHP'.format(path_project)
    path_rasterized_out='{}/SHP_RASTERIZED'.format(path_project)
    path_coco_prefix='/Users/seongsu/u/oates-r0/eric/SENTINEL1'
    path_tile_with_null='{}/TILED_WITH_NULL'.format(path_project)
    path_tile_without_null='{}/TILED_WITHOUT_NULL'.format(path_project)

elif os.uname().sysname=='Linux':
    #TODO: Change accordingly
    path_project='/u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS'
    path_shp='{}/SHP'.format(path_project)
    path_rasterized_out='{}/SHP_RASTERIZED_V2'.format(path_project)
    path_coco_prefix='/u/oates-r0/eric/SENTINEL1'
    path_tile_with_null='{}/TILED_WITH_NULL_V2'.format(path_project)
    path_tile_without_null='{}/TILED_WITHOUT_NULL_V2'.format(path_project)

print('path_project:',path_project)

if not os.path.exists(path_rasterized_out):
    os.makedirs(path_rasterized_out)
if not os.path.exists(path_tile_with_null):
    os.makedirs(path_tile_with_null)
if not os.path.exists(path_tile_without_null):
    os.makedirs(path_tile_without_null)

#Shapefile name example:
#gl_007_171231-180106-180112-180118_008958-020029-009133-020204_T050832_T050832.shp
list_shp=glob.glob('{}/*.shp'.format(path_shp))
form_path_pair='{PARENT}/{YEAR}/{DT}d/Track{ORBIT}/Track{ID0}-{ID1}_T{TIME0}'
form_coco='coco{ID0}-{ID1}-{ID2}-{ID3}_T{TIME1}.flat.topo_off.deramp.geo.psfilt'
form_pwrbmp='{ID0}.pwr1.geo.bmp'
form_command_rasterize='gdal_rasterize -burn {BURN} -ot Byte -tr {RX} {RY} -te {XMIN} {YMIN} {XMAX} {YMAX} -co compress=LZW {SHP_IN} {TIFF_OUT}'


#output tile template
coco_tile_out=grup.raster()
coco_tile_out.nx=nx_tile
coco_tile_out.ny=ny_tile
coco_tile_out.set_proj_by_EPSG(3031)
coco_tile_out.option=['COMPRESS=LZW']
coco_tile_out.z=np.zeros((3,ny_tile,nx_tile)).astype(np.float32)


delineation_tile_out=grup.raster()
delineation_tile_out.clone(coco_tile_out)

coco_tile_out.numbands=3

for i,filename_shp in enumerate(list_shp):
    print(i+1,':',filename_shp)

    #/Users/seongsu/Desktop/ACCESS/SHP/
    
    seg_filename_shp=filename_shp.split('/')[-1].split('_')
    seg_dates=seg_filename_shp[2].split('-')
    seg_id=seg_filename_shp[3].split('-')

    str_time0=seg_filename_shp[4].replace('T','')
    str_time1=seg_filename_shp[5].replace('T','').replace('.shp','')

    str_track=seg_filename_shp[1]
    list_year=[None]*4
    list_year[0]=int('20'+seg_dates[0][0:2])
    list_year[1]=int('20'+seg_dates[1][0:2])
    list_year[2]=int('20'+seg_dates[2][0:2])
    list_year[3]=int('20'+seg_dates[3][0:2])
    year_dir=np.array(list_year).max()

    date0=datetime.datetime.strptime(seg_dates[0],'%y%m%d')
    date1=datetime.datetime.strptime(seg_dates[1],'%y%m%d')
    date2=datetime.datetime.strptime(seg_dates[2],'%y%m%d')
    date3=datetime.datetime.strptime(seg_dates[3],'%y%m%d')
    dt=int((date1.timestamp()-date0.timestamp())/86400+0.5)
    
    path_gl=form_path_pair.format(PARENT=path_coco_prefix,
                                  YEAR=year_dir,
                                  DT=dt,
                                  ORBIT=str_track,
                                  ID0=seg_id[0],
                                  ID1=seg_id[1],
                                  TIME0=str_time0)
    
    filename_coco=form_coco.format(ID0=seg_id[0],
                                   ID1=seg_id[1],
                                   ID2=seg_id[2],
                                   ID3=seg_id[3],
                                   TIME1=str_time1)

    filename_pwr1=form_pwrbmp.format(ID0=seg_id[0])

    if os.path.exists('{}/{}'.format(path_gl,filename_coco)):
        print('Corresponding coco found: {}/{}'.format(path_gl,filename_coco))

        filename_tiff=filename_shp.replace('.shp','.tif')
        #Load DEM_gc_par in path_gl
        print('Loading DEM_gc_par')
        dem_gc_par=geo_param()
        dem_gc_par.load('{}/DEM_gc_par'.format(path_gl))

        #Rasterize the shp file
        print('Rasterizing shp file')
        '''gdal_rasterize
           -burn {BURN} 
           -ot Byte -tr {RX} {RY} 
           -te {XMIN} {YMIN} {XMAX} {YMAX} -co compress=LZW
           {SHP_IN} {TIFF_OUT}'''
        x0=dem_gc_par.xmin-dem_gc_par.xposting/2
        x1=x0+dem_gc_par.xposting*dem_gc_par.npix
        y1=dem_gc_par.ymax-dem_gc_par.yposting/2
        y0=y1+dem_gc_par.yposting*dem_gc_par.nrec
        
        
        command_rasterize=form_command_rasterize.format(BURN=dn_burn,
                                                        RX=dem_gc_par.xposting,
                                                        RY=-float(dem_gc_par.yposting),
                                                        XMIN=x0,
                                                        YMIN=y0,
                                                        XMAX=x1,
                                                        YMAX=y1,
                                                        SHP_IN=filename_shp,
                                                        TIFF_OUT=filename_tiff)
        if os.path.exists(filename_tiff) and (not flag_overwrite_rasterized_shp):
            print('Rasterized tiff exists:',filename_tiff)
        else:
            os.remove(filename_tiff)
            subprocess.call(command_rasterize,shell=True)

        print('Loading rasterized geotiff')
        geotiff_delineation=grup.raster(filename_tiff)

        #load the original coco
        print('Loading Original coco')
        raster_coco=np.fromfile('{}/{}'.format(path_gl,filename_coco),dtype=np.complex64).byteswap().reshape(dem_gc_par.nrec,dem_gc_par.npix)

        #load the geocoded pwrbmp
        raster_backscatter=misc.imread('{}/{}'.format(path_gl,filename_pwr1))
        print('Loading completed')

        #chop the image
        #chop_img(raster_coco,nx_tile,ny_tile,-1,1)
        grid_x=np.arange(0,raster_coco.shape[1],nx_tile)
        if not dirx:
            grid_x=grid_x+raster_coco.shape[1]-grid_x[-1]
        
        grid_y=np.arange(0,raster_coco.shape[0],ny_tile)
        if not diry:
            grid_y=grid_y+raster_coco.shape[0]-grid_y[-1]

        for i in range(len(grid_y)-1):
            for j in range(len(grid_x)-1):
                print('x:{}-{}, y:{}-{}'.format(grid_x[j],grid_x[j+1],grid_y[i],grid_y[i+1]))
                #chop coco
                #chop rasterized delineation
                path_tile_out='{PATH_TILE}/DIR{DIRX:d}{DIRY:d}'.format(PATH_TILE=path_tile_with_null,
                                                                       DIRX=dirx,
                                                                       DIRY=diry)
                #if not os.path.exists(path_tile_out):
                #    os.makedirs(path_tile_out)


                filename_base=filename_shp.split('/')[-1].replace('.shp','')
                filename_chopped_coco='{PATH_OUT}/coco_{BASE}_x{IDX:04d}_y{IDY:04d}_DIR{DIRX:d}{DIRY:d}.tif'.\
                                      format(PATH_OUT=path_tile_with_null,
                                             BASE=filename_base,
                                             IDX=grid_x[j],
                                             IDY=grid_y[i],
                                             DIRX=dirx,
                                             DIRY=diry)
                                             
                filename_chopped_delineation='{PATH_OUT}/delineation_{BASE}_x{IDX:04d}_y{IDY:04d}_DIR{DIRX:d}{DIRY:d}.tif'.\
                                             format(PATH_OUT=path_tile_with_null,
                                             BASE=filename_base,
                                             IDX=grid_x[j],
                                             IDY=grid_y[i],
                                             DIRX=dirx,
                                             DIRY=diry)

                #print('COCO tile       : {}'.format(filename_chopped_coco))
                #print('DELINEATION tile: {}\n'.format(filename_chopped_delineation))

                #Calculate the geotransform information
                #
                coco_sub=raster_coco[grid_y[i]:grid_y[i+1],grid_x[j]:grid_x[j+1]]

                #afasaf
                coco_tile_out.z[0,:,:]=coco_sub.real
                coco_tile_out.z[1,:,:]=coco_sub.imag
                coco_tile_out.z[2,:,:]=raster_backscatter[grid_y[i]:grid_y[i+1],grid_x[j]:grid_x[j+1]]

                delineation_tile_out.z=geotiff_delineation.z[grid_y[i]:grid_y[i+1],grid_x[j]:grid_x[j+1]]
                gtf_list=[geotiff_delineation.GeoTransform[0]+grid_x[j]*geotiff_delineation.GeoTransform[1],
                          geotiff_delineation.GeoTransform[1],
                          0,
                          geotiff_delineation.GeoTransform[3]+grid_y[i]*geotiff_delineation.GeoTransform[5],
                          0,
                          geotiff_delineation.GeoTransform[5]]
                
                coco_tile_out.GeoTransform=tuple(gtf_list)
                delineation_tile_out.GeoTransform=tuple(gtf_list)

                coco_tile_out.write(filename_chopped_coco)
                delineation_tile_out.write(filename_chopped_delineation)

                #check the delineation tile contains meaningful info
                if delineation_tile_out.z.max()>0:
                    coco_tile_out.write(filename_chopped_coco.replace('WITH','WITHOUT'))
                    delineation_tile_out.write(filename_chopped_delineation.replace('WITH','WITHOUT'))
    else:
        print('COCO not found: {}/{}'.format(path_gl,filename_coco))



    #gl_007_180629-180705-180705-180711_011583-022654-022654-011758_T050836_T050921.shp
    #coco011583-022654-011758-022829_T050836.flat.topo_off.deramp.geo.psfilt
    #coco011583-022654-011758-022829_T050901.flat.topo_off.deramp.geo.psfilt
    #coco011583-022654-022654-011758_T050946.flat.topo_off.deramp.geo.psfilt
    #coco011583-022654-022654-011758_T050857.flat.topo_off.deramp.geo.psfilt
    #coco011583-022654-022654-011758_T050921.flat.topo_off.deramp.geo.psfilt


