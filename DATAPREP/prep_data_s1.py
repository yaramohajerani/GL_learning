#!/usr/bin/env python3

import glob
import grup
import os
import numpy as np
import datetime
import subprocess
import sys

if __name__=='__main__':
    str_usage='''
    prep_data_s1.py -p /Users/username/ACCESS -dir 11 -nx 512 -ny 512 -ox 0 -oy 0
    prep_data_s1.py -p /Users/username/ACCESS -c /u/oates-r0/eric/SENTINEL1 -dir 11 -nx 512 -ny 512 -ox 0 -oy 0
    prep_data_s1.py -p /Users/username/ACCESS -dir 01 -nx 512 -ny 512 -ox 0 -oy 0 #tiling in inversed x direction
    prep_data_s1.py -p /Users/username/ACCESS -dir 10 -nx 512 -ny 512 -ox 0 -oy 0 #tiling in inversed y direction
    prep_data_s1.py -p /Users/username/ACCESS -dir 11 -nx 512 -ny 512 -ox 256 -oy 256 #staggered tile

    Parameter description:
    -p : Project directory, in which the subdirectory 'SHP' is located (with the shape file in it)
    -c : Parent directory of coco data to find
    -dir: Tiling direction
    -ox, oy: Offset in x and y coordinates for staggered tiling
    '''

    #default setting
    #Tiling
    nx_tile=512
    ny_tile=512
    dirx=True #Chopping from left to right when True, other way arounw when False
    diry=True #Chopping from upper to bottom when True, other way arounw when False
    offset_x=0
    offset_y=0
    flag_overwrite_rasterized_shp=True
    dn_burn=255

    
    path_project=''
    path_coco_prefix=''

    path_shp='{}/SHP'.format(path_project)
    path_rasterized_out='{}/SHP_RASTERIZED'.format(path_project)
    path_tile_with_null='{}/TILED_WITH_NULL'.format(path_project)
    path_tile_without_null='{}/TILED_WITHOUT_NULL'.format(path_project)

    if len(sys.argv)==1:
        #Print out the usage when there is no input parameters
        print(str_usage)

    else:
        #parse the input argument
        idoi=1
        while idoi<len(sys.argv):
            if sys.argv[idoi]=='-p':
                path_project=sys.argv[idoi+1]
                path_shp='{}/SHP'.format(path_project)
                path_rasterized_out='{}/SHP_RASTERIZED'.format(path_project)
                path_tile_with_null='{}/TILED_WITH_NULL'.format(path_project)
                path_tile_without_null='{}/TILED_WITHOUT_NULL'.format(path_project)
                idoi+=2
            elif sys.argv[idoi]=='-c':
                path_coco_prefix=path_project=sys.argv[idoi+1]
                idoi+=2
            elif sys.argv[idoi]=='-dir':
                dirx=bool(int(sys.argv[idoi+1][0]))
                diry=bool(int(sys.argv[idoi+1][1]))
                idoi+=2
            elif sys.argv[idoi]=='-nx':
                nx_tile=int(sys.argv[idoi+1])
                idoi+=2
            elif sys.argv[idoi]=='-ny':
                ny_tile=int(sys.argv[idoi+1])
                idoi+=2
            elif sys.argv[idoi]=='-ox':
                offset_x=int(sys.argv[idoi+1])
                idoi+=2
            elif sys.argv[idoi]=='-oy':
                offset_y=int(sys.argv[idoi+1])
                idoi+=2
        
        #check if the input argument makes sense
        if path_project=='' or path_coco_prefix=='':
            print('ERROR: Either ptoject path or coco prefix was not provided. These are mandatory.')
            exit(1)
        else:
            #print out the parameters
            print('               Project path:',path_project)
            print('             Shapefile path:',path_shp)
            print(' Output tile path (w/ null):',path_tile_with_null)
            print('Output tile path (w/o null):',path_tile_without_null)

        #Create the output path if not exist
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
        delineation_tile_out=grup.raster()
        delineation_tile_out.clone(coco_tile_out)

        for i,filename_shp in enumerate(list_shp):
            print(i+1,':',filename_shp)

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
                xmin=None
                ymax=None
                xposting=None
                yposting=None
                nx_src=None
                ny_src=None
                with open('{}/DEM_gc_par'.format(path_gl)) as fin:
                    lines_in=fin.readlines()
                    for line_in in lines_in:
                        if 'width:' in line_in:
                            nx_src=int(line_in.split()[1])
                        elif 'nlines:' in line_in:
                            ny_src=int(line_in.split()[1])
                        elif 'corner_north:' in line_in:
                            ymax=float(line_in.split()[1])
                        elif 'corner_east:' in line_in:
                            xmin=float(line_in.split()[1])
                        elif 'post_north:' in line_in:
                            yposting=float(line_in.split()[1])
                        elif 'post_east:' in line_in:
                            xposting=float(line_in.split()[1])

                #Rasterize the shp file
                print('Rasterizing shp file')
                x0=xmin-xposting/2
                x1=x0+xposting*nx_src
                y1=ymax-yposting/2
                y0=y1+yposting*ny_src
                
                command_rasterize=form_command_rasterize.format(BURN=dn_burn,
                                                                RX=xposting,
                                                                RY=-float(yposting),
                                                                XMIN=x0,
                                                                YMIN=y0,
                                                                XMAX=x1,
                                                                YMAX=y1,
                                                                SHP_IN=filename_shp,
                                                                TIFF_OUT=filename_tiff)
                if os.path.exists(filename_tiff) and (not flag_overwrite_rasterized_shp):
                    print('Rasterized tiff exists:',filename_tiff)
                else:
                    if os.path.exists(filename_tiff):
                        os.remove(filename_tiff)
                    print(command_rasterize)
                    subprocess.call(command_rasterize,shell=True)

                print('Loading rasterized geotiff')
                geotiff_delineation=grup.raster(filename_tiff)

                #load the original coco
                print('Loading Original coco')
                raster_coco=np.fromfile('{}/{}'.format(path_gl,filename_coco),dtype=np.complex64).byteswap().reshape(ny_src,nx_src)

                #load the geocoded pwrbmp
                print('Loading done')

                #chop the image
                grid_x=np.arange(offset_x,raster_coco.shape[1],nx_tile)
                if not dirx:
                    grid_x=np.flipud(np.arange(raster_coco.shape[1]-offset_x,0,-nx_tile))
                
                grid_y=np.arange(offset_y,raster_coco.shape[0],ny_tile)
                if not diry:
                    grid_y=np.flipud(np.arange(raster_coco.shape[0]-offset_y,0,-ny_tile))

                print('nx={},ny={}'.format(raster_coco.shape[1],raster_coco.shape[0]))
                print('x grid:',grid_x)
                print('y grid:',grid_y)
                for i in range(len(grid_y)-1):
                    for j in range(len(grid_x)-1):
                        print('x:{}-{}, y:{}-{}'.format(grid_x[j],grid_x[j+1],grid_y[i],grid_y[i+1]))
                        #chop coco
                        #chop rasterized delineation
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

                        #Calculate the geotransform information
                        coco_tile_out.z=raster_coco[grid_y[i]:grid_y[i+1],grid_x[j]:grid_x[j+1]]
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


