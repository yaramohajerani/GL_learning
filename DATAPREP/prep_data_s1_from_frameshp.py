#to be moved to GL_learing/DATAPREP

import gsup
import grup
import glob
import prep_data
import sys
import os

#Extract the list of pair strings from a shapefile's attribute table
def extract_pairstr(filename_shp):
    shpin=gsup.shapefileobj(filename_shp)
    print(shpin.nfeatures,'features in the input shapefile')
    
    list_pairstr=[None]*shpin.nfeatures
    for i,feat in enumerate(shpin.layer):
        #print(feat.GetField('PATH'))
        list_pairstr[i]=feat.GetField('PATH')

    return list_pairstr


if __name__=='__main__':
    if 'Victoria' in os.uname().nodename:
        list_pairstr=extract_pairstr('/Users/seongsu/Desktop/ACCESS/FRAMES_TO_PROCESS/S1_2018_12d_for_Yara/2018_12d_PS_GL_only_for_Yara.shp')
        path_project='/Users/seongsu/Desktop/ACCESS/S1_2018_12d'
    elif os.uname().nodename.split('.')[0] in ['pennell','oates','hobbs','bakutis','mawson','walgreen','fram']:
        list_pairstr=extract_pairstr('/u/oates-r0/eric/ACCESS_S1_12d_2018/FRAMES_TO_PROCESS/S1_2018_12d_for_Yara/2018_12d_PS_GL_only_for_Yara.shp')
        path_project='/u/oates-r0/eric/ACCESS_S1_12d_2018'

    if not os.path.exists(path_project):
        os.makedirs(path_project)
    if not os.path.exists(path_project+'/TILES'):
        os.makedirs(path_project+'/TILES')
    if not os.path.exists(path_project+'/SRCTIFF'):
        os.makedirs(path_project+'/SRCTIFF')

    s1procopobj=prep_data.sentinel1()

    for i,pairstr in list_pairstr:
        print(pairstr)
        #grap gl file in the pair directory
        #gl_164_180919-181001-181001-181013_012790-012965-012965-013140_T233047_T233048.tif
        list_gl_tif=glob.glob('{}/GL/gl_*.tif')

        print(pairstr,'-',len(list_gl_tif),'DInSAR tiff file(s) found.')

        for gl_tif in list_gl_tif:
            #parse gl name into coco
            filename_coco=s1procopobj.gl2coco(os.path.basename(gl_tif))
            coco_src=s1procopobj.load_coco(filename_coco)
            #def split_raster(grupraster_in,path_out,prefix_outfile,nx_tile=512,ny_tile=512,dirx=True,diry=True,offset_x=0,offset_y=0,epsg_tile=3031)
            str_prefix=('coco_'+os.path.basename(gl_tif)).replace('.tif','')
            prep_data.split_raster(coco_src,path_project+'/TILES',str_prefix,dirx=True,diry=True,offset_x=0,offset_y=0,epsg_tile=3031)
            prep_data.split_raster(coco_src,path_project+'/TILES',str_prefix,dirx=True,diry=False,offset_x=0,offset_y=0,epsg_tile=3031)
            prep_data.split_raster(coco_src,path_project+'/TILES',str_prefix,dirx=False,diry=True,offset_x=0,offset_y=0,epsg_tile=3031)
            prep_data.split_raster(coco_src,path_project+'/TILES',str_prefix,dirx=False,diry=False,offset_x=0,offset_y=0,epsg_tile=3031)
            prep_data.split_raster(coco_src,path_project+'/TILES',str_prefix,dirx=True,diry=True,offset_x=256,offset_y=256,epsg_tile=3031)
            prep_data.split_raster(coco_src,path_project+'/TILES',str_prefix,dirx=True,diry=False,offset_x=256,offset_y=256,epsg_tile=3031)
            prep_data.split_raster(coco_src,path_project+'/TILES',str_prefix,dirx=False,diry=True,offset_x=256,offset_y=256,epsg_tile=3031)
            prep_data.split_raster(coco_src,path_project+'/TILES',str_prefix,dirx=False,diry=False,offset_x=256,offset_y=256,epsg_tile=3031)
            






    


    



    




