#!/usr/bin/env python3

import glob
import tarfile
import sys
import os

def get_tile_list(path_tiles):
    list_tiles_fullpath=glob.glob('{}/coco_gl*.tif'.format(path_tiles))
    
    list_tiles=[None]*len(list_tiles_fullpath)
    print('Reformatting tile list.')
    for i,tilename_fullpath in enumerate(list_tiles_fullpath):
        list_tiles[i]=os.path.basename(tilename_fullpath)

    return list_tiles


def DinSAR_dict_from_tile_list(list_tiles):
    list_dinsar_all=[None]*len(list_tiles)
    print('Finding unique coco names.')
    for i, tile in enumerate(list_tiles):
        seg_tile=tile.split('_')
        list_dinsar_all[i]='_'.join(seg_tile[:-3])
    
    list_dinsar_uniq=list(set(list_dinsar_all))

    #build up the dict frame
    print('Reorganizing the list into dict.')
    dict_dinsar={}
    for dinsar in list_dinsar_uniq:
        dict_dinsar[dinsar]=[]

    #stack up the tile names in each key in dict
    for i, tile in enumerate(list_tiles):
        seg_tile=tile.split('_')
        tile_sub='_'.join(seg_tile[:-3])
        dict_dinsar[tile_sub].append(tile)

    return dict_dinsar


def tar_tiles_by_coco(dict_dinsaro,path_tiles,path_tar,flag_remove_source_tiles):
    num_dinsar=len(dict_dinsar.keys())
    for i,dinsar in enumerate(dict_dinsar.keys()):
        print('{}/{} - {}'.format(i+1,num_dinsar,dinsar))
        filename_tar='{}/{}.tar'.format(path_tar,dinsar)
        
        with tarfile.open(filename_tar,'w') as tin:
            os.chdir(path_tiles)
            for tilename in dict_dinsar[dinsar]:
                try:
                    tin.add(tilename)
                    if flag_remove_source_tiles:
                        os.remove(tilename)
                except:
                    print('Cannot put into tar:',tilename)


if __name__=='__main__':
    str_usage='''
    archive_tiles.py [tile directory] [output tar file directory]
    archive_tiles.py [tile directory] [output tar file directory] --remove-files
    
    '''

    path_tiles=sys.argv[1]
    path_tar=sys.argv[2]
    if len(sys.argv)==4 and sys.argv[-1]=='--remove-files':
        flag_remove_source_tiles=True
    else:
        flag_remove_source_tiles=False

    if not os.path.exists(path_tar):
        print('Output directory does not exist. Creating:',path_tar)
        os.makedirs(path_tar)
        
    list_tiles=get_tile_list(path_tiles)
    dict_dinsar=DinSAR_dict_from_tile_list(list_tiles)
    tar_tiles_by_coco(dict_dinsar,path_tiles,path_tar,flag_remove_source_tiles)
    print('Processing completed')









