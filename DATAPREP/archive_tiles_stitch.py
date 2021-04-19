#!/usr/bin/env python3

import glob
import tarfile
import sys
import os

def get_tile_list(path_tiles):
    list_tiles_fullpath=glob.glob('{}/gl*.*'.format(path_tiles))
    
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
        #list_dinsar_all[i]='_'.join(seg_tile[:-3])
        list_dinsar_all[i]='_'.join(seg_tile[:6])
    list_dinsar_uniq=list(set(list_dinsar_all))

    #build up the dict frame
    print('Reorganizing the list into dict.')
    dict_dinsar={}
    for dinsar in list_dinsar_uniq:
        dict_dinsar[dinsar]=[]

    #stack up the tile names in each key in dict
    for i, tile in enumerate(list_tiles):
        seg_tile=tile.split('_')
        #tile_sub='_'.join(seg_tile[:-3])
        tile_sub='_'.join(seg_tile[:6])
        dict_dinsar[tile_sub].append(tile)

    num_dinsar=len(list_dinsar_uniq)

    print('Sorting')
    for i,dinsar in enumerate(dict_dinsar.keys()):
        print('{} of {}'.format(i+1, num_dinsar),end='\r')
        dict_dinsar[dinsar].sort()
    print('\n')
    return dict_dinsar


def tar_tiles_by_coco(dict_dinsar,path_tiles,path_tar,flag_remove_source_tiles):
    num_dinsar=len(dict_dinsar.keys())
    for i,dinsar in enumerate(dict_dinsar.keys()):
        print('{}/{} - {}'.format(i+1,num_dinsar,dinsar))
        filename_tar='{}/{}.tar'.format(path_tar,dinsar)
        
        with tarfile.open(filename_tar,'a') as tin:
            os.chdir(path_tiles)
            for tilename in dict_dinsar[dinsar]:
                try:
                    tin.add(tilename)
                    if flag_remove_source_tiles:
                        os.remove(tilename)
                except:
                    print('Cannot put into tar:',tilename)


def export_tilelist(dict_dinsar,path_out):
    for i,dinsar in enumerate(dict_dinsar.keys()):
        filename_list='{}/{}_list.txt'.format(path_out,dinsar)
        with open(filename_list,'w+') as fout:
            for tilename in dict_dinsar[dinsar]:
                fout.write(tilename+'\n')


if __name__=='__main__':
    str_usage='''
    archive_tiles.py [tile directory] [output tar file directory]
    archive_tiles.py [tile directory] [output tar file directory] --list_only
    archive_tiles.py [tile directory] [output tar file directory] --remove-files
    archive_tiles.py [tile directory] [output tar file directory] --verbose
    
    '''

    #default parameters
    path_tiles=None
    path_out=None

    flag_list_only=False
    flag_remove_files=False

    flag_proceed=False

    #parse the input arguments
    path_tiles=sys.argv[1]
    path_out=sys.argv[2]

    NOI=3
    while NOI<len(sys.argv):
        if sys.argv[NOI]=='--list_only':
            flag_list_only=True
            NOI+=1
        elif sys.argv[NOI]=='--remove-files':
            flag_remove_files=True
            NOI+=1
        else:
            print('Canot recognize the argument:',sys.argv[NOI])
            break
    


    #determine whether or not to proceed
    if path_tiles is None or path_out is None:
        flag_proceed=False
    else:
        flag_proceed=True

    #Initiate the processing when desirable
    if flag_proceed:
        pass
        #display the perceved arguments information
        print('############### Argument informtaion ###############')
        print('Tile(s) directory   :',path_tiles)
        print('Output directory    :',path_out)
        print('flag_list_only      :',flag_list_only)
        print('flag_remove_files   :',flag_remove_files)

        if not os.path.exists(path_out):
            print('Output directory does not exist. Creating:',path_out)
            os.makedirs(path_out)
            
        list_tiles=get_tile_list(path_tiles)
        dict_dinsar=DinSAR_dict_from_tile_list(list_tiles)

        if flag_list_only:
            export_tilelist(dict_dinsar,path_out)
        else:
            tar_tiles_by_coco(dict_dinsar,path_tiles,path_out,flag_remove_files)

        print('Processing completed')
    else:
        print('Insufficient information. Cannot proceed.\n')
        print('############### Argument informtaion ###############')
        print('Tiles'' directory   :',path_tiles)
        print('Output directory    :',path_out)
        print('flag_list_only      :',flag_list_only)
        print('flag_remove_files   :',flag_remove_files)

