#Data prep script for Pope, Smith, Kohler
#!/usr/bin/env bash

$HOME/CODES/ACCESS/prep_data_s1.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK -dir 11 -nx 512 -ny 512 -ox 0 -oy 0
$HOME/CODES/ACCESS/prep_data_s1.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK -dir 10 -nx 512 -ny 512 -ox 0 -oy 0
$HOME/CODES/ACCESS/prep_data_s1.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK  -dir 01 -nx 512 -ny 512 -ox 0 -oy 0
$HOME/CODES/ACCESS/prep_data_s1.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK  -dir 00 -nx 512 -ny 512 -ox 0 -oy 0

#staggered grid
$HOME/CODES/ACCESS/prep_data_s1.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK  -dir 11 -nx 512 -ny 512 -ox 256 -oy 256
$HOME/CODES/ACCESS/prep_data_s1.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK  -dir 10 -nx 512 -ny 512 -ox 256 -oy 256
$HOME/CODES/ACCESS/prep_data_s1.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK  -dir 01 -nx 512 -ny 512 -ox 256 -oy 256
$HOME/CODES/ACCESS/prep_data_s1.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK  -dir 00 -nx 512 -ny 512 -ox 256 -oy 256

$HOME/CODES/ACCESS/prep_data_s1_unusedDInSAR.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK_UNUSED -l /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK/list_Track010_unused.txt -dir 11 -nx 512 -ny 512 -ox 0 -oy 0
$HOME/CODES/ACCESS/prep_data_s1_unusedDInSAR.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK_UNUSED -l /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK/list_Track010_unused.txt -dir 10 -nx 512 -ny 512 -ox 0 -oy 0
$HOME/CODES/ACCESS/prep_data_s1_unusedDInSAR.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK_UNUSED -l /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK/list_Track010_unused.txt -dir 01 -nx 512 -ny 512 -ox 0 -oy 0
$HOME/CODES/ACCESS/prep_data_s1_unusedDInSAR.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK_UNUSED -l /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK/list_Track010_unused.txt -dir 00 -nx 512 -ny 512 -ox 0 -oy 0

$HOME/CODES/ACCESS/prep_data_s1_unusedDInSAR.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK_UNUSED -l /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK/list_Track010_unused.txt -dir 11 -nx 512 -ny 512 -ox 256 -oy 256
$HOME/CODES/ACCESS/prep_data_s1_unusedDInSAR.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK_UNUSED -l /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK/list_Track010_unused.txt -dir 10 -nx 512 -ny 512 -ox 256 -oy 256
$HOME/CODES/ACCESS/prep_data_s1_unusedDInSAR.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK_UNUSED -l /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK/list_Track010_unused.txt -dir 01 -nx 512 -ny 512 -ox 256 -oy 256
$HOME/CODES/ACCESS/prep_data_s1_unusedDInSAR.py -p /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK_UNUSED -l /u/pennell-z1/eric/SEONGSU_SCRATCH/ACCESS_PSK/list_Track010_unused.txt -dir 00 -nx 512 -ny 512 -ox 256 -oy 256

