#!/usr/bin/env bash

$HOME/CODES/ACCESS/prep_data_s1.py -dir 11 -nx 512 -ny 512 -ox 0 -oy 0
$HOME/CODES/ACCESS/prep_data_s1.py -dir 10 -nx 512 -ny 512 -ox 0 -oy 0
$HOME/CODES/ACCESS/prep_data_s1.py -dir 01 -nx 512 -ny 512 -ox 0 -oy 0
$HOME/CODES/ACCESS/prep_data_s1.py -dir 00 -nx 512 -ny 512 -ox 0 -oy 0

#staggered grid
$HOME/CODES/ACCESS/prep_data_s1.py -dir 11 -nx 512 -ny 512 -ox 256 -oy 256
$HOME/CODES/ACCESS/prep_data_s1.py -dir 10 -nx 512 -ny 512 -ox 256 -oy 256
$HOME/CODES/ACCESS/prep_data_s1.py -dir 01 -nx 512 -ny 512 -ox 256 -oy 256
$HOME/CODES/ACCESS/prep_data_s1.py -dir 00 -nx 512 -ny 512 -ox 256 -oy 256



