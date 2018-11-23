#!/bin/bash
#PBS -q gpuk -o submission_poster.out -e submission_poster.err -l walltime=200:00:00
cd $PBS_O_WORKDIR
use cuda8

./main.o -dt 1e-4 -output /home/user/work/coupled_model/output -vertex-file ./voronoi/2K/vertices.txt -ridge-file ./voronoi/2K/ridges.txt -orientation-file ./voronoi/2K/oris.txt -save-each-percent 0.01 -max-grains-percent 0.9 -solver MULTISTEP -inner-resolution 8 -lambda 0.01 -mu 1  -kappa0 1e-6  -N0 1000  -Dkappa0 1e12 -de 0.1
