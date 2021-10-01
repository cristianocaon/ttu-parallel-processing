#!/bin/bash

mpicc -o $1 ./$1.c
mpirun -np 8 ./$1
