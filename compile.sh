#!/bin/bash
module load cuda
nvcc kernel.cu cupSODA.cu input_reader.cpp stoc2det.cpp -gencode=arch=compute_20,code=compute_20 -O3 -o cupSODA --use_fast_math --cudart static
nvcc kernel.cu cupSODA.cu input_reader.cpp stoc2det.cpp -gencode=arch=compute_35,code=compute_35 -O3 -o cupSODA35 --use_fast_math  --cudart static


