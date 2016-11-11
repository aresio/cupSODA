#cupSODA release 1.1.0

## ABOUT

cupSODA is a black-box deterministic simulator of biological systems that exploits the remarkable memory bandwidth and computational capability of GPUs. 
cupSODA allows to efficiently execute in parallel large numbers of simulations, which are usually required to investigate the emergent dynamics of a given biological system under different conditions.
cupSODA works by automatically deriving the system of ordinary differential equations from a reaction-based mechanistic model, defined according to the mass-action kinetics, and then exploiting the numerical integration algorithm, LSODA. 


## DEPENDENCIES

Just the Nvidia CUDA library (version >7.0).


##  COMPILATION

A cupSODA binary can be compiled on any supported architecture (e.g., GNU/Linux, Microsoft Windows, Apple OS/X) using the following compilation command:

```bash
nvcc kernel.cu cupSODA.cu input_reader.cpp stoc2det.cpp -gencode=arch=compute_20,code=compute_20 -O3 -o cupSODA --use_fast_math
```

The command above would create a binary executable file runnable on GPUs with _at least_ a compute capability equal to 2.0. Please note that a specific compute capability, supporting additional functionality, can be targeted by using the ```gencode``` argument. For instance, to target the compute capability 3.5, the following argument can be passed to ```nvcc```:

```bash
-gencode=arch=compute_35,code=compute_35
```

## LAUNCHING CUPSODA

cupSODA is designed to be launched from the command line. The arguments are: 

`cupSODA input_folder blocks output_folder prefix gpu fitness memory_configuration debug`

where

* `input_folder` is the path to the directory containing the input model;
* `blocks` is the number of CUDA blocks used to distribute the requested parallel threads;
* `output_folder` is the path to the directory that will store the output dynamics of the simulations;
* `prefix` is the file name of the output files. A number, corresponding the thread, will be automatically appended to the filename by cupSODA;
* `debug` enables debug information: 1 outputs everything, 2 outputs everything but LSODA istate values for each thread.

Further information about the `gpu`, `fitness` and `memory_configuration` arguments, along with the specifications of the input files, can be found at the following address:

https://docs.google.com/document/d/1gPq-mYk-IP-bVmiMZewGPmTJ6nMCH8al1nNr7OaBsv4/edit?usp=sharing

## LICENSE

BSD License


## CONTACT 

nobile@disco.unimib.it
