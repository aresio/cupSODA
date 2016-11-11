/*

kernel.cu: cupSODA's main file.

cupSODA is black-box deterministic simulator of biological systems that 
exploits the remarkable memory bandwidth and computational capability of GPUs.
cupSODA allows to efficiently execute in parallel large numbers of simulations, 
which are usually required to investigate the emergent dynamics of a given 
biological system under different conditions. cupSODA works by automatically 
deriving the system of ordinary differential equations from a reaction-based 
mechanistic model, defined according to the mass-action kinetics, and then 
exploiting the numerical integration algorithm, LSODA.

See file COPYING for copyright and licensing information.

Bibliography:

- Nobile M.S., Cazzaniga P., Besozzi D., Mauri G.: GPU-accelerated simulations of
mass-action kinetics models with cupSODA. The Journal of Supercomputing, 
vol. 69, issue 1, pp.17-24, 2014

- Petzold L.: Automatic selection of methods for solving stiff and nonstiff 
systems of ordinary differential equations. SIAM Journal of Scientific and
Statistical Computing, 4(1):136-148, 1983

Direct link to the cupSODA paper: http://link.springer.com/article/10.1007/s11227-014-1208-8

*/

#define NOMINMAX

#include <algorithm>
#include <stdio.h>
#include <cmath>		// std::min, std::max
#include <iostream>
#include <fstream>
#include <vector>
#include "constants.h"
#include "service_stuff.h"
#include "cupSODA.h"
#include "input_reader.h"
#include "stoc2det.h"
#include <cuda.h>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <string>

// #define VERBOSE
#define DUMP
#define USE_NEW_ODE


/* Pointer to compressed ODE on the GPU */
char* device_compressed_odes;

/* Int2String conversion utility */
std::string convertInt(int number)
{
   std::stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

int main(int argc, char** argv)  
{

	/* This function verifies that all non-optional arguments are correctly passed to cupSODA */
	if ( !CheckArguments(argc, argv)) {
		exit(-1);
	}
	
	/* The following code determines and sets the optional values. */
	unsigned int GPU = atoi(argv[5]);

	bool just_fitness = true;
	bool print_fitness = false;
	bool print_yukifitness = false;
	bool print_nomanfitness = false;
	bool dump_fft = false;
	bool print_dynamics = false;

	if (argc>6) just_fitness  = std::string(argv[6])!="0";
	if (argc>6) print_fitness = std::string(argv[6])=="2";
	if (argc>6) print_yukifitness = std::string(argv[6])=="3";
	if (argc>6) print_nomanfitness = std::string(argv[6])=="4";
	if (argc>6) dump_fft = std::string(argv[6])=="5";
	if (argc>6) print_dynamics = std::string(argv[6])=="6";
	
	bool ACTIVATE_SHARED_MEMORY = false;
	bool ACTIVATE_CONSTANT_MEMORY = false;
	
	if (argc>7) {
		ACTIVATE_SHARED_MEMORY = std::string(argv[7])!="0";
		ACTIVATE_CONSTANT_MEMORY = std::string(argv[7])=="2";
	}

	int DUMP_DEBUG = 0;
	float EA_ITERATION =-1;

	if (argc>8) {
		DUMP_DEBUG = atoi(std::string(argv[8]).c_str());		
	}

	if (argc>9) {
		EA_ITERATION = atof( std::string(argv[9]).c_str() );
	}

	/* The st2det object load and contains all the information related to the model. */
	st2det* s2d = new st2det();
	s2d->blocks = atoi(argv[2]);		// TODO
	bool a = s2d->open_files(argv[1], argv[3], argv[4], true, DUMP_DEBUG, just_fitness, ACTIVATE_CONSTANT_MEMORY, !(dump_fft|print_yukifitness|print_nomanfitness));
	
	
	if (a!=true) {
		perror("ERROR while opening input files:");
		exit(-3);
	} else {
		if (!(print_fitness|print_yukifitness|print_nomanfitness|print_dynamics)) 		printf(" * All files correct\n");		
	}

	if (!(print_fitness|print_yukifitness|print_nomanfitness|print_dynamics)) {
		if ( just_fitness ) {
			printf(" * Fitness calculation: ENABLED \n");	
		} else {
			printf(" * Fitness calculation: DISABLED \n");
		}
	}

	if (print_dynamics) {
		// printf(" * Dynamics redirected to stdout.\n");
		just_fitness = false;
	}

	/* This function sets all the constant values in the GPU, storing them in constant memory. */
	SetConstants(s2d->species, s2d->reactions, s2d->ODE_lun, s2d->JAC_lun, s2d->species_to_sample.size(), s2d->time_instants.size(), s2d->repetitions, s2d->experiments, s2d->threads, false);

	conc_t*			host_X;
	param_t*		device_flattenODE;
	unsigned int*	device_offsetODE;
	param_t*		device_flattenJAC;
	unsigned int*	device_offsetJAC;
	unsigned int*	device_species_to_sample;
	conc_t*			device_X;
	conc_t*			device_target;

	double* device_fitness;
	double* host_fitness;
	char* device_swarms;

	/* Allocate and store the ODE system and the Jacobian matrix */
	cudaMalloc((void**)&device_flattenODE,sizeof(param_t)*s2d->ODE_lun);
	CudaCheckError();
	cudaMemcpy(device_flattenODE,s2d->ODE,sizeof(param_t)*s2d->ODE_lun,cudaMemcpyHostToDevice);
	CudaCheckError();
	cudaMalloc((void**)&device_flattenJAC,sizeof(param_t)*s2d->JAC_lun);
	CudaCheckError();
	cudaMemcpy(device_flattenJAC,s2d->JAC,sizeof(param_t)*s2d->JAC_lun,cudaMemcpyHostToDevice);
	CudaCheckError();


	// Allocate and store the offsets used by the ODE and Jacobian representations */
	cudaMalloc((void**)&device_offsetODE,sizeof(unsigned int)*s2d->species);
	CudaCheckError();
	cudaMemcpy(device_offsetODE,s2d->ODE_offset,sizeof(unsigned int)*s2d->species,cudaMemcpyHostToDevice);
	CudaCheckError();	
	cudaMalloc((void**)&device_offsetJAC,sizeof(unsigned int)*s2d->species*s2d->species);
	CudaCheckError();
	cudaMemcpy(device_offsetJAC,s2d->JAC_offset,sizeof(unsigned int)*s2d->species*s2d->species,cudaMemcpyHostToDevice);
	CudaCheckError();


	/* Allocates the memory space for samples and the target time-series */
	cudaMalloc((void**)&device_X, sizeof( conc_t ) * s2d->species_to_sample.size() * s2d->time_instants.size() * s2d->threads );
	CudaCheckErrorRelease();
	cudaMalloc((void**)&device_target, sizeof( conc_t ) * s2d->repetitions * s2d->experiments * s2d->target_quantities * s2d->time_instants.size() );
	CudaCheckError();
	cudaMemcpy(device_target,s2d->global_time_series, sizeof( conc_t ) * s2d->repetitions * s2d->experiments * s2d->target_quantities * s2d->time_instants.size() ,cudaMemcpyHostToDevice);
	CudaCheckError();
	host_X = (conc_t*) malloc ( sizeof( conc_t ) * s2d->species_to_sample.size() * s2d->time_instants.size() * s2d->threads );
	if (host_X==NULL) {
		perror("ERROR allocating states\n");
		exit(-14);
	}
	memset( host_X, 0, sizeof( conc_t ) * s2d->species_to_sample.size() * s2d->time_instants.size() * s2d->threads );
	
	/* Load the initial state of the system, according to input files */
	for (unsigned int ss=0; ss<s2d->species_to_sample.size(); ss++) {
		for (unsigned int t=0; t<s2d->threads; t++) {
			host_X[ s2d->threads*ss + t ] = s2d->X[ s2d->species_to_sample[ss] ]; 
		}
	}	
	cudaMemcpy(device_X,host_X,sizeof(conc_t) * s2d->species_to_sample.size() * s2d->time_instants.size() * s2d->threads, cudaMemcpyHostToDevice );
	CudaCheckError();
	

	/* Allocate and store the species to be sampled */
	cudaMalloc((void**)&device_species_to_sample, sizeof( unsigned int ) * s2d->species_to_sample.size() );
	cudaMemcpy(device_species_to_sample,&s2d->species_to_sample[0],sizeof(unsigned int) * s2d->species_to_sample.size(),cudaMemcpyHostToDevice);
	CudaCheckError();
	
		
	///// DEBUG //////
	int* h_debug = (int*) malloc ( sizeof(int)*s2d->threads );
	int* d_debug;
	cudaMalloc((void**)&d_debug,sizeof(int)*s2d->threads);	
	CudaCheckError();  
	///// DEBUG //////

	
	 /* Local variables */
	 double* constants = (double*) malloc( sizeof(double) *s2d->threads*s2d->reactions );
     double *t = (double*)malloc(sizeof(double)*s2d->threads);
	 double *y = (double*)malloc(sizeof(double)*s2d->species*s2d->threads);
     int *jt = (int*)malloc(sizeof(int)*s2d->threads);
     int *neq = (int*)malloc(sizeof(int)*s2d->threads);
	 int *liw = (int*)malloc(sizeof(int)*s2d->threads);
	 int *lrw = (int*)malloc(sizeof(int)*s2d->threads);     
	 double *atol = (double*)malloc(sizeof(double)*s2d->species*s2d->threads);
     int *itol =(int*) malloc(sizeof(int)*s2d->threads);
	 int *iopt =(int*) malloc(sizeof(int)*s2d->threads);
     double *rtol = (double*)malloc(sizeof(double)*s2d->threads);
     int *iout =(int*) malloc(sizeof(int)*s2d->threads);
     double *tout =(double*) malloc(sizeof(double)*s2d->threads);
     int *itask = (int*)malloc(sizeof(int)*s2d->threads);
		 	 
	 // const int lrn = 20 + 16*s2d->species;
	 const int lrs = 22+9*s2d->species + (s2d->species*s2d->species);
	 // const int LRW = max(lrn,lrs);
	 const int LRW = 22+s2d->species*std::max(16,(int)(s2d->species+9));	 
	 const int LIW = 20+s2d->species;	 
	 
	 
	double *rwork = (double*)malloc(sizeof(double)*LRW*s2d->threads);
	memset(rwork, 0, sizeof(double)*LRW*s2d->threads); // TEST

	int *iwork = (int*) malloc(sizeof(int)*LIW*s2d->threads);     
	memset(iwork, 0, sizeof(int)*LIW*s2d->threads); // TEST

	 int *istate = (int*)malloc(sizeof(int)*s2d->threads);
	struct cuLsodaCommonBlock* common = (struct cuLsodaCommonBlock*) malloc(sizeof(struct cuLsodaCommonBlock)*s2d->threads);
	struct cuLsodaCommonBlock *Hcommon = common;


	/* Pointers to GPU's global memory areas of LSODA data structures */
	double* device_constants;
	double	*_Dt;
	double	*_Dy;	
	int	*_Djt;
	int	*_Dneq;
	int	*_Dliw;
	int	*_Dlrw;
    double	*_Datol;
    int	*_Ditol;
	int	*_Diopt;
    double	*_Drtol;
    int	*_Diout;
    double	*_Dtout;
    int	*_Ditask;
	int	*_Diwork;	
    double	*_Drwork;	
	int	*_Distate;
	struct cuLsodaCommonBlock *_Dcommon;	
	
	/* Method instantiations for Derivative and Jacobian functions to send to template */
	myFex fex;
	myJex jex;

	/* Assignment of initial values to locals */
	for (unsigned int i=0; i<s2d->threads; i++) {
		
		 // iwork[i*LIW+5] = 100000;		// default
		iwork[i*LIW+5] = s2d->max_steps;

		for (unsigned r=0; r<s2d->reactions; r++) 
			constants[i*s2d->reactions+r] = s2d->c_matrix[ i*s2d->reactions+r ];

		/* Number of ODEs */
		neq[i] = s2d->species;

		/* Initial quantities */
		for (unsigned int s=0; s<s2d->species; s++)
			y[i*s2d->species+s] = s2d->X[i*s2d->species+s];

		/* Initial time */
		t[i] = (double)0.;
				
		/* Error tolerances */
		itol[i] = 2;
		rtol[i] = s2d->rtol;						
		for (unsigned int s=0; s<s2d->species; s++) {
			atol[i*s2d->species+s] = s2d->atol[s];			
		}		

		/* Standard LSODA execution */
		itask[i] = 1;
		istate[i] = 1;
		iopt[i] = 0;
		lrw[i] = LRW;
		liw[i] = LIW;
		jt[i] = 2;
	}
	cuLsodaCommonBlockInit(Hcommon, s2d->threads);
	
	/* Allocate the global memory for LSODA data structures, replicated for each thread,
	   and store the values loaded from local files. */
	cudaMalloc((void**)&device_constants,sizeof(double)*s2d->threads*s2d->reactions);	cudaMemcpy(device_constants,constants,sizeof(double)*s2d->threads*s2d->reactions,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Dt,sizeof(double)*s2d->threads);								cudaMemcpy(_Dt,t,sizeof(double)*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Dy,sizeof(double)*s2d->species*s2d->threads);					cudaMemcpy(_Dy,y,sizeof(double)*s2d->species*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Djt,sizeof(int)*s2d->threads);									cudaMemcpy(_Djt,jt,sizeof(int)*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Dneq,sizeof(int)*s2d->threads);								cudaMemcpy(_Dneq,neq,sizeof(int)*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Dliw,sizeof(int)*s2d->threads);								cudaMemcpy(_Dliw,liw,sizeof(int)*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Dlrw,sizeof(int)*s2d->threads);								cudaMemcpy(_Dlrw,lrw,sizeof(int)*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Datol,sizeof(double)*s2d->species*s2d->threads);				cudaMemcpy(_Datol,atol,sizeof(double)*s2d->species*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Ditol,sizeof(int)*s2d->threads);								cudaMemcpy(_Ditol,itol,sizeof(int)*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Diopt,sizeof(int)*s2d->threads);								cudaMemcpy(_Diopt,iopt,sizeof(int)*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Drtol,sizeof(double)*s2d->threads);							cudaMemcpy(_Drtol,rtol,sizeof(double)*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Diout,sizeof(int)*s2d->threads);								cudaMemcpy(_Diout,iout,sizeof(int)*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Dtout,sizeof(double)*s2d->threads);							cudaMemcpy(_Dtout,tout,sizeof(double)*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Ditask,sizeof(int)*s2d->threads);								cudaMemcpy(_Ditask,itask,sizeof(int)*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Diwork,sizeof(int)*LIW*s2d->threads);							cudaMemcpy(_Diwork,iwork,sizeof(int)*LIW*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Drwork,sizeof(double)*LRW*s2d->threads);						cudaMemcpy(_Drwork,rwork,sizeof(double)*LRW*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Distate,sizeof(int)*s2d->threads);								cudaMemcpy(_Distate,istate,sizeof(int)*s2d->threads,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Dcommon,sizeof(struct cuLsodaCommonBlock)*s2d->threads);		cudaMemcpy(_Dcommon,Hcommon,sizeof(struct cuLsodaCommonBlock)*s2d->threads, cudaMemcpyHostToDevice);
	CudaCheckError()  ;
	
	unsigned int sh_memory_bytes;

	/* Check for available shared memory: if the execution hierarchy (i.e., number of threads per block)
	   cannot be launched with the current configuration, then abort. */
	if (ACTIVATE_SHARED_MEMORY) {
		sh_memory_bytes = sizeof(double)*s2d->species*s2d->tpb + sizeof(double)*s2d->tpb;
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, GPU);
		if (sh_memory_bytes > prop.sharedMemPerBlock ) {
			printf("ERROR: insufficient shared memory (%d).\n", sh_memory_bytes);
			exit(ERROR_INSUFF_SHARED_MEMORY);
		}
	} else {
		sh_memory_bytes = 0;
	}

	SetODEarray( s2d );

	// Code for profiling
	cudaEvent_t start, stop;
	if (!just_fitness) {
		cudaEventCreate(&start);  
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
	}
	
	// LSODA documentation reads: "write a main program which calls subroutine lsoda once for each point at which answers are desired"
	// For this reason, I use a for cycle that goes through the set of sampling temporal instants.
	for (unsigned int ti=0; ti<s2d->time_instants.size(); ti++) 
	{

		/* Si può parallelizzare: TODO */		
		for (unsigned int i=0; i<s2d->threads; i++) tout[i] = s2d->time_instants[ti];		
		
		cudaMemcpy(_Dtout,tout,sizeof(double)*s2d->threads,cudaMemcpyHostToDevice);

		// printf(" * Time step to %f.\n", tout[0]);
					
		dim3 BlocksPerGrid(s2d->blocks,1,1);
		dim3 ThreadsPerBlock(s2d->tpb,1,1);
		
		// CUDA LSODA entry point 
		cuLsoda<<<BlocksPerGrid,ThreadsPerBlock,sh_memory_bytes>>>
			(fex, _Dneq, _Dy, _Dt, _Dtout, _Ditol, _Drtol, _Datol, _Ditask, _Distate, _Diopt, _Drwork, _Dlrw, _Diwork, _Dliw, 
			 jex, _Djt, _Dcommon, d_debug, device_compressed_odes, device_flattenODE, device_offsetODE, device_constants, device_X, ti, 
			 device_species_to_sample, device_flattenJAC, device_offsetJAC, ACTIVATE_SHARED_MEMORY, ACTIVATE_CONSTANT_MEMORY);
		CudaCheckError();

		/* Print debug information (if requested), for each thread */
		if (DUMP_DEBUG==2) {
			cudaMemcpy(istate,_Distate, sizeof(int)*s2d->threads,cudaMemcpyDeviceToHost);
			printf(" * Dumping istates:\n");
			for (unsigned int th=0; th<s2d->threads; th++) {				
				printf("Thread %d: istate %d", th, istate[th]);
				switch(istate[th]) {
				case 1: printf(" (First step) "); break; 
				case 2: printf(" (OK!) "); break; 
				case -1: printf (" (excess of word done) "); break;
				case -2: printf (" (excess of accuracy requested) "); break;
				case -3: printf (" (illegal input detected) "); break;
				case -4: printf (" (repeated error test failures) "); break;
				case -5: printf (" (convergence failure) "); break;
				case -6: printf (" (error weight became zero) "); break;
				case -7: printf (" (work space insufficient to finish) "); break;
				default:
					printf(" (UNKNOWN LSODA ERROR) "); break;
				};
				printf("\n");
			}

			cudaMemcpy(iwork,_Diwork, sizeof(int)*LIW*s2d->threads,cudaMemcpyDeviceToHost);

			for (unsigned int th=0; th<s2d->threads; th++) {
				printf("[Thr%d] steps so far: %d, ODE evals: %d, Jac evals: %d.\n", th, iwork[th*20+10], iwork[th*20+11], iwork[th*20+12]);
			}

			printf("\n");
		}

    }

	if ((!just_fitness) && (!print_dynamics)) {
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		float tempo;
		cudaEventElapsedTime( &tempo, start, stop );
		tempo /= 1000;
		printf("Running time: %f seconds\n", tempo);
		cudaEventDestroy(start); 
		cudaEventDestroy(stop); 
	}
	
	unsigned int DEV_CONST_SAMPLESLUN = (unsigned int)s2d->species_to_sample.size();

	cudaThreadSynchronize();

	// Calculate the FT of the signals (i.e., time-series)
	if (dump_fft) {

		// calculate_fft(s2d, device_X, "prova_fft");
		exit(0);

	}

	/* If we are just calculating a fitness value, avoid creating and dumping to output files of simulations */
	if (just_fitness) {

		host_fitness = (double*) malloc ( sizeof(double) * s2d->threads );

		cudaMalloc((void**)&device_fitness,sizeof(double)*s2d->threads);	
		cudaMalloc((void**)&device_swarms, sizeof(char)  *s2d->threads);	
		CudaCheckError();

		cudaMemcpy(device_swarms, s2d->thread2experiment, sizeof(char)*s2d->threads, cudaMemcpyHostToDevice);
		CudaCheckError();
		
		dim3 BlocksPerGrid(s2d->blocks,1,1);
		dim3 ThreadsPerBlock(s2d->tpb,1,1);

		if (print_yukifitness)  {
			// calculateFitnessYuki<<<BlocksPerGrid,ThreadsPerBlock>>>( device_X, device_fitness);
			CudaCheckError();			
		} else if (print_nomanfitness) {
			calculateFitnessNoman<<<BlocksPerGrid,ThreadsPerBlock>>>( device_X, device_fitness, EA_ITERATION);
			CudaCheckError();			
		} 	else {
			calculateFitness<<<BlocksPerGrid,ThreadsPerBlock>>>( device_X, device_target, device_fitness, device_swarms );
			CudaCheckError();
		}

		// cudaMemcpy(device_swarms,s2d->thread2experiment,sizeof(char)*s2d->threads, cudaMemcpyDeviceToHost);
		cudaMemcpy(host_fitness,device_fitness,sizeof(double)*s2d->threads, cudaMemcpyDeviceToHost);
		CudaCheckError();

		// if we are not just printing on video the fitness values, open the output file "pref_allfit"
		if ( !(print_fitness | print_yukifitness | print_nomanfitness) ) {

		#ifdef _WIN32
			std::string comando_crea_cartella("md ");
			comando_crea_cartella.append(s2d->DEFAULT_OUTPUT);		
			system(comando_crea_cartella.c_str());
		#else
			std::string comando_crea_cartella("mkdir ");
			comando_crea_cartella.append(s2d->DEFAULT_OUTPUT);
			system(comando_crea_cartella.c_str());
		#endif


			std::string outputfile(s2d->DEFAULT_OUTPUT);
			outputfile.append("/pref_allfit");
		
			std::ofstream dump2(outputfile.c_str());

			if (!dump2.is_open()) {
				printf("Path: %s.\n", outputfile.c_str());
				perror("Unable to save fitness file 'pref_allfit', aborting.");
				exit(-17);
			}

			// verify!!
			for (unsigned int sw=0; sw<s2d->blocks; sw++) {
				for (unsigned int part=0; part<s2d->tpb; part++) {
					dump2 << host_fitness[sw*s2d->tpb + part] << "\t";					
				}
				dump2 << "\n";
			}
			dump2.close();

		} else {

			// verify!!
			for (unsigned int sw=0; sw<s2d->blocks; sw++) {
				for (unsigned int part=0; part<s2d->tpb; part++) {
					std::cout << host_fitness[sw*s2d->tpb + part] << "\t";					
				}
				std::cout << "\n";
			}

		}

		free(host_fitness);
		cudaFree(device_fitness);	
		cudaFree(device_swarms);
		cudaFree(device_target);


	}  // end if just fitness

	if (!just_fitness) {

		// No fitness: let's save the output of simulations on the hard disk
		cudaMemcpy(host_X,device_X, sizeof(conc_t) * s2d->species_to_sample.size() * s2d->threads * s2d->time_instants.size(), cudaMemcpyDeviceToHost);

		if (print_dynamics) {

			for ( unsigned int tid=0; tid<s2d->actual_threads; tid++ ) {
				
				std::cout << std::setprecision(15);
		
				unsigned int larg = s2d->threads;
				unsigned int DEV_CONST_SAMPLESLUN = (unsigned int) s2d->species_to_sample.size();		
	
				for (unsigned int campione=0; campione<s2d->time_instants.size(); campione++) {
					std::cout << s2d->time_instants[campione] << "\t";
					for (unsigned int s=0; s<s2d->species_to_sample.size(); s++) {				
						std::cout << host_X[ ACCESS_SAMPLE ];
						if ( s!=s2d->species_to_sample.size()-1 )
							std::cout << "\t";
					}
					std::cout << "\n";
				}
				std::cout << "\n";

			}

		} else {

			// """crossplatform""" folder creation (TODO)
			#ifdef _WIN32
				std::string comando_crea_cartella("md ");
				comando_crea_cartella.append(s2d->DEFAULT_OUTPUT);		
				system(comando_crea_cartella.c_str());
			#else
				std::string comando_crea_cartella("mkdir ");
				comando_crea_cartella.append(s2d->DEFAULT_OUTPUT);
				system(comando_crea_cartella.c_str());
			#endif

			for ( unsigned int tid=0; tid<s2d->actual_threads; tid++ ) {
				
				std::string outputfile(s2d->DEFAULT_OUTPUT);
				outputfile.append("/");
				outputfile.append(s2d->DEFAULT_PREFIX);
				outputfile.append("_");
				outputfile.append( convertInt(tid) );

				std::ofstream dump2(outputfile.c_str());

				if (! dump2.good() ) {
					perror("ERROR: cannot save dynamics");
					exit(-2);
				}
	 
				dump2 << std::setprecision(15);
		
				unsigned int larg = s2d->threads;
				unsigned int DEV_CONST_SAMPLESLUN = (unsigned int) s2d->species_to_sample.size();		
	
				for (unsigned int campione=0; campione<s2d->time_instants.size(); campione++) {
					dump2 << s2d->time_instants[campione] << "\t";
					for (unsigned int s=0; s<s2d->species_to_sample.size(); s++) {				
						dump2 << host_X[ ACCESS_SAMPLE ];
						if ( s!=s2d->species_to_sample.size()-1 )
							dump2 << "\t";
					}
					dump2 << "\n";
				}
				dump2.close();

			} // end for

		} // end print fitness

	} // end not just fitness

	// release memory on the CPU
	free(host_X);
		
	// release memory on the GPU
	cudaFree(device_X);
	cudaFree(device_compressed_odes);
	cudaFree(device_constants);
	cudaFree(device_flattenJAC);
	cudaFree(device_flattenODE);
	cudaFree(device_offsetJAC);
	cudaFree(device_offsetODE);
	cudaFree(device_species_to_sample);	

	    return 0;
} 


