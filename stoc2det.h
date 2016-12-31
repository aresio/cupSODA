/*
stoc_2_det.h: conversion of molecule-based models into concentration-based models.
See file COPYING for copyright and licensing information.
*/


#ifndef __STOC2DET__
#define __STOC2DET__

#include <string>
#include <vector>

#define N_A 6.0221415E23 

#define MAX_ODE_LUN 4000
#define MAX_JAC_LUN 4000

typedef double param_t;
typedef double tau_t;
typedef double conc_t;

class st2det {

public:

	std::string DEFAULT_FOLDER;
	std::string DEFAULT_OUTPUT;
	std::string DEFAULT_NAME;
	std::string DEFAULT_PREFIX;
	
	double c_moles_molec;
	double volume;
	tau_t time_max;

	unsigned int species;
	unsigned int reactions;
	unsigned int threads;
	unsigned int actual_threads;
	unsigned int blocks;
	unsigned int tpb;

	
	/// Vettori usati da cupSODA
	
	// stato del sistema
	conc_t* X;

	// quali specie campionare...
	std::vector<unsigned int> species_to_sample;

	// ...e quando
	std::vector<tau_t> time_instants;

	// con che precisione
	double rtol;
	std::vector<double> atol;

	// vettore corrispondenze thread <--> condizione sperimentale
	char* thread2experiment;

	// matrice serie temporali
	double* global_time_series;

	// sistema di equazioni differenziali e jacobiano (in formato compresso)
	param_t* ODE;
	param_t* JAC;
	param_t* c_matrix;

	// offset delle varie equazioni differenziali
	unsigned int* ODE_offset;
	unsigned int* JAC_offset;

	// lunghezza dei sistemi in formato compresso
	unsigned int ODE_lun;
	unsigned int JAC_lun;

	// dati sperimentali
	unsigned int experiments;
	unsigned int repetitions;
	unsigned int target_quantities;

	unsigned int max_steps;


	/// Constructor.
	st2det() {
		this->DEFAULT_FOLDER = "./input";
		this->DEFAULT_OUTPUT = "./ODE_output";
		this->DEFAULT_NAME = "ODE";
		this->c_moles_molec = 0;

		this->species = 0;		
		this->reactions = 0;
		this->threads = 0;
		this->blocks = 0;
		this->tpb = 0;

		this->time_max = 0;

		this->ODE_lun = 0;
		this->JAC_lun = 0;

		this->rtol = (double)1e-4;
		this->max_steps = 0;

		// this->time_instants.push_back(0);

		this->experiments = 0;
		this->repetitions = 0;
		this->target_quantities = 0;

	}

	// bool open_files( std::string folder, std::string output, std::string prefix, bool use_cmatrix, bool dump, bool just_fit, bool use_constant);
	bool st2det::open_files( std::string folder, std::string output, std::string prefix, bool use_cmatrix, bool dump, bool just_fit, bool use_constant, bool traditional_fitness);
	void dump_odes();
	void dump_jac(bool print_to_video=false);

private:
	void tokenize( std::string s, std::vector<std::string>& v );
	unsigned int fattoriale(int n);

};


#endif