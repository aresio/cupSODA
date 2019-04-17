/*
stoc_2_det.h: conversion of molecule-based models into concentration-based models.
See file COPYING for copyright and licensing information.
*/


#include "stoc2det.h"
#include <iostream>
#include <ostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <math.h>
#include <cstdlib>
#include <cstring>

#include "constants.h"

// #define VERBOSE
#define FULL_DUMP

double potenza(double v, unsigned int n) {
	if (n==0) return 1;
	if (n==1) return v;
	if (n>1) return v*potenza(v,n-1);
}

void st2det::tokenize( std::string s, std::vector<std::string>& v )  {
    std::istringstream buf(s.c_str());
    for(std::string token; getline(buf, token, '\t'); )
        v.push_back(token);
}

unsigned int st2det::fattoriale(int n) {
	if (n<2) return 1;
	else return n*fattoriale(n-1);
}

void st2det::dump_odes() {

	std::ofstream derlog("derivatives_log");

	unsigned int pos =0;

	for (unsigned int s=0; s<this->species; s++) {

		// derlog << "dX[" << s << "]/dt = ";
		printf("%d) ", s);
		
		while( pos<this->ODE_offset[s] ) {
			printf("%.1f ", this->ODE[pos]);
			pos ++;
		}
		printf("\n");
		// pos ++;

	}

	// system("pause");

	pos = 0;

	derlog << "// Dump ordinary differential equations\n";
	printf(" * DUMP ODE\n");

	// unsigned int pos=0;
	unsigned int reazione=0;

	for (unsigned int s=0; s<this->species; s++) {

		printf("dX[%d]/dt = ", s);
		derlog << "dy[" << s << "]=";

		if (pos==this->ODE_offset[s]) {
			derlog << "0";
		}
		
		while( pos<this->ODE_offset[s] ) {

			unsigned int i;
					
			// variazione
			if (this->ODE[pos]>0)  {
				printf("+");
				derlog << "+";
			}

			printf(" %d ", (int)this->ODE[pos]);
			derlog << (int)this->ODE[pos];

			// costante cinetica
			printf(" * k[%d] ", (int)this->ODE[pos+1]);
			derlog << "*p[" << (int)this->ODE[pos+1] << "]";

			// fattore conversione
			printf(" * %f ", this->ODE[pos+2]);
			derlog << "*" << this->ODE[pos+2];

			// concentrazioni
			for( i=0; i<this->ODE[pos+3]; i++ ) {
				printf(" * X[%d] ", (int)this->ODE[pos+4+i]);
				derlog << "*y[" << (int)this->ODE[pos+4+i] << "]";
			}
		
			pos += 4 + i;

		} 

		printf("\n");
		derlog << ";\n";
	}

	derlog.close();

}

void st2det::dump_jac(bool print_to_video) {

	// return;

	std::ofstream derlog("derivatives_log", std::ofstream::app);

	if (print_to_video) printf("\n * DUMP JACOBIANO\n");
	derlog << "\n// Dump Jacobian matrix\n";

	/*
	derlog << "Offsets:\n";
	for (unsigned int i=0; i<this->species*this->species; i++) {
		derlog << this->JAC_offset[i] << " ";
	}

	derlog << "\nCodifica:\n";
	unsigned int o=0;
	for (unsigned int i=0; i<this->JAC_lun; i++) {		
		while ( this->JAC_offset[o]==i ) {			
			o++;
			// derlog << "\n";
		}
		derlog << this->JAC[i] << " ";
	}
	derlog << "\n\n";
	*/

	unsigned int pos = 0;
	
	for (unsigned int s1=0; s1<this->species; s1++) { 

		for (unsigned int s2=0; s2<this->species; s2++) { 

			// s1 * DEV_CONST_SPECIES + s2

			if (print_to_video) printf("dX[s1 * DEV_CONST_SPECIES + s2] = ", s2, s1);
			derlog << "dX[" << (s1 * this->species + s2) << "] = ";

			// if (print_to_video) printf("dX[%d][%d] = ", s2, s1);
			// derlog << "dX[" << s2 << "][" << s1 << "] = ";
			

			if ( pos == this->JAC_offset[s1*this->species+s2]  ) {
				if (print_to_video)  printf("0");
				derlog << 0;
			}

			while( pos<this->JAC_offset[s1*this->species+s2] ) {

				unsigned int i;
					
				// variazione
				if (this->JAC[pos]>0) {
					if (print_to_video) printf("+");
					derlog << "+";
				}
				if (print_to_video) printf(" %d ", (int)this->JAC[pos]);
				derlog << (int)this->JAC[pos] << " ";

				// costante cinetica
				if (print_to_video) printf(" * k[%d] ", (int)this->JAC[pos+1]);
				derlog << " * k[" << (int)this->JAC[pos+1] << "] ";

				// fattore conversione
				if (print_to_video) printf(" * (fc)%f ", this->JAC[pos+2]);
				derlog << " * " << this->JAC[pos+2] << " ";

				// concentrazioni
				for( i=0; i<this->JAC[pos+3]; i++ ) {
					if (print_to_video) printf(" * X[%d] ", (int)this->JAC[pos+4+i]);
					derlog << " *X[" << (int)this->JAC[pos+4+i] << "] ";
				}
		
				pos += 4 + i;

			}

			if (print_to_video) printf("\n");
			derlog << ";\n";

		}				
	}

	
}


bool st2det::open_files( std::string folder, std::string output, std::string prefix, bool use_cmatrix, bool dump, bool just_fit, bool use_constant, bool traditional_fitness) {

	std::string v;	
	
	if (!folder.empty()) {
		this->DEFAULT_FOLDER = folder;
	} 

	if (!output.empty()) {
		this->DEFAULT_OUTPUT = output;
	}

	if (!prefix.empty()) {
		this->DEFAULT_PREFIX = prefix;
	}

	bool StochasticModel = true;
	


	std::ifstream kindfile;
	kindfile.open( (this->DEFAULT_FOLDER+"/modelkind").c_str() );
	if (!kindfile.is_open()) {
		if (dump) printf("WARNING: cannot open modelkind file, assuming stochastic\n");		
	} else {
		getline(kindfile, v);		
		StochasticModel = (v == "stochastic");
		if (dump) {
			if (StochasticModel) {
				if (dump) printf(" * Stochastic model detected\n");
			} else {
				if (dump) printf(" * Deterministic model detected\n");
			}
		}
		kindfile.close();
	}
	
	/// step 1: volume
	if (StochasticModel) {
		std::ifstream volumefile;
		// printf("%s\n", (this->DEFAULT_FOLDER+"volume").c_str());
		volumefile.open( (this->DEFAULT_FOLDER+"/volume").c_str() );
		if (!volumefile.is_open()) {
			perror("ERROR: cannot open volume file");
			return false;
		}	
		getline(volumefile, v);	
		// printf("%s\n", v.c_str());
		this->volume = atof( v.c_str() ); 
		if (dump)
			printf(" * Read volume %e\n", this->volume);
		volumefile.close();
		this->c_moles_molec = this->volume*N_A;
		if (dump)
			printf(" * Conversion constant: %e\n", this->c_moles_molec);
	}
		
	


	/// step 2a: matrices pre-processing
	std::ifstream leftfile;
	std::vector<std::string> a;
	leftfile.open((this->DEFAULT_FOLDER+"/left_side").c_str());
	if (!leftfile.is_open()) {
		perror("ERROR: cannot open left matrix");
		return false;
	}

	getline(leftfile, v);
	tokenize(v,a);
	this->species = a.size();
	leftfile.close();

	leftfile.open( (this->DEFAULT_FOLDER+"/left_side").c_str() );
	while( leftfile.good() ) {
		getline(leftfile, v);	
		if (v.length()>0)
			reactions++;
	}
	/*tokenize(v,a);
	this->species = a.size();*/
	if (dump)
		printf (" * Detected %d species\n", this->species);
	
	leftfile.close();
	if (dump)
		printf(" * Detected %d reactions\n", reactions);

	/// step 2: left matrix	
	int* left_matrix = (int*)  malloc ( sizeof(int)*this->species*reactions );
	int* right_matrix = (int*) malloc ( sizeof(int)*this->species*reactions );
	int* var_matrix = (int*)   malloc ( sizeof(int)*this->species*reactions );

	leftfile.open( (this->DEFAULT_FOLDER+"/left_side").c_str() );
	// while ( leftfile.good() ) {
		for (unsigned int r=0; r<reactions; r++) {
			getline(leftfile, v);
			a.clear();
			tokenize(v,a);
			// printf("%s\n", v.c_str());
			for (unsigned s=0; s<species; s++) {
				left_matrix[r*species+s] = atoi(a[s].c_str());
			}
		}
	//}
	leftfile.close();

	std::ifstream rightfile;
	rightfile.open( (this->DEFAULT_FOLDER+"/right_side").c_str() );
	if (! rightfile.is_open()) {
		perror("ERROR: cannot open right matrix");
		return false;
	}
	//while ( rightfile.good() ) {
		for (unsigned int r=0; r<reactions; r++) {
			getline(rightfile, v);
			a.clear();
			tokenize(v,a);
			// printf("%s\n", v.c_str());
			for (unsigned s=0; s<species; s++) {
				right_matrix[r*species+s] = atoi(a[s].c_str());
			}
		}
	//}
	rightfile.close();

	for (unsigned int r=0; r<reactions; r++) {
		for (unsigned int c=0; c<species; c++) {
			var_matrix[r*species+c] = right_matrix[r*species+c] - left_matrix[r*species+c];
		}
	}

	if (dump)
		printf(" * Left, right and var matrices loaded\n");

	/// step 3 : c_matrix 
	// pre-processing old-style
	std::ifstream cm_file;
	cm_file.open( (this->DEFAULT_FOLDER+"/c_matrix").c_str() );
	if (! cm_file.is_open()) {
		perror("ERROR: cannot open c_matrix");
		return false;
	}
	while ( cm_file.good() ) {
		getline(cm_file, v);
		if ( v.size()>2 )
			threads++;
	}
	cm_file.close();

	if (dump)
		printf(" * %d threads necessary (according to c_matrix)\n", threads);

	// experimental
	this->actual_threads = this->threads;	
	this->tpb = ceil( (float)this->threads / this->blocks);
	this->threads = this->tpb * this->blocks;

	if (dump)
		printf(" * Will launch %d threads on %d blocks\n", this->tpb, this->blocks);



	this->c_matrix = (param_t*) malloc ( sizeof(param_t)*threads*reactions);

	cm_file.open( (this->DEFAULT_FOLDER+"/c_matrix").c_str() );
	for (unsigned int t=0; t<threads;  t++) {
		getline(cm_file, v);
		a.clear();
		tokenize(v,a);
		// printf("%s\n", v.c_str());
		for (unsigned r=0; r<reactions; r++) {
			// this->c_matrix[r*threads+t] = atof(a[r].c_str());
			// this->c_matrix[t*reactions+r] = atof(a[r].c_str())/this->c_moles_molec;
			this->c_matrix[t*reactions+r] = atof(a[r].c_str());
		}
	}
	cm_file.close();

	if (dump)
		printf(" * Kinetic constants for %d threads loaded\n", threads);

	unsigned int* mx_matrix = (unsigned int*) malloc( sizeof(unsigned int) * threads * species);
	this->X = (conc_t*) malloc( sizeof(conc_t) * threads * species);

	/// Step 4: MX_0
	std::ifstream mx_file;
	mx_file.open( (this->DEFAULT_FOLDER+"/MX_0").c_str() );
	for (unsigned int t=0; t<threads;  t++) {
		getline(mx_file, v);
		a.clear();
		tokenize(v,a);
		// printf("%s\n", v.c_str());
		for (unsigned s=0; s<species; s++) {

			if (StochasticModel) {
				// mx_matrix[s*threads+t] = atoi(a[s].c_str());
				mx_matrix[t*species+s] = (unsigned int)atof(a[s].c_str());
				// X[s*threads+t] =(float) mx_matrix[s*threads+t]/this->c_moles_molec;			
				X[t*species+s] =(conc_t)(mx_matrix[t*species+s])/(this->c_moles_molec);
				// X[t*species+s] =(conc_t)(mx_matrix[t*species+s]);
			} else {				
				X[t*species+s] = atof(a[s].c_str());

				/*mx_matrix[t*species+s] = atof(a[s].c_str());
				X[t*species+s] = mx_matrix[t*species+s];*/
			}

			
		}
	}
	mx_file.close();

	/// Step 5: feed
	// unsigned int* feed_vector = (unsigned int*) malloc( sizeof(unsigned int)*species);
	conc_t* feed_vector = (conc_t*) malloc( sizeof(conc_t)*species);
	std::ifstream feed_file;
	feed_file.open( (this->DEFAULT_FOLDER+"/M_feed").c_str() );
	if (! feed_file.is_open()) {
		memset( feed_vector, 0, sizeof(conc_t)*species);		
	} else {
		getline(feed_file, v);
		a.clear();
		tokenize(v,a);
		if (a.size()<species) {
			printf("ERROR: M_feed file is wrong (%d species)\n", a.size());
			return false;
		}
		for (unsigned int s=0; s<species; s++) {
			feed_vector[s] = atof(a[s].c_str());
			if (feed_vector[s]!=0) {
				for (unsigned int t=0; t<threads; t++) {				
					if (StochasticModel) 
						X[t*species+s] = feed_vector[s] / (this->c_moles_molec);
					else
						X[t*species+s] = feed_vector[s];
				}
			}
		}
		feed_file.close();

		if (dump)
			printf(" * Feed vector loaded\n");
	}

	/// Step 6: constants
	std::ifstream tmax_file;
	tmax_file.open( (this->DEFAULT_FOLDER+"/time_max").c_str() );
	if (! tmax_file.is_open()) {
		perror("ERROR:  cannot open time max: ");
		return false;	
	} else {
		getline(tmax_file, v);
		time_max = atof(v.c_str());
		tmax_file.close();

		if (dump)
			printf(" * time_max set to %f\n", time_max);
	}


	///  Step 7: system
	std::vector<param_t> vettore_ode;
	std::vector<param_t> vettore_jac;
	std::vector<unsigned int> vettore_offset;
	std::vector<unsigned int> vettore_jac_offset;
	unsigned int last_jump =  0;
	unsigned int last_jump_jac =  0;


	if (dump)
			printf(" * Conversion constant (POST): %e\n", this->c_moles_molec);

	// per ogni specie: dobbiamo costruire l'equazione differenziale
	for (unsigned int s =0; s<species; s++ ) {

#ifdef VERBOSE
		printf(" dX[%d]/dt = ", s);
#endif	

		// se la specie è in feed (buffer), allora non cambia mai concentrazione
		if (feed_vector[s]!=0) {
#ifdef VERBOSE
			printf("0 (feed)\n");
#endif 
			vettore_offset.push_back( vettore_ode.size() );
			continue;
		}


		// analizzo quali specie influenzino la specie
		for (unsigned int r=0; r<reactions; r++ ) {
			 int ordine_reazione = 0;
			unsigned int tot_fatt = 1;
			for (unsigned int s2=0; s2<species; s2++ ) {
				ordine_reazione += left_matrix[r*species+s2];	
				tot_fatt *= fattoriale( left_matrix[r*species+s2] );
			}
			// math.pow(self.c_moles_molec, ordine_reazione-1) / tot_fatt
			// param_t fattore = potenza( this->c_moles_molec, ordine_reazione-1 );
			param_t fattore = pow( this->c_moles_molec, ordine_reazione-1 );

			fattore /= tot_fatt;


			if (!StochasticModel) fattore = 1;		// chiedere a Dario

			// param_t fattore = std::powf( this->c_moles_molec, ordine_reazione-1) / (float)tot_fatt;

			//printf(" %f\n", fattore);
			if ( left_matrix[r*species+s] != 0 ) {
				if ( var_matrix[r*species+s] < 0 ) {
#ifdef VERBOSE
					printf("%d*p[%d]*%f", var_matrix[r*species+s],r,fattore);
#endif
					vettore_ode.push_back( var_matrix[r*species+s] );		// mult
					vettore_ode.push_back( r );								// stoc const
					vettore_ode.push_back( fattore );						// fattore conversione
					if (fattore==0) {
						printf("WARNING: detected conversion factor zero\n");
						// system("pause");
					}
					vettore_ode.push_back( 0 );								// N variabili
					last_jump = vettore_ode.size()-1;
				
			

					for (unsigned int s2=0; s2<species; s2++) {
						// if ( left_matrix[r*species+s2] != 0 ) {
						if ( left_matrix[r*species+s2] != 0 ) {

	#ifdef VERBOSE
							printf("*X[%d]",s2);
	#endif
							vettore_ode[last_jump]+=left_matrix[r*species+s2];
							for (unsigned espo=0; espo<left_matrix[r*species+s2]; espo++) 
								vettore_ode.push_back( s2 );
						} 					 
					}
				}

			} // end if

#ifdef VERBOSE
			printf(" ");
#endif

			if ( right_matrix[r*species+s] != 0 ) {
				if ( var_matrix[r*species+s] > 0 ) {

#ifdef VERBOSE
					printf("%d*p[%d]*%f", var_matrix[r*species+s],r,fattore);
#endif
					vettore_ode.push_back( var_matrix[r*species+s] );		// mult
					vettore_ode.push_back( r );								// stoc const
					vettore_ode.push_back( fattore );						// fattore conversione
					if (fattore==0) {
						printf("WARNING: detected conversion factor zero\n");
						// system("pause");
					}
					vettore_ode.push_back( 0 );								// N variabili
					last_jump = vettore_ode.size()-1;
				
			

					for (unsigned int s2=0; s2<species; s2++) {
						if ( left_matrix[r*species+s2] != 0 ) {

	#ifdef VERBOSE
							printf("*X[%d]",s2);
	#endif

							vettore_ode[last_jump]+=left_matrix[r*species+s2];
							for (unsigned espo=0; espo<left_matrix[r*species+s2]; espo++) 
								vettore_ode.push_back( s2 );
						} 					
					}

				}
			} // end if

#ifdef VERBOSE
			printf(" ");
#endif 
			
		} // end for reactions

		vettore_offset.push_back( vettore_ode.size() );

#ifdef VERBOSE
		printf("\n");
#endif


	} // end for species

	// save technical stuff 
	this->ODE_lun = vettore_ode.size();

	// If the encoding of the ODEs is larger than the allocated constant memory, we quit.
	// In the future, we will switch to global memory.
	if (this->ODE_lun> MAX_ODE_LUN ) {
		if (use_constant) {
			printf("ERROR: encoding of ODEs is greater than the maximum available chunk of constant memory, quitting.\n");
			exit(ERROR_INSUFF_CONSTANT_MEMORY_ODE);
		}
	}

	this->ODE_offset = (unsigned int*) malloc ( sizeof(unsigned int) * this->species );
	for (unsigned int s=0; s<this->species; s++) {
		this->ODE_offset[s] = vettore_offset[s];
	}

	this->ODE = (param_t*) malloc ( sizeof(param_t) * vettore_ode.size() );
	for (unsigned int i=0; i<vettore_ode.size(); i++) {
		this->ODE[i] = vettore_ode[i];
	}
	
#ifdef VERBOSE
	
#endif

#ifdef FULL_DUMP
	if (dump)
		this->dump_odes();
#endif

	//// JACOBIANO

	// E' una matrice specie X specie.
	//
	// Per ogni coppia (specie1,specie2) devo vedere se nel vettore compresso dell'equazione differenziale della specie 1
	// si trova, nei campi delle specie coinvolte, la specie2. Si può procedere a salti, perché la struttura della matrice
	// compressa è nota a priori. Nel caso la specie sia presente, torno indietro e copio i dati nella nuova matrice 
	// compressa dello jacobiano, assolutamente identica tranne per il fatto che specie1 compare n-1 volte.
	// printf ("\nJACOBIANO: \n");

	unsigned int abs_pos=0;
	
	for ( unsigned int s1=0; s1<species; s1++ ) {	

		// printf(" Deriving species %d\n", s1);
	
		unsigned int pos=0;
		unsigned int i;
		
		for ( unsigned int equazione=0; equazione<species; equazione++ ) {
			
#ifdef VERBOSE
			printf("d[%d][%d] = ", equazione, s1);
#endif

			while( pos<this->ODE_offset[equazione] ) {			

				unsigned int species_hits = 0;

				pos+=3;				
				abs_pos+=3;
				for (i=0; i<(int)ODE[pos]; i++) {				
					
					// trovata la specie: calcoliamo la derivata
					if ( (int)ODE[pos+1+i]==s1 ) {						
						species_hits ++ ;					
					}
				}

				// la specie chimica era presente nelle ODE?
				if ( species_hits>0 ) {
				
					int from = pos-3;

					// salvo i pre-dati
					for (unsigned int i=from; i<from+3; i++)  {
						// vettore_jac.push_back((int)ODE[i]);
						vettore_jac.push_back(ODE[i]);
						if ((i==2) && (vettore_jac.back() == 0) ) {
							printf("ARGH!");
							// system("pause");
						}
					}
					
#ifdef VERBOSE
					if ( (int)ODE[from]>0)	printf("+");					
					printf("%d", (int)ODE[from]);
					printf(" * k[%d] ", (int)ODE[from+1]);
					printf(" * %f ", ODE[from+2]); 
#endif
										
					if (species_hits>0) vettore_jac.push_back((int)ODE[pos]-1);
					else vettore_jac.push_back(0);
					
				} // end if 		

				bool trovato = false;
				unsigned int position_multipier = vettore_jac.size()-2;

				if ( species_hits>0 ) {

					for (unsigned int i=pos+1; i<pos+ODE[pos]+1; i++) {

						if (vettore_jac[ position_multipier ]==0) {
								printf("MMM!\n");
								// system("pause");
						}
					
						if ( (int)ODE[i]==s1 && !trovato ) {
							trovato = true;
							// vettore_jac[abs_pos-1] *= species_hits;
							vettore_jac[ position_multipier ] *= species_hits;
							if (vettore_jac[ position_multipier ]==0) {
								printf("ERROR!\n");
								// system("pause");
							}
							continue;
						}
						
						if ( (int)ODE[i]==s1 && trovato  )  {
							vettore_jac.push_back((int)ODE[i]);
#ifdef VERBOSE
							printf(" * X[%d] ", (int)ODE[i]);
#endif
							continue;
						}
						
						vettore_jac.push_back((int)ODE[i]);
#ifdef VERBOSE
						printf(" * X[%d] ", (int)ODE[i]);
#endif
					}

				}	 // end if species hits

				/* if (species_hits==0) {
					//vettore_jac.push_back((int)ODE[i]);
					// printf("%d ?!\n", pos);
					// pos-=2;
				} else 		 {	*/					
					pos+=i+1;
					abs_pos+=i+1;
				// }

			}

			vettore_jac_offset.push_back( vettore_jac.size() );

#ifdef VERBOSE
			printf("\n");
#endif
		
		}

	}

	// save technical stuff 
	this->JAC_lun = vettore_jac.size();

	if (this->JAC_lun> MAX_JAC_LUN ) {
		if (use_constant) {
			printf("ERROR: encoding of Jacobian matrix is greater than the maximum available chunk of constant memory, quitting.\n");
			exit(ERROR_INSUFF_CONSTANT_MEMORY_JAC);
		}
	}

	this->JAC_offset = (unsigned int*) malloc ( sizeof(unsigned int) * species * species );
	for (unsigned int s=0; s<species*species; s++) {
		this->JAC_offset[s] = vettore_jac_offset[s];
	}

	this->JAC = (param_t*) malloc ( sizeof(param_t) * vettore_jac.size() );
	for (unsigned int i=0; i<vettore_jac.size(); i++) {
		this->JAC[i] = vettore_jac[i];
	}

#ifdef FULL_DUMP
	if (dump)
		this->dump_jac();
#endif

	/// specie da campionare
	std::ifstream csvector_file;
	csvector_file.open( (this->DEFAULT_FOLDER+"/cs_vector").c_str() );
	if (! csvector_file.is_open()) {
		if (dump) perror("WARNING: cannot open cs_vector: ");
		if (dump) printf("Will proceed anyway by sampling all the species\n");
		for (unsigned int i=0; i<this->species; i++ ) 
			this->species_to_sample.push_back(i);
		// return false;	
	} else {
		while ( csvector_file.good() ) {
			getline(csvector_file, v);
			if (! v.empty()) 
				this->species_to_sample.push_back ( atoi( v.c_str() ) );			
		}		
		csvector_file.close();
	}

	if (dump)
		printf(" * Will sample %d species\n", this->species_to_sample.size());


	/// istanti di campionamento
	std::ifstream tvector_file;
	tvector_file.open( (this->DEFAULT_FOLDER+"/t_vector").c_str() );
	if (! tvector_file.is_open()) {
		perror("ERROR: cannot open t_vector: ");
		return false;	
	} else {
		while ( tvector_file.good() ) {
			getline(tvector_file, v);
			if (v.empty()) continue;
			this->time_instants.push_back ( atof( v.c_str() ) );			
		}		
		tvector_file.close();
	}

	if (dump)
		printf(" * Will store %d samples \n", this->time_instants.size());


	
	/// errore assoluto
	std::ifstream atol_file;
	atol_file.open( (this->DEFAULT_FOLDER+"/atol_vector").c_str() );
	if (! atol_file.is_open()) {
		if (dump) perror("WARNING: cannot open atol_vector, using default (1E-6 for each species): ");
		for (unsigned s=0; s<species; s++) 
			this->atol.push_back( (double)1e-6 ); 
		// return false;	
	} else {
		while ( atol_file.good() ) {
			getline(atol_file, v);
			if (! v.empty()) 
				this->atol.push_back ( atof( v.c_str() ) );			
		}		
		atol_file.close();
		if (this->atol.size()!=species) {
			if (dump)
				printf("ERROR: number of absolute tolerances (%d) is different from the number of chemical species (%d).\n", this->atol.size(), species);
			return false;
		}
	}

	if (dump)
		printf(" * ATOL vector created\n");


	/// Step ?: relative error
	std::ifstream rtol_file;
	rtol_file.open( (this->DEFAULT_FOLDER+"/rtol").c_str() );
	if (! rtol_file.is_open()) {
		if (dump) perror("WARNING:  cannot open rtol file: ");
		// return false;	
	} else {
		getline(rtol_file, v);
		this->rtol = atof(v.c_str());
		rtol_file.close();
				
	}

	if (dump)
		printf(" * RTOL set to %e\n", this->rtol);

	// ogni vettore è associato alla sua condizione sperimentale (aka sciame)
	this->thread2experiment = (char*) malloc ( sizeof(char)*threads );

	if (just_fit) {
		if (traditional_fitness)  {

			/// Step ??: apertura serie temporali (if any)
			/// la serie è lunga T_vector
			/// dobbiamo leggere le informazioni su il numero di esperimenti, e ripetizioni
			std::ifstream ts_rep_file;
			ts_rep_file.open( (this->DEFAULT_FOLDER+"/ts_rep").c_str() );
			if (! ts_rep_file.is_open() ) {
				if (dump) perror("WARNING: cannot open ts_rep file: ");
			} else {
				getline(ts_rep_file, v);
				this->repetitions = atoi(v.c_str());	
				ts_rep_file.close();
			}

			std::ifstream ts_numtgt_file;
			ts_numtgt_file.open( (this->DEFAULT_FOLDER+"/ts_numtgt").c_str() );
			if (! ts_numtgt_file.is_open() ) {
				if (dump) perror("WARNING: cannot open ts_numtgt file: ");
			} else {
				getline(ts_numtgt_file, v);
				this->target_quantities = atoi(v.c_str());	
				ts_numtgt_file.close();
			}

	
	
			std::ifstream tts_file;
			tts_file.open( (this->DEFAULT_FOLDER+"/tts_vector").c_str() );
			if (! tts_file.is_open() ) {
				if (dump) perror("WARNING: cannot open tts_vector file: ");
			} else {
				char pp=0;
				unsigned int last=0;
				while( tts_file.good() ) {
					getline(tts_file, v);
					if (v.size()<1) continue;
					for (unsigned int kk=last; kk<atoi(v.c_str()); kk++) {
						this->thread2experiment[kk]=pp;
					}
					last =  atoi(v.c_str());
					pp++;
					// this->experiments = atoi(v.c_str());	
				}		
		
				this->experiments = pp;

				if (last!=this->threads) {
					printf("ERROR! cannot assign threads to initial conditions (check tts_vector file): aborting\n");
					system("pause");
					exit(-10);
				}

				if (dump) {
					printf(" * Threads assigned to conditions:\n");
					for (unsigned kk=0; kk<threads; kk++) {
						printf("%d\t", this->thread2experiment[kk]);
					}
					printf("\n");
				}
				tts_file.close();
			}

	


			if (dump)
				printf(" * Experiments: %d, repetitions: %d target quantities: %d\n", this->experiments, this->repetitions, this->target_quantities);


			/// sappiamo le righe (len time_instants), sappiamo le colonne (exp * rep * len cs_vector) 
			this->global_time_series = (double*) malloc ( sizeof(double) * 
				this->experiments * 
					this->repetitions * 
						this->target_quantities * 
							this->time_instants.size() );


			std::ifstream ts_matrix_file;
			ts_matrix_file.open( (this->DEFAULT_FOLDER+"/ts_matrix").c_str() );
			if (! ts_matrix_file.is_open() ) {
				if (dump) perror("WARNING: cannot open ts file (fitness unavailable)");
				if (just_fit) {
					exit(ERROR_TIMESERIES_NOT_SPECIFIED);
				}
			} else {
		
				unsigned int riga =0;
				unsigned int colonne = this->experiments * this->repetitions * this ->target_quantities;

				while( ts_matrix_file.good() ) {

					getline(ts_matrix_file, v);
					a.clear();
					tokenize(v,a);
					if (a.size()<1) continue;
					for (unsigned int kk=1; kk<colonne+1; kk++ ) {
						this->global_time_series[ riga*colonne + kk-1 ] = atof(a[kk].c_str());
					}
					riga++;

				}

				if (dump) printf(" * Time series loaded and assigned to threads\n");
				ts_matrix_file.close();

			}

		} // end traditional fitness

	} // end just fitness

	/// Step 6b: constants
	std::ifstream maxs_file;
	maxs_file.open( (this->DEFAULT_FOLDER+"/max_steps").c_str() );
	if (! maxs_file.is_open()) {
		if (dump) perror("WARNING:  cannot open max number of steps: ");
		// return false;	
	} else {
		getline(maxs_file, v);
		max_steps = atof(v.c_str());
		maxs_file.close();

		if (dump)
			printf(" * max integration steps set to %d\n", max_steps);

		this->max_steps = max_steps;
	}

	return true;
}
