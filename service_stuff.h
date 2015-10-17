/*
service_stuff.h: service functions for dump and conversions.
See file COPYING for copyright and licensing information.
*/


#ifndef __SERVICE_STUFF__
#define __SERVICE_STUFF__

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>


template <typename T>
std::string NumberToString ( T Number )
{
	std::stringstream ss;
	ss << Number;
	return ss.str();
}

std::vector<std::string> files_vector;
// std::vector<std::ofstream*> files_descriptor_vector;

void create_dump_files(unsigned int quanti, std::string pa, unsigned int gpu) {

	for (unsigned int i=0; i<quanti; i++) {

		std::string p(pa);
		p.append( "_" );
		p.append( NumberToString(i) );
		p.append( "_GPU" );
		p.append( NumberToString(gpu) );
		files_vector.push_back(p);
		// std::cout << files_vector.back() << std::endl;
		std::ofstream* myfile = new std::ofstream(p.c_str());

		if (! myfile->is_open() ) {
			perror("ERROR: cannot open output file. Directory does not exist? ");
			exit(-1);
		}

		myfile->close();
						
	}

}

void write_dump_files( std::vector< std::vector<double>* > v ) {

	// v è un vettore dei vettori timestamp + stato corrente	

	for (unsigned int i=0; i<v.size(); i++) {		
		std::ofstream* myfile =  new std::ofstream();
		myfile->open( files_vector[i].c_str() , std::fstream::app);

		for (unsigned int j=0; j<v[i]->size(); j++) {			
			*myfile << v[i]->at(j) << "\t"; 
			// printf("%d, %d, %d\n", i,j, );
		}

		*myfile << "\n";
		myfile->close();
	}
	
}

void close_dump_files(void) {

	/*
	for (unsigned int i=0; i<files_descriptor_vector.size(); i++) {
		*files_descriptor_vector[i] << "\n";
		files_descriptor_vector[i]->close();
	}
	*/
}


#endif
