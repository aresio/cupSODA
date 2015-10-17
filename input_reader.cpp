/*
input_reader.cpp: parser for the input files.
See file COPYING for copyright and licensing information.
*/


#include "input_reader.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <istream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>


std::vector<int>* split_string_integers(std::string& value, char separator, unsigned int* totale)
{
	int LENG = value.size();
    std::vector<int>* result = new std::vector<int>;
	std::string::size_type pos;
	*totale = 0;
	while (pos = value.find_first_of(separator) != -1) 
    {
        // std::string::size_type pos = value.find_first_of(separator);
        // value = value.substr(0, pos) + "%" + value.substr(pos+1);
		// value = value.substr(0, pos);
		int n = atoi((const char*)value.substr(0, pos).c_str());
		result->push_back(n);
		*totale += n;
		value = value.substr(pos+1);
    }
	int n = atoi((const char*)value.substr(0).c_str());
	result->push_back(n);
	*totale += n;
    return result;
}


bool InputReader::OpenInputFile(std::string p) {

	
	std::vector<std::vector<int>*>   left_matrix;
	std::vector<std::vector<int>*>   right_matrix;
	unsigned int rows = 0;
	unsigned int cols = 0;


	/* MATRICE SINISTRA */
	std::ifstream left_file((p+"/left_side").c_str());

	/* is there the left matrix? */
	if (!left_file.is_open()) {
		perror("ERROR: cannot open input file");
		printf("%s\n", (p+"/left_side").c_str());
		return false;
	};

	/* read matrix */
	std::vector<int> totals_left;
	while (left_file.good()) {
		std::string riga ;
		std::getline(left_file, riga);		
		unsigned int total_left = 0;
		std::vector<int>* ret = split_string_integers(riga, '\t', &total_left);
		totals_left.push_back(total_left);		
		left_matrix.push_back(ret);
	};

	left_file.close();

	rows = left_matrix.size();
	cols = left_matrix.back()->size();
	
	/* Initialize compressed ODE matrix */
	for ( unsigned int i=0; i<cols; i++ ) {
		this->comp_ODE.push_back( new std::vector<int> );
	}

	/* 
		RE-parsing matrice sinistra 
	*/
	for (unsigned int r=0; r<rows; r++) {
		for (unsigned int c=0; c<cols; c++) {
			switch ( left_matrix[r]->at(c) ) {

			case 0:
				break;

			case 1:

				this->comp_ODE[ c ]->push_back( totals_left[ r ] +1 );
				this->comp_ODE[ c ]->push_back( -(r+1) );

				for ( unsigned int cl=0; cl<cols; cl++ ) {
					switch ( left_matrix[r]->at(cl) ) {

					case 0: 
						break;
					case 1:
						this->comp_ODE[ c ]->push_back( cl );
						break;
					case 2:
						this->comp_ODE[ c ]->push_back( cl );
						this->comp_ODE[ c ]->push_back( cl );
						break;
					default: 
						perror("ERROR: unsupported case");
						break;
					}
				}

				break;

			default:
				perror("ERROR: unsupported case");
				break;
			}
		}
	}

	
	/* 
		MATRICE DESTRA 
		La matrice destra genera direttamente i prodotti delle reazioni, 
		compaiono tutti con una costante positiva e possono essere aggiunti on-fly.	
	*/
	std::ifstream right_file((p+"/right_side").c_str());		

	/* is there the right matrix? */
	if (!right_file.is_open()) {
		perror("ERROR: cannot open input file");
		printf("%s\n", (p+"/right_side").c_str());
		return false;
	};

	/* read matrix */	
	unsigned int total_right = 0;
	while (right_file.good()) {
		std::string riga ;
		std::getline(right_file, riga);		
		std::vector<int>* ret = split_string_integers(riga, '\t', &total_right);
		right_matrix.push_back(ret);

		/* 
			ret is a single row of the stoichometric matrix.
			we parse it and put the corresponding value into the compressed matrix.
		*/
		for ( unsigned c=0; c<cols; c++ ) {
			
			switch ( ret->at(c) ) {

				case 0: 
					break;
				case 1: 
					/* 
						We have counted how many stoichiometric values are different from 0 
						in the left matrix. We use this information to fill the first 
						value of the compressed matrix. We make +1 because we also consider
						the (unknown) kinetic constant.
					*/
					this->comp_ODE[ c ]->push_back( totals_left[ right_matrix.size()-1 ] +1 );
					this->comp_ODE[ c ]->push_back( right_matrix.size() );
					for (unsigned int cl=0; cl<cols; cl++) {
						switch ( left_matrix[ right_matrix.size()-1 ]->at(cl) ) {
							case 0: 
								break;
							case 1:
								this->comp_ODE[ c ]->push_back( cl );
								break;
							case 2:
								this->comp_ODE[ c ]->push_back( cl );
								this->comp_ODE[ c ]->push_back( cl );
								break;
							default:
								perror("ERROR: unsupported case");
								break;
						} // end switch
					} // end for
					break;
				case 2:
					perror("ERROR: unsupported case");
					break;
				default:
					perror("ERROR: unsupported case");
					break;

			} // end switch		
		} // end for		
	}; // end while

	/* Terminators */
	for ( unsigned int i=0; i<cols; i++) 
		this->comp_ODE[ i ]->push_back( 0 );
	
	right_file.close();


	return true;
}



unsigned int InputReader::GetCompODEMaxSize() {
	unsigned int mass = 0;
	for (unsigned int i=0; i<this->comp_ODE.size(); i++) {
		if ( this->comp_ODE[i]->size() > mass )
			mass = this->comp_ODE[i]->size();
	}
	return mass;
}

unsigned int InputReader::GetCompODESize() {
	unsigned int mass = 0;
	for (unsigned int i=0; i<this->comp_ODE.size(); i++) {
		mass += this->comp_ODE[i]->size();
	}
	return mass;
}
