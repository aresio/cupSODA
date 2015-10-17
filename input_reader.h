/*
input_reader.h: parser for the input files.
See file COPYING for copyright and licensing information.
*/


#ifndef __INPUTREADER__
#define __INPUTREADER__

#include <string>
#include <vector>

class InputReader {

public:
	unsigned int GetNumberReactions() { return this->Reactions; };
	unsigned int GetNumberSpecies() { return this->Species; };
	unsigned int GetCompODEMaxSize();
	unsigned int GetCompODESize();

	bool OpenInputFile(std::string);

	void CreateODEs();

// private:
	unsigned int Reactions;
	unsigned int Species;

	std::vector< std::vector<int>* > comp_ODE;
	

};


#endif