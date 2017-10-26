#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char * argv[]) {
	unsigned int N;

	N = (unsigned int) atoi(argv[1]);

	std::ofstream f;
	std::string filename = std::to_string(N) + ".txt";
	f.open (filename);
	f << "Writing this to a file.\n";
	f.close();
	return 0;
}