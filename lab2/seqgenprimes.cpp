#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

int main(int argc, char * argv[]) {
	unsigned int N;

	N = (unsigned int) atoi(argv[1]);

	// create array of bools; true is prime
	// we will set non primes to false
	bool* primes = new bool[N+1]; 

	for(int j=2; j <= N; j++) {
		primes[j] = true;
	}

	for(int p=2; p <= floor((N+1)/2); p++) {
		if(primes[p] == true) {
			for(int i=p*2; i <= N; i+=p) {
				primes[i] = false;
			}
		}
	}

	//print output
	std::ofstream f;
	std::string filename = std::to_string(N) + ".txt";
	f.open (filename);
	//skip 0 and 1 are not primes
	for(int p=2; p <= N; p++) {
		if(primes[p]) {
			f << std::to_string(p) << " ";
		}
	}
	// f << "Writing this to a file.\n";
	f.close();
	return 0;
}