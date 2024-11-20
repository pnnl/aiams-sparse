  
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <cassert>
#include <unistd.h>

//using namespace std;

void readweights();
void readinput();

void parseCommandLine(int argc, char** const argv);

//#define BALANCE 30 //BALANCE LAYER 0 FOR EVERY LAYER COMMENT OUT FOR TURN OFF
//#define OUTOFCORE //COMMENT THIS OUT IF YOU HAVE ENOUGH MEMORY
//#define OVERLAP //WORKS ONLY WHEN OUTOFCORE IS ENABLED


#ifndef INDPREC
#define INDPREC int
#endif
#ifndef VALPREC
#define VALPREC float
#endif
#ifndef FEATPREC
#define FEATPREC float
#endif
