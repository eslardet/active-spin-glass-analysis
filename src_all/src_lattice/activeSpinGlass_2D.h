// activeSpinGlass_2D.h
// ====================
// Declaration of all global variables necessary to run activeSpinGlass_2D.cpp
// Created by Thibault Bertrand on 2022-04-19
// Last update by TB 2022-04-19

#ifndef activeSpinGlass_2D_h
#define activeSpinGlass_2D_h

std::fstream inputFile,initposFile;
std::fstream logFile,posFile,posExactFile,forceFile,couplingFile;

// Input parameters
int nPart;
unsigned int seed;
double Rp;
double rotD;
bool savePos,saveForce,saveCoupling;
char initMode,couplingMode;
double dT,DT,DTex,eqT,simulT,startT;
int Nsimul,Neq,Nskip,Nskipexact;
double xmin,xmax;
double ymin,ymax;
double Lx,Ly;

// Coupling constant parameters
double K0; // mode 'C': K0: Coupling constant
double KAA,KAB,KBB; // mode 'T': KAA: Coupling constant for A-A interactions 
                    //           KAB: Coupling constant for A-B interactions 
                    //           KBB: Coupling constant for B-B interactions
double KAVG,STDK; // mode 'G', 'F', 'A': KAVG: Average coupling constant
                  //                     STDK: Standard deviation of coupling constant

///////////////
// saveFrame //
///////////////
// Saves to file a frame
inline void saveFrame(std::vector<double> pp, double tt, std::fstream& File)
{
    File << tt << std::endl;
    for(int i=0 ; i<nPart ; i++)
    {
        File << pp[i] << std::endl;
    }
}

////////////////
// saveHeader //
////////////////
// Saves the header for the collision file
inline void saveHeader(std::fstream& File)
{
	File << nPart << std::endl;
    File << seed << std::endl;
    File << rotD << std::endl;
    File << Rp << std::endl;
    File << xmin << '\t' << xmax << std::endl;
    File << ymin << '\t' << ymax << std::endl;
}

#endif