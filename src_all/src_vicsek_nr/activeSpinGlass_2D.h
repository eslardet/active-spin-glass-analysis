// activeSpinGlass_2D.h
// ====================
// Declaration of all global variables necessary to run activeSpinGlass_2D.cpp
// Created by Thibault Bertrand on 2022-04-19
// Last update by TB 2022-04-19

#ifndef activeSpinGlass_2D_h
#define activeSpinGlass_2D_h

std::fstream inputFile,initposFile;
std::fstream logFile,posFile,posExactFile,forceFile,couplingFile,seedFile;

// Input parameters
int nPart;
unsigned int seed;
double phi,noise,vp;
double Rp;
bool savePos,saveForce,saveCoupling,saveInitPos;
char initMode,couplingMode,intMethod;
double dT,DT,DTex,eqT,simulT,startT;
int Nsimul,Neq,Nskip,Nskipexact;
double xmin,xmax;
double ymin,ymax;
double xTy,Lx,Ly;

// Coupling constant parameters
double K0; // mode 'C': K0: Coupling constant
double KAB,KBA,KBC,KCB,KAC,KCA;
double KAVG,STDK; // mode 'G', 'A': KAVG: Average coupling constant
                  //                STDK: Standard deviation of coupling constant
// double alpha; // Fraction of particles with +K0 coupling for mode F

///////////////
// saveFrame //
///////////////
// Saves to file a frame
inline void saveFrame(std::vector<double> xx, std::vector<double> yy, std::vector<double> pp, double tt, std::fstream& File)
{
    File << tt << std::endl;
    for(int i=0 ; i<nPart ; i++)
    {
        File << xx[i] << '\t' << yy[i] << '\t' << pp[i] << std::endl;
    }
}

////////////////
// saveHeader //
////////////////
// Saves the header for the collision file
inline void saveHeader(std::fstream& File)
{
	File << nPart << std::endl;
    File << phi << std::endl;
    File << seed << std::endl;
    File << Rp << std::endl;
    File << xmin << '\t' << xmax << std::endl;
    File << ymin << '\t' << ymax << std::endl;
}

#endif
