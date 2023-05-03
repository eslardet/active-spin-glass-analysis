// abdSG2D_functions.h
// ====================
// Declaration of all functions necessary to run activeSpinGlass_2D.cpp
// Created by Thibault Bertrand on 2022-04-19
// Last update by TB 2022-04-19

#ifndef abdSG2Dfunctions_h
#define abdSG2Dfunctions_h

// #include <boost/iostreams/filtering_streambuf.hpp>
// #include <boost/iostreams/copy.hpp>
// #include <boost/iostreams/filter/gzip.hpp>

///////////////////////////////
// Define external variables //
///////////////////////////////
extern std::fstream initposFile,logFile,couplingFile, posExactFile;

extern int nPart;
extern unsigned int seed;
extern double phi,noise,vp;
extern char Rp;
extern bool saveCoupling,savePos,saveInitPos;
extern char initMode,couplingMode,intMethod;
extern double dT,DT,DTex,eqT,simulT,startT;
extern int Nsimul,Neq,Nskip,Nskipexact;
extern double Lx,xmin,xmax;
extern double Ly,ymin,ymax;
extern double xTy;

extern double K0; // mode 'C': K0: Coupling constant
extern double KAA,KAB,KBB; // mode 'T': KAA: Coupling constant for A-A interactions 
                           //           KAB: Coupling constant for A-B interactions 
                           //           KBB: Coupling constant for B-B interactions
extern double KAVG,STDK; // mode 'G', 'F', 'A': KAVG: Average coupling constant
                         //                     STDK: Standard deviation of coupling constant

/////////////////////////////
// Define global variables //
/////////////////////////////
const double PI = 3.141592653589793238463;
static double L; // Box size

// Define variables needed for the SRK
static std::vector<double> X,Fx;
static std::vector<double> Y,Fy;
static std::vector<double> P,Fp;

// Define the coupling constant array
static std::vector< std::vector<double> > K;

////////////////////////
// Inline Definitions //
////////////////////////
template<class T>
inline const T sgn(const T &a)
	{return ( 0 < a ) - ( a < 0 ) ; }

template<class T>
inline const T SQR(const T a) {return a*a;}

template<class T>
inline const T MAX(const T &a, const T &b)
        {return b > a ? (b) : (a);}

inline float MAX(const double &a, const float &b)
        {return b > a ? (b) : float(a);}

inline float MAX(const float &a, const double &b)
        {return b > a ? float(b) : (a);}

template<class T>
inline const T MIN(const T &a, const T &b)
        {return b < a ? (b) : (a);}

inline float MIN(const double &a, const float &b)
        {return b < a ? (b) : float(a);}

inline float MIN(const float &a, const double &b)
        {return b < a ? float(b) : (a);}

////////////////////////////
// Functions Declarations //
////////////////////////////
std::string currentDateTime(void);
void checkParameters(void);
void initialize(std::vector<double>&,std::vector<double>&,std::vector<double>&);
void initialConditionsRandom(std::vector<double>&,std::vector<double>&,std::vector<double>&);
void initialConditionsSim(std::vector<double>&,std::vector<double>&,std::vector<double>&);
void allocateSRKmem(void);
bool checkOverlap(std::vector<double>,std::vector<double>);
double volumeFraction(void);
void SRK2(std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&);
void EM(std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&);
std::vector<float> force(std::vector<double>,std::vector<double>&,std::vector<double>&,std::vector<double>&);
void activeBrownianDynamics(std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,double&);
void finalize(void);

///////////////////
// saveInitFrame //
///////////////////
// Saves to file a frame
inline void saveInitFrame(std::vector<double> x, std::vector<double> y, std::vector<double> p, std::fstream& File) 
{

    File << nPart << std::endl;
    File << phi << std::endl;
    File << seed << std::endl;
    File << Rp << std::endl; 
    File << xmin << '\t' << xmax << std::endl;
    File << ymin << '\t' << ymax << std::endl;
    File << 0 << std::endl;

    for(int i=0 ; i<nPart ; i++)
    {
        File << x[i] << '\t' << y[i] << '\t' << p[i] << std::endl;
    }
}


///////////////////
// saveCouplings //
///////////////////
// Saves to file the coupling constants
inline void saveCouplings(std::vector< std::vector<double> > k, std::fstream& File) 
{
    
    for(int i=0 ; i<nPart ; i++)
    {
        for(int j=i+1 ; j<nPart ; j++){
            File << k[i][j] << std::endl; 
        }
    }
}



#endif