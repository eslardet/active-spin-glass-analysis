// abdSG2D_functions.h
// ====================
// Declaration of all functions necessary to run activeSpinGlass_2D.cpp
// Created by Thibault Bertrand on 2022-04-19
// Last update by EL 2023-10-09

#ifndef abdSG2Dfunctions_h
#define abdSG2Dfunctions_h

// #include <boost/iostreams/filtering_streambuf.hpp>
// #include <boost/iostreams/copy.hpp>
// #include <boost/iostreams/filter/gzip.hpp>

///////////////////////////////
// Define external variables //
///////////////////////////////
extern std::fstream initposFile,logFile,couplingFile,posExactFile,seedFile;

extern int nPart;
extern unsigned int seed;
extern double phi,noise,vp;
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
extern double KAVG,STDK; // mode 'G', 'A': KAVG: Average coupling constant
                         //                STDK: Standard deviation of coupling constant
// extern double alpha; // Fraction of particles with +K0 coupling for mode F

/////////////////////////////
// Define global variables //
/////////////////////////////
const double PI = 3.141592653589793238463;
static double L; // Box size
static int nCell,mx,my; // number of cells on each direction

// Interaction radii
static double beta,betasq;
extern double Rp;
static double rp,rpsq;

// Define variables needed for the SRK
static std::vector<double> X,Fx;
static std::vector<double> Y,Fy;
static std::vector<double> P,Fp;

// Define the coupling constant array
static std::vector<double> K;

// Neighbor list variables
static double rl,rc; // radius of the neighbor list
static double rlsq;
static std::vector<int> nl; // nl contains the neighbor list
static std::vector<int> cl; // cl contains the number of neighbors for each particle
static std::vector<double> xl,yl; // contains positions of particles at last neighbor list build
static std::vector<double> sqDisp;

// Cell linked list variables
static double lx = rl; // width of the cell
static double ly = rl; // height of the cell
const int nNeighbor = 5;

static std::vector<int> head; // head holds the index of first index of 
static std::vector<int> lscl; // lscl vector implementation of linked-cell list 
static std::vector< std::vector<int> > mp; // contains list of neighboring cells for each cell

// FIRE/IC parameters
static const double fTOL=1.e-10;
static const double pTOL=1.e-6;
static const double eps=1.e-3;
static const double phi_drop = 0.6;

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

inline int lCellIndex(int ii, int jj)
{
    return ( ii+mx )%mx + ( ( jj+my )%my )*mx ;
}

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
void buildMap(void);
bool checkNL(std::vector<double>,std::vector<double>);
void updateNL(std::vector<double>,std::vector<double>);
void SRK2(std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&);
void EM(std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&);
std::vector<float> force(std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>&,std::vector<double>&,std::vector<double>&);
void activeBrownianDynamics(std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,double&);
double fHarmonic(std::vector<double>&,std::vector<double>&);
void dfHarmonic(std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&);
void fire(std::vector<double> &px, std::vector<double> &py, const double dT0, const double ftol, 
		  double &fret, double func(std::vector<double> &,std::vector<double> &), 
		  void dfunc(std::vector<double> &, std::vector<double> &, std::vector<double> &, std::vector<double> &));
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
inline void saveCouplings(std::vector<double> k, std::fstream& File) 
{
    for (const auto &e : k) File << e << std::endl;
}

inline unsigned long long getIndex(int i, int j)
{
    unsigned long long i_long = i, j_long = j, N_long = nPart;
    if (i>j)
    {
        return j_long*(N_long-1) - j_long*(j_long-1)/2 + i_long - j_long - 1;
    }
    else
    {
        return i_long*(N_long-1) - i_long*(i_long-1)/2 + j_long - i_long - 1;
    }
}


#endif