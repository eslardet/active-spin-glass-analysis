// abdSG2D_functions.cpp
// =====================
// Definition of all functions necessary to run activeSpinGlassL_2D.cpp
// Created by Thibault Bertrand on 2022-04-19
// Last update by TB 2022-04-19

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <limits>
#include <random>
#include <array>
#include <ctime>
#include "math.h"
#include "abdSG2D_functions.h"

using namespace std;

///////////////////////
// Define functions  //
///////////////////////

//////////////////////////////
// Random number generation //
//////////////////////////////
random_device rd;
mt19937 rnd_gen;

uniform_real_distribution<double> uniDist(0.0,1.0);
normal_distribution<double> normDist(0.0,1.0);
uniform_real_distribution<double> whiteNoise(-1.0,1.0);

/////////////////////
// currentDateTime //
/////////////////////
// Returns the current date and time
string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://www.cplusplus.com/reference/clibrary/ctime/strftime/
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    
    return buf;
}

//////////////////////
// Check Parameters //
//////////////////////
// This function checks the parameters combination
void checkParameters()
{
    
    switch(initMode)
    {

        case 'R' : // Random configuration
            logFile << "Initializing in mode 'R', particles placed uniformly in the box" << endl;
            break;

        case 'S' : // Starting from previous simulation
            logFile << "Initializing in mode 'S', starting off from previous simulation" << endl;
            break;

        case 'L' : // Lattice configuration (hexagonal)
            logFile << "Initializing in mode 'L', hexagonal lattice" << endl;
            break;

        default :
            cerr << "Invalid Initialization Mode!" << endl;
            cerr << " --> Valid modes are : 'R', 'S', 'L' ... " << endl;
            ::exit(1);

    }

    switch(couplingMode)
    {
        case 'C' : // Constant coupling
            logFile << "Initializing couplings in mode 'C', couplings are constant" << endl;
            break;

        case 'T' : // Two-populations
            logFile << "Initializing couplings in mode 'T', two populations with reciprocal couplings (K_AA, K_AB and K_BB)" << endl;
            break;

        case 'G' : // Gaussian distributed couplings
            logFile << "Initializing couplings in mode 'G', Gaussian distributed couplings" << endl;
            break;

        case 'F' : // Normally distributed ferromagnetic couplings
            logFile << "Initializing couplings in mode 'F', normally distributed ferromagnetic couplings" << endl;
            break;

        case 'A' : // Normally distributed antiferromagnetic couplings
            logFile << "Initializing couplings in mode 'A', normally distributed antiferromagnetic couplings" << endl;
            break;

        // case 'B' : // Bimodal Gaussian distributed couplings
        //     logFile << "Initializing couplings in mode 'B', Bimodal Gaussian distributed couplings" << endl;
        //     break;            

        // case 'N' : // Non-reciprocal couplings
        //     logFile << "Initializing couplings in mode 'N', non-reciprocal couplings" << endl;
        //     break;   
        
        default :
            cerr << "Invalid Couplings Initialization Mode!" << endl;
            cerr << " --> Valid modes are : 'C', 'T', 'G', 'F', 'A' ... " << endl;
            ::exit(1);

    }
}

////////////////
// initialize //
////////////////
// Initialize positions, and velocities
void initialize(vector<double>& x, vector<double>& y, vector<double>& p)
{
    double KK;

    // Seed the random engines
    rnd_gen.seed (seed);

    // Initialize particle hard-core radius
    beta = 1.0;
    betasq = beta*beta;

    // Initialize Vicsek interaction
    rp = (Rp+0.1)*beta; // to allow for rounding errors
    rpsq = rp*rp;


    // Neighbor list radius
    // rc=rp as particles do not move
    rc = rp;
    rl = rc; 
    rlsq = rl*rl;

    // initialize particles positions & polarities
    switch(initMode)
    {
        case 'L' : // Particles placed on a hexagonal lattice
            initialConditionsLattice(x,y,p);
            break;
    }
    // Allocation of memory
    allocateSRKmem();

    // Initialize the coupling array
    switch(initMode)
    {
        case 'L' :
            switch(couplingMode)
            {
                case 'C' : // Constant coupling
                    for(int i=0 ; i<nPart ; i++){
                        K[i][i] = 0.0;
                        for(int j=i+1 ; j<nPart ; j++){
                            K[i][j] = K0; 
                            K[j][i] = K0; 
                        }
                    }
                    break;

                case 'T' : // Two-populations
                    for(int i=0 ; i<nPart ; i++){
                        K[i][i] = 0.0;
                        for(int j=i+1 ; j<nPart ; j++){
                            if(i<nPart/2.0){
                                if(j<nPart/2.0){
                                    K[i][j] = KAA;
                                    K[j][i] = KAA;
                                }else{
                                    K[i][j] = KAB;
                                    K[j][i] = KAB;
                                }
                            }else{
                                K[i][j] = KBB;
                                K[j][i] = KBB;
                            }
                        }
                    }
                    break;

                case 'G' : // Gaussian distributed couplings
                    for(int i=0 ; i<nPart ; i++){
                        K[i][i] = 0.0;
                        for(int j=i+1 ; j<nPart ; j++){
                            KK = KAVG + STDK*normDist(rnd_gen);
                            K[i][j] = KK;
                            K[j][i] = KK;
                        }
                    }
                    break;

                case 'F' : // Normally distributed ferromagnetic couplings
                    for(int i=0 ; i<nPart ; i++){
                        K[i][i] = 0.0;
                        for(int j=i+1 ; j<nPart ; j++){
                            do{
                                KK = KAVG + STDK*normDist(rnd_gen);    
                            }while (KK<0.0);
                            K[i][j] = KK;
                            K[j][i] = KK;
                        }
                    }
                    break;

                case 'A' : // Normally distributed antiferromagnetic couplings
                    for(int i=0 ; i<nPart ; i++){
                        K[i][i] = 0.0;
                        for(int j=i+1 ; j<nPart ; j++){
                            do{
                                KK = KAVG + STDK*normDist(rnd_gen);    
                            }while (KK>0.0);
                            K[i][j] = KK;
                            K[j][i] = KK;
                        }
                    }
                    break;
            }

            if (saveCoupling) {
                couplingFile.open("coupling",ios::out);
                if(couplingFile.fail())
                {cerr<<"Failed to open couplings file!"<<endl; ::exit(1);}
                couplingFile.precision(8);
                saveCouplings(K,couplingFile);
                couplingFile.close();
            }
            break;
    }       
}


/////////////////////////////
// initialConditionsLattice //
/////////////////////////////
// Initialize positions of particles on a hexagonal lattice (filled up completely)
void initialConditionsLattice(vector<double>& x, vector<double>& y, vector<double>& p)
{
    int Nx, Ny;
    startT = 0.0;

    // Open file to write initial conditions
    initposFile.open("initpos",ios::out);
    if (initposFile.fail()) 
    {cerr << "Can't open initial positions file!" << endl; ::exit(1);}
    initposFile.precision(8);

    Nx = ceil(sqrt(nPart));
    Ny = Nx;
    nPart = SQR(Nx);

    // Calculate size of the box
    Lx = Nx*beta;
    Ly = sqrt(3)/2 * Ny*beta;

    
    xmin = 0.0;
    xmax = Lx;
    ymin = 0.0;
    ymax = Ly;

    // Timing
    Neq = (int) ceil(eqT/dT);
    Nsimul = (int) ceil(simulT/dT);
    Nskip = (int) ceil(DT/dT);
    Nskipexact = (int) ceil(DTex/dT);
    logFile << "Neq = " << Neq << ", Nsimul = " << Nsimul << " and Nskip = " << Nskip << endl;

    // Initialize particles on a hexagonal lattice and random orientations
    for (int i=0 ; i<Ny ; i++) {
        for (int j=0 ; j<Nx ; j++) {
            y[i*Nx+j] = i*sqrt(3)/2 * beta + beta/2;
            x[i*Nx+j] = j*beta;
            if (i%2==1) {
                x[i*Nx+j] += beta/2;
            }
        }
    }

    for (int i=0 ; i<nPart ; i++) {
        p[i] = 2.0*PI*uniDist(rnd_gen); 
    }

    // Initialize lengthscales related to the cell list
    lx = rl; 
    ly = rl;
    mx = (int)floor(Lx/lx);
    my = (int)floor(Ly/ly);
    lx = Lx/double(mx);
    ly = Ly/double(my);
    nCell = mx*my;

    // Allocation of memory
    allocateSRKmem();   

    // Build map of cells
    buildMap();

    // Create neighbour lists (no need to reupdate later as particles are stationary)
    updateNL(x,y);

    // Save initial conditions
    saveInitFrame(x,y,p,initposFile);
    initposFile.close();

    return; 
}

////////////////////
// allocateSRKmem //
////////////////////
// Allocate memory for SRK2
void allocateSRKmem(void)
{    
    X.resize(nPart);    
    Y.resize(nPart);
    P.resize(nPart);
    
    Fx.resize(nPart);
    Fy.resize(nPart);
    Fp.resize(nPart);

    xl.resize(nPart,0.0);    
    yl.resize(nPart,0.0);

    cl.resize(nPart+1);
    sqDisp.resize(nPart);

    head.resize(nCell);
    lscl.resize(nPart);
    mp.resize(nCell, vector<int>(nNeighbor));

    K.resize(nPart, vector<double>(nPart));

    return;
}

//////////////////
// checkOverlap //
//////////////////
// Checks for overlaps between particles
bool checkOverlap(vector<double> x, vector<double> y)
{
    double xij,yij,rij;
    bool overlap = false;

    for (int i=0 ; i<nPart ; i++) {
        for (int j=i+1 ; j<nPart ; j++) {

            xij = x[i]-x[j];
            xij = xij - Lx*rint(xij/Lx);

            if (fabs(xij) < beta) {

                yij = y[i]-y[j];
                yij = yij - Ly*rint(yij/Ly);

                rij = sqrt(SQR(xij)+SQR(yij));

                if(rij < beta) return true;
            }
        }
    }
    return overlap;
}

//////////////
// buildMap //
//////////////
// Build the map of neighbors for interaction computation
void buildMap(void)
{
    int iMap;

    for(int i=0 ; i<mx ; i++) {
        for(int j=0 ; j<my ; j++) {

            iMap = lCellIndex(i,j);

            mp[iMap][0] = iMap;
            mp[iMap][1] = lCellIndex( i   , j+1 );
            mp[iMap][2] = lCellIndex( i+1 , j+1 );
            mp[iMap][3] = lCellIndex( i+1 , j   );
            mp[iMap][4] = lCellIndex( i+1 , j-1 );

        }
    }

}

/////////////
// checkNL //
/////////////
// This function returns the maximum displacement over the particles since the last build of the 
// neighbor list

bool checkNL(vector<double> x, vector<double> y)
{

    double maxsqDisp;

    for (int i=0 ; i<nPart ; i++) {
        sqDisp[i] = SQR(x[i]-xl[i])+SQR(y[i]-yl[i]);
    }
    maxsqDisp = *max_element(sqDisp.begin(),sqDisp.end());
        
    return maxsqDisp > SQR((rl - rc)/2.0);

}

//////////////
// updateNL //
//////////////
// Builds the neighbor list
void updateNL(vector<double> x, vector<double> y)
{
    double xij,yij,rijsq;
    double xi,yi;
    int j,nC;
    double xcb,ycb;
    int iC;
    int k=0;

    // (1) Clear existing neighbor list
    nl.clear();

    // (2) Build linked-cell list
    for (int i=0 ; i<nCell ; i++) {
        head[i]=-1;
    }
    for (int i=0 ; i<nPart ; i++) {
        xcb = x[i]-Lx*floor(x[i]/Lx);
        ycb = y[i]-Ly*floor(y[i]/Ly);
        iC = lCellIndex( int(xcb/lx) , int(ycb/ly) );
        lscl[i] = head[iC];
        head[iC] = i;
    }

    // (3) Update the neighbor list
    for (int i=0 ; i<nPart ; i++) {

        cl[i] = k;

        xi = x[i];
        yi = y[i];

        // Loop over all molecules below i in the current cell
        j = lscl[i];
        while( j > -1) {

            xij = xi-x[j];
            xij = xij - Lx*rint(xij/Lx);

            if (fabs(xij) <= rl) {

                yij = yi-y[j];
                yij = yij - Ly*rint(yij/Ly);

                rijsq = SQR(xij)+SQR(yij);

                if (rijsq <= rlsq) {
                    nl.push_back(j);
                    k++;
                }
            }

            j = lscl[j];
        }

        xcb = xi-Lx*floor(xi/Lx);
        ycb = yi-Ly*floor(yi/Ly);
        iC = lCellIndex( int(xcb/lx) , int(ycb/ly) );

        // Loop over all molecules in neighboring cells
        for (int jC=1 ; jC<=4 ; jC++) {
            nC = mp[iC][jC];
            j = head[nC];

            while( j > -1) {

                xij = xi-x[j];
                xij = xij - Lx*rint(xij/Lx);

                if (fabs(xij) <= rl) {

                    yij = yi-y[j];
                    yij = yij - Ly*rint(yij/Ly);

                    rijsq = SQR(xij)+SQR(yij);

                    if (rijsq <= rlsq) {
                        nl.push_back(j);
                        k++;
                    }
                }

                j = lscl[j];
            }

        }

    }
    cl[nPart]=nl.size();

    // (4) Update the saved positions
    for (int i=0 ; i<nPart ; i++) {
        xl[i] = x[i];
        yl[i] = y[i];
    }

    return;
}

//////////
// SRK2 //
//////////
// Integrator for overdamped Langevin equation -- Stochastic Runge-Kutta 2nd order
void SRK2(vector<double>& p, vector<double>& fp)
{
    double sig_R = sqrt(2.0*rotD*dT);

    // Calculate Forces on particle i at positions {r_i}, F_i({r_i(t)})
    force(p,fp);

    // Calculate updated positions
    for (int i=0 ; i<nPart ; i++ ) {
        P[i] = p[i] + fp[i]*dT + sig_R*normDist(rnd_gen);
    }

    // Calculate Forces on particle i at positions {R_i}, F_i({R_i(t)})
    force(P,Fp);

    // Calculate Final updated positions
    for (int i=0 ; i<nPart ; i++ ) {
        p[i] += (fp[i]+Fp[i])/2.0*dT + sig_R*normDist(rnd_gen);
    }

    return;
}

////////
// EM //
////////
// Integrator for Force Balance equation -- Euler-Mayurama
void EM(vector<double>& p, vector<double>& fp)
{
    double sig_R = sqrt(2.0*rotD*dT);

    // Calculate Forces on particle i at positions {r_i}, F_i({r_i(t)})
    force(p,fp);

    // Calculate updated positions
    for (int i=0 ; i<nPart ; i++ ) {
        p[i] = p[i] + fp[i]*dT + sig_R*normDist(rnd_gen);
    }

    return;
}

///////////
// force //
///////////
// consists of : alignment interactions
void force(vector<double> pp, vector<double>& ffp)
{
    // double xij,yij,rij,rijsq;
    double pi,pj,pij,Kij;
    double ff;

    for (int i=0 ; i<nPart ; i++) {
        ffp[i] = 0.0;
    }

    
    for (int i=0 ; i<nPart ; i++) {

        pi = pp[i];

        // ffp[i] = 0.0;

        for (int j=cl[i] ; j<cl[i+1] ; j++) {
            pj = pp[nl[j]];
            Kij = K[i][nl[j]];
            pij = pi-pj;
            ff = -Kij*sin(pij);

            ffp[i]     += ff;
            ffp[nl[j]] -= ff;
        }
    }

    return;
}

//////////////////////
// brownianDynamics //
//////////////////////
// Proceeds to one timestep integration of the equation of motion
void activeBrownianDynamics(vector<double>& p, vector<double>& fp, double& t)
{
    // Force Balance equation
    EM(p,fp);
    t += dT;
    return;
}

