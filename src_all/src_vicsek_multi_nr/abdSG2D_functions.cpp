// abdSG2D_functions.cpp
// =====================
// Definition of all functions necessary to run activeSpinGlassL_2D.cpp
// Created by Thibault Bertrand on 2022-04-19
// Last update by EL 2023-10-09

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
normal_distribution<double> whiteNoise(0.0,1.0);

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

        case 'A' : // Aligned configuration
            logFile << "Initializing in mode 'A', particles placed uniformly in the box fully aligned" << endl;
            break;

        default :
            cerr << "Invalid Initialization Mode!" << endl;
            cerr << " --> Valid modes are : 'R' ... " << endl;
            ::exit(1);

    }


    switch(couplingMode)
    {
        case 'C' : // Constant coupling
            logFile << "Initializing couplings in mode 'C', couplings are constant" << endl;
            break;

        case 'T' : // Three populations
            logFile << "Initializing couplings in mode 'T', three populations with reciprocal couplings A,B,C and no coupling within population" << endl;
            break;

        case 'G' : // Gaussian distributed couplings
            logFile << "Initializing couplings in mode 'G', Gaussian distributed couplings" << endl;
            break;

        case 'F' : // Normally distributed ferromagnetic couplings
            logFile << "Initializing couplings in mode 'F', fraction mode" << endl;
            break;

        case 'A' : // Normally distributed antiferromagnetic couplings
            logFile << "Initializing couplings in mode 'A', non-reciprocal couplings" << endl;
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

    switch(intMethod)
    {
        case 'E' :
            logFile << "Initializing Euler-Maruyama method for solving the SDE" << endl;
            break;

        case 'S' :
            logFile << "Initializing 2nd order Stochastic Runge-Kutta method for solving the SDE" << endl;
            break;

        default:            
            cerr << "Invalid method for SDE solving!" << endl;
            cerr << " --> Valid modes are : 'E', 'S' " << endl;
            ::exit(1);
    }
}

////////////////
// initialize //
////////////////
// Initialize positions, and velocities
void initialize(vector<double>& x, vector<double>& y, vector<double>& p)
{
    double KK, KK2;
    double alpha, Kp, Kn;
    unsigned long long index;

    // Seed the random engines
    rnd_gen.seed (seed);
    
    beta = 1.0;
    betasq = beta*beta;

    // Initialize Vicsek interaction
    rp = Rp;
    rpsq = rp*rp;

    // Neighbor list radius
    rc = rp;
    rl = rc+0.5;
    rlsq = rl*rl;

    // initialize particles positions & polarities
    if (initMode == 'A') {
        initialConditionsAligned(x,y,p);
    }
    else {
        initialConditionsRandom(x,y,p);
    }

    // Initialize the coupling array
    switch(couplingMode)
    {
        case 'C' : // Constant coupling
            for(int i=0 ; i<nPart ; i++){
                // K[i][i] = 0.0;
                for(int j=i+1 ; j<nPart ; j++){
                    index = getIndex(i,j);
                    K[index] = K0; 
                }
            }
            break;

        // case 'T' : // Two-populations
        //     for(int i=0 ; i<nPart ; i++){
        //         // K[i][i] = 0.0;
        //         for(int j=i+1 ; j<nPart ; j++){
        //             index = getIndex(i,j);
        //             if(i<nPart/2.0){
        //                 if(j<nPart/2.0){
        //                     K[index] = KAA;
        //                 }else{
        //                     K[index] = KAB;
        //                 }
        //             }else{
        //                 K[index] = KBB;
        //             }
        //         }
        //     }
        //     break;

        case 'T' : // Three-populations
            for(int i=0 ; i<nPart ; i++){
                // K[i][i] = 0.0;
                for(int j=i+1 ; j<nPart ; j++){
                    index = getIndex(i,j);
                    if(i<nPart/3.0){
                        if(j<nPart/3.0){
                            K[index] = 0;
                        }else if (j<2*nPart/3.0){
                            K[index] = KAB;
                        }else {
                            K[index] = KAC;
                        }
                    }else if (i<2*nPart/3.0){
                        if (nPart/3.0<=j<2*nPart/3.0) {
                            K[index] = 0;
                        }else if (j>=2*nPart/3.0) {
                            K[index] = KBC;
                        }
                    else{
                        if (j>=2*nPart/3.0) {
                            K[index] = 0;
                        }
                    }
                }
            }
            }
            break;

        case 'G' : // Gaussian distributed couplings
            for(int i=0 ; i<nPart ; i++){
                // K[i][i] = 0.0;
                for(int j=i+1 ; j<nPart ; j++){
                    index = getIndex(i,j);
                    KK = KAVG + STDK*normDist(rnd_gen);
                    K[index] = KK;
                }
            }
            break;

        case 'F' : // Fraction of particles ferro and anti-ferro magnetic
            // alpha = SQR(KAVG-K1)/(SQR(STDK)+SQR(KAVG-K1));
            // K0 = (SQR(STDK)+KAVG*(KAVG-K1))/(KAVG-K1);
            if(STDK==0) {
                Kp = KAVG;
                Kn = 0;
                alpha = 1;
            }
            else {
                alpha = 0.5 - 0.5 * erf(-KAVG/(sqrt(2)*STDK));
                Kp = KAVG + STDK*sqrt((1-alpha)/alpha);
                Kn = KAVG - STDK*sqrt(alpha/(1-alpha));
            }
            cout << " ----> alpha = " << alpha << "; K+ = " << Kp << "; K- = " << Kn << endl;
            for(int i=0 ; i<nPart ; i++){
                // K[i][i] = 0.0;
                for(int j=i+1 ; j<nPart ; j++){
                    index = getIndex(i,j);
                    if(uniDist(rnd_gen) < alpha) {
                        KK = Kp; // Kp is alpha proportion (positive one K+)
                        }
                    else{
                        KK = Kn; // Kn is negative one K-
                    }
                    K[index] = KK;
                }
            }
            break;

        case 'A' : // Normally distributed antiferromagnetic couplings
            {cerr<<"Need to use full coupling store version of code!"<<endl; ::exit(1);}
            break;
    }

    if (initMode == 'S') 
    {
        initialConditionsSim(x,y,p);
    }

    // Save seed used for rest of simulation
    seedFile.open("seed_p",ios::out);
    if(seedFile.fail())
    {cerr<<"Failed to open seed file!"<<endl; ::exit(1);}
    seedFile << seed << std::endl;
    seedFile.close();
}

void finalize(void)
{
    if (saveCoupling) {
    couplingFile.open("coupling",ios::out);
    if(couplingFile.fail())
    {cerr<<"Failed to open couplings file!"<<endl; ::exit(1);}
    couplingFile.precision(4);
    saveCouplings(K,couplingFile);
    couplingFile.close();
    }
}

/////////////////////////////
// initialConditionsRandom //
/////////////////////////////
// Initialize positions of particles at random location using FIRE
void initialConditionsRandom(vector<double>& x, vector<double>& y, vector<double>& p)
{
    double U;
    double dTF = 0.1;

    startT = 0.0;

    // Open file to write initial conditions
    if (saveInitPos) {
        initposFile.open("initpos",ios::out);
        if (initposFile.fail()) 
        {cerr << "Can't open initial positions file!" << endl; ::exit(1);}
        initposFile.precision(8);
    }

    // Calculate size of the box
    L = sqrt(double(nPart)/(phi*xTy));
    
    xmin = 0.0;
    xmax = xTy*L;
    ymin = 0.0;
    ymax = L;

    Lx = xTy*L;
    Ly =     L;

    if (eqT>simulT)
    {cerr << "eqT is larger than simulT!" << endl; ::exit(1);}


    // Timing
    Neq = (int) ceil(eqT/dT);
    Nsimul = (int) ceil((simulT-eqT)/dT);
    Nskip = (int) ceil(DT/dT);
    Nskipexact = (int) ceil(DTex/dT);
    logFile << "Neq = " << Neq << ", Nsimul = " << Nsimul << " and Nskip = " << Nskip << endl;
    logFile << "Volume fraction is phi = " << phi << endl;

    // Initialize particles at random positions
    for (int i=0 ; i<nPart ; i++) {
        x[i] = Lx*uniDist(rnd_gen);
        y[i] = Ly*uniDist(rnd_gen);
        p[i] = 2.0*PI*uniDist(rnd_gen); 
    }

    // // Save initial conditions
    // if (saveInitPos) {
    //     saveInitFrame(x,y,p,initposFile);
    // }

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

    // // Proceed to one-step energy minimization via FIRE
    // U = -1.0;
    // fire(x,y,dTF,fTOL,U,fHarmonic,dfHarmonic);

    // Save initial conditions
    if (saveInitPos) {
        saveInitFrame(x,y,p,initposFile);
        initposFile.close();
    }
    return; 
}

/////////////////////////////
// initialConditionsRandom //
/////////////////////////////
// Initialize positions of particles at random location using FIRE
void initialConditionsAligned(vector<double>& x, vector<double>& y, vector<double>& p)
{
    double U;
    double dTF = 0.1;

    startT = 0.0;

    // Open file to write initial conditions
    if (saveInitPos) {
        initposFile.open("initpos",ios::out);
        if (initposFile.fail()) 
        {cerr << "Can't open initial positions file!" << endl; ::exit(1);}
        initposFile.precision(8);
    }

    // Calculate size of the box
    L = sqrt(double(nPart)/(phi*xTy));
    
    xmin = 0.0;
    xmax = xTy*L;
    ymin = 0.0;
    ymax = L;

    Lx = xTy*L;
    Ly =     L;

    if (eqT>simulT)
    {cerr << "eqT is larger than simulT!" << endl; ::exit(1);}


    // Timing
    Neq = (int) ceil(eqT/dT);
    Nsimul = (int) ceil((simulT-eqT)/dT);
    Nskip = (int) ceil(DT/dT);
    Nskipexact = (int) ceil(DTex/dT);
    logFile << "Neq = " << Neq << ", Nsimul = " << Nsimul << " and Nskip = " << Nskip << endl;
    logFile << "Volume fraction is phi = " << phi << endl;

    // Initialize particles at random positions
    for (int i=0 ; i<nPart ; i++) {
        x[i] = Lx*uniDist(rnd_gen);
        y[i] = Ly*uniDist(rnd_gen);
        p[i] = 0; // Not random orientations
    }

    // // Save initial conditions
    // if (saveInitPos) {
    //     saveInitFrame(x,y,p,initposFile);
    // }

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

    // // Proceed to one-step energy minimization via FIRE
    // U = -1.0;
    // fire(x,y,dTF,fTOL,U,fHarmonic,dfHarmonic);

    // Save initial conditions
    if (saveInitPos) {
        saveInitFrame(x,y,p,initposFile);
        initposFile.close();
    }
    return; 
}

/////////////////////////////
// initialConditionsSim //
/////////////////////////////
// Initialize positions of particles from exact positions file of previous simulation
void initialConditionsSim(vector<double>& x, vector<double>& y, vector<double>& p)
{
    unsigned int seed_p;
    // Find seed
    seedFile.open("seed_p",ios::in);
    if (seedFile.fail())
    // {cerr << "Can't open seed file!" << endl; ::exit(1);}
    {seed_p = seed;}
    seedFile >> seed_p;
    seedFile.close();

    // Change seed
    seed = seed_p+1000;
    rnd_gen.seed (seed);

    // Calculate size of the box
    L = sqrt(double(nPart)/(phi*xTy));
    
    xmin = 0.0;
    xmax = xTy*L;
    ymin = 0.0;
    ymax = L;

    Lx = xTy*L;
    Ly =     L;

    // Open file to read initial conditions
    posExactFile.open("pos_exact",ios::in);
    if (posExactFile.fail()) 
    {cerr << "Can't open exact positions file!" << endl; ::exit(1);}

    // Initialize particles by reading file
    posExactFile >> startT;
    for(int i=0 ; i<nPart ; i++)
    {
        posExactFile >> x[i] >> y[i] >> p[i];
    }

    posExactFile.close();

    if (ceil(startT)>=simulT)
    {cerr << "Already simulated up to simulT! Exact position file time is " << startT << endl; ::exit(1);}

    // Timing
    if (eqT==0)
        {Neq = (int) 0;
        Nsimul = (int) ceil((simulT-startT)/dT);}
    else if (eqT<startT)
        {cerr << "eqT is smaller than startT so running longer!" << endl;
        Neq = (int) 0;
        Nsimul = (int) ceil((simulT-eqT)/dT);}
    else
        {Neq = (int) ceil((eqT-startT)/dT);
        Nsimul = (int) ceil((simulT-eqT)/dT);}
    Nskip = (int) ceil(DT/dT);
    Nskipexact = (int) ceil(DTex/dT);
    logFile << "Neq = " << Neq << ", Nsimul = " << Nsimul << " and Nskip = " << Nskip << endl;
    logFile << "Volume fraction is phi = " << phi << endl;

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

//// Change to 2D //////
////////////////////
// volumeFraction //
////////////////////
// Calculates volume fraction 
double volumeFraction(void)
{
    double phi;
    double boxsize = Lx*Ly;
    phi = nPart/ boxsize;
    return phi;
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
void SRK2(vector<double>& x, vector<double>& fx,
          vector<double>& y, vector<double>& fy, 
          vector<double>& p, vector<double>& fp)
{
    double sig_T = 0.0;
    double sig_R = noise*sqrt(dT);
    vector<float> nei(nPart); // number of neighbours

    // Check the neighbor list and update if necessary
    if ( checkNL(x,y) ) {
        updateNL(x,y);
    }

    // Calculate Forces on particle i at positions {r_i}, F_i({r_i(t)})
    std::vector<float> nnei = force(x,y,p,fx,fy,fp);

    // Calculate updated positions
    for (int i=0 ; i<nPart ; i++ ) {
        X[i] = x[i] + fx[i]*dT + sig_T*normDist(rnd_gen);
        Y[i] = y[i] + fy[i]*dT + sig_T*normDist(rnd_gen);
        if (nnei[i] == 0) {
            P[i] = p[i] + sig_R*whiteNoise(rnd_gen);
        }
        else {
            P[i] = p[i] + fp[i]*dT/(nnei[i]) + sig_R*whiteNoise(rnd_gen);
        }
    }



    // Calculate Forces on particle i at positions {R_i}, F_i({R_i(t)})
    std::vector<float> NNei = force(X,Y,P,Fx,Fy,Fp);

    // Calculate Final updated positions
    for (int i=0 ; i<nPart ; i++ ) {
        x[i] += (fx[i]+Fx[i])/2.0*dT + sig_T*normDist(rnd_gen);
        y[i] += (fy[i]+Fy[i])/2.0*dT + sig_T*normDist(rnd_gen);
        p[i] += sig_R*whiteNoise(rnd_gen);
        if (nnei[i] != 0) {
            p[i] += fp[i]/(2.0*nnei[i])*dT;
        }
        if (NNei[i] != 0) {
            p[i] += Fp[i]/(2.0*NNei[i])*dT;
        }
    }
    return;
}

////////
// EM //
////////
// Integrator for Force Balance equation -- Euler-Mayurama
void EM(vector<double>& x, vector<double>& fx,
        vector<double>& y, vector<double>& fy, 
        vector<double>& p, vector<double>& fp)
{
    double sig_T = 0.0;
    double sig_R = noise*sqrt(dT);

    // Check the neighbor list and update if necessary
    if ( checkNL(x,y) ) {
        updateNL(x,y);
    }

    // Calculate Forces on particle i at positions {r_i}, F_i({r_i(t)})
    std::vector<float> nnei = force(x,y,p,fx,fy,fp);

    // Calculate updated positions
    for (int i=0 ; i<nPart ; i++ ) {
        x[i] = x[i] + fx[i]*dT + sig_T*normDist(rnd_gen);
        y[i] = y[i] + fy[i]*dT + sig_T*normDist(rnd_gen);
        if (nnei[i] == 0) {
            p[i] = p[i] + sig_R*whiteNoise(rnd_gen);
        }
        else {
            p[i] = p[i] + fp[i]*dT/nnei[i] + sig_R*whiteNoise(rnd_gen);
        }
    }

    return;
}

///////////
// force //
///////////
// consists in : Lennard-Jones potential between particles, active propulsion and alignment interactions
std::vector<float> force(vector<double> xx, vector<double> yy, vector<double> pp,
                        vector<double>& ffx, vector<double>& ffy, vector<double>& ffp)
{
    double xij,yij,rij,rijsq;
    double pi,pj,pij,Kij;
    double ff;

    std::vector<float> nei(nPart); // number of neighbours


    for (int i=0 ; i<nPart ; i++) {
        pi = pp[i];
        // Self-propelling force
        ffx[i] = vp*cos(pi);
        ffy[i] = vp*sin(pi);
        ffp[i] = 0.0;
    }

    for (int i=0 ; i<nPart ; i++) {

        pi = pp[i];
        // // Self-propelling force
        // ffx[i] = vp*cos(pi);
        // ffy[i] = vp*sin(pi);
        // ffp[i] = 0.0;

        for (int j=cl[i] ; j<cl[i+1] ; j++) {

            xij = xx[i]-xx[nl[j]];
            xij = xij - Lx*rint(xij/Lx);

            if (fabs(xij) <= rc) { // rc = rp
                yij = yy[i]-yy[nl[j]];
                yij = yij - Ly*rint(yij/Ly);

                rijsq = SQR(xij)+SQR(yij);

                // Vicsek alignment
                if (rijsq <= rpsq){
                    pj = pp[nl[j]];
                    if (i == nl[j]){
                        Kij = 0.0;
                    }
                    else {
                        Kij = K[getIndex(i,nl[j])];
                    }
                    // Kij = K[i][nl[j]];
                    pij = pi-pj;
                    ff = -Kij*sin(pij);

                    ffp[i]     += ff;
                    // ffp[nl[j]] += ff;  // + instead of - , this is what makes this asymmetric/ non-reciprocal

                    nei[i] += 1.0;
                    nei[nl[j]] += 1.0;
                }
            }
        }
    }
    return nei;
}

//////////////////////
// brownianDynamics //
//////////////////////
// Proceeds to one timestep integration of the equation of motion
void activeBrownianDynamics(vector<double>& x, vector<double>& y, vector<double>& p, 
                               vector<double>& fx, vector<double>& fy, vector<double>& fp, 
                               double& t)
{
    // Force Balance equation
    switch (intMethod)
    {
    case 'E':
        EM(x,fx,y,fy,p,fp);
        break;
    
    case 'S':
        SRK2(x,fx,y,fy,p,fp);
        break;
    }
    t += dT;
    return;
}

///////////////
// fHarmonic //
///////////////
// Contains the core of the function to minimize - Harmonic potential
// r is a 2N vector which N first entries correspond to x-axis coordinates of the particles and the following N entries to the y-coordinates
double fHarmonic(vector<double>& rx, vector<double>& ry)
{
    double xij,yij,rij,rijsq;
    double fp=0.0;

    for (int i=0 ; i<nPart ; i++) {
        for (int j=cl[i] ; j<cl[i+1] ; j++) {
            xij = rx[i]-rx[nl[j]];
            xij = xij - Lx*rint(xij/Lx);
            if(fabs(xij)<beta){
                yij = ry[i]-ry[nl[j]];
                yij = yij - Ly*rint(yij/Ly);
                if(fabs(yij)<beta){
                    rijsq = SQR(xij)+SQR(yij);

                    if (rijsq < betasq){
                        rij = sqrt(rijsq);
                        fp += pow((1.0-rij/beta),2.0)/2.0;
                    }
                }
            }
        }
    }

    return fp;             
}

////////////////
// dfHarmonic //
////////////////
// gradient function - Harmonic potential
// r is a 2N vector which N first entries correspond to x-axis coordinates of the particles and the following N entries to the y-coordinates
// f is a 2N vector which N first entries correspond to x-components of the forces and the following N entries to the y-components
void dfHarmonic(vector<double>& rx, vector<double>& ry, vector<double>& fx, vector<double>& fy)
{
    double xij,yij,rij,rijsq;
    double fp;

    for (int i=0 ; i<nPart ; i++) {
        fx[i] = 0.0;
        fy[i] = 0.0;
    }

    for (int i=0 ; i<nPart ; i++) {
        for (int j=cl[i] ; j<cl[i+1] ; j++) {
            xij = rx[i]-rx[nl[j]];
            xij = xij - Lx*rint(xij/Lx);
            if(fabs(xij)<beta){
                yij = ry[i]-ry[nl[j]];
                yij = yij - Ly*rint(yij/Ly);
                if(fabs(yij)<beta){
                    rijsq = SQR(xij)+SQR(yij);

                    if (rijsq < betasq){
                        
                        rij = sqrt(rijsq);
                        fp = (1.0-rij/beta)/beta;

                        fx[i]     += fp*xij/rij;
                        fx[nl[j]] -= fp*xij/rij;

                        fy[i]     += fp*yij/rij;
                        fy[nl[j]] -= fp*yij/rij;

                    }
                }
            }
        }
    }
    
    return;
}


//////////
// fire //
//////////
// Fast Inertial Relaxation Engine
void fire(vector<double> &px, vector<double> &py, const double dT0, const double ftol, double &fret, double func(vector<double> &,vector<double> &), void dfunc(vector<double> &, vector<double> &, vector<double> &, vector<double> &))
{

    // Local variables - FIRE Algorithm parameters
    const int Nmin = 5; // Minimal number of steps since last P<0
    const double finc = 1.1; // Increase factor in the timestepping
    const double fdec = 0.5; // Decrease factor in the timestepping
    const double astart = 0.1;
    const double fa = 0.99;
    const double dTmax = 0.1;
    const int nStepMax = 1.e6;

    bool notConverged = true;
    double dt = dT0;
    double P;
    double alph = astart; 
    int nStep,nStop;
    double normV,normF;
    double Fmax;

    vector<double> vx (nPart,0.0); 
    vector<double> vy (nPart,0.0); 
    vector<double> ax (nPart,0.0); 
    vector<double> ay (nPart,0.0); 
    vector<double> fx (nPart,0.0); 
    vector<double> fy (nPart,0.0); 

    nStop = 0;
    nStep = 0;

    while(notConverged && nStep < nStepMax){

        nStep++;

        // Step 0: Check the neighbor list and update if necessary
        if ( checkNL(px,py) ) {
            updateNL(px,py);
        }

        // Step 1: Update the positions
        for(int i=0 ; i<nPart ; i++){
            px[i] += vx[i]*dt + 0.5*SQR(dt)*fx[i];
            py[i] += vy[i]*dt + 0.5*SQR(dt)*fy[i];
            ax[i] = fx[i];
            ay[i] = fy[i];
        }

        // Step 2: Compute new forces
        dfunc(px,py,fx,fy);

        // Step 3: Update the velocities
        normV = 0.0;
        normF = 0.0;
        Fmax = 0.0;
        for(int i=0 ; i<nPart ; i++){
            vx[i] += 0.5*(fx[i]+ax[i])*dt;
            vy[i] += 0.5*(fy[i]+ay[i])*dt;
            normV += SQR(vx[i]) + SQR(vy[i]);
            normF += SQR(fx[i]) + SQR(fy[i]);
            Fmax = MAX(Fmax,sqrt(SQR(fx[i]) + SQR(fy[i])));
        }
        normV = sqrt(normV);
        normF = sqrt(normF);

        // Step 4: Check for convergence
        if(Fmax < ftol){
            cout << "Done with FIRE, Fmax = " << Fmax << " and ftol = " << ftol << endl;
            fret = func(px,py);
            return; 
        }

        // Step 5: FIRE algorithm
        P = 0.0;
        for(int i=0 ; i<nPart ; i++){
            P += fx[i]*vx[i] + fy[i]*vy[i];
        }

        for(int i=0 ; i<nPart ; i++){
            vx[i] = (1-alph)*vx[i] + alph*fx[i]*normV/normF;
            vy[i] = (1-alph)*vy[i] + alph*fy[i]*normV/normF;
        }

        if( P < 0.0 ){
            nStop = nStep;
            for(int i = 0 ; i<nPart ; i++){
                vx[i] = 0.0;    
                vy[i] = 0.0;    
            } 
            dt *= fdec;
            alph = astart;
        }else if( P >= 0.0 && nStep-nStop > Nmin ){
            dt = MIN(dt*finc,dTmax);
            alph *= fa;
        }

    }

    cerr << "Maximum number of iterations exceeded in FIRE!"<< endl;
    ::exit(1);

    return;
}