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
    double KK;

    // Seed the random engines
    rnd_gen.seed (seed);

    // // initialize particles positions & polarities
    // switch(initMode)
    // {
    //     case 'R' : // Particles placed randomly in the box
    initialConditionsRandom(x,y,p);

    // Allocation of memory
    allocateSRKmem();

    // Initialize the coupling array
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

            // if (saveCoupling) {
            //     couplingFile.open("coupling",ios::out);
            //     if(couplingFile.fail())
            //     {cerr<<"Failed to open couplings file!"<<endl; ::exit(1);}
            //     couplingFile.precision(4);
            //     saveCouplings(K,couplingFile);
            //     couplingFile.close();
            // }
            // break;

    if (initMode == 'S') 
    {
        initialConditionsSim(x,y,p);
        allocateSRKmem();
    }
    //     case 'S' : // Particles configured starting from previous simulation
    //         // initial condition stuff (inc reading pos_exact file and coupling file)

    //         initialConditionsSim(x,y,p);

    //         allocateSRKmem();

    //         // read coupling file (not necessary for constant mode)
    //         switch(couplingMode)
    //         {
    //             case 'C' : // Constant coupling
    //                 for(int i=0 ; i<nPart ; i++){
    //                     K[i][i] = 0.0;
    //                     for(int j=i+1 ; j<nPart ; j++){
    //                         K[i][j] = K0; 
    //                         K[j][i] = K0; 
    //                     }
    //                 }
    //                 break;

    //             case 'T' : // Two-populations
    //                 for(int i=0 ; i<nPart ; i++){
    //                     K[i][i] = 0.0;
    //                     for(int j=i+1 ; j<nPart ; j++){
    //                         if(i<nPart/2.0){
    //                             if(j<nPart/2.0){
    //                                 K[i][j] = KAA;
    //                                 K[j][i] = KAA;
    //                             }else{
    //                                 K[i][j] = KAB;
    //                                 K[j][i] = KAB;
    //                             }
    //                         }else{
    //                             K[i][j] = KBB;
    //                             K[j][i] = KBB;
    //                         }
    //                     }
    //                 }
    //                 break;

    //             default : // Other random coupling modes
    //                 couplingFile.open("coupling", ios::in);
    //                 if(couplingFile.fail())
    //                 {cerr<<"Failed to open couplings file!"<<endl; ::exit(1);}
    //                 for(int i=0 ; i<nPart ; i++)
    //                 {
    //                     for(int j=i+1 ; j<nPart ; j++){
    //                         couplingFile >> K[i][j];        
    //                     }
    //                 }
    //                 couplingFile.close();
    //                 break;

    //         }
    // }
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
// Initialize positions of particles at random location
void initialConditionsRandom(vector<double>& x, vector<double>& y, vector<double>& p)
{

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

    // Timing
    Neq = (int) ceil(eqT/dT);
    Nsimul = (int) ceil(simulT/dT);
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

    // Save initial conditions
    if (saveInitPos) {
        saveInitFrame(x,y,p,initposFile);
    }

    // Allocation of memory
    allocateSRKmem();   

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

    if (startT>simulT)
    {cerr << "Already simulated up to simulT! Exact position file time is " << startT << endl; ::exit(1);}

    posExactFile.close();


    // Timing
    Neq = (int) ceil(eqT/dT);
    Nsimul = (int) ceil((simulT-startT)/dT);
    Nskip = (int) ceil(DT/dT);
    Nskipexact = (int) ceil(DTex/dT);
    logFile << "Neq = " << Neq << ", Nsimul = " << Nsimul << " and Nskip = " << Nskip << endl;
    logFile << "Volume fraction is phi = " << phi << endl;

    // Allocation of memory
    allocateSRKmem();   

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

    // Calculate Forces on particle i at positions {r_i}, F_i({r_i(t)})
    std::vector<float> nnei = force(p,fx,fy,fp);

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
    std::vector<float> NNei = force(P,Fx,Fy,Fp);

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

    // Calculate Forces on particle i at positions {r_i}, F_i({r_i(t)})
    std::vector<float> nnei = force(p,fx,fy,fp);

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
std::vector<float> force(vector<double> pp,
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

        for (int j=0 ; j<i ; j++) {

            pj = pp[j];
            Kij = K[i][j];
            pij = pi-pj;
            ff = -Kij*sin(pij);

            ffp[i] += ff;
            ffp[j] -= ff;

            nei[i] += 1.0;
            nei[j] += 1.0;
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