//  activeSpinGlass_2D.cpp
//  ======================
//  Code performing 2D active brownian dynamics 
//	This code includes: 
//		-overdamped langevin dynamics for both position and orientations
//      -Purely repulsive Lennard-Jones interactions between particles
//      -Vicsek interactions with generalized coupling constants
//      -self-propelled particles via a constant velocity vp
//      -Hybrid cell-linked list and Verlet neighbor lists
//
//  Created by Thibault Bertrand on 2022-04-19
//  Last update by TB 2022-04-19

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <functional>
#include "math.h"
#include "activeSpinGlass_2D.h"
#include "abdSG2D_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

void signal_handler(int signo);

using namespace std;

int main(int argc, char *argv[])
{
int sig;

setlinebuf(stdout);

/* On MacOS, if you try to catch SIGCONT, then it's the only signal you will ever see... 
 * Even when you send -HUP you get to see [19] 
 *
 * Also, don't even try to catch SIGKILL. */
for (sig=1; sig<=32; sig++)
  if ((sig!=SIGCONT) && (sig!=SIGKILL))
    signal(sig, signal_handler);

/* From here, do your thing. 
 * The pause(3) is here only do stop until a signal comes.
 * This needs to be removed in production code. */
////////////////
//Main Program//
////////////////

	//////////////////////////////
	// Parsing input parameters //
	//////////////////////////////
	if (argc < 2) {
    	cerr << "Usage: " << argv[0] << " INPUT FILENAME" << endl;
    	exit(1);
	} else {
	    inputFile.open(argv[1],ios::in);
	    if (inputFile.fail()) 
	    {cerr << "Can't open input parameters file!" << endl; exit(1);}
	}


    ///////////////////////////////////////////
    // Output Files & Parsing of input files //
    ///////////////////////////////////////////

    // Simulation Log
    logFile.open("simu.log",ios::out);
    if(logFile.fail())
    {cerr<<"Failed to open log file!"<<endl; exit(1);}
    logFile.precision(8);

    cout.precision(8);

    logFile << "------------------------------------------------------------" << endl;
    logFile << "activeSpinGlass_2D (v1) Simulation Started" << endl;
    logFile << currentDateTime() << endl;
    logFile << "------------------------------------------------------------" << endl;

    // Get input parameters from input parameters file
    inputFile >> nPart;
    logFile << " --> Total Number of particles = " << nPart << endl;

    inputFile >> phi;
    logFile << " --> Volume fraction, phi = " << phi << endl;

    inputFile >> seed;
    logFile << " --> Seed = " << seed << endl;

    inputFile >> noise;
    logFile << " --> Noise = " << noise << endl;  

    inputFile >> vp;
    logFile << " --> Particle velocity = " << vp << endl;  

    inputFile >> Rp;
    logFile << " --> Ratio of Vicsek interaction radius to particle size = " << Rp << endl;    

    inputFile >> xTy;
    logFile << " --> Ratio of Lx/Ly = " << xTy << endl;

    logFile << "------------------------------------------------------------" << endl;

    inputFile >> initMode; 
    logFile << " --> Initial Conditions Mode = " << initMode << endl; 

    logFile << "------------------------------------------------------------" << endl;

    inputFile >> couplingMode; 
    logFile << " --> Coupling Constants Mode = " << couplingMode << endl; 

    // Add the coupling constants parameters
    switch(couplingMode)
    {
        case 'C' : // Constant coupling
            inputFile >> K0; 
            logFile << " ----> Coupling Constant K0 = " << K0 << endl; 
            break;

        case 'T' : // Two-populations
            inputFile >> KAA; 
            inputFile >> KAB;
            inputFile >> KBB;
            logFile << " ----> Coupling Constants, KAA = " << KAA << ", KAB = " << KAB << ", KBB = " << KBB << endl; 
            break;

        case 'G' : // Gaussian distributed couplings
            inputFile >> KAVG; 
            inputFile >> STDK;            
            logFile << " ----> Average Coupling Constants, KAVG = " << KAVG << " and standard deviation, STDK = " << STDK << endl; 
            break;

        case 'F' : // Fraction of ferro and anti-ferro magnetic couplings
            inputFile >> K1;
            inputFile >> KAVG; 
            inputFile >> STDK;            
            cout << " ----> K- = " << K1 << ", average Coupling Constants, KAVG = " << KAVG << " and standard deviation, STDK = " << STDK << endl; 
            break;

        case 'A' : // Normally distributed non-reciprocal couplings
            inputFile >> KAVG; 
            inputFile >> STDK;            
            logFile << " ----> Average Coupling Constants, KAVG = " << KAVG << " and standard deviation, STDK = " << STDK << endl; 
            break;
        
        default :
            cerr << "Invalid Couplings Initialization Mode!" << endl;
            cerr << " --> Valid modes are : 'C', 'T', 'G', 'F', 'A' ... " << endl;
            exit(1);
    }

	logFile << "------------------------------------------------------------" << endl;

    inputFile >> dT; 
    logFile << " --> Timestep, dt = "  << dT << endl;

    inputFile >> DT; 
    logFile << " --> Recording Timestep, DT = "  << DT << endl;

    inputFile >> DTex;
    logFile << " --> Recording Timestep (precise) = " << DTex << endl;

	inputFile >> eqT; 
	logFile << " --> Equilibration Time = " << eqT << endl;   

    inputFile >> simulT; 
	logFile << " --> Simulation Time = " << simulT << endl;   

	logFile << "------------------------------------------------------------" << endl;

    inputFile >> savePos; 
    logFile << " --> savePos = " << savePos << endl;

    inputFile >> saveInitPos;
    logFile << " --> saveInitPos = " << saveInitPos << endl;

    inputFile >> saveForce; 
    logFile << " --> saveForce = " << saveForce << endl;

    inputFile >> saveCoupling; 
    logFile << " --> saveCoupling = " << saveCoupling << endl;

    logFile << "------------------------------------------------------------" << endl;

    inputFile >> intMethod;
    logFile << " --> SDE solving method = " << intMethod << endl; 

    logFile << '\n';
    
    inputFile.close();


	///////////////////////
    // Declare Variables //
    ///////////////////////
    vector<double> x(nPart); // x-positions
    vector<double> y(nPart); // y-positions
    vector<double> p(nPart); // heading 
    
    vector<double> fx(nPart); // x-force
    vector<double> fy(nPart); // y-force
    vector<double> fp(nPart); // torque 

    double t,t0;
    double tphi;

    ///////////////////////
    // Initialize System //
    ///////////////////////
    // Initialize
    checkParameters(); // Check validity of the parameters combination
    initialize(x,y,p); // Initialize configuration

    // Checks
    tphi = volumeFraction();
    logFile << "Check on the simulation after initialization: " << endl;
    logFile << " --> Volume fraction = " << tphi << endl;

    // Tape Files
    bool append_pos = false;
    if (savePos) {
        ifstream ifile;
        ifile.open("pos");
        if(ifile && initMode == 'S' && eqT == 0) {
            append_pos = true;
        }

        if (append_pos == true) {
            posFile.open("pos",ios::app);
            if(posFile.fail())
            {cerr<<"Failed to open positions file!"<<endl; exit(1);}
            posFile.precision(8);
        }
        else {
            posFile.open("pos",ios::out);
            if(posFile.fail())
            {cerr<<"Failed to open positions file!"<<endl; exit(1);}
            posFile.precision(8);
            saveHeader(posFile);
        }
        
    }
    if (saveForce) {
        forceFile.open("force",ios::out);
        if(forceFile.fail())
        {cerr<<"Failed to open forces file!"<<endl; exit(1);}
        forceFile.precision(8);
        saveHeader(forceFile);
    }

    ////////////////
    // Time loop  //
    ////////////////
    t = startT;

	// (1) Equilibration
    for(int ne=0 ; ne<Neq ; ne++) {
        // full equilibration
        activeBrownianDynamics(x,y,p,fx,fy,fp,t);
    }

    // t0 = t;
    if (savePos && append_pos==0) {
		saveFrame(x,y,p,t,posFile);
        saveFrame(x,y,p,t,posExactFile);
    }
    if (saveForce) {
        saveFrame(fx,fy,fp,t,forceFile);
    }     
    
    // (2) Recording
    cout << "Starting to record simulation results:" << endl;
    cout << " --> " << flush;
    for(int ns=0 ; ns<Nsimul ; ns++) {
        // Move to the next timestep
        activeBrownianDynamics(x,y,p,fx,fy,fp,t);
        // Save data if necessary                
        if ( (ns+1)%Nskip == 0 ) {
            if (savePos) {
                saveFrame(x,y,p,t,posFile);
            }
            if (saveForce) {
                saveFrame(fx,fy,fp,t,forceFile);
            }     
        }
        if ((ns+1)%Nskipexact == 0 ){
            if (savePos) {
                posExactFile.open("pos_exact", ios::out);
                if(posExactFile.fail())
                {cerr<<"Failed to open exact positions file!"<<ns<<endl; exit(1);}
                posExactFile.precision(17);
                saveFrame(x,y,p,t,posExactFile);
                posExactFile.close();
            }     
        }
        if ( (ns+1) % int(floor(Nsimul/10)) == 0) {
            cout << "|" << flush;
        }           
    }     


    // Save coupling file if required
    finalize();

	///////////////////
    // Closing Files //
    ///////////////////
    logFile.close(); // Simulation Log
    if (savePos) { posFile.close(); }
    if (saveForce) { forceFile.close(); }
    
    cout << endl << "Simulation successful, with nPart = " << nPart << ", phi = " << phi << ", seed = " << seed << ", noise = " << noise << ", vp = " << vp << ", Rp = " << Rp << ", couplingMode = " << couplingMode << endl;
    switch(couplingMode)
    {
        case 'C' : // Constant coupling
            inputFile >> K0; 
            cout << " ----> Coupling Constant K0 = " << K0 << endl; 
            break;

        case 'T' : // Two-populations
            inputFile >> KAA; 
            inputFile >> KAB;
            inputFile >> KBB;
            cout << " ----> Coupling Constants, KAA = " << KAA << ", KAB = " << KAB << ", KBB = " << KBB << endl; 
            break;

        case 'G' : // Gaussian distributed couplings
            inputFile >> KAVG; 
            inputFile >> STDK;            
            cout << " ----> Average Coupling Constants, KAVG = " << KAVG << " and standard deviation, STDK = " << STDK << endl; 
            break;

        case 'F' : // Fraction mode
            inputFile >> K1;
            inputFile >> KAVG; 
            inputFile >> STDK;            
            cout << " ----> K- = " << K1 << ", average Coupling Constants, KAVG = " << KAVG << " and standard deviation, STDK = " << STDK << endl; 
            break;

        case 'A' : // Normally distributed antiferromagnetic couplings
            inputFile >> KAVG; 
            inputFile >> STDK;            
            cout << " ----> Average Coupling Constants, KAVG = " << KAVG << " and standard deviation, STDK = " << STDK << endl; 
            break;
        
        default :
            cerr << "Invalid Couplings Initialization Mode!" << endl;
            cerr << " --> Valid modes are : 'C', 'T', 'G', 'F', 'A' ... " << endl;
            exit(1);
    }

    return 0;

}



void signal_handler(int signo)
{
  fprintf(stdout, "Received signal [%i]\n", signo);
  /* The next line means that you bail out from whatever signal,
   * maybe this is not desired. */
  exit(EXIT_FAILURE);
}



