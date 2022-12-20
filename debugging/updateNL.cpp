#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <limits>
#include <random>
#include <array>
#include <ctime>
#include <iomanip>
#include "math.h"

using namespace std;

///////////////////////
// Define functions  //
///////////////////////

//////////////////////////////
// Random number generation //
//////////////////////////////
// random_device rd;
// mt19937 rnd_gen;

// uniform_real_distribution<double> uniDist(0.0,1.0);
// normal_distribution<double> normDist(0.0,1.0);

// //////
// void updateNL(std::vector<double>,std::vector<double>);
// void initialize(std::vector<double>&,std::vector<double>&,std::vector<double>&);
// void initialConditionsRandom(std::vector<double>&,std::vector<double>&,std::vector<double>&);


// //////

// // Neighbor list variables
// static double rl,rc; // radius of the neighbor list
// static double rlsq;
// static std::vector<int> nl; // nl contains the neighbor list
// static std::vector<int> cl; // cl contains the number of neighbors for each particle
// static std::vector<double> xl,yl; // contains positions of particles at last neighbor list build
// static std::vector<double> sqDisp;





///// Main /////

int main()
{
	// ///////////////////////
    // // Declare Variables //
    // ///////////////////////

    int nPart = 5000;
    vector<double> x(nPart); // x-positions
    vector<double> y(nPart); // y-positions
    vector<double> p(nPart); // heading 
    
    // vector<double> fx(nPart); // x-force
    // vector<double> fy(nPart); // y-force
    // vector<double> fp(nPart); // torque 

    // double t,t0;
    // double tphi;

    // // Cell linked list variables
    // static double lx = rl; // width of the cell
    // static double ly = rl; // height of the cell
    // const int nNeighbor = 5;

    // static std::vector<int> head; // head holds the index of first index of 
    // static std::vector<int> lscl; // lscl vector implementation of linked-cell list 
    // static std::vector< std::vector<int> > mp; // contains list of neighboring cells for each cell
    /////////////


    string file = "pos";
    ifstream fin(file.c_str());

    double xi, yi, pi;
    int i=0;
    string line;
    istringstream iss;
    cout << setprecision(8);
    while (getline(fin, line)) {
        iss.clear();
        iss.str(line);
        iss >> xi >> yi >> pi;
        x[i] = xi;
        y[i] = yi;
        p[i] = pi;
        i += 1;
    }
    fin.close();

    return 0;
}





// /////////////////////////////
// // initialConditionsFile //
// /////////////////////////////
// // Initialize positions of particles from position file
// void initialConditionsFile(vector<double>& x, vector<double>& y, vector<double>& p)
// {
//     double U;
//     double dTF = 0.1;

//     // Open file to write initial conditions
//     initposFile.open("initpos",ios::out);
//     if (initposFile.fail()) 
//     {cerr << "Can't open initial positions file!" << endl; exit(1);}
//     initposFile.precision(8);

//     // Calculate size of the box
//     L = sqrt(double(nPart)*PI*SQR(beta/2.0)/(phi*xTy));
    
//     xmin = 0.0;
//     xmax = xTy*L;
//     ymin = 0.0;
//     ymax = L;

//     Lx = xTy*L;
//     Ly =     L;

//     // Timing
//     Neq = (int) ceil(eqT/dT);
//     Nsimul = (int) ceil(simulT/dT);
//     Nskip = (int) ceil(DT/dT);
//     logFile << "Neq = " << Neq << ", Nsimul = " << Nsimul << " and Nskip = " << Nskip << endl;
//     logFile << "Volume fraction is phi = " << phi << endl;

//     // Initialize particles at random positions
//     for (int i=0 ; i<nPart ; i++) {
//         x[i] = Lx*uniDist(rnd_gen);
//         y[i] = Ly*uniDist(rnd_gen);
//         p[i] = 2.0*PI*uniDist(rnd_gen); 
//     }

//     // Save initial conditions
//     saveInitFrame(x,y,p,initposFile);

//     // Initialize lengthscales related to the cell list
//     lx = rl; 
//     ly = rl;
//     mx = (int)floor(Lx/lx);
//     my = (int)floor(Ly/ly);
//     lx = Lx/double(mx);
//     ly = Ly/double(my);
//     nCell = mx*my;

//     // Allocation of memory
//     allocateSRKmem();   

//     // Build map of cells
//     buildMap();

//     // Proceed to one-step energy minimization via FIRE
//     U = -1.0;
//     fire(x,y,dTF,fTOL,U,fHarmonic,dfHarmonic);

//     // Save initial conditions
//     saveInitFrame(x,y,p,initposFile);
//     initposFile.close();

//     return; 
// }

// //////////////
// // updateNL //
// //////////////
// // Builds the neighbor list
// void updateNL(vector<double> x, vector<double> y)
// {
//     double xij,yij,rijsq;
//     double xi,yi;
//     int j,nC;
//     double xcb,ycb;
//     int iC;
//     int k=0;

//     // (1) Clear existing neighbor list
//     nl.clear();

//     // (2) Build linked-cell list
//     for (int i=0 ; i<nCell ; i++) {
//         head[i]=-1;
//     }
//     for (int i=0 ; i<nPart ; i++) {
//         xcb = x[i]-Lx*floor(x[i]/Lx);
//         ycb = y[i]-Ly*floor(y[i]/Ly);
//         iC = lCellIndex( int(xcb/lx) , int(ycb/ly) );
//         lscl[i] = head[iC];
//         head[iC] = i;
//     }

//     // (3) Update the neighbor list
//     for (int i=0 ; i<nPart ; i++) {

//         cl[i] = k;

//         xi = x[i];
//         yi = y[i];

//         // Loop over all molecules below i in the current cell
//         j = lscl[i];
//         while( j > -1) {

//             xij = xi-x[j];
//             xij = xij - Lx*rint(xij/Lx);

//             if (fabs(xij) <= rl) {

//                 yij = yi-y[j];
//                 yij = yij - Ly*rint(yij/Ly);

//                 rijsq = SQR(xij)+SQR(yij);

//                 if (rijsq <= rlsq) {
//                     nl.push_back(j);
//                     k++;
//                 }
//             }

//             j = lscl[j];
//         }

//         xcb = xi-Lx*floor(xi/Lx);
//         ycb = yi-Ly*floor(yi/Ly);
//         iC = lCellIndex( int(xcb/lx) , int(ycb/ly) );

//         // Loop over all molecules in neighboring cells
//         for (int jC=1 ; jC<=4 ; jC++) {
//             nC = mp[iC][jC];
//             j = head[nC];

//             while( j > -1) {

//                 xij = xi-x[j];
//                 xij = xij - Lx*rint(xij/Lx);

//                 if (fabs(xij) <= rl) {

//                     yij = yi-y[j];
//                     yij = yij - Ly*rint(yij/Ly);

//                     rijsq = SQR(xij)+SQR(yij);

//                     if (rijsq <= rlsq) {
//                         nl.push_back(j);
//                         k++;
//                     }
//                 }

//                 j = lscl[j];
//             }

//         }

//     }
//     cl[nPart]=nl.size();

//     // (4) Update the saved positions
//     for (int i=0 ; i<nPart ; i++) {
//         xl[i] = x[i];
//         yl[i] = y[i];
//     }

//     return;
// }
