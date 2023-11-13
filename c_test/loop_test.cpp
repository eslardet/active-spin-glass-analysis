#include <iostream>
#include<math.h>
using namespace std;

// unsigned long long getIndex(int i, int j, int nPart) {
//     unsigned long long i_long = i, j_long = j, N_long = nPart;
//     if (i>j)
//     {
//         return j_long*(N_long-1) - j_long*(j_long-1)/2 + i_long - j_long - 1;
//     }
//     else
//     {
//         return i_long*(N_long-1) - i_long*(i_long-1)/2 + j_long - i_long - 1;
//     }
// }

int main()
{   
    int nPart = 6;
    static std::vector< std::vector<double> > K;
    K.resize(nPart, vector<double>(nPart));
    int KAB = 1;
    int KBA = -1;
    int KBC = 2;
    int KCB = -2;
    int KCA = 3;
    int KAC = -3;
    // for (int i=0 ; i<nPart ; i++){
    //     for (int j=0 ; j<nPart ; j++) {
    //         K[i][j] = i;
    //     }
    // }

    for(int i=0 ; i<nPart ; i++){
        for(int j=i+1; j<nPart ; j++){
            if(i<nPart/3){
                if(j<nPart/3){ // A-A
                    K[i][j] = 0.0;
                    K[j][i] = 0.0;
                    cout << "AA" << i << j << endl;
                }else if (j<2*nPart/3){ // A-B
                    K[i][j] = KAB;
                    K[j][i] = KBA;
                    cout << "AB" << i << j << endl;
                }else { // A-C
                    K[i][j] = KAC;
                    K[j][i] = KCA;
                    cout << "AC" << i << j << endl;
                }
            }else if (nPart/3.0<=i && i<2*nPart/3){ 
                if (nPart/3.0<=j && j<2*nPart/3) { // B-B
                    K[i][j] = 0.0;
                    K[j][i] = 0.0;
                    cout << "BB" << i << j << endl;
                }
                if (j>=2*nPart/3) { // B-C
                K[i][j] = KBC;
                K[j][i] = KCB;
                cout << "BC" << i << j << endl;
                }
            }
            else{
                if (j>=2*nPart/3) { // C-C
                    K[i][j] = 0.0;
                    K[j][i] = 0.0;
                    cout << "CC" << i << j << endl;
                }
            }
        }
    }

    for (int i=0 ; i<nPart ; i++){
        for (int j=0 ; j<nPart ; j++) {
            cout << K[i][j] << endl;
        }
    }


    return 0;
}


