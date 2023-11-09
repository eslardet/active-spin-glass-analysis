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
    std::bitset<1000> K;
    std::vector<bool> P;
    std::vector<float> F;
    P.resize(1000);
    F.resize(1000);
    for (int i=0; i<100; i++)
    {
        K[i] = 1;
        P[i] = 1;
        F[i] = 1.54355;
    }

    cout << sizeof(K) << endl;
    cout << std::vector::capacity(P) << endl;
    cout << std::vector::capacity(F) << endl;
    return 0;
}


