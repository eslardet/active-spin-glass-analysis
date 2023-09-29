#include <iostream>
#include<math.h>

unsigned long long getIndex(int i, int j, int nPart) {
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

int main()
{
    unsigned long long m;
    std::vector<double> k;
    std::cout << m.max_size() << std::endl;
    std::cout << k.max_size() << std::endl;

    m = getIndex(2,4,6);
    // std::cout << m << std::endl;

    // m = getIndex(3,0);
    // std::cout << m << std::endl;

    return 0;
}


