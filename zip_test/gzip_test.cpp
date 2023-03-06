#include <fstream>
#include <iostream>
// #include <boost/iostreams/filtering_streambuf.hpp>
// #include <boost/iostreams/copy.hpp>
// #include <boost/iostreams/filter/gzip.hpp>

int main()
{
    using namespace std;

    // std::ifstream inStream("coupling", std::ios_base::in);
    // std::ofstream outStream("coupling.gzip", std::ios_base::out);
    // boost::iostreams::filtering_streambuf< boost::iostreams::input> in;
    // in.push( boost::iostreams::gzip_compressor());
    // in.push( inStream );
    // boost::iostreams::copy(in, outStream);
    std::cout << "Hello World!";
    return 0;
}
