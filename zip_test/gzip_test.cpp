#include <fstream>
#include <iostream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

// int main()
// {
//     using namespace std;

//     std::ifstream inStream("coupling", std::ios_base::in);
//     std::ofstream outStream("coupling.gzip", std::ios_base::out);
//     boost::iostreams::filtering_streambuf< boost::iostreams::input> in;
//     in.push( boost::iostreams::gzip_compressor(
//         boost::iostreams::gzip_params(9)
//     ));
//     in.push( inStream );
//     boost::iostreams::copy(in, outStream);
//     std::cout << "Hello World!";
//     return 0;
// }

int main()
{
    using namespace std;

    std::ofstream file("coupling", std::ios_base::out | std::ios_base::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> outbuf;
    outbuf.push(boost::iostreams::gzip_compressor());
    outbuf.push(file);

    ostream out(&outbuf);
    out << "This is a test text!\n";
    boost::iostreams::close(outbuf);
    file.close()
}
