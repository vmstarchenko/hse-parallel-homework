#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
namespace mpi = boost::mpi;

int main() {
    mpi::environment env;
    mpi::communicator world;
    std::cout << "I am process " << world.rank() << " on " << world.size()
              << "." << std::endl;
    return 0;
}
