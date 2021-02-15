#include <metis.h>

int main( int argc, char* argv[] )
{
  static_assert(IDXTYPEWIDTH==64, "Metis has to be build with 64-bit integer support!");
  static_assert(REALTYPEWIDTH==64, "Metis has to be build with 64-bit floating point precision!");
  return 0;
}
