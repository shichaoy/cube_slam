/**
 * @author  Daniel Maturana
 * @year    2015
 *
 * @attention Copyright (c) 2015
 * @attention Carnegie Mellon University
 * @attention All rights reserved.
 *
 **@=*/


#include <iostream>
#include "tictoc_profiler/profiler.hpp"

long stupid_fibonacci(int i) {
  if (i==1 || i==2) { return 1; }
  return stupid_fibonacci(i-1)+stupid_fibonacci(i-2);
}

long less_stupid_fibonacci(int i) {
  if (i==1 || i==2) { return 1; }
  long f_im1=1, f_im2=1, f_i=2;
  for (int ctr=3; ctr < i; ++ctr) {
    f_im2 = f_im1;
    f_im1 = f_i;
    f_i = f_im1+f_im2;
  }
  return f_i;
}

int main(int argc, char *argv[]) {
  ca::Profiler::enable();

  for (int i=0; i < 5; ++i) {
    ca::Profiler::tictoc("stupid_fibonacci");
    long f1 = stupid_fibonacci(40+i);
    ca::Profiler::tictoc("stupid_fibonacci");
    std::cerr << "f1 = " << f1 << std::endl;

    ca::Profiler::tictoc("less_stupid_fibonacci");
    long f2 = less_stupid_fibonacci(40+i);
    ca::Profiler::tictoc("less_stupid_fibonacci");
    std::cerr << "f2 = " << f2 << std::endl;

    std::cerr << "\n";
  }

  //ca::Profiler::print_all(std::cerr);
  ca::Profiler::print_aggregated(std::cerr);
  return 0;
}
