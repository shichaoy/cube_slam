/**
 * @author  Daniel Maturana
 * @year    2015
 *
 * @attention Copyright (c) 2015
 * @attention Carnegie Mellon University
 * @attention All rights reserved.
 *
 **@=*/


#ifndef PROFILER_HPP_JUEKLQ5B
#define PROFILER_HPP_JUEKLQ5B

#include <stdint.h>
#include <time.h>

#include <map>
#include <vector>
#include <string>
#include <iostream>

#include <boost/chrono.hpp>

#include <ros/ros.h>

#define CA_FUN_TICTOC {ca::Profiler::tictoc(__PRETTY_FUNCTION__);}

namespace ca
{

typedef boost::chrono::time_point<boost::chrono::system_clock> tictoc_timestamp_t;

struct ProfilerEntry {
  tictoc_timestamp_t start_time;
  tictoc_timestamp_t end_time;
  ProfilerEntry(tictoc_timestamp_t _start_time) :
      start_time(_start_time),
      end_time()
  { }
};

struct AggregatedProfilerEntry {
  double total_ms;
  double min_ms;
  double max_ms;
  double avg_ms;
  int num_calls;
  std::string name;
};

// TODO what about templating on whether the profiler is enabled?
// Then disabled version should absolutely have no cost
class Profiler {
 public:
  typedef std::vector<ProfilerEntry> ProfilerEntries;
  typedef typename std::map<std::string, ProfilerEntries> ProfilerEntryMap;
  typedef std::map<std::string, bool> RunningMap;

 public:

  //static int64_t tictoc(const char * name);
  static int64_t tictoc(const std::string& name);

  static void enable();
  static void disable();

  static void aggregate_entries(std::vector<AggregatedProfilerEntry>* aggregated);

  static void print_aggregated(std::ostream& os);
  static void print_aggregated_csv(std::ostream& os);
  static void print_all(std::ostream& os);
  static void publish_messages(ros::NodeHandle& nh);

 private:
  Profiler(const Profiler& other);
  Profiler& operator=(const Profiler& other);

 private:
  static bool enabled_;

  static ProfilerEntryMap entries_;
  static RunningMap running_;
};

}
#endif /* end of include guard: PROFILER_HPP_JUEKLQ5B */
