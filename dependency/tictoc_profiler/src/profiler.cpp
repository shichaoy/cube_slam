/**
 * @author  Daniel Maturana
 * @year    2015
 *
 * @attention Copyright (c) 2015
 * @attention Carnegie Mellon University
 * @attention All rights reserved.
 *
 **@=*/


#include "tictoc_profiler/profiler.hpp"

#include <iomanip>
#include <limits>
#include <algorithm>

#include <boost/foreach.hpp>

#include <sys/time.h>

#include "tictoc_profiler/ProfilerEntry.h"

namespace ca
{

// initialize static data
bool Profiler::enabled_ = false;
Profiler::ProfilerEntryMap Profiler::entries_;
Profiler::RunningMap Profiler::running_;

static bool aggregated_profiler_entry_total_cmp(const AggregatedProfilerEntry& a,
                                                const AggregatedProfilerEntry& b) {
  return (a.avg_ms > b.avg_ms);
};

void Profiler::enable() { Profiler::enabled_ = true; }
void Profiler::disable() { enabled_ = false; }

int64_t Profiler::tictoc(const std::string& name) {
  // this will be std::chrono
  namespace chrono = boost::chrono;

  // TODO alternative where an iterator is returned,
  // to avoid lookup at toc()
  if (!enabled_) { return 0; }
  tictoc_timestamp_t timestamp( boost::chrono::system_clock::now() );
  RunningMap::iterator running_itr(running_.find(name));
  // if entry is nonexistent, or existent but not running,
  // create a new profiler entry and mark it as running
  if (running_itr == running_.end() ||
      !running_itr->second) {
    entries_[name].push_back(ProfilerEntry(timestamp));
    running_[name] = true;
    return 0;
  }
  // if entry is existent and running, stop it.
  ProfilerEntry& entry(entries_[name].back());
  entry.end_time = timestamp;
  running_itr->second = false;

  //chrono::nanoseconds delta_ns(entry.end_time-entry.start_time);

  typedef chrono::duration<int64_t, boost::micro> microseconds;
  microseconds delta_us = chrono::duration_cast<microseconds>(entry.end_time-entry.start_time);
  return delta_us.count();
}

void Profiler::aggregate_entries(std::vector<AggregatedProfilerEntry>* aggregated) {
  // this will be std::chrono
  namespace chrono = boost::chrono;
  if (!enabled_) { return; }
  for (ProfilerEntryMap::iterator itr(entries_.begin()), end_itr(entries_.end());
       itr != end_itr;
       ++itr) {
    AggregatedProfilerEntry ag;
    ag.name = itr->first;
    ag.min_ms = std::numeric_limits<double>::max();
    ag.max_ms = -1;
    ag.total_ms = 0.;
    ProfilerEntries& name_entries(itr->second);
    ag.num_calls = static_cast<int>(name_entries.size());
    BOOST_FOREACH( const ProfilerEntry& entry, name_entries ) {
      // TODO a proper usage of boost::chrono representation
      typedef chrono::duration<double, boost::milli> milliseconds;
      double delta_ms = (chrono::duration_cast<milliseconds>(entry.end_time-entry.start_time)).count();
      ag.total_ms += delta_ms;
      ag.min_ms = std::min(ag.min_ms, delta_ms);
      ag.max_ms = std::max(ag.max_ms, delta_ms);
    }
    ag.avg_ms = ag.total_ms / ag.num_calls;
    aggregated->push_back(ag);
  }
}

void Profiler::print_aggregated(std::ostream& os) {
  if (!enabled_) { return; }
  std::vector<AggregatedProfilerEntry> aggregated;
  aggregate_entries(&aggregated);
  std::sort(aggregated.begin(), aggregated.end(), aggregated_profiler_entry_total_cmp);
  os << "\n\n";
  os << std::setw(40) << std::setfill(' ') << "Description";
  os << std::setw(15) << std::setfill(' ') << "Calls";
  os << std::setw(15) << std::setfill(' ') << "Total ms";
  os << std::setw(15) << std::setfill(' ') << "Avg ms";
  os << std::setw(15) << std::setfill(' ') << "Min ms";
  os << std::setw(15) << std::setfill(' ') << "Max ms";
  os << "\n";
  for (size_t i=0; i < aggregated.size(); ++i) {
    AggregatedProfilerEntry& ag(aggregated[i]);
    os << std::setw(40) << std::setfill(' ') << ag.name;
    os << std::setw(15) << std::setprecision(5) << std::setfill(' ') << ag.num_calls;
    os << std::setw(15) << std::setprecision(5) << std::setfill(' ') << ag.total_ms;
    os << std::setw(15) << std::setprecision(5) << std::setfill(' ') << ag.avg_ms;
    os << std::setw(15) << std::setprecision(5) << std::setfill(' ') << ag.min_ms;
    os << std::setw(15) << std::setprecision(5) << std::setfill(' ') << ag.max_ms;
    os << "\n";
  }
  os << "\n";
}

void Profiler::print_aggregated_csv(std::ostream& os) {
  if (!enabled_) { return; }
  std::vector<AggregatedProfilerEntry> aggregated;
  aggregate_entries(&aggregated);
  std::sort(aggregated.begin(), aggregated.end(), aggregated_profiler_entry_total_cmp);
  os << "description";
  os << ",calls";
  os << ",total_ms";
  os << ",avg_ms";
  os << ",min_ms";
  os << ",max_ms";
  os << "\n";
  BOOST_FOREACH( AggregatedProfilerEntry& ag, aggregated ) {
    os << ag.name;
    os << "," << ag.num_calls;
    os << "," << ag.total_ms;
    os << "," << ag.avg_ms;
    os << "," << ag.min_ms;
    os << "," << ag.max_ms;
    os << "\n";
  }
  os << "\n";
}

void Profiler::print_all(std::ostream& os) {
  if (!enabled_) { return; }
  os << "start_time; ";
  os << "description; ";
  os << "duration";
  os << "\n";
  for (ProfilerEntryMap::iterator itr = entries_.begin();
       itr != entries_.end();
       ++itr) {
    std::string name(itr->first);
    std::vector<ProfilerEntry>& name_entries(itr->second);
    BOOST_FOREACH( const ProfilerEntry& entry, name_entries ) {
      os << entry.start_time << "; ";
      os << name << "; ";
      // TODO what will this do?
      os << (entry.end_time - entry.start_time);
      os << "\n";
    }
  }
}

void Profiler::publish_messages(ros::NodeHandle& nh) {
  namespace chrono = boost::chrono;
  ros::Publisher pub = nh.advertise<tictoc_profiler::ProfilerEntry>("profiler", 20);
  int seq = 0;
  for (ProfilerEntryMap::iterator itr = entries_.begin();
       itr != entries_.end();
       ++itr) {
    std::string name(itr->first);
    std::vector<ProfilerEntry>& name_entries(itr->second);
    BOOST_FOREACH( const ProfilerEntry& entry, name_entries ) {
      tictoc_profiler::ProfilerEntry entry_msg;
      entry_msg.seq = seq++;
      entry_msg.name = name;
      // TODO this is in nanoseconds, not microseconds as before.
      entry_msg.start_time = entry.start_time.time_since_epoch().count();
      entry_msg.end_time = entry.end_time.time_since_epoch().count();
      typedef chrono::duration<double, boost::milli> milliseconds;
      double delta_ms = (chrono::duration_cast<milliseconds>(entry.end_time-entry.start_time)).count();
      entry_msg.delta_time_ms = delta_ms;
      pub.publish(entry_msg);
    }
  }
}

}
