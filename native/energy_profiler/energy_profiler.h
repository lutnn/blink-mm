#ifndef JIANYU_ENERGY_PROFILER_H_
#define JIANYU_ENERGY_PROFILER_H_

#include <atomic>
#include <thread>

namespace jianyu {

class EnergyProfiler {
 public:
  EnergyProfiler(int interval);

  ~EnergyProfiler();

  void Start();

  void Resume();

  void Pause();

  void Stop();

  double GetAvgPower();

  double GetPower();

 private:
  std::thread thread_;

  int count_;

  double total_power_;

  std::atomic<double> power_;

  std::atomic<bool> profiling_;

  std::atomic<bool> finished_;

  int interval_;
};

}  // namespace jianyu

#endif