#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include "energy_profiler/energy_profiler.h"

std::atomic<bool> stop(false);

void Profile(float interval_ms, const std::string &csv_path) {
  int interval_us = interval_ms * 1e3;

  std::unique_ptr<jianyu::EnergyProfiler> energy_profiler;
  energy_profiler.reset(new jianyu::EnergyProfiler(0));

  auto fp = fopen(csv_path.c_str(), "w");

  fprintf(fp, "time_s,power_w\n");
  fflush(fp);

  energy_profiler->Start();

  auto start = std::chrono::high_resolution_clock::now();

  while (!stop.load()) {
    double power = energy_profiler->GetPower();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    double time_s =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() /
        1e6;
    fprintf(fp, "%.6lf,%.6lf\n", time_s, power);
    fflush(fp);

    std::this_thread::sleep_for(std::chrono::microseconds(interval_us));
  }

  energy_profiler.reset(nullptr);
  fclose(fp);
}

int main(int argc, char **argv) {
  auto thread = std::thread(Profile, 10, argv[1]);

  while (1) {
    std::string s;
    std::cin >> s;
    if (s == "stop") {
      break;
    }
  }

  stop.store(true);
  thread.join();
  return 0;
}