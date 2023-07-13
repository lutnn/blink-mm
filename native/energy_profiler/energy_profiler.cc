#include "energy_profiler/energy_profiler.h"

#include <fstream>
#include <string>

namespace jianyu {

namespace {

const std::string USB_CURRENT = "/sys/class/power_supply/usb/current_now";
const std::string USB_CURRENT_FALLBACK =
    "/sys/class/power_supply/usb/input_current_now";
const std::string USB_VOLTAGE = "/sys/class/power_supply/usb/voltage_now";
const std::string BAT_CURRENT = "/sys/class/power_supply/battery/current_now";
const std::string BAT_VOLTAGE = "/sys/class/power_supply/battery/voltage_now";

int ReadFromFile(std::string filename) {
  std::ifstream fs(filename);
  std::string content;
  fs >> content;
  if (content.empty()) {
    return ReadFromFile(USB_CURRENT_FALLBACK);
  }
  return std::stoi(content);
}

double ReadPower() {
  double usb_current = ReadFromFile(USB_CURRENT);
  double usb_voltage = ReadFromFile(USB_VOLTAGE);
  double bat_current = ReadFromFile(BAT_CURRENT);
  double bat_voltage = ReadFromFile(BAT_VOLTAGE);

  return usb_current / 1000000. * usb_voltage / 1000000. +
         bat_current / 1000000. * bat_voltage / 1000000.;
}

}  // namespace

EnergyProfiler::EnergyProfiler(int interval)
    : count_(0),
      total_power_(.0),
      power_(.0),
      profiling_(false),
      finished_(false),
      interval_(interval) {
  thread_ = std::thread([this]() {
    while (true) {
      if (finished_.load()) {
        return;
      }
      if (profiling_.load()) {
        count_ += 1;
        double power = ReadPower();
        total_power_ += power;
        power_.store(power);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(interval_));
    }
  });
}

EnergyProfiler::~EnergyProfiler() { Stop(); }

void EnergyProfiler::Start() { profiling_.store(true); }

void EnergyProfiler::Resume() { Start(); }

void EnergyProfiler::Pause() { profiling_.store(false); }

void EnergyProfiler::Stop() {
  finished_.store(true);
  thread_.join();
}

double EnergyProfiler::GetAvgPower() {
  return total_power_ / (count_ == 0 ? 1 : count_);
}

double EnergyProfiler::GetPower() { return power_.load(); }

}  // namespace jianyu