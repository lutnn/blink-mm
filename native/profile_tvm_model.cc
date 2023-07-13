#include <dlpack/dlpack.h>
#include <stdint.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <chrono>
#include <thread>

int64_t Memory() {
  rusage ret;
  getrusage(RUSAGE_SELF, &ret);
  return ret.ru_maxrss;
}

int main(int argc, char** argv) {
  int num_cores = std::thread::hardware_concurrency();

  DLDevice dev{kDLCPU, num_cores - 1};
  tvm::runtime::Module mod_factory =
      tvm::runtime::Module::LoadFromFile(argv[1]);
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  int times;
  if (argc >= 3) {
    if (std::string(argv[2]) == "inf") {
      times = std::numeric_limits<int>::max();
    } else {
      times = std::stoi(argv[2]);
    }
  } else {
    times = 100;
  }
  for (int i = 0; i < times; i++) {
    run();
  }

  std::cout << Memory() << std::endl;

  return 0;
}