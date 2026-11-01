#ifndef HECATE_SUPPORT_COMPILATIONPROGRESS_H
#define HECATE_SUPPORT_COMPILATIONPROGRESS_H

#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <mutex>

namespace hecate {
bool compilationProgressEnabled();

// Opt-in, wall-clock progress; no IR or plaintext payloads are printed.
class CompilationProgress {
  using Clock = std::chrono::steady_clock;
  const char *name;
  bool enabled;
  Clock::time_point start = Clock::now(), last = start;
  inline static std::mutex outputMutex;

public:
  explicit CompilationProgress(const char *name, bool active = true)
      : name(name), enabled(active && compilationProgressEnabled()) {
    if (enabled)
      write("begin", 0, 0);
  }
  ~CompilationProgress() {
    if (enabled)
      write("end", 0, 0);
  }
  void update(uint64_t done, uint64_t total, bool force = false) {
    if (!enabled)
      return;
    auto now = Clock::now();
    if (force || now - last >= std::chrono::seconds(5)) {
      last = now;
      write("progress", done, total);
    }
  }

private:
  void write(const char *event, uint64_t done, uint64_t total) {
    std::lock_guard<std::mutex> lock(outputMutex);
    llvm::errs() << "[compile] " << name << ' ' << event << " elapsed_s="
                 << std::chrono::duration<double>(Clock::now() - start).count();
    if (total)
      llvm::errs() << " completed=" << done << '/' << total;
    llvm::errs() << '\n';
  }
};
} // namespace hecate
#endif
