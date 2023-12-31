// https://github.com/apache/tvm/blob/main/src/relay/quantize/calibrate.cc

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

std::vector<float> SmoothDistribution(const std::vector<float>& p,
                                      const float eps = 0.0001) {
  std::vector<size_t> is_zeros(p.size());
  std::vector<size_t> is_nonzeros(p.size());
  {
    auto it = p.begin();
    std::generate(is_zeros.begin(), is_zeros.end(),
                  [&it]() { return static_cast<size_t>(*(it++) == 0.f); });
  }
  {
    auto it = p.begin();
    std::generate(is_nonzeros.begin(), is_nonzeros.end(),
                  [&it]() { return static_cast<size_t>(*(it++) != 0.f); });
  }
  size_t n_zeros = std::accumulate(is_zeros.begin(), is_zeros.end(), 0);
  size_t n_nonzeros = p.size() - n_zeros;
  if (!n_nonzeros) {
    // The discrete probability distribution is malformed. All entries are 0.
    return std::vector<float>();
  }
  float eps1 =
      eps * static_cast<float>(n_zeros) / static_cast<float>(n_nonzeros);
  if (eps1 >= 1.0) return std::vector<float>();
  auto ret = p;
  for (size_t i = 0; i < p.size(); i++) {
    ret[i] += eps * is_zeros[i] - eps1 * is_nonzeros[i];
  }
  return ret;
}

float ComputeEntropy(float* p, float* q, size_t size) {
  float p_sum = std::accumulate(p, p + size, 0.f);
  float q_sum = std::accumulate(q, q + size, 0.f);
  float ret = 0;
  for (size_t i = 0; i < size; i++) {
    p[i] /= p_sum;
    q[i] /= q_sum;
    if (p[i] && q[i]) ret += p[i] * std::log(p[i] / q[i]);
  }
  return ret;
}

float MinimizeKL(const std::vector<int>& hist,
                 const std::vector<float>& hist_edges, int num_bins,
                 int num_quantized_bins) {
  const int zero_bin_idx = num_bins / 2;
  const int num_half_quantized_bins = num_quantized_bins / 2;
  std::vector<float> thresholds(num_bins / 2 + 1 - num_quantized_bins / 2, 0.f);
  std::vector<float> divergence(thresholds.size(), 0.f);
  std::vector<float> quantized_bins(num_quantized_bins, 0);
  for (int i = num_quantized_bins / 2; i < zero_bin_idx + 1; ++i) {
    const int p_bin_idx_start = zero_bin_idx - i;
    const int p_bin_idx_stop = zero_bin_idx + i + 1;
    thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop];

    std::vector<int> sliced_nd_hist(p_bin_idx_stop - p_bin_idx_start);
    std::vector<float> p(sliced_nd_hist.size());
    p[0] = 0;
    p.back() = 0;
    for (int j = 0; j < num_bins; j++) {
      if (j <= p_bin_idx_start) {
        p[0] += hist[j];
      } else if (j >= p_bin_idx_stop) {
        p.back() += hist[j];
      } else {
        sliced_nd_hist[j - p_bin_idx_start] = hist[j];
        p[j - p_bin_idx_start] = hist[j];
      }
    }
    // calculate how many bins should be merged to generate quantized
    // distribution q
    const auto num_merged_bins = sliced_nd_hist.size() / num_quantized_bins;
    for (int j = 0; j < num_quantized_bins; j++) {
      const int start = j * num_merged_bins;
      const int stop = (j + 1) * num_merged_bins;
      quantized_bins[j] = std::accumulate(sliced_nd_hist.begin() + start,
                                          sliced_nd_hist.begin() + stop, 0);
    }
    quantized_bins.back() += std::accumulate(
        sliced_nd_hist.begin() +
            static_cast<int>(num_quantized_bins * num_merged_bins),
        sliced_nd_hist.end(), 0);
    // expand quantized_bins into p.size bins
    std::vector<float> q(sliced_nd_hist.size(), 0);
    for (int j = 0; j < num_quantized_bins; j++) {
      const int start = j * num_merged_bins;
      const int stop = (j == num_quantized_bins - 1)
                           ? q.size()
                           : ((j + 1) * num_merged_bins);
      int norm = std::count_if(sliced_nd_hist.begin() + start,
                               sliced_nd_hist.begin() + stop,
                               [](size_t i) { return i != 0; });
      if (norm) {
        for (int k = start; k < stop; k++) {
          if (p[k]) q[k] = quantized_bins[j] / norm;
        }
      }
    }
    p = SmoothDistribution(p);
    q = SmoothDistribution(q);

    if (!q.size()) {
      divergence[i - num_half_quantized_bins] =
          std::numeric_limits<float>::infinity();
    } else {
      divergence[i - num_half_quantized_bins] =
          ComputeEntropy(p.data(), q.data(), p.size());
    }
  }
  auto min_divergence_idx =
      std::distance(divergence.begin(),
                    std::min_element(divergence.begin(), divergence.end()));
  return thresholds[min_divergence_idx];
}

PYBIND11_MODULE(blink_mm_native_lib, m) { m.def("minimize_kl", &MinimizeKL); }