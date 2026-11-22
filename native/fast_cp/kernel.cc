// Small-observable prediction path derived from PyMatching v2.4.0, Apache-2.0.
// Upstream's C++ implementation is compiled as a separate translation unit.
#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

// Internal helpers from the pinned mwpm_decoding.cc.
void process_timeline_until_completion(pm::Mwpm&, const std::vector<uint64_t>&);
pm::MatchingResult shatter_blossoms_for_all_detection_events_and_extract_obs_mask_and_weight(
    pm::Mwpm&, const std::vector<uint64_t>&);

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
thread_local std::string last_error;
struct Decoder {
    pm::Mwpm graph;
    size_t detectors, observables;
    std::vector<uint64_t> hits;
    std::vector<int64_t> edges;
    std::vector<uint8_t> bits;
    std::vector<uint8_t> zero_result;
    bool zero_cached = false;
    explicit Decoder(const stim::DetectorErrorModel& dem)
        : graph(pm::detector_error_model_to_mwpm(dem, pm::NUM_DISTINCT_WEIGHTS, true, true)),
          detectors(dem.count_detectors()), observables(dem.count_observables()), bits(observables),
          zero_result((observables+7)/8) {}

    void solve(uint8_t* out, bool fast) {
        if (!fast || observables > 64) {
            std::fill(bits.begin(), bits.end(), 0);
            pm::total_weight_int weight = 0;
            pm::decode_detection_events(graph, hits, bits.data(), weight, /*edge_correlations=*/true);
            for (size_t k=0; k<observables; ++k)
                out[k/8] |= bits[k] << (k%8);
            return;
        }
        edges.clear();
        pm::decode_detection_events_to_edges(graph, hits, edges);
        auto& weights = graph.flooder.graph;
        try {
            weights.reweight_for_edges(edges);
            // Up to 64 observables are accumulated directly by the matching flooder.
            // No search-flooder weight is read between here and restoration.
            process_timeline_until_completion(graph, hits);
            auto res = shatter_blossoms_for_all_detection_events_and_extract_obs_mask_and_weight(graph, hits);
            auto& negative = graph.flooder.negative_weight_detection_events;
            if (!negative.empty())
                res += shatter_blossoms_for_all_detection_events_and_extract_obs_mask_and_weight(graph, negative);
            auto mask = res.obs_mask ^ graph.flooder.negative_weight_obs_mask;
            for (size_t k=0; k<(observables+7)/8; ++k)
                out[k] = static_cast<uint8_t>(mask >> (8*k));
        } catch (...) {
            weights.undo_reweights();
            throw;
        }
        weights.undo_reweights();
    }
};
}
EXPORT const char* cp_error() { return last_error.c_str(); }
EXPORT void* cp_create(const char* text) {
    try { return new Decoder(stim::DetectorErrorModel(text)); }
    catch (const std::exception& ex) { last_error = ex.what(); return nullptr; }
}
EXPORT void cp_destroy(void* ptr) { delete static_cast<Decoder*>(ptr); }
EXPORT int cp_decode(void* ptr, const uint8_t* shots, size_t rows, size_t width,
                     uint8_t* out, size_t out_width, int fast, int cache_zero) {
    try {
        auto& d = *static_cast<Decoder*>(ptr);
        if (width != (d.detectors+7)/8 || out_width != (d.observables+7)/8)
            throw std::invalid_argument("incorrect packed width");
        for (size_t i=0; i<rows; ++i) {
            d.hits.clear();
            for (size_t j=0; j<width; ++j) {
                unsigned byte = shots[i*width+j];
                while (byte) {
                    size_t id = j*8 + std::countr_zero(byte);
                    if (id >= d.detectors) throw std::invalid_argument("nonzero padding");
                    d.hits.push_back(id);
                    byte &= byte-1;
                }
            }
            uint8_t* dest = out+i*out_width;
            if (cache_zero && d.hits.empty() && d.zero_cached) {
                std::copy(d.zero_result.begin(), d.zero_result.end(), dest);
            } else {
                std::fill(dest, dest+out_width, 0);
                d.solve(dest, fast);
                if (cache_zero && d.hits.empty()) {
                    std::copy(dest, dest+out_width, d.zero_result.begin());
                    d.zero_cached = true;
                }
            }
        }
        return 0;
    } catch (const std::exception& ex) { last_error = ex.what(); return -1; }
}
