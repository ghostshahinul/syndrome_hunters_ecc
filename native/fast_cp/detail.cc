// Instrument the two-pass <=64-observable Sparse Blossom call on real shots.
// This separate profiling library is never loaded into the throughput benchmark.
#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"
#include <bit>
#include <chrono>
#include <string>

void process_timeline_until_completion(pm::Mwpm&, const std::vector<uint64_t>&);
pm::MatchingResult shatter_blossoms_for_all_detection_events_and_extract_obs_mask_and_weight(
    pm::Mwpm&, const std::vector<uint64_t>&);

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;
static thread_local std::string error;
EXPORT const char* cp_detail_error() { return error.c_str(); }
EXPORT int cp_detail(const char* model, const uint8_t* rows, size_t shots, size_t width,
                     uint8_t* pred, double* sums, uint64_t* counts) {
    try {
        stim::DetectorErrorModel dem(model);
        if (dem.count_observables() != 1 || width != (dem.count_detectors()+7)/8)
            throw std::invalid_argument("profiler expects one observable and packed detector rows");
        auto g = pm::detector_error_model_to_mwpm(dem, pm::NUM_DISTINCT_WEIGHTS, true, true);
        for (size_t i=0; i<shots; ++i) {
            std::vector<uint64_t> hits;
            for (size_t j=0; j<width; ++j) {
                unsigned b = rows[i*width+j];
                while (b) { hits.push_back(j*8+std::countr_zero(b)); b &= b-1; }
            }
            auto start = Clock::now();
            std::vector<int64_t> edges;
            pm::decode_detection_events_to_edges(g, hits, edges);
            auto first = Clock::now();
            g.flooder.graph.reweight_for_edges(edges);
            auto flood = Clock::now();
            g.search_flooder.graph.reweight_for_edges(edges);
            auto search = Clock::now();
            process_timeline_until_completion(g, hits);
            auto res = shatter_blossoms_for_all_detection_events_and_extract_obs_mask_and_weight(g, hits);
            if (!g.flooder.negative_weight_detection_events.empty())
                res += shatter_blossoms_for_all_detection_events_and_extract_obs_mask_and_weight(
                    g, g.flooder.negative_weight_detection_events);
            pred[i] = (res.obs_mask ^ g.flooder.negative_weight_obs_mask) & 1;
            auto second = Clock::now();
            counts[0] += edges.size()/2;
            counts[1] += g.flooder.graph.previous_weights.size();
            counts[2] += g.search_flooder.graph.previous_weights.size();
            counts[3] += g.flooder.graph.previous_weights.empty();
            g.flooder.graph.undo_reweights();
            g.search_flooder.graph.undo_reweights();
            auto end = Clock::now();
            sums[0] += Seconds(first-start).count();
            sums[1] += Seconds(flood-first).count();
            sums[2] += Seconds(search-flood).count();
            sums[3] += Seconds(second-search).count();
            sums[4] += Seconds(end-second).count();
        }
        return 0;
    } catch (const std::exception& ex) { error = ex.what(); return -1; }
}
