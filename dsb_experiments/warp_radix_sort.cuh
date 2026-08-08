#pragma once

#include <cuda_runtime.h>
#include <cub/block/radix_rank_sort_operations.cuh>

// WarpRadixSort: stable LSD radix sort for a warp of 32 threads.
// All 32 threads in the warp must call Sort cooperatively with identical arguments.
// Only RadixBits<=5 is supported (NUM_BINS must not exceed WARP_THREADS=32).
template <typename KeyT, int KeyBits = sizeof(KeyT) * 8, int RadixBits = 5>
class WarpRadixSort
{
    static constexpr unsigned int FULL_MASK = 0xffffffffu;
    static constexpr int WARP_THREADS       = 32;
    static constexpr int NUM_BINS           = 1 << RadixBits;
    static constexpr int NUM_PASSES         = (KeyBits + RadixBits - 1) / RadixBits;
    static_assert(NUM_BINS <= WARP_THREADS,
        "WarpRadixSort: RadixBits must be <= 5 (NUM_BINS must not exceed WARP_THREADS=32)");

    using traits                 = cub::detail::radix::traits_t<KeyT>;
    using BitsT                  = typename traits::bit_ordered_type;
    using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;

    static __device__ __forceinline__ BitsT ToBits(KeyT k) {
        return bit_ordered_conversion::to_bit_ordered(
            cub::detail::identity_decomposer_t{}, ::cuda::std::bit_cast<BitsT>(k));
    }

    static __device__ __forceinline__ KeyT FromBits(BitsT b) {
        b = bit_ordered_conversion::from_bit_ordered(
            cub::detail::identity_decomposer_t{}, b);
        return ::cuda::std::bit_cast<KeyT>(b);
    }

    static __device__ __forceinline__
    unsigned int ExtractDigit(BitsT key, int pass)
    {
        return (key >> (pass * RadixBits)) & (unsigned int)(NUM_BINS - 1);
    }

    // Narrows mask by one radix bit: ANDs with ballot if bit==1, ~ballot if bit==0.
    static __device__ __forceinline__
    unsigned int UpdateBinMask(unsigned int mask, unsigned int ballot, unsigned int bit)
    {
        return mask & (bit ? ballot : ~ballot);
    }

public:
    // Construct a WarpRadixSort without any shared memory.
    __device__ WarpRadixSort() {}

    // Sort keys, using tmp_keys as scratch space.
    // Sorted results end up in keys.
    __device__ void Sort(KeyT *keys, KeyT *tmp_keys, int num_items)
    {
        const int lane_id       = threadIdx.x & 31;
        const int n_items_padded = (num_items + WARP_THREADS - 1) & ~(WARP_THREADS - 1);

        // Phase 1: single scan over all items, building all NUM_PASSES histograms.
        // hist[p] accumulates the count of items whose pass-p digit equals lane_id.
        int hist[NUM_PASSES];
        #pragma unroll
        for (int p = 0; p < NUM_PASSES; p++)
            hist[p] = 0;

        for (int i = lane_id; i < n_items_padded; i += WARP_THREADS) {
            const bool valid              = (i < num_items);
            BitsT raw_key = valid ? ToBits(keys[i]) : BitsT(0);
            const unsigned int valid_mask = __ballot_sync(FULL_MASK, valid);
            #pragma unroll
            for (int p = 0; p < NUM_PASSES; p++) {
                unsigned int digit    = ExtractDigit(raw_key, p);
                unsigned int bin_mask = valid_mask;
                #pragma unroll
                for (int b = 0; b < RadixBits; b++) {
                    unsigned int ballot = __ballot_sync(FULL_MASK, valid & (digit >> b) & 1u);
                    bin_mask = UpdateBinMask(bin_mask, ballot, (unsigned int)(lane_id >> b) & 1u);
                }
                if (lane_id < NUM_BINS) hist[p] += __popc(bin_mask);
            }
        }

        // Phase 2: exclusive warp prefix scan per pass using __shfl_up_sync.
        // After this, hist[p] in lane l = exclusive prefix sum for bin l in pass p.
        #pragma unroll
        for (int p = 0; p < NUM_PASSES; p++) {
            int val = hist[p];
            #pragma unroll
            for (int offset = 1; offset < WARP_THREADS; offset <<= 1) {
                int n = __shfl_up_sync(FULL_MASK, val, offset);
                val = (lane_id >= offset) ? val + n : val;
            }
            hist[p] = val - hist[p];
        }

        // Phase 3: scatter passes with double buffering.
        // hist[p] is reused as a running scatter pointer for bin lane_id in pass p.
        // Intermediate stores decode back to KeyT so each pass loads valid KeyT values.
        for (int p = 0; p < NUM_PASSES; p++) {
            KeyT *read_ptr  = (p % 2 == 0) ? keys : tmp_keys;
            KeyT *write_ptr = (p % 2 == 0) ? tmp_keys : keys;

            for (int i = lane_id; i < n_items_padded; i += WARP_THREADS) {
                const bool valid              = (i < num_items);
                BitsT raw_key      = valid ? ToBits(read_ptr[i]) : BitsT(0);
                unsigned int digit = ExtractDigit(raw_key, p);
                const unsigned int valid_mask = __ballot_sync(FULL_MASK, valid);

                // Single loop builds both masks: same_mask (for within_rank) and
                // bin_mask (for advancing the scatter pointer for bin == lane_id).
                unsigned int same_mask = valid_mask;
                unsigned int bin_mask  = valid_mask;
                #pragma unroll
                for (int b = 0; b < RadixBits; b++) {
                    unsigned int ballot = __ballot_sync(FULL_MASK, valid & (digit >> b) & 1u);
                    same_mask = UpdateBinMask(same_mask, ballot, (digit    >> b) & 1u);
                    bin_mask  = UpdateBinMask(bin_mask,  ballot, (unsigned int)(lane_id >> b) & 1u);
                }

                // Rank of this lane among all lanes with the same digit.
                int within_rank = __popc(same_mask & ((1u << lane_id) - 1));

                // Fetch the current scatter base from the lane that owns this digit's bin.
                int base = __shfl_sync(FULL_MASK, hist[p], digit);

                if (valid) write_ptr[base + within_rank] = FromBits(raw_key);

                // Advance scatter pointer for the bin this lane owns (bin == lane_id).
                if (lane_id < NUM_BINS) hist[p] += __popc(bin_mask);
            }
        }

        // If NUM_PASSES is odd the sorted data landed in the original tmp_keys buffer.
        if (NUM_PASSES % 2 == 1) {
            for (int i = lane_id; i < num_items; i += WARP_THREADS)
                keys[i] = tmp_keys[i];
        }
    }
};
