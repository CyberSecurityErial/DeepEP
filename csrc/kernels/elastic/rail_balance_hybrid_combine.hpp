#pragma once

#include <climits>
#include <memory>

#include <nccl.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/layout.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

// Isolated force-v1 Hybrid combine specialization. The public CombineRuntime
// and its recursive include hash remain untouched.
struct RailBalanceHybridCombineSpec {
    int num_sms;
    int num_scaleup_warps;
    int num_forward_warps;
    int num_scaleout_ranks;
    int num_scaleup_ranks;
    int hidden;
    int num_max_tokens_per_rank;
    int num_experts;
    int num_topk;
    int num_qps;
    int64_t num_timeout_cycles;
    int num_smem_bytes;
};

class RailBalanceHybridCombineRuntime final:
    public jit::LaunchRuntime<RailBalanceHybridCombineRuntime> {
public:
    struct Args {
        RailBalanceHybridCombineSpec spec;

        // Legacy Hybrid combine ABI.
        nv_bfloat16* x;
        float* topk_weights;
        int* src_metadata;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* token_metadata_at_forward;
        int* channel_linked_list;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* buffer;
        void* workspace;

        // Precomputed base of the registered symmetric proxy-return arena.
        // Gate #2 validates every p against Pcap before dispatch publication;
        // the committed kernel intentionally carries neither plan nor status.
        void* rail_balance_proxy_return_base;

        int scaleout_rank_idx;
        int scaleup_rank_idx;
        int num_reduced_tokens;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        const auto& s = args.spec;
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_hybrid_combine.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_hybrid_combine_impl<
            false, true, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
            s.num_sms,
            s.num_scaleup_warps,
            s.num_forward_warps,
            s.num_scaleout_ranks,
            s.num_scaleup_ranks,
            s.hidden,
            s.num_max_tokens_per_rank,
            s.num_experts,
            s.num_topk,
            s.num_qps,
            s.num_timeout_cycles);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.x, args.topk_weights,
            args.src_metadata,
            args.psum_num_recv_tokens_per_scaleup_rank,
            args.token_metadata_at_forward,
            args.channel_linked_list,
            args.nccl_dev_comm, args.nccl_window,
            args.buffer, args.workspace,
            args.rail_balance_proxy_return_base,
            args.scaleout_rank_idx, args.scaleup_rank_idx,
            args.num_reduced_tokens));
    }
};

static void validate_rail_balance_hybrid_combine_spec(
        const RailBalanceHybridCombineSpec& spec) {
    EP_HOST_ASSERT(spec.num_sms >= 2);
    EP_HOST_ASSERT(spec.num_scaleup_warps > 0 and
                   spec.num_scaleup_warps == spec.num_forward_warps);
    EP_HOST_ASSERT(spec.num_scaleout_ranks >= 2 and
                   spec.num_scaleout_ranks <= 32);
    EP_HOST_ASSERT(spec.num_scaleup_ranks >= 2 and
                   spec.num_scaleup_ranks <= 32);
    EP_HOST_ASSERT(spec.hidden > 0 and spec.hidden % 256 == 0 and
                   spec.hidden <= INT_MAX /
                       static_cast<int>(sizeof(nv_bfloat16)));
    EP_HOST_ASSERT(spec.num_max_tokens_per_rank > 0);
    EP_HOST_ASSERT(spec.num_experts > 0 and
                   spec.num_experts %
                       (spec.num_scaleout_ranks * spec.num_scaleup_ranks) == 0);
    const int num_ranks =
        spec.num_scaleout_ranks * spec.num_scaleup_ranks;
    EP_HOST_ASSERT(num_ranks <= layout::WorkspaceLayout::kNumMaxRanks);
    EP_HOST_ASSERT(
        spec.num_experts <= layout::WorkspaceLayout::kNumMaxExperts);
    EP_HOST_ASSERT(
        spec.num_experts / num_ranks <=
            layout::WorkspaceLayout::kNumMaxExpertsPerRank);
    EP_HOST_ASSERT(spec.num_topk >= 1 and spec.num_topk <= 32);
    EP_HOST_ASSERT(spec.num_qps > 0);
    EP_HOST_ASSERT(spec.num_timeout_cycles > 0);
    EP_HOST_ASSERT(spec.num_smem_bytes > 0);
    EP_HOST_ASSERT(
        static_cast<int64_t>(spec.num_scaleout_ranks) *
            spec.num_scaleup_ranks * spec.num_max_tokens_per_rank <= INT_MAX);
    EP_HOST_ASSERT(
        static_cast<int64_t>(spec.num_sms) * spec.num_forward_warps <=
            deep_ep::kNumMaxChannels);
    EP_HOST_ASSERT(
        (static_cast<int64_t>(spec.num_scaleup_warps) +
         spec.num_forward_warps) * 32 <= 1024);

    const auto token_layout = layout::TokenLayout(
        spec.hidden * sizeof(nv_bfloat16), 0, spec.num_topk, false);
    EP_HOST_ASSERT(
        (static_cast<int64_t>(spec.num_scaleup_warps) +
         spec.num_forward_warps) * token_layout.get_num_bytes<true>() <=
            spec.num_smem_bytes);
}

static RailBalanceHybridCombineRuntime::Args
make_rail_balance_hybrid_combine_args(
        const RailBalanceHybridCombineSpec& spec) {
    validate_rail_balance_hybrid_combine_spec(spec);
    const int num_threads =
        (spec.num_scaleup_warps + spec.num_forward_warps) * 32;
    return {
        .spec = spec,
        .x = nullptr,
        .topk_weights = nullptr,
        .src_metadata = nullptr,
        .psum_num_recv_tokens_per_scaleup_rank = nullptr,
        .token_metadata_at_forward = nullptr,
        .channel_linked_list = nullptr,
        .nccl_dev_comm = {nullptr},
        .nccl_window = nullptr,
        .buffer = nullptr,
        .workspace = nullptr,
        .rail_balance_proxy_return_base = nullptr,
        .scaleout_rank_idx = 0,
        .scaleup_rank_idx = 0,
        .num_reduced_tokens = 0,
        // Preserve the legacy Hybrid combine clustered/cooperative contract.
        .launch_args = jit::LaunchArgs(
            spec.num_sms, num_threads, spec.num_smem_bytes,
            2 - (spec.num_sms % 2), true),
    };
}

static std::shared_ptr<jit::KernelRuntime>
prepare_rail_balance_hybrid_combine(
        const RailBalanceHybridCombineSpec& spec) {
    const auto args = make_rail_balance_hybrid_combine_args(spec);
    const auto key = fmt::format(
        "rail_balance_hybrid_combine_force_v1_"
        "sm{}_sw{}_fw{}_d{}_g{}_h{}_m{}_e{}_k{}_q{}_t{}",
        spec.num_sms, spec.num_scaleup_warps, spec.num_forward_warps,
        spec.num_scaleout_ranks, spec.num_scaleup_ranks,
        spec.hidden, spec.num_max_tokens_per_rank,
        spec.num_experts, spec.num_topk,
        spec.num_qps, spec.num_timeout_cycles);
    return jit::compiler->build(
        key, RailBalanceHybridCombineRuntime::generate(args));
}

// Probe the immutable legacy root without touching CombineRuntime::generate:
// its function-static recursive include hash must not be initialized by a
// synthetic geometry.
class RailBalanceHybridLegacyCombineProbeRuntime final:
    public jit::LaunchRuntime<
        RailBalanceHybridLegacyCombineProbeRuntime> {
public:
    struct Args {
        RailBalanceHybridCombineSpec spec;
    };

    static std::string generate_impl(const Args& args) {
        const auto& s = args.spec;
        return fmt::format(R"(
#include <deep_ep/impls/hybrid_combine.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &hybrid_combine_impl<
            false, true, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
            s.num_sms,
            s.num_scaleup_warps,
            s.num_forward_warps,
            s.num_scaleout_ranks,
            s.num_scaleup_ranks,
            s.hidden,
            s.num_max_tokens_per_rank,
            s.num_experts,
            s.num_topk,
            s.num_qps,
            s.num_timeout_cycles);
    }
};

static std::shared_ptr<jit::KernelRuntime>
prepare_rail_balance_hybrid_legacy_combine_probe(
        const RailBalanceHybridCombineSpec& spec) {
    validate_rail_balance_hybrid_combine_spec(spec);
    const RailBalanceHybridLegacyCombineProbeRuntime::Args args = {
        .spec = spec,
    };
    const auto key = fmt::format(
        "rail_balance_hybrid_combine_legacy_probe_v1_"
        "sm{}_sw{}_fw{}_d{}_g{}_h{}_m{}_e{}_k{}_q{}_t{}",
        spec.num_sms, spec.num_scaleup_warps, spec.num_forward_warps,
        spec.num_scaleout_ranks, spec.num_scaleup_ranks,
        spec.hidden, spec.num_max_tokens_per_rank,
        spec.num_experts, spec.num_topk,
        spec.num_qps, spec.num_timeout_cycles);
    return jit::compiler->build(
        key, RailBalanceHybridLegacyCombineProbeRuntime::generate(args));
}

// Compile-only local gate. A one-node topology cannot truthfully launch these
// scaleout>1 Rail/Gin specializations.
static pybind11::dict rail_balance_hybrid_combine_codegen_test(
        const int& num_scaleout_ranks,
        const int& num_scaleup_ranks,
        const int& hidden,
        const int& num_topk,
        const int& num_sms,
        const int& num_channels_per_sm,
        const int& num_experts) {
    EP_HOST_ASSERT(num_scaleout_ranks > 0 and num_scaleup_ranks > 0);
    EP_HOST_ASSERT(num_scaleout_ranks <= INT_MAX / num_scaleup_ranks);
    const int num_ranks = num_scaleout_ranks * num_scaleup_ranks;
    EP_HOST_ASSERT(num_experts > 0 and num_experts % num_ranks == 0);

    const RailBalanceHybridCombineSpec spec = {
        .num_sms = num_sms,
        .num_scaleup_warps = num_channels_per_sm,
        .num_forward_warps = num_channels_per_sm,
        .num_scaleout_ranks = num_scaleout_ranks,
        .num_scaleup_ranks = num_scaleup_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = 8192,
        .num_experts = num_experts,
        .num_topk = num_topk,
        .num_qps = 9,
        .num_timeout_cycles = 1000000000ll,
        .num_smem_bytes = jit::device_runtime->get_num_smem_bytes(),
    };
    const auto force_args = make_rail_balance_hybrid_combine_args(spec);
    const auto force_code =
        RailBalanceHybridCombineRuntime::generate(force_args);
    const auto force_runtime = prepare_rail_balance_hybrid_combine(spec);
    EP_HOST_ASSERT(force_runtime != nullptr);

    const RailBalanceHybridLegacyCombineProbeRuntime::Args legacy_args = {
        .spec = spec,
    };
    const auto legacy_code =
        RailBalanceHybridLegacyCombineProbeRuntime::generate(legacy_args);
    const auto legacy_runtime =
        prepare_rail_balance_hybrid_legacy_combine_probe(spec);
    EP_HOST_ASSERT(legacy_runtime != nullptr);

    const auto token_layout = layout::TokenLayout(
        hidden * sizeof(nv_bfloat16), 0, num_topk, false);
    pybind11::dict result;
    result["code"] = force_code;
    result["legacy_code"] = legacy_code;
    result["num_threads"] = force_args.launch_args.num_threads;
    result["num_channels"] = num_sms * num_channels_per_sm;
    result["num_forward_metadata_dims"] = 3 + 2 * num_topk;
    result["combine_token_bytes"] = token_layout.get_num_bytes<false>();
    result["use_expanded_layout"] = false;
    result["allow_multiple_reduction"] = true;
    result["direct_proxy_return_base"] = true;
    return result;
}

static void register_rail_balance_hybrid_combine_apis(
        pybind11::module_& m) {
    m.def(
        "_rail_balance_hybrid_combine_codegen_test",
        &rail_balance_hybrid_combine_codegen_test,
        pybind11::arg("num_scaleout_ranks"),
        pybind11::arg("num_scaleup_ranks"),
        pybind11::arg("hidden"),
        pybind11::arg("num_topk"),
        pybind11::arg("num_sms") = 64,
        pybind11::arg("num_channels_per_sm") = 4,
        pybind11::arg("num_experts") = 256);
}

}  // namespace deep_ep::elastic
