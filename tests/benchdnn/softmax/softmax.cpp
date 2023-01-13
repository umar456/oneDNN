/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <random>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "softmax/softmax.hpp"

namespace softmax {

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    auto dst_d = dnn_mem_t::init_md(
            prb->ndims, prb->dims.data(), prb->ddt, prb->dtag);

    dnnl_alg_kind_t alg_kind = dnnl_softmax_accurate;
    if (prb->alg == LOGSOFTMAX) alg_kind = dnnl_softmax_log;

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    if (prb->dir & FLAG_FWD) {
        auto src_d = dnn_mem_t::init_md(
                prb->ndims, prb->dims.data(), prb->sdt, prb->stag);

        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;

        DNN_SAFE_STATUS(dnnl_softmax_forward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, prop, alg_kind,
                init_pd_args.src_md ? init_pd_args.src_md : src_d, dst_d,
                prb->axis, dnnl_attr));
    } else {
        // Re-create dst_md with source tag if dst was not specified, immitating
        // default value.
        if (prb->dtag == tag::any) {
            dst_d = dnn_mem_t::init_md(
                    prb->ndims, prb->dims.data(), prb->ddt, prb->stag);
        }

        auto diff_src_d = dnn_mem_t::init_md(
                prb->ndims, prb->dims.data(), prb->sdt, tag::any);
        auto diff_dst_d = dnn_mem_t::init_md(
                prb->ndims, prb->dims.data(), prb->ddt, tag::any);

        DNN_SAFE_STATUS(dnnl_softmax_backward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, alg_kind, diff_src_d,
                diff_dst_d, dst_d, prb->axis, init_pd_args.hint, dnnl_attr));
    }

    return dnnl_success;
}

int fill_data_fwd(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    int64_t outer_size = 0, inner_size = 0, axis_size = 0;
    get_sizes(prb, outer_size, inner_size, axis_size);

    // Fill data the way it tests two modes: max_val < 0 and max_val >= 0;
    // Test max_val < 0 by using only negative numbers to check correct max_val
    // subtraction, mostly if library used signed value, not abs.
    // Test max_val >= 0 by exceeding `exp_ovfl_arg` value to check answer
    // does not contain +infinity (nan).
    // Distribute several top-1 values to check softmax works right. Also use
    // bit more top-2 values so they contribute in final exp sum as well. Fill
    // much more values with top-3 to check we apply correct maths for whole
    // input.
    // Filling data such way prevents cancellation error for LOGSOFTMAX due to
    // log(sum(x_j)) won't be close to zero as in case of single top-1 value.

    // Do fixed partitioning to have same filling for any number of threads.
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(outer_size, n_chunks);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, outer_size);
        std::minstd_rand msr(idx_start + 1);
        msr.discard(1);
        std::vector<std::uniform_int_distribution<>> igen_top_fp {
                std::uniform_int_distribution<>(1, 2),
                std::uniform_int_distribution<>(2, 5),
                std::uniform_int_distribution<>(5, 8)};
        std::vector<std::uniform_int_distribution<>> igen_top_int8 {
                std::uniform_int_distribution<>(1, 1),
                std::uniform_int_distribution<>(1, 1),
                std::uniform_int_distribution<>(0, 4)};
        std::vector<std::uniform_int_distribution<>> igen_top
                = dnnl_data_type_size(prb->ddt) == 1 ? igen_top_int8
                                                     : igen_top_fp;
        const int sign = (idx_chunk % 2 != 0 && prb->sdt != dnnl_u8) ? -1 : 1;
        const int exp_ovfl_arg = 88 * sign;
        std::vector<int> top_val {
                exp_ovfl_arg + 2, exp_ovfl_arg + 1, exp_ovfl_arg};

        for_(int64_t idx = idx_start; idx < idx_end; ++idx)
        for (int64_t in = 0; in < inner_size; in++) {
            std::vector<int64_t> n_top {
                    igen_top[0](msr), igen_top[1](msr), igen_top[2](msr)};
            int i = 2;
            int64_t n_sum = n_top[0] + n_top[1] + n_top[2];
            // Adjust number of top elements to fit axis_size if needed
            while (n_sum > axis_size) {
                n_sum -= n_top[i];
                n_top[i] -= std::min(n_top[i], n_sum + n_top[i] - axis_size);
                n_sum += n_top[i];
                i--;
            }
            // If number of top elements is less the axis_size, set a random
            // index to start dense filling from.
            std::uniform_int_distribution<> igen_as_idx(0, axis_size - n_sum);
            msr.discard(2);
            int64_t axis_idx_start = igen_as_idx(msr);

            i = 0;
            for (int64_t as = 0; as < axis_size; as++) {
                auto offset = inner_size * (idx * axis_size + as) + in;
                float value = INT_MIN;
                if (as >= axis_idx_start && as < axis_idx_start + n_sum) {
                    value = top_val[i];
                    n_top[i]--;
                    if (n_top[i] == 0) i++;
                }
                mem_fp.set_elem(offset,
                        round_to_nearest_representable(mem_dt.dt(), value));
            }
        }
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_data_bwd(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, int seed) {
    const auto nelems = mem_fp.nelems();
    const int range = 128;

    // to avoid any cancellation erros it's better to have d_dst and dst of
    // different signs (refer to ref computations).
    // softmax := (d_dst - SUM (d_dst * dst); keep +d_dst and -dst.
    // logsoftmax := d_dst - exp(dst) * SUM (d_dst); keep -d_dst and +dst.
    // seed decides about the sign.
    const float sign = seed % 2 == 0 ? 1.f : -1.f;
    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((11 * i) + 37 + 19 * seed) % range;
        const float value = sign * gen / range;
        mem_fp.set_elem(i, value);
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_scales(
        const attr_t &attr, int arg, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    assert(mem_dt.nelems() == mem_fp.nelems());

    const auto &scales = attr.scales.get(arg);

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);
    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(idx_start + 1);
        int_seed.discard(1);

        std::uniform_int_distribution<> gen(-5, 5);

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            int pow2 = gen(int_seed);
            int pow2_shift = 1 << std::abs(pow2);
            const float gen_val = pow2 < 0 ? (1.f / pow2_shift) : pow2_shift;
            const float fixed_val = scales.scale;
            const float val = nelems == 1 ? fixed_val : gen_val;
            mem_fp.set_elem(idx, val);
        }
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->sdt, prb->ddt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res);
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // See `skip_invalid_inplace` for details.
    if (prb->inplace) {
        skip_invalid_inplace(res, prb->sdt, prb->ddt, prb->stag, prb->dtag);
        if (res->state == SKIPPED) return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    const auto trh_dt = (prb->dir & FLAG_FWD) ? prb->ddt : prb->sdt;
    const float trh_coeff_log = prb->alg == LOGSOFTMAX ? 5 : 1;
    const float trh_coeff_f32 = trh_dt == dnnl_f32 ? 10.f : 1.f;
    const float trh_coeff_bwd = (prb->dir & FLAG_FWD) ? 1.f : 4.f;
    const float trh = trh_coeff_log * trh_coeff_bwd * trh_coeff_f32
            * epsilon_dt(trh_dt);
    cmp.set_threshold(trh);

    const int64_t axis_size = prb->dims[prb->axis];
    const int64_t n_zeros = (prb->ddt == dnnl_s8 || prb->ddt == dnnl_u8)
            ? (axis_size - 1)
            : MAX2(0, axis_size - 8);
    float zero_trust_percent = 100.f * n_zeros / axis_size;
    // Note:
    // * Logsoftmax over axis of size `1` does not make any sense.
    // * Logsoftmax for u8 dst does not make any sense either.
    if (prb->alg == LOGSOFTMAX && (axis_size == 1 || prb->ddt == dnnl_u8))
        zero_trust_percent = 100.f;
    if (prb->dir & FLAG_BWD) zero_trust_percent = 30.f;
    cmp.set_zero_trust_percent(zero_trust_percent);

    const auto softmax_add_check
            = [&](const compare::compare_t::driver_check_func_args_t &args) {
                  // SSE4.1 and OpenCL rdiff tolerance is too high for
                  // certain scenarios.
                  return args.diff < epsilon_dt(args.dt);
              };
    cmp.set_driver_check_function(softmax_add_check);
}

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_DST,
    };
    static const std::vector<int> exec_bwd_args = {
            DNNL_ARG_DST,
            DNNL_ARG_DIFF_SRC,
            DNNL_ARG_DIFF_DST,
    };
    return (dir & FLAG_FWD) ? exec_fwd_args : exec_bwd_args;
};

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res, dir_t dir,
        dnnl_primitive_t prim_ref) {
    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        ref_mem_map.emplace(
                exec_arg, dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_data_fwd(prb, mem, ref_mem), WARN);
                break;
            case DNNL_ARG_DST:
                if (dir & FLAG_BWD) {
                    const bool neg_sign = prb->alg == SOFTMAX ? true : false;
                    SAFE(fill_data_bwd(prb, mem, ref_mem, neg_sign), WARN);
                }
                break;
            case DNNL_ARG_DIFF_DST: {
                const bool neg_sign = prb->alg == SOFTMAX ? true : false;
                SAFE(fill_data_bwd(prb, mem, ref_mem, !neg_sign), WARN);
            } break;
            case DNNL_ARG_SCRATCHPAD: break;
            default: { // Process all attributes here
                bool is_scales_arg = (exec_arg & DNNL_ARG_ATTR_SCALES);
                if (is_scales_arg) {
                    int exec_src_arg = exec_arg ^ DNNL_ARG_ATTR_SCALES;
                    SAFE(fill_scales(prb->attr, exec_src_arg, mem, ref_mem),
                            WARN);
                }
            } break;
        }
        // Don't keep reference memory if it is not used further.
        if (!is_bench_mode(CORR)) ref_mem_map.clear();
    }

    // Drop destination memory for in-place case. `args` will take care of rest.
    const bool inplace_fwd = prb->inplace && (prb->dir & FLAG_FWD);
    const bool inplace_bwd = prb->inplace && (dir & FLAG_BWD);
    if (inplace_fwd) {
        mem_map[DNNL_ARG_DST] = dnn_mem_t();
    } else if (inplace_bwd) {
        mem_map[DNNL_ARG_DIFF_SRC] = dnn_mem_t();
    }

    if (!is_bench_mode(CORR)) return OK;

    // Use inplace reference computation every time.
    if (dir & FLAG_FWD) {
        ref_mem_map.emplace(DNNL_ARG_DST, dnn_mem_t());
    } else {
        ref_mem_map.emplace(DNNL_ARG_DIFF_SRC, dnn_mem_t());
    }

    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb) {
    std::vector<data_kind_t> check_kinds;
    if (prb->dir & FLAG_FWD) {
        check_kinds = {DST};
    } else if (prb->dir & FLAG_BWD) {
        check_kinds = {SRC};
    } else {
        assert(!"unexpected!");
        SAFE_V(FAIL);
    }
    assert(!check_kinds.empty());
    return check_kinds;
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prb->ctx_init, prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
    if (is_bench_mode(INIT)) return OK;

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args<prb_t>(mem_map, prb, prim, supported_exec_args(prb->dir));
    SAFE(init_ref_memory_args(ref_mem_map, mem_map, prim, prb, res, prb->dir),
            WARN);

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (is_bench_mode(CORR)) {
        check_correctness(
                prb, get_kinds_to_check(prb), args, ref_args, setup_cmp, res);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace softmax
