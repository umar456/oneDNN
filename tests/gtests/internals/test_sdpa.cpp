
/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <dnnl_test_common.hpp>
#include <gtest/gtest.h>

#include <oneapi/dnnl/dnnl.hpp>

#include <oneapi/dnnl/experimental/dnnl_experimental.hpp>

#include <memory>
#include <random>

using std::vector;

struct sdpa_dims_t {
    memory::dim mb;
    memory::dim seq_len;
    memory::dim head_num;
    memory::dim head_size;
    memory::dim query_num;
};

using dnnl::algorithm;
using dnnl::matmul;
using dnnl::memory;
using dnnl::primitive_attr;
using dnnl::softmax_forward;

#define COMPLAIN_DNNL_ERROR_AND_EXIT(what, status) \
    do { \
        printf("[%s:%d] `%s` returns oneDNN error: %s.\n", __FILE__, __LINE__, \
                what, dnnl_status2str(status)); \
        printf("Example failed.\n"); \
        exit(1); \
    } while (0)

#define COMPLAIN_EXAMPLE_ERROR_AND_EXIT(complain_fmt, ...) \
    do { \
        printf("[%s:%d] Error in the example: " complain_fmt ".\n", __FILE__, \
                __LINE__, __VA_ARGS__); \
        printf("Example failed.\n"); \
        exit(2); \
    } while (0)

#undef CHECK
#define CHECK(f) \
    do { \
        dnnl_status_t s_ = f; \
        if (s_ != dnnl_success) COMPLAIN_DNNL_ERROR_AND_EXIT(#f, s_); \
    } while (0)

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (!handle) throw std::runtime_error("handle is nullptr.");

#ifdef DNNL_WITH_SYCL
    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
        if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
            auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
            auto dst = buffer.get_host_access();
            uint8_t *dst_ptr = dst.get_pointer();
            if (!dst_ptr)
                throw std::runtime_error("get_pointer returned nullptr.");
            for (size_t i = 0; i < size; ++i)
                dst_ptr[i] = ((uint8_t *)handle)[i];
        } else {
            assert(mkind == dnnl::sycl_interop::memory_kind::usm);
            uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
            if (!dst_ptr)
                throw std::runtime_error("get_data_handle returned nullptr.");
            if (is_cpu_sycl) {
                for (size_t i = 0; i < size; ++i)
                    dst_ptr[i] = ((uint8_t *)handle)[i];
            } else {
                auto sycl_queue
                        = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                sycl_queue.memcpy(dst_ptr, handle, size).wait();
            }
        }
        return;
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
        void *mapped_ptr = mem.map_data();
        if (mapped_ptr) std::memcpy(mapped_ptr, handle, size);
        mem.unmap_data(mapped_ptr);
        return;
    }
#endif

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
        for (size_t i = 0; i < size; ++i)
            dst[i] = ((uint8_t *)handle)[i];
        return;
    }

    assert(!"not expected");
}

// this is changed from the fill_random() function in matmul_perf.cpp.
void fill_random(std::vector<float> &out) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

    for (size_t i = 0; i < out.size(); i += nrand) {
        size_t chunk = std::min(nrand, out.size() - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

// initialize the mask with first 3/4 elements with 0s and the last 1/4 elements
// with -inf.
void fill_mask(std::vector<float> &mask, size_t seq_len) {
    const size_t pos = seq_len * 3 / 4;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (i % seq_len < pos)
            mask[i] = 0.f;
        else
            mask[i] = -1 * std::numeric_limits<float>::infinity();
    }
}

void print_test_case(memory::data_type dt, const sdpa_dims_t &p) {
    std::cout << '[' << std::setw(4) << dnnl_dt2str(memory::convert_to_c(dt));
    std::cout << " mb = " << p.mb << ", seq_len = " << p.seq_len
              << ", head_num = " << p.head_num
              << ", head_size = " << p.head_size
              << ", query_num = " << p.query_num;
    std::cout << "] " << std::flush;
}

inline dnnl_dim_t product(const std::vector<int64_t> &dims) {
    return dims.empty() ? 0
                        : std::accumulate(dims.begin(), dims.end(),
                                (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
}

vector<float> bench_sdpa_primitives(engine::kind ekind, memory::data_type dt,
        const sdpa_dims_t &p, double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);
    print_test_case(dt, p);

    // Create execution dnnl::engine.
    dnnl::engine eng(ekind, 0);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // Prepare input and output shapes to construct the sdpa graph.
    const memory::dims q_sz = {p.mb, p.head_num, p.query_num, p.head_size};
    const memory::dims k_sz = {p.mb, p.head_num, p.head_size, p.seq_len};
    const memory::dims v_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const memory::dims score_sz = {p.mb, p.head_num, p.query_num, p.seq_len};
    const memory::dims scale_sz = {1, 1, 1, 1};
    const memory::dims mask_sz = {p.mb, 1, p.query_num, p.seq_len};

    // score = query x key.T
    // scaled_score = score / scale
    // masked_score = scaled_score + mask
    // All combined in a single matmul primitive.
    auto query_md = memory::desc(q_sz, dt, memory::format_tag::abcd);
    auto key_md = memory::desc(k_sz, dt, memory::format_tag::abdc);
    auto score_md = memory::desc(score_sz, dt, memory::format_tag::abcd);
    auto scale_md = memory::desc(scale_sz, dt, memory::format_tag::abcd);
    auto mask_md = memory::desc(mask_sz, dt, memory::format_tag::abcd);

    dnnl::primitive_attr bmm1_attr;
    bmm1_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    dnnl::post_ops bmm1_po;
    bmm1_po.append_binary(algorithm::binary_div, scale_md);
    bmm1_po.append_binary(algorithm::binary_add, mask_md);
    bmm1_attr.set_post_ops(bmm1_po);

    auto bmm1_pd = matmul::primitive_desc(
            eng, query_md, key_md, score_md, bmm1_attr);
    auto bmm1_prim = matmul(bmm1_pd);

    // attention_probs = softmax(masked_score)
    primitive_attr softmax_attr;
    softmax_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto softmax_pd = softmax_forward::primitive_desc(eng,
            dnnl::prop_kind::forward_inference, algorithm::softmax_accurate,
            score_md, score_md, /* axis = */ score_md.get_ndims() - 1,
            softmax_attr);
    auto softmax_prim = softmax_forward(softmax_pd);

    // attention_output = attention_probs x value
    auto value_md = memory::desc(v_sz, dt, memory::format_tag::abcd);
    auto output_md = memory::desc(q_sz, dt, memory::format_tag::abcd);
    primitive_attr bmm2_attr;
    bmm2_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto bmm2_pd = matmul::primitive_desc(
            eng, score_md, value_md, output_md, bmm2_attr);
    auto bmm2_prim = matmul(bmm2_pd);

    // Create memory objects
    auto m_query = memory(query_md, eng);
    auto m_key = memory(key_md, eng);
    auto m_scale = memory(scale_md, eng);
    auto m_mask = memory(mask_md, eng);
    auto m_value = memory(value_md, eng);
    auto m_output = memory(output_md, eng);

    // Allocate user data.
    std::vector<float> query_data(product(q_sz));
    std::vector<float> key_data(product(k_sz));
    std::vector<float> scale_data(product(scale_sz), std::sqrt(p.head_size));
    std::vector<float> mask_data(product(mask_sz));
    std::vector<float> value_data(product(v_sz));
    std::vector<float> output_data(product(q_sz));

    fill_random(query_data);
    fill_random(key_data);
    fill_random(value_data);
    fill_mask(mask_data, static_cast<size_t>(p.seq_len));

    // Write data to tensor object's handle.
    write_to_dnnl_memory(query_data.data(), m_query);
    write_to_dnnl_memory(key_data.data(), m_key);
    write_to_dnnl_memory(scale_data.data(), m_scale);
    write_to_dnnl_memory(mask_data.data(), m_mask);
    write_to_dnnl_memory(value_data.data(), m_value);

    size_t max_scratchpad_size = 0;
    auto bmm1_scratchpad = bmm1_pd.scratchpad_desc().get_size();
    auto softmax_scratchpad = softmax_pd.scratchpad_desc().get_size();
    auto bmm2_scratchpad = bmm2_pd.scratchpad_desc().get_size();
    for (auto &sz : {bmm1_scratchpad, softmax_scratchpad, bmm2_scratchpad}) {
        if (max_scratchpad_size < sz) max_scratchpad_size = sz;
    }
    auto scratchpad_md
            = memory::desc({static_cast<memory::dim>(max_scratchpad_size)},
                    memory::data_type::u8, memory::format_tag::a);

    // allocate intermediate memory
    auto m_score = memory(score_md, eng);
    auto m_scratchpad = memory(scratchpad_md, eng);

    const auto loop = [&]() {
        // each primitive will use all threads
        bmm1_prim.execute(strm,
                {{DNNL_ARG_SRC, m_query}, {DNNL_ARG_WEIGHTS, m_key},
                        {DNNL_ARG_DST, m_score},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                m_scale},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                m_mask},
                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});

        softmax_prim.execute(strm,
                {{DNNL_ARG_SRC, m_score}, {DNNL_ARG_DST, m_score},
                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});

        bmm2_prim.execute(strm,
                {{DNNL_ARG_SRC, m_score}, {DNNL_ARG_WEIGHTS, m_value},
                        {DNNL_ARG_DST, m_output},
                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop();

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    loop();
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return {};

    // Timing runs.
    static const int min_runs = 4;
    const int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++)
        loop();
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    double avg_time = (duration.count() - dur_first.count()) / runs;
    std::cout << "primitive runs: " << runs + 1 << "; ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;
}

TEST(SDPA, SDPA) {

    sdpa_dims_t p = {32, 384, 16, 64, 384};
    memory::data_type dt = memory::data_type::f16;
    memory::data_type kdt = memory::data_type::f16; //memory::data_type::s4;
    memory::data_type vdt = memory::data_type::f16; //memory::data_type::s8;
    memory::data_type scale_dt = memory::data_type::f16;
    bool invert_scale = false;

    // Create execution dnnl::engine.
    dnnl::engine eng(engine::kind::gpu, 0);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // Prepare input and output shapes to construct the sdpa graph.
    const memory::dims q_sz = {p.mb, p.head_num, p.query_num, p.head_size};
    const memory::dims k_sz = {p.mb, p.head_num, p.head_size, p.seq_len};
    const memory::dims v_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const memory::dims score_sz = {p.mb, p.head_num, p.query_num, p.seq_len};
    const memory::dims scale_sz = {1, 1, 1, 1};
    const memory::dims mask_sz = {p.mb, 1, p.query_num, p.seq_len};

    // score = query x key.T
    // scaled_score = score / scale
    // masked_score = scaled_score + mask
    // All combined in a single matmul primitive.
    auto query_md = memory::desc(q_sz, dt, memory::format_tag::abcd);
    auto key_md = memory::desc(k_sz, kdt, memory::format_tag::abdc);
    auto score_md = memory::desc(score_sz, dt, memory::format_tag::abcd);
    auto scale_md = memory::desc(scale_sz, dt, memory::format_tag::abcd);
    auto mask_md = memory::desc(mask_sz, dt, memory::format_tag::abcd);
    auto value_md = memory::desc(v_sz, vdt, memory::format_tag::abcd);
    auto output_md = memory::desc(q_sz, dt, memory::format_tag::abcd);

    // Create memory objects
    auto m_query = memory(query_md, eng);
    auto m_key = memory(key_md, eng);
    auto m_scale = memory(scale_md, eng);
    auto m_mask = memory(mask_md, eng);
    auto m_value = memory(value_md, eng);
    auto m_output = memory(output_md, eng);

    // Allocate user data.
    std::vector<float> query_data(product(q_sz));
    std::vector<float> key_data(product(k_sz));
    std::vector<float> scale_data(product(scale_sz), std::sqrt(p.head_size));
    std::vector<float> mask_data(product(mask_sz));
    std::vector<float> value_data(product(v_sz));
    std::vector<float> output_data(product(q_sz));

    fill_random(query_data);
    fill_random(key_data);
    fill_random(value_data);
    fill_mask(mask_data, static_cast<size_t>(p.seq_len));

    // Write data to tensor object's handle.
    write_to_dnnl_memory(query_data.data(), m_query);
    write_to_dnnl_memory(key_data.data(), m_key);
    write_to_dnnl_memory(scale_data.data(), m_scale);
    write_to_dnnl_memory(mask_data.data(), m_mask);
    write_to_dnnl_memory(value_data.data(), m_value);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);

    using namespace dnnl::experimental;
    auto sdpa_pd = sdpa::primitive_desc(eng, query_md, key_md, value_md,
            &mask_md, scale_dt, output_md, invert_scale, 1, attr);
    auto sdpa_p = sdpa(sdpa_pd);

    sdpa_p.execute(strm,
            {{DNNL_ARG_QUERIES, m_query}, {DNNL_ARG_KEYS, m_key},
                    {DNNL_ARG_VALUES, m_value}, {DNNL_ARG_SCALE, m_scale},
                    {DNNL_ARG_ATTN_MASK, m_mask}, {DNNL_ARG_DST, m_output}});
    strm.wait();
}
