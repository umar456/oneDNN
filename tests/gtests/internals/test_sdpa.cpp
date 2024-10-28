
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

#include <common/sdpa.hpp>

#include <memory>
#include <random>

using std::vector;

struct sdpa_dims_t {
    memory::dim mb;
    memory::dim seq_len;
    memory::dim head_num;
    memory::dim head_size;
    memory::dim query_num;
    int group_size;
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
template <typename T>
inline void write_to_dnnl_memory(const T *handle, dnnl::memory &mem,
        primitive_attr *attr = nullptr, dnnl::memory *scale_attr = nullptr) {
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
        if (mem.get_desc().get_data_type() != dnnl_f32
                && std::is_same<T, float>::value) {
            dnnl::stream s(eng);
            memory mem_f32_mem(
                    {mem.get_desc().get_dims(), memory::data_type::f32,
                            mem.get_desc().get_strides()},
                    eng);
            write_to_dnnl_memory<float>(
                    (const float *)handle, mem_f32_mem, attr, scale_attr);

            if (attr) {
                dnnl::reorder(mem_f32_mem, mem, *attr)
                        .execute(s,
                                {{DNNL_ARG_SRC, mem_f32_mem},
                                        {DNNL_ARG_DST, mem},
                                        {DNNL_ARG_DST | DNNL_ARG_ATTR_SCALES,
                                                *scale_attr}});
            } else {
                dnnl::reorder(mem_f32_mem, mem).execute(s, mem_f32_mem, mem);
            }

            s.wait();
        } else {
            // PC: this branch is identical to the one above
            void *mapped_ptr = mem.map_data();
            if (mapped_ptr) std::memcpy(mapped_ptr, handle, size);
            mem.unmap_data(mapped_ptr);
        }
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

template <typename T>
void fill_random_quantized(std::vector<T> &out) {
    static std::vector<T> random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_int_distribution<int> dist_f(-4, 4);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

    for (size_t i = 0; i < out.size(); i += nrand) {
        size_t chunk = std::min(nrand, out.size() - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

void fill_random_scales(std::vector<float> &out) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_int_distribution<int> dist_f(-16, 16);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator) * 0.25f;
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
    std::cout << " mb(n) = " << p.mb << ", seq_len(k) = " << p.seq_len
              << ", head_num = " << p.head_num
              << ", head_size(d/v) = " << p.head_size
              << ", query_num(q) = " << p.query_num
              << ", group_size = " << p.group_size;
    std::cout << "] " << std::flush;
}

void print_mem(const dnnl::memory &mem, std::string name = "") {

    auto desc = mem.get_desc();
    auto dims = desc.get_dims();
    auto strides = desc.get_strides();
    printf("%sbegin\n", name.c_str());
    printf("dims   : %zu %6ld %6ld %6ld %6ld\n", dims.size(), dims[0], dims[1],
            dims[2], dims[3]);
    printf("strides: %zu %6ld %6ld %6ld %6ld\n", strides.size(), strides[0],
            strides[1], strides[2], strides[3]);
    void *mapped_ptr_ = (void *)mem.map_data();
    printf("        i:    ");
    for (int i = 0; i < dims[3]; i++) {

        switch ((int)desc.get_data_type()) {

            case dnnl_s4: printf("%2d", i); break;
            case dnnl_s8: printf("%4d", i); break;
            case dnnl_f16: printf("%8d", i); break;
        }
    }
    printf("\n");

    switch ((int)desc.get_data_type()) {

        case dnnl_s4: {
            char *mapped_ptr = (char *)mem.map_data();
            for (int l = 0; l < dims[0]; l++) {
                for (int k = 0; k < dims[1]; k++) {
                    for (int j = 0; j < dims[2]; j++) {
                        printf("(%2d, %2d, %3d): ", l, k, j);
                        for (int i = 0; i < dims[3] / 2; i++) {
                            auto offset = l * strides[0] + k * strides[1]
                                    + j * strides[2] / 2 + i * strides[3];
                            printf("%2d%2d", (mapped_ptr[offset] & 0x0f),
                                    ((mapped_ptr[offset] & 0xf0) >> 4));
                        }
                        printf("\n");
                    }
                }
            }
        } break;

        case dnnl_s8: {
            char *mapped_ptr = (char *)mapped_ptr_;
            for (int l = 0; l < dims[0]; l++) {
                for (int k = 0; k < dims[1]; k++) {
                    for (int j = 0; j < dims[2]; j++) {
                        printf("(%2d, %2d, %3d): ", l, k, j);
                        for (int i = 0; i < dims[3]; i++) {
                            printf("%4d",
                                    (mapped_ptr[l * strides[0] + k * strides[1]
                                            + j * strides[2]
                                            + i * strides[3]]));
                        }
                        printf("\n");
                    }
                }
            }
        } break;
        case dnnl_f16: {
            using dnnl::impl::float16_t;
            float16_t *mapped_ptr = (float16_t *)mapped_ptr_;

            for (int l = 0; l < dims[0]; l++) {
                for (int k = 0; k < dims[1]; k++) {
                    for (int j = 0; j < dims[2]; j++) {
                        printf("(%2d, %2d, %3d): ", l, k, j);
                        for (int i = 0; i < dims[3]; i++) {
                            printf("%+1.4f ",
                                    (mapped_ptr[l * strides[0] + k * strides[1]
                                            + j * strides[2] + i * strides[3]]
                                                    .f()));
                        }
                        printf("\n");
                    }
                }
            }
        } break;
        default: throw std::runtime_error("Not supported");
    }
    mem.unmap_data(mapped_ptr_);
    printf("%send\n", name.c_str());
}

inline dnnl_dim_t product(const std::vector<int64_t> &dims) {
    return dims.empty() ? 0
                        : std::accumulate(dims.begin(), dims.end(),
                                (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
}

struct sdpa_tensors {
    memory m_query, m_key, m_scale, m_mask, m_value, m_output;
    memory m_key_quantized, m_value_quantized, m_output_quantized;

    memory m_reorder_scale_attr, m_key_scales, m_key_zp, m_value_scales,
            m_value_zp;
    dnnl::primitive_attr reorder_attr, sdpa_attr, sdpa_attr_quantized,
            sdpa_kq_attr_quantized, sdpa_vs_attr_quantized;
};

vector<float> dequantize(const vector<float> &input,
        const vector<int> &zero_points, const vector<float> &scales,
        int group_size) {

    int groups = zero_points.size();
    vector<float> out(input.size());
    for (int g = 0; g < groups; g++)
        for (int i = g * group_size; i < g * group_size + group_size; i++)
            out[i] = (input[i] - zero_points[g]) * scales[g];
    return out;
}

sdpa_tensors get_descriptors(dnnl::engine &eng, sdpa_dims_t p,
        memory::data_type dt, memory::data_type kdt, memory::data_type vdt) {

    sdpa_tensors out;

    print_test_case(kdt, p);
    std::cout << std::endl;

    // Prepare input and output shapes to construct the sdpa graph.
    const memory::dims q_sz = {p.mb, p.head_num, p.query_num, p.head_size};
    const memory::dims k_sz = {p.mb, p.head_num, p.head_size, p.seq_len};
    const memory::dims v_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const memory::dims score_sz = {p.mb, p.head_num, p.query_num, p.seq_len};
    const memory::dims scale_sz = {1, 1, 1, 1};
    const memory::dims key_scales_sz
            = {k_sz[0], k_sz[1], k_sz[2] / p.group_size, k_sz[3]};
    const memory::dims val_scales_sz
            = {v_sz[0], v_sz[1], v_sz[2], v_sz[3] / p.group_size};
    const memory::dims mask_sz = {p.mb, 1, p.query_num, p.seq_len};

    // score = query x key.T
    // scaled_score = score / scale
    // masked_score = scaled_score + mask
    // All combined in a single matmul primitive.
    auto query_md = memory::desc(q_sz, dt, memory::format_tag::abcd);
    auto key_md = memory::desc(k_sz, dt, memory::format_tag::abdc);
    auto value_md = memory::desc(v_sz, dt, memory::format_tag::abcd);
    auto score_md = memory::desc(score_sz, dt, memory::format_tag::abcd);
    auto scale_md = memory::desc(scale_sz, dt, memory::format_tag::abcd);
    auto reorder_scale_attr_md = memory::desc(
            key_scales_sz, memory::data_type::f32, memory::format_tag::abcd);
    auto key_quantized_md = memory::desc(k_sz, kdt, memory::format_tag::abdc);
    auto key_scales_md = memory::desc(
            key_scales_sz, memory::data_type::f16, memory::format_tag::abdc);
    auto key_zp_md = memory::desc(key_scales_sz, kdt, memory::format_tag::abdc);
    auto val_quantized_md = memory::desc(v_sz, vdt, memory::format_tag::abcd);
    auto val_scales_md = memory::desc(
            val_scales_sz, memory::data_type::f16, memory::format_tag::abcd);
    auto val_zp_md = memory::desc(val_scales_sz, vdt, memory::format_tag::abcd);
    auto mask_md = memory::desc(mask_sz, dt, memory::format_tag::abcd);
    auto output_md = memory::desc(q_sz, dt, memory::format_tag::abcd);

    // Create memory objects
    out.m_query = memory(query_md, eng);
    out.m_key = memory(key_md, eng);
    out.m_scale = memory(scale_md, eng);
    out.m_reorder_scale_attr = memory(reorder_scale_attr_md, eng);
    out.m_key_quantized = memory(key_quantized_md, eng);
    out.m_key_scales = memory(key_scales_md, eng);
    out.m_key_zp = memory(key_zp_md, eng);
    out.m_value_quantized = memory(val_quantized_md, eng);
    out.m_value_scales = memory(val_scales_md, eng);
    out.m_value_zp = memory(val_zp_md, eng);
    out.m_mask = memory(mask_md, eng);
    out.m_value = memory(value_md, eng);
    out.m_output = memory(output_md, eng);
    out.m_output_quantized = memory(output_md, eng);

    // Allocate user data.
    std::vector<float> query_data(product(q_sz));
    std::vector<float> scale_data(product(scale_sz), std::sqrt(p.head_size));
    std::vector<float> reorder_key_scale_data(product(key_scales_sz), 1.f);
    std::vector<float> key_quantized_data(product(k_sz));
    std::vector<float> val_quantized_data(product(v_sz));
    std::vector<float> key_scale_data(product(key_scales_sz), 1.f);
    std::vector<float> val_scale_data(product(val_scales_sz), 1.f);
    std::vector<int> key_zp_data(product(key_scales_sz), 0);
    std::vector<int> val_zp_data(product(val_scales_sz), 0);
    std::vector<float> mask_data(product(mask_sz));
    std::vector<float> output_data(product(q_sz));

    out.reorder_attr.set_scales_mask(DNNL_ARG_DST, 0);
    //out.reorder_attr.set_scales(DNNL_ARG_DST,
    //(1 << 0) + (1 << 1) + (1 << 2) + (1 << 3), {1, 1, group_size, 1},
    //memory::data_type::f32);

    out.sdpa_attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);

    out.sdpa_attr_quantized.set_scratchpad_mode(dnnl::scratchpad_mode::library);
    //out.sdpa_kq_attr_quantized.set_scales(DNNL_ARG_WEIGHTS, 1 << 3, {1, 1, p.group_size, 1}, memory::data_type::f16);
    out.sdpa_vs_attr_quantized.set_scales(DNNL_ARG_WEIGHTS, 1 << 3,
            {1, 1, 1, p.group_size}, memory::data_type::f16);
    //out.sdpa_kq_attr_quantized.set_zero_points(DNNL_ARG_WEIGHTS, 1 << 3, {1, 1, p.group_size, 1}, memory::data_type::s8);
    out.sdpa_vs_attr_quantized.set_zero_points(DNNL_ARG_WEIGHTS, 1 << 3,
            {1, 1, 1, p.group_size}, memory::data_type::s8);

    fill_random(query_data);
    fill_random_quantized(key_quantized_data);
    fill_random_quantized(val_quantized_data);
    //fill_random_scales(key_scale_data);
    fill_random_scales(val_scale_data);
    //fill_random_quantized(key_zp_data);
    fill_random_quantized(val_zp_data);
    fill_mask(mask_data, static_cast<size_t>(p.seq_len));

#if 1
    auto &Q = query_data;
    auto &K = key_quantized_data;
    auto &V = val_quantized_data;
    auto &Ks = key_scale_data;
    auto &Vs = val_scale_data;
    auto &Kz = key_zp_data;
    auto &Vz = val_zp_data;
    auto d = p.head_size;
    auto k = p.seq_len;
    auto q = p.query_num;

    int kr = -1, kc = -1, qr = -1, qc = -1, vr = -1, vc = -1, xb = 0;
    int ksr = -1, ksc = -1, kzr = -1, kzc = -1, vsr = -1, vsc = -1, vzr = -1,
        vzc = -1;
    if (getenv("KR")) kr = atoi(getenv("KR"));
    if (getenv("KC")) kc = atoi(getenv("KC"));
    if (getenv("KSR")) ksr = atoi(getenv("KSR"));
    if (getenv("KSC")) ksc = atoi(getenv("KSC"));
    if (getenv("KZR")) kzr = atoi(getenv("KZR"));
    if (getenv("KZC")) kzc = atoi(getenv("KZC"));
    if (getenv("QR")) qr = atoi(getenv("QR"));
    if (getenv("QC")) qc = atoi(getenv("QC"));
    if (getenv("VR")) vr = atoi(getenv("VR"));
    if (getenv("VC")) vc = atoi(getenv("VC"));
    if (getenv("VSR")) vsr = atoi(getenv("VSR"));
    if (getenv("VSC")) vsc = atoi(getenv("VSC"));
    if (getenv("VZR")) vzr = atoi(getenv("VZR"));
    if (getenv("VZC")) vzc = atoi(getenv("VZC"));
    if (getenv("XB")) xb = atoi(getenv("XB"));

    if (kr >= 0 || kc >= 0) {
        kr = std::max(kr, 0);
        kc = std::max(kc, 0);
        if (getenv("KX")) {
            for (int kr_ = 0; kr_ < k; kr_++)
                for (int kc_ = 0; kc_ < d; kc_++)
                    if (kr_ >= kr || kc_ >= kc) K[kr_ * d + kc_] = 0;
        } else {
            for (auto &k : K)
                k = 0;
            K[xb * d * k + kr * d + kc] = 1;
        }
    }
    if (ksr >= 0 || ksc >= 0) {
        ksr = std::max(ksr, 0);
        ksc = std::max(ksc, 0);
        for (auto &ks : Ks)
            ks = 0;
        Ks[(xb * d * k + ksr * d) / p.group_size + ksc] = 1;
    }
    if (kzr >= 0 || kzc >= 0) {
        kzr = std::max(kzr, 0);
        kzc = std::max(kzc, 0);
        for (auto &kz : Kz)
            kz = 0;
        Kz[(xb * d * k + kzr * d) / p.group_size + kzc] = 2;
    }
    if (qr >= 0 || qc >= 0) {
        qr = std::max(qr, 0);
        qc = std::max(qc, 0);
        if (getenv("QX")) {
            for (int qr_ = 0; qr_ < d; qr_++)
                for (int qc_ = 0; qc_ < q; qc_++)
                    if (qr_ >= qr || qc_ >= qc) Q[qr_ + qc_ * d] = 0;
        } else {
            for (auto &q : Q)
                q = 0;
            Q[xb * d * q + qr + qc * d] = 1;
        }
    }
    if (vr >= 0 || vc >= 0) {
        vr = std::max(vr, 0);
        vc = std::max(vc, 0);
        if (getenv("VX")) {
            for (int vr_ = 0; vr_ < d; vr_++)
                for (int vc_ = 0; vc_ < k; vc_++)
                    if (vr_ >= vr || vc_ >= vc) V[vr_ + vc_ * d] = 0;
        } else {
            for (auto &v : V)
                v = 0;
            V[xb * d * k + vr + vc * d] = 1;
        }
    }
    if (vsr >= 0 || vsc >= 0) {
        vsr = std::max(vsr, 0);
        vsc = std::max(vsc, 0);
        for (auto &vs : Vs)
            vs = 0;
        Vs[(xb * d * k + vsc * d) / p.group_size + vsr] = 1;
    }
    if (vzr >= 0 || vzc >= 0) {
        vzr = std::max(vzr, 0);
        vzc = std::max(vzc, 0);
        for (auto &vz : Vz)
            vz = 0;
        Vz[(xb * d * k + vzc * d) / p.group_size + vzr] = 1;
    }
#endif

    printf("quantized: ");
    for (int i = 0; i < 10; i++) {
        printf("%8.2f", key_quantized_data[i]);
    }
    printf("\nscales: ");
    for (int i = 0; i < 10; i++) {
        printf("%8.2f", key_scale_data[i]);
    }
    printf("\nzp: ");
    for (int i = 0; i < 10; i++) {
        printf("%4d", key_zp_data[i]);
    }
    auto key_data = dequantize(
            key_quantized_data, key_zp_data, key_scale_data, p.group_size);
    printf("\ndequantized: ");
    for (int i = 0; i < 10; i++) {
        printf("%8.2f", key_data[i]);
    }
    printf("\n");
    auto value_data = dequantize(
            val_quantized_data, val_zp_data, val_scale_data, p.group_size);

    write_to_dnnl_memory(mask_data.data(), out.m_mask);
    write_to_dnnl_memory(scale_data.data(), out.m_scale);

    // Write data to tensor object's handle.
    write_to_dnnl_memory(key_data.data(), out.m_key);
    write_to_dnnl_memory(value_data.data(), out.m_value);
    write_to_dnnl_memory(query_data.data(), out.m_query);
    //write_to_dnnl_memory(reorder_key_scale_data.data(), out.m_reorder_scale_attr);

    write_to_dnnl_memory(key_quantized_data.data(), out.m_key_quantized);
    write_to_dnnl_memory(val_quantized_data.data(), out.m_value_quantized);
    write_to_dnnl_memory(key_zp_data.data(), out.m_key_zp);
    write_to_dnnl_memory(val_zp_data.data(), out.m_value_zp);
    write_to_dnnl_memory(key_scale_data.data(), out.m_key_scales);
    write_to_dnnl_memory(val_scale_data.data(), out.m_value_scales);
    //print_mem(out.m_key_zp, "key_zp_data");
    //print_mem(out.m_key, "key");
    //print_mem(out.m_query, "query");

    //print_mem(out.m_query, "query");
    //print_mem(out.m_key, "key");
    //print_mem(out.m_value, "value");

    return out;
}
sdpa_dims_t p = {.mb = 1,
        .seq_len = 128, // k
        .head_num = 1,
        .head_size = 64, // d
        .query_num = 128, // q
        .group_size = 32};

TEST(SDPA, compares8tof16) {

#if 1
    if (getenv("B")) p.mb = atoi(getenv("B"));
    if (getenv("H")) p.head_num = atoi(getenv("H"));
    if (getenv("D")) p.head_size = atoi(getenv("D"));
    if (getenv("S")) p.seq_len = p.query_num = atoi(getenv("S"));
    if (getenv("Q")) p.query_num = atoi(getenv("Q"));
#endif

    // Create execution dnnl::engine.
    dnnl::engine eng(engine::kind::gpu, 0);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    memory::data_type dt = memory::data_type::f16;
    memory::data_type kdt = memory::data_type::f16;
    memory::data_type vdt = memory::data_type::s8;
    memory::data_type scale_dt = memory::data_type::f16;
    bool invert_scale = false;

    sdpa_tensors t = get_descriptors(eng, p, dt, kdt, vdt);

    using namespace dnnl::experimental;
    auto mask = t.m_mask.get_desc();
    auto sdpas8_pd = sdpa::primitive_desc(eng, t.m_query.get_desc(),
            t.m_key_quantized.get_desc(), t.m_value_quantized.get_desc(), &mask,
            scale_dt, t.m_output_quantized.get_desc(), invert_scale, 1,
            t.sdpa_attr_quantized, t.sdpa_kq_attr_quantized,
            t.sdpa_vs_attr_quantized);
    auto sdpas8_p = sdpa(sdpas8_pd);

    auto sdpaf16_pd = sdpa::primitive_desc(eng, t.m_query.get_desc(),
            t.m_key.get_desc(), t.m_value.get_desc(), &mask, scale_dt,
            t.m_output.get_desc(), invert_scale, 1, t.sdpa_attr);
    auto sdpaf16_p = sdpa(sdpaf16_pd);

    //print_mem(t.m_key_scales, "key_scale_attr");
    //print_mem(t.m_key_zp, "key_zero_points");
    //print_mem(t.m_value_scales, "value_scale_attr");
    //print_mem(t.m_value_zp, "value_zero_points");

#if 1
    if (!::getenv("SKIP_S8"))
#endif
        sdpas8_p.execute(strm,
                {{DNNL_ARG_QUERIES, t.m_query},
                        {DNNL_ARG_KEYS, t.m_key_quantized},
                        //{DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS, t.m_key_scales},
                        //{DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS, t.m_key_zp},

                        {DNNL_ARG_VALUES, t.m_value_quantized},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES,
                                t.m_value_scales},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES,
                                t.m_value_zp},

                        {DNNL_ARG_SCALE, t.m_scale},
                        {DNNL_ARG_ATTN_MASK, t.m_mask},
                        {DNNL_ARG_DST, t.m_output_quantized}});
#if 1
    if (!::getenv("SKIP_F16"))
#endif
        sdpaf16_p.execute(strm,
                {{DNNL_ARG_QUERIES, t.m_query}, {DNNL_ARG_KEYS, t.m_key},
                        {DNNL_ARG_VALUES, t.m_value},
                        {DNNL_ARG_SCALE, t.m_scale},
                        {DNNL_ARG_ATTN_MASK, t.m_mask},
                        {DNNL_ARG_DST, t.m_output}});
    strm.wait();
    //print_mem(t.m_output, "output");

#if 1
    if (::getenv("SKIP_CHECK")) return;
#endif

    float16_t *mapped_ptr_f16 = (float16_t *)t.m_output.map_data();
    float16_t *mapped_ptr_s8 = (float16_t *)t.m_output_quantized.map_data();

    auto dims = t.m_output.get_desc().get_dims();
    auto strides = t.m_output.get_desc().get_strides();

    int mismatches = 0;
    for (int l = 0; l < dims[0]; l++) {
        for (int k = 0; k < dims[1]; k++) {
            for (int j = 0; j < dims[2]; j++) {
                for (int i = 0; i < dims[3]; i++) {
                    auto offset = l * strides[0] + k * strides[1]
                            + j * strides[2] + i * strides[3];
                    auto o_f16 = mapped_ptr_f16[offset].f();
                    auto o_s8 = mapped_ptr_s8[offset].f();
                    if (o_f16 != o_s8 && mismatches++ < 20)
                        fprintf(stderr,
                                "Mismatch at (%d,%d,%d,%d): computed %f vs. "
                                "%f\n",
                                l, k, j, i, o_s8, o_f16);
                }
            }
        }
    }

    t.m_output.unmap_data(mapped_ptr_f16);
    t.m_output_quantized.unmap_data(mapped_ptr_s8);
    ASSERT_EQ(mismatches, 0);
}
