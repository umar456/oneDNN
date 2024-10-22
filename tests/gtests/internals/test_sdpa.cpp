
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
template<typename T>
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
      if (mem.get_desc().get_data_type() != dnnl_f32 && std::is_same<T, float>::value) {
            dnnl::stream s(eng);
            memory mem_f32_mem(
                    {mem.get_desc().get_dims(), memory::data_type::f32,
                            memory::format_tag::abcd},
                    eng);
            write_to_dnnl_memory<float>((const float*)handle, mem_f32_mem, attr, scale_attr);

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
      } else if (( mem.get_desc().get_data_type() == dnnl_f32 && std::is_same<T, float>::value )
              || (mem.get_desc().get_data_type() == dnnl_s32 && std::is_same<T, int>::value)) {
          void *mapped_ptr = mem.map_data();
          if (mapped_ptr) std::memcpy(mapped_ptr, handle, size);
          mem.unmap_data(mapped_ptr);
      } else {
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

    memory m_reorder_scale_attr, m_key_scales, m_key_zp, m_value_scales,
            m_value_zp;
    dnnl::primitive_attr reorder_attr, sdpa_attr;
};

void quantize(int &zero_point, float &scale, const float *begin, const float *end, float *obegin, int utype_max, int stype_max) {
    float min = std::numeric_limits<float>::infinity();
    float max = -1 * std::numeric_limits<float>::infinity();
    float sum = 0;
    const float *start = begin;
    while (start != end) {
        float val = *start;
        min = std::min(val, min);
        max = std::max(val, max);
        sum += val;
        start++;
    }
    scale = (max - min) / utype_max;
    zero_point = 0;//(min / -scale) - stype_max;
    start = begin;
    while (start != end) {
        *obegin = round( (*start / scale) + zero_point );
        obegin++;
        start++;
    }
}

vector<float> dequantize(vector<float> &input, vector<int> &zero_points,
                         vector<float> &scales, int group_size, int offset = -1) {

    int groups = zero_points.size();
    if (offset == -1) {
        vector<float> out(input.size());
        for (int g = 0; g < groups; g++) {
            for (int i = g * group_size; i < g * group_size + group_size; i++) {
                out[i] = (input[i] - zero_points[g]) * scales[g];
            }
        }
        return out;
    } else {
        vector<float> out(group_size);
        int g = offset / group_size;
        for (int i = offset, ii = 0; i < offset + group_size; i++, ii++) {
            out[ii] = (input[i] - zero_points[g]) * scales[g];
        }
        return out;
    }
}


void quantize_and_write(dnnl::memory &out, dnnl::memory &out_scales, dnnl::memory &out_zp,
                        const dnnl::memory::desc &md, const vector<float> &data, int group_size) {
    switch ((int)md.get_data_type()) {
    case dnnl_s4: {
        vector<float> quantized(data.size());
        vector<int> zero_points(data.size() / group_size);
        vector<float> scales(data.size() / group_size);
        for (int i = 0; i < data.size(); i += group_size) {

            printf("\noriginal:    ");
            for (int ii = i; ii < i + group_size; ii++) {
                printf("%f ", data[ii]);
            }
            quantize(zero_points[i / group_size], scales[i / group_size],
                    &data[i], &data[i + group_size], &quantized[i], 15, 8);
            printf("\nquantize: ");
            for (int ii = i; ii < i + group_size; ii++) {
                printf("%f ", quantized[ii]);
            }
            printf("\nzero_points: ");
            for (int ii = i / group_size; ii < i / group_size + 1; ii++) {
                printf("%d ", zero_points[ii]);
            }
            printf("\nscales: ");
            for (int ii = i / group_size; ii < i / group_size + 1; ii++) {
                printf("%f ", scales[ii]);
            }
            auto de = dequantize(quantized, zero_points, scales, group_size, i);
            printf("\ndequantized: ");
            for (int i = 0; i < de.size(); i++) {
              printf("%f ", de[i]);
            }
        }
        write_to_dnnl_memory(quantized.data(), out);
        write_to_dnnl_memory(scales.data(), out_scales);
        write_to_dnnl_memory(zero_points.data(), out_zp);

        break;
    }
    case dnnl_s8: {
        vector<float> quantized(data.size());
        vector<int> zero_points(data.size() / group_size);
        vector<float> scales(data.size() / group_size);

        for (int i = 0; i < data.size(); i += group_size) {

            //printf("\noriginal:    ");
            //for (int ii = i; ii < i + group_size; ii++) {
            //    printf("%f ", data[ii]);
            //}
            quantize(zero_points[i / group_size], scales[i / group_size],
                    &data[i], &data[i + group_size], &quantized[i], 255,
                    128);
            //printf("\nquantize: ");
            //for (int ii = i; ii < i + group_size; ii++) {
            //    printf("%f ", quantized[ii]);
            //}
            //printf("\nzero_points: ");
            //for (int ii = i / group_size; ii < i / group_size + 2; ii++) {
            //    printf("%d ", zero_points[ii]);
            //}
            //printf("\nscales: ");
            //for (int ii = i / group_size; ii < i / group_size + 2; ii++) {
            //    printf("%f ", scales[ii]);
            //}
            //auto de = dequantize(quantized, zero_points, scales, group_size, i);
            //printf("\ndequantized: ");
            //for (int i = 0; i < de.size(); i++) {
            //  printf("%f ", de[i]);
            //}
        }

        write_to_dnnl_memory(quantized.data(), out);
        write_to_dnnl_memory(scales.data(), out_scales);
        write_to_dnnl_memory(zero_points.data(), out_zp);
    } break;
    default:
      write_to_dnnl_memory(data.data(), out);//, &out.reorder_attr, &out.m_reorder_scale_attr);
      //write_to_dnnl_memory(key_scale_attr_data.data(), out_scales);
    }
}


sdpa_tensors get_descriptors(dnnl::engine &eng, sdpa_dims_t p,
        memory::data_type dt, memory::data_type kdt, memory::data_type vdt) {

    sdpa_tensors out;

    print_test_case(kdt, p);

    // Prepare input and output shapes to construct the sdpa graph.
    const memory::dims q_sz = {p.mb, p.head_num, p.query_num, p.head_size};
    const memory::dims k_sz = {p.mb, p.head_num, p.head_size, p.seq_len};
    const memory::dims v_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const memory::dims score_sz = {p.mb, p.head_num, p.query_num, p.seq_len};
    const memory::dims scale_sz = {1, 1, 1, 1};
    const memory::dims key_scales_sz = {k_sz[0], k_sz[1], k_sz[2], k_sz[3] / p.group_size};
    const memory::dims val_scales_sz = {v_sz[0], v_sz[1], v_sz[2], v_sz[3] / p.group_size};
    const memory::dims mask_sz = {p.mb, 1, p.query_num, p.
    seq_len};

    // score = query x key.T
    // scaled_score = score / scale
    // masked_score = scaled_score + mask
    // All combined in a single matmul primitive.
    auto query_md = memory::desc(q_sz, dt, memory::format_tag::abcd);
    auto key_md = memory::desc(k_sz, kdt, memory::format_tag::abcd);
    auto score_md = memory::desc(score_sz, dt, memory::format_tag::abcd);
    auto scale_md = memory::desc(scale_sz, dt, memory::format_tag::abcd);
    auto reorder_scale_attr_md = memory::desc(
            key_scales_sz, memory::data_type::f32, memory::format_tag::abcd);
    auto key_scales_md = memory::desc(
            key_scales_sz, memory::data_type::f16, memory::format_tag::abcd);
    auto key_zp_md = memory::desc(key_scales_sz, kdt, memory::format_tag::abcd);
    auto val_scales_md = memory::desc(
            val_scales_sz, memory::data_type::f16, memory::format_tag::abcd);
    auto val_zp_md = memory::desc(val_scales_sz, vdt, memory::format_tag::abcd);
    auto mask_md = memory::desc(mask_sz, dt, memory::format_tag::abcd);
    auto value_md = memory::desc(v_sz, vdt, memory::format_tag::abcd);
    auto output_md = memory::desc(q_sz, dt, memory::format_tag::abcd);

    // Create memory objects
    out.m_query = memory(query_md, eng);
    out.m_key = memory(key_md, eng);
    out.m_scale = memory(scale_md, eng);
    out.m_reorder_scale_attr = memory(reorder_scale_attr_md, eng);
    out.m_key_scales = memory(key_scales_md, eng);
    out.m_key_zp = memory(key_zp_md, eng);
    out.m_value_scales = memory(val_scales_md, eng);
    out.m_value_zp = memory(val_zp_md, eng);
    out.m_mask = memory(mask_md, eng);
    out.m_value = memory(value_md, eng);
    out.m_output = memory(output_md, eng);

    // Allocate user data.
    std::vector<float> query_data(product(q_sz));
    std::vector<float> key_data(product(k_sz));
    std::vector<float> scale_data(product(scale_sz), std::sqrt(p.head_size));
    std::vector<float> reorder_key_scale_attr_data(
            product(key_scales_sz), 1.f);
    std::vector<float> key_scale_attr_data(product(key_scales_sz), 1.f);
    std::vector<float> val_scale_attr_data(product(val_scales_sz), 1.f);
    std::vector<float> mask_data(product(mask_sz));
    std::vector<float> value_data(product(v_sz));
    std::vector<float> output_data(product(q_sz));

    out.reorder_attr.set_scales_mask(DNNL_ARG_DST, 0);
    //out.reorder_attr.set_scales(DNNL_ARG_DST,
    //(1 << 0) + (1 << 1) + (1 << 2) + (1 << 3), {1, 1, group_size, 1},
    //memory::data_type::f32);

    out.sdpa_attr.set_scales(DNNL_ARG_KEYS, 1 << 3, {1, 1, 1, p.group_size}, memory::data_type::f16);
    //out.sdpa_attr.set_scales(DNNL_ARG_VALUES, 1 << 3, {1, 1, 1, p.group_size}, memory::data_type::f16);
    //out.sdpa_attr.set_zero_points(DNNL_ARG_KEYS, 1 << 3, {1, 1, 1, group_size}, memory::data_type::s8);
    //out.sdpa_attr.set_zero_points(DNNL_ARG_VALUES, 1 << 3, {1, 1, 1, group_size}, memory::data_type::s8);
    out.sdpa_attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);

    //fill_random(query_data);

    printf("Q: p.mb(%ld), p.head_num(%ld), p.query_num(%ld), "
           "p.head_size(%ld)\n",
            p.mb, p.head_num, p.query_num, p.head_size);
    for (int i = 0; i < p.mb; ++i) {
        for (int ii = 0; ii < p.head_num; ++ii) {
            for (int iii = 0; iii < p.query_num; ++iii) {
                for (int iiii = 0; iiii < p.head_size; ++iiii) {
                    query_data[i * p.head_num * p.query_num * p.head_size
                            + ii * p.query_num * p.head_size + iii * p.head_size
                            + iiii]
                            = (iii * p.head_size + iiii) * 0.1;
                }
            }
        }
    }

    fill_random(key_data);
    fill_random(value_data);
    fill_mask(mask_data, static_cast<size_t>(p.seq_len));

    write_to_dnnl_memory(mask_data.data(), out.m_mask);
    write_to_dnnl_memory(scale_data.data(), out.m_scale);

    // Write data to tensor object's handle.
    write_to_dnnl_memory(query_data.data(), out.m_query);
    //write_to_dnnl_memory(reorder_key_scale_attr_data.data(), out.m_reorder_scale_attr);

    quantize_and_write(out.m_key, out.m_key_scales, out.m_key_zp, key_md, key_data, p.group_size);
    quantize_and_write(out.m_value, out.m_value_scales, out.m_value_zp, value_md, value_data, p.group_size);
    //print_mem(out.m_query, "query");
    //print_mem(out.m_key, "key");
    //print_mem(out.m_value, "value");

    //if (key_md.get_data_type() == dnnl_s4
    //        || key_md.get_data_type() == dnnl_s8) {

    //} else {
    //    write_to_dnnl_memory(key_data.data(), out.m_key);
    //}

    //print_mem(out.m_key, "key");
    //print_mem(out.m_query, "query");

    //write_to_dnnl_memory(val_scale_attr_data.data(), out.m_value_scales);
    //print_mem(out.m_value_scales, "val_scale_attr");

    //if (value_md.get_data_type() == dnnl_s8) {
    //    //write_to_dnnl_memory(value_data.data(), out.m_value, &out.reorder_attr, &out.m_reorder_scale_attr);
    //    write_to_dnnl_memory(value_data.data(), out.m_value);
    //} else {
    //    write_to_dnnl_memory(value_data.data(), out.m_value);
    //}

    return out;
}
sdpa_dims_t p = {.mb = 1,
        .seq_len = 128,
        .head_num = 16,
        .head_size = 64,
        .query_num = 128,
        .group_size = 32};

//TEST(SDPA, primitive) {
//    //sdpa_dims_t p = {32, 384, 16, 64, 384};
//    memory::data_type dt = memory::data_type::f16;
//    memory::data_type kdt = memory::data_type::f16;
//    memory::data_type vdt = memory::data_type::f16;
//    memory::data_type scale_dt = memory::data_type::f16;
//    bool invert_scale = false;
//
//    // Create execution dnnl::engine.
//    dnnl::engine eng(engine::kind::gpu, 0);
//    // Create dnnl::stream.
//    dnnl::stream strm(eng);
//
//    sdpa_tensors t = get_descriptors(eng, p, dt, kdt, vdt);
//
//    dnnl::primitive_attr bmm1_attr;
//    bmm1_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
//    dnnl::post_ops bmm1_po;
//    bmm1_po.append_binary(algorithm::binary_div, t.m_scale.get_desc());
//    bmm1_po.append_binary(algorithm::binary_add, t.m_mask.get_desc());
//    bmm1_attr.set_post_ops(bmm1_po);
//
//    const memory::dims score_sz = {p.mb, p.head_num, p.query_num, p.seq_len};
//    auto score_md = memory::desc(score_sz, dt, memory::format_tag::abcd);
//    auto m_score = memory(score_md, eng);
//
//    auto bmm1_pd = matmul::primitive_desc(
//            eng, t.m_query.get_desc(), t.m_key.get_desc(), score_md, bmm1_attr);
//    auto bmm1_prim = matmul(bmm1_pd);
//
//    // attention_probs = softmax(masked_score)
//    primitive_attr softmax_attr;
//    softmax_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
//    auto softmax_pd = softmax_forward::primitive_desc(eng,
//            dnnl::prop_kind::forward_inference, algorithm::softmax_accurate,
//            score_md, score_md, /* axis = */ score_md.get_ndims() - 1,
//            softmax_attr);
//    auto softmax_prim = softmax_forward(softmax_pd);
//
//    // attention_output = attention_probs x value
//    primitive_attr bmm2_attr;
//    bmm2_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
//    auto bmm2_pd = matmul::primitive_desc(eng, score_md, t.m_value.get_desc(),
//            t.m_output.get_desc(), bmm2_attr);
//    auto bmm2_prim = matmul(bmm2_pd);
//
//    size_t max_scratchpad_size = 0;
//    auto bmm1_scratchpad = bmm1_pd.scratchpad_desc().get_size();
//    auto softmax_scratchpad = softmax_pd.scratchpad_desc().get_size();
//    auto bmm2_scratchpad = bmm2_pd.scratchpad_desc().get_size();
//    for (auto &sz : {bmm1_scratchpad, softmax_scratchpad, bmm2_scratchpad}) {
//        if (max_scratchpad_size < sz) max_scratchpad_size = sz;
//    }
//    auto scratchpad_md
//            = memory::desc({static_cast<memory::dim>(max_scratchpad_size)},
//                    memory::data_type::u8, memory::format_tag::a);
//
//    // allocate intermediate memory
//    auto m_scratchpad = memory(scratchpad_md, eng);
//
//    const auto loop = [&]() {
//        // each primitive will use all threads
//        bmm1_prim.execute(strm,
//                {{DNNL_ARG_SRC, t.m_query}, {DNNL_ARG_WEIGHTS, t.m_key},
//                        {DNNL_ARG_DST, m_score},
//                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
//                                t.m_scale},
//                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, t.m_mask},
//                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});
//
//        softmax_prim.execute(strm,
//                {{DNNL_ARG_SRC, m_score}, {DNNL_ARG_DST, m_score},
//                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});
//
//        bmm2_prim.execute(strm,
//                {{DNNL_ARG_SRC, m_score}, {DNNL_ARG_WEIGHTS, t.m_value},
//                        {DNNL_ARG_DST, t.m_output},
//                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});
//    };
//
//    // Warmup run.
//    // Execute primitives of sdpa.
//    loop();
//
//    // Wait for the computation to finish.
//    strm.wait();
//
//    //print_mem(t.m_output, "output");
//}
//
//TEST(SDPA, f16) {
//    //sdpa_dims_t p = {32, 384, 16, 64, 384};
//    memory::data_type dt = memory::data_type::f16;
//    memory::data_type kdt = memory::data_type::f16;
//    memory::data_type vdt = memory::data_type::f16;
//    memory::data_type scale_dt = memory::data_type::f16;
//    bool invert_scale = false;
//
//    // Create execution dnnl::engine.
//    dnnl::engine eng(engine::kind::gpu, 0);
//    // Create dnnl::stream.
//    dnnl::stream strm(eng);
//
//    sdpa_tensors t = get_descriptors(eng, p, dt, kdt, vdt);
//
//    dnnl::primitive_attr attr;
//    attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);
//
//    using namespace dnnl::experimental;
//    auto mask = t.m_mask.get_desc();
//    auto sdpa_pd = sdpa::primitive_desc(eng, t.m_query.get_desc(),
//            t.m_key.get_desc(), t.m_value.get_desc(), &mask, scale_dt,
//            t.m_output.get_desc(), invert_scale, 1, attr);
//    auto sdpa_p = sdpa(sdpa_pd);
//
//    sdpa_p.execute(strm,
//            {{DNNL_ARG_QUERIES, t.m_query}, {DNNL_ARG_KEYS, t.m_key},
//                    {DNNL_ARG_VALUES, t.m_value}, {DNNL_ARG_SCALE, t.m_scale},
//                    {DNNL_ARG_ATTN_MASK, t.m_mask},
//                    {DNNL_ARG_DST, t.m_output}});
//    strm.wait();
//
//    //print_mem(t.m_output, "outputf16");
//}
//
//TEST(SDPA, s4) {
//    //sdpa_dims_t p = {32, 384, 16, 64, 384};
//
//    // Create execution dnnl::engine.
//    dnnl::engine eng(engine::kind::gpu, 0);
//    // Create dnnl::stream.
//    dnnl::stream strm(eng);
//
//    memory::data_type dt = memory::data_type::f16;
//    memory::data_type kdt = memory::data_type::s8;
//    memory::data_type vdt = memory::data_type::f16;
//    memory::data_type scale_dt = memory::data_type::f16;
//    bool invert_scale = false;
//
//    sdpa_tensors t = get_descriptors(eng, p, dt, kdt, vdt);
//
//    using namespace dnnl::experimental;
//    auto mask = t.m_mask.get_desc();
//    auto sdpa_pd = sdpa::primitive_desc(eng, t.m_query.get_desc(),
//            t.m_key.get_desc(), t.m_value.get_desc(), &mask, scale_dt,
//            t.m_output.get_desc(), invert_scale, 1, t.sdpa_attr);
//    auto sdpa_p = sdpa(sdpa_pd);
//
//
//    print_mem(t.m_key_scales, "key_scale_attr");
//    //print_mem(t.m_key_zp, "key_zero_points");
//    //print_mem(t.m_value_scales, "value_scale_attr");
//    //print_mem(t.m_value_zp, "value_zero_points");
//
//
//    sdpa_p.execute(strm,
//            {{DNNL_ARG_QUERIES, t.m_query},
//             {DNNL_ARG_KEYS, t.m_key},
//             {DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS, t.m_key_scales},
//             //{DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS, t.m_key_zp},
//
//             {DNNL_ARG_VALUES, t.m_value},
//             //{DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES, t.m_value_scales},
//             //{DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES, t.m_value_zp},
//
//             {DNNL_ARG_SCALE, t.m_scale}, {DNNL_ARG_ATTN_MASK, t.m_mask},
//             {DNNL_ARG_DST, t.m_output}});
//    strm.wait();
//    //print_mem(t.m_output, "outputs8");
//}


TEST(SDPA, compares8tof16) {
    //sdpa_dims_t p = {32, 384, 16, 64, 384};

    // Create execution dnnl::engine.
    dnnl::engine eng(engine::kind::gpu, 0);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    memory::data_type dt = memory::data_type::f16;
    memory::data_type kdt = memory::data_type::s8;
    memory::data_type vdt = memory::data_type::f16;
    memory::data_type scale_dt = memory::data_type::f16;
    bool invert_scale = false;

    sdpa_tensors ts8 = get_descriptors(eng, p, dt, kdt, vdt);

    sdpa_tensors tf16 = get_descriptors(eng, p, dt, memory::data_type::f16, vdt);

    using namespace dnnl::experimental;
    auto mask = ts8.m_mask.get_desc();
    auto sdpas8_pd = sdpa::primitive_desc(eng, ts8.m_query.get_desc(),
            ts8.m_key.get_desc(), ts8.m_value.get_desc(), &mask, scale_dt,
            ts8.m_output.get_desc(), invert_scale, 1, ts8.sdpa_attr);
    auto sdpas8_p = sdpa(sdpas8_pd);

    auto maskf16 = tf16.m_mask.get_desc();
    auto sdpaf16_pd = sdpa::primitive_desc(eng, tf16.m_query.get_desc(),
            tf16.m_key.get_desc(), tf16.m_value.get_desc(), &maskf16, scale_dt,
            tf16.m_output.get_desc(), invert_scale, 1, tf16.sdpa_attr);
    auto sdpaf16_p = sdpa(sdpaf16_pd);


    print_mem(ts8.m_key_scales, "key_scale_attr");
    //print_mem(ts8.m_key_zp, "key_zero_points");
    //print_mem(ts8.m_value_scales, "value_scale_attr");
    //print_mem(ts8.m_value_zp, "value_zero_points");

    sdpas8_p.execute(strm,
            {{DNNL_ARG_QUERIES, ts8.m_query}, {DNNL_ARG_KEYS, ts8.m_key},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS, ts8.m_key_scales},
                    //{DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS, ts8.m_key_zp},

                    {DNNL_ARG_VALUES, ts8.m_value},
                    //{DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES, ts8.m_value_scales},
                    //{DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES, ts8.m_value_zp},

                    {DNNL_ARG_SCALE, ts8.m_scale},
                    {DNNL_ARG_ATTN_MASK, ts8.m_mask},
                    {DNNL_ARG_DST, ts8.m_output}});
    sdpaf16_p.execute(strm,
            {{DNNL_ARG_QUERIES, tf16.m_query}, {DNNL_ARG_KEYS, tf16.m_key},
                    {DNNL_ARG_VALUES, tf16.m_value},
                    {DNNL_ARG_SCALE, tf16.m_scale},
                    {DNNL_ARG_ATTN_MASK, tf16.m_mask},
                    {DNNL_ARG_DST, tf16.m_output}});
    strm.wait();
    //print_mem(ts8.m_output, "outputs8");

    float16_t *mapped_ptr_f16 = (float16_t *)tf16.m_output.map_data();
    float16_t *mapped_ptr_s8 = (float16_t*)ts8.m_output.map_data();

    auto dims = tf16.m_output.get_desc().get_dims();
    auto strides = tf16.m_output.get_desc().get_strides();

    for (int l = 0; l < dims[0]; l++) {
        for (int k = 0; k < dims[1]; k++) {
            for (int j = 0; j < dims[2]; j++) {
                for (int i = 0; i < dims[3]; i++) {
                    auto offset = l * strides[0] + k * strides[1]
                            + j * strides[2] + i * strides[3];
                            EXPECT_NEAR(mapped_ptr_f16[offset].f(),
                                        mapped_ptr_s8[offset].f(), 0.0003f)
                              << "AT (l:" << l << ",k:" << k << ",j:" << j << ",i:"<< i <<")";
                }
            }
        }
    }

    tf16.m_output.unmap_data(mapped_ptr_f16);
    ts8.m_output.unmap_data(mapped_ptr_s8);

}
