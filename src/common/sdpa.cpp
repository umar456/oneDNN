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

#include <common/sdpa.h>

#include "c_types_map.hpp"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"
#include "sdpa_pd.hpp"
#include "sdpa_types.hpp"
#include "sdpa_utils.hpp"

using dnnl::impl::status_t;
using namespace dnnl::impl;

namespace {
status_t sdpa_attr_check(const sdpa_desc_t &desc, const engine_t *engine,
        const primitive_attr_t *attr, const primitive_attr_t *k_attr,
        const primitive_attr_t *v_attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;
    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;
    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Check attributes
    //const data_type_t src_dt = desc.src_desc.data_type;
    //const data_type_t wei_dt = desc.weights_desc.data_type;
    //const data_type_t dst_dt = desc.dst_desc.data_type;

    return status::success;

    // auto attr_mask = smask_t::post_ops | smask_t::sum_dt | smask_t::dropout
    //         | smask_t::rounding_mode;
    // // Matmul supports scales for floating point data types
    // attr_mask |= smask_t::scales_runtime;
    // attr_mask |= smask_t::scales_runtime_data_type;

    // const bool src_is_int8
    //         = utils::one_of(src_dt, data_type::s8, data_type::u8);
    // if (src_is_int8) attr_mask |= smask_t::zero_points_runtime;

    // // Matmul supports zero points for floating point data types as part of
    // // weights decompression.
    // const bool wei_is_int = utils::one_of(
    //         wei_dt, data_type::s8, data_type::u8, data_type::s4, data_type::u4);
    // if (wei_is_int) {
    //     attr_mask |= smask_t::zero_points_runtime_data_type;
    //     attr_mask |= smask_t::zero_points_runtime_groups;
    //     attr_mask |= smask_t::scales_runtime_groups;
    // }

    // // Matmul supports fpmath mode
    // attr_mask |= smask_t::fpmath_mode;

    // VCHECK_MATMUL_UNIMPL(attr->has_default_values(attr_mask, dst_dt),
    //         VERBOSE_UNSUPPORTED_ATTR);

    // int ndims_src = desc.src_desc.ndims;
    // int ndims_wei = desc.weights_desc.ndims;
    // assert(ndims_src >= 2);
    // assert(ndims_wei >= 2);
    // int src_qmask_M = 1 << (ndims_src - 2);
    // int src_qmask_K = 1 << (ndims_src - 1);

    // int wei_qmask_K = 1 << (ndims_wei - 2);
    // int wei_qmask_N = 1 << (ndims_wei - 1);

    // // Check scales
    // if (!attr->scales_.has_default_values()) {
    //     const auto &sc = attr->scales_;
    //     const int mask_src = sc.get(DNNL_ARG_SRC).mask_;
    //     const int mask_wei = sc.get(DNNL_ARG_WEIGHTS).mask_;
    //     const int mask_dst = sc.get(DNNL_ARG_DST).mask_;

    //     // Check allowed masks.
    //     VCHECK_MATMUL_UNIMPL(utils::one_of(mask_src, 0, src_qmask_K,
    //                                  src_qmask_M + src_qmask_K)
    //                     && utils::one_of(mask_wei, 0, wei_qmask_N,
    //                             wei_qmask_N + wei_qmask_K)
    //                     && mask_dst == 0,
    //             VERBOSE_UNSUPPORTED_SCALES_CFG);
    //     // Check dependency between scales.
    //     // Source scales groups are supported for int8 source and must divide
    //     // or be divided by weights groups when both are greater than 1.
    //     const auto src_scale_group_k = (mask_src & src_qmask_K)
    //             ? sc.get(DNNL_ARG_SRC).group_dims_[1]
    //             : 1;
    //     const auto wei_scale_group_k = (mask_wei & wei_qmask_K)
    //             ? sc.get(DNNL_ARG_WEIGHTS).group_dims_[0]
    //             : 1;
    //     const bool groups_are_divisible = IMPLICATION(
    //             src_scale_group_k > 1 && wei_scale_group_k > 1,
    //             (src_scale_group_k % wei_scale_group_k == 0)
    //                     || (wei_scale_group_k % src_scale_group_k == 0));
    //     VCHECK_MATMUL_UNIMPL(IMPLICATION(src_scale_group_k > 1,
    //                                  src_is_int8 && groups_are_divisible),
    //             VERBOSE_UNSUPPORTED_SCALES_CFG);
    // }

    // // Check zero points
    // if (!attr->zero_points_.has_default_values()) {
    //     const auto &zp = attr->zero_points_;
    //     int mask_src = 0, mask_wei = 0, mask_dst = 0;
    //     zp.get(DNNL_ARG_SRC, &mask_src);
    //     zp.get(DNNL_ARG_WEIGHTS, &mask_wei);
    //     zp.get(DNNL_ARG_DST, &mask_dst);

    //     VCHECK_MATMUL_UNIMPL(mask_src == 0
    //                     || (desc.src_desc.ndims == 2 && mask_src == 1 << 1),
    //             VERBOSE_UNSUPPORTED_ZP_CFG);
    //     VCHECK_MATMUL_UNIMPL(utils::one_of(mask_wei, 0, wei_qmask_N,
    //                                  wei_qmask_N + wei_qmask_K),
    //             VERBOSE_UNSUPPORTED_ZP_CFG);
    //     VCHECK_MATMUL_UNIMPL(mask_dst == 0
    //                     || (desc.dst_desc.ndims == 2 && mask_dst == 1 << 1),
    //             VERBOSE_UNSUPPORTED_ZP_CFG);

    //     if (utils::one_of(zp.get_data_type(DNNL_ARG_WEIGHTS), data_type::s4,
    //                 data_type::u4)) {
    //         dim_t k = desc.weights_desc.dims[ndims_wei - 2];
    //         dim_t n = desc.weights_desc.dims[ndims_wei - 1];
    //         VCHECK_MATMUL_UNIMPL(
    //                 IMPLICATION(mask_wei & wei_qmask_K, k % 2 == 0),
    //                 VERBOSE_UNSUPPORTED_ZP_CFG);
    //         VCHECK_MATMUL_UNIMPL(
    //                 IMPLICATION(mask_wei & wei_qmask_N, n % 2 == 0),
    //                 VERBOSE_UNSUPPORTED_ZP_CFG);
    //     }
    // }

    // // Check post-ops
    // if (!attr->post_ops_.has_default_values()) {
    //     const auto &po = attr->post_ops_;
    //     using namespace primitive_kind;
    //     VCHECK_MATMUL_UNIMPL(
    //             po.has_default_values({binary, eltwise, prelu, sum}),
    //             VERBOSE_UNSUPPORTED_POSTOP);

    //     // Check sum
    //     VCHECK_MATMUL_UNIMPL(
    //             po.check_sum_consistency(dst_dt, src_is_int8, true),
    //             VERBOSE_UNSUPPORTED_POSTOP);
    // }

    return status::success;
}

} // namespace

dnnl_status_t dnnl_sdpa_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc_iface, dnnl_engine_t engine,
        const_dnnl_memory_desc_t query_desc, const_dnnl_memory_desc_t key_desc,
        const_dnnl_memory_desc_t value_desc,
        const_dnnl_memory_desc_t dst_desc, const_dnnl_memory_desc_t mask_desc,
        dnnl_data_type_t scale_dt, bool invert_scale, dnnl_dim_t kv_head_number,
        const_dnnl_primitive_attr_t attr, const_dnnl_primitive_attr_t kq_attr,
        const_dnnl_primitive_attr_t vs_attr) {
    dnnl::impl::sdpa_desc_t sdpa_desc
            = dnnl::impl::create_sdpa_desc(query_desc, key_desc, value_desc,
                    dst_desc, mask_desc, (dnnl::impl::data_type_t)scale_dt, 1,
                    kq_attr, vs_attr, invert_scale);
    sdpa_attr_check(sdpa_desc, engine, attr, kq_attr, vs_attr);
    return dnnl::impl::primitive_desc_create(primitive_desc_iface, engine,
            (const dnnl::impl::op_desc_t *)&sdpa_desc, nullptr, attr);
}
