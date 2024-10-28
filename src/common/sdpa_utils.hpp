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

#ifndef COMMON_SDPA_UTILS_HPP
#define COMMON_SDPA_UTILS_HPP

#include "oneapi/dnnl/dnnl.h"
#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/sdpa_types.hpp"
#include "common/sdpa.h"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

#define VCHECK_SDPA(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, sdpa, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

static inline sdpa_desc_t create_sdpa_desc(const memory_desc_t *q_md,
        const memory_desc_t *k_md, const memory_desc_t *v_md,
        const memory_desc_t *dst_md, const memory_desc_t *attn_mask_md,
        data_type_t scale_dt, dim_t kv_head_number, const primitive_attr_t *kq_attr,
        const primitive_attr_t *vs_attr, bool invert_scale = false) {
    auto sdpa_desc = sdpa_desc_t();
    sdpa_desc.primitive_kind = primitive_kind::sdpa;
    sdpa_desc.q_desc = *q_md;
    sdpa_desc.k_desc = *k_md;
    if(kq_attr) {
        sdpa_desc.kq_scales = kq_attr->scales_.get(DNNL_ARG_WEIGHTS);
        sdpa_desc.kq_zero_points = kq_attr->zero_points_;
    }
    if (vs_attr) {
        sdpa_desc.vs_scales = vs_attr->scales_.get(DNNL_ARG_WEIGHTS);
        sdpa_desc.vs_zero_points = vs_attr->zero_points_;
    }
    sdpa_desc.v_desc = *v_md;
    sdpa_desc.dst_desc = *dst_md;
    if (attn_mask_md) sdpa_desc.attn_mask_desc = *attn_mask_md;
    sdpa_desc.scale_dt = scale_dt;
    sdpa_desc.invert_scale = invert_scale;
    sdpa_desc.kv_head_number = kv_head_number;
    return sdpa_desc;
}

static inline status_t create_sdpa_pd(
        std::shared_ptr<primitive_desc_t> &sdpa_pd_, engine_t *engine,
        const memory_desc_t *q_md, const memory_desc_t *k_md,
        const memory_desc_t *v_md, const memory_desc_t *dst_md,
        const memory_desc_t *attn_mask_md, data_type_t scale_dt,
        bool invert_scale, const primitive_attr_t *attr, dim_t kv_head_number,
        const primitive_attr_t *kq_attr, const primitive_attr_t *vs_attr) {
    auto sdpa_desc = create_sdpa_desc(q_md, k_md, v_md, dst_md, attn_mask_md,
            scale_dt, kv_head_number, kq_attr, vs_attr, invert_scale);

    int ndims = dst_md->ndims;
    int r = ndims - 2, c = ndims - 1;
    if (!utils::everyone_is(ndims, q_md->ndims, k_md->ndims, v_md->ndims))
        return status::invalid_arguments;
    if (q_md->dims[c] != k_md->dims[r]) return status::invalid_arguments;
    if (k_md->dims[c] != v_md->dims[r]) return status::invalid_arguments;
    if (dst_md->dims[r] != q_md->dims[r] || dst_md->dims[c] != v_md->dims[c])
        return status::invalid_arguments;

    if (attr == nullptr) attr = &default_attr();

    primitive_attr_t sdpa_attr = *attr;
    //primitive_attr_t sdpa_k_attr = (kq_attr) ? *kq_attr : default_attr();
    //primitive_attr_t sdpa_vs_attr = (vs_attr) ? *vs_attr : default_attr();

    //auto k_zero_points = kq_attr->zero_points_;
    //VCHECK_SDPA(IMPLICATION(!types::is_integral_dt(k_md->data_type),
    //                    k_zero_points.has_default_values(DNNL_ARG_WEIGHTS)),
    //        VERBOSE_UNSUPPORTED_ZP_CFG);
    //// TODO(umar): additional checks for scales

    //auto v_zero_points = v_attr->zero_points_;
    //VCHECK_SDPA(IMPLICATION(!types::is_integral_dt(v_md->data_type),
    //                    v_zero_points.has_default_values(DNNL_ARG_WEIGHTS)),
    //        VERBOSE_UNSUPPORTED_ZP_CFG);
    //// TODO(umar): additional checks for scales

    primitive_desc_iterator_t it(
            engine, (op_desc_t *)&sdpa_desc, &sdpa_attr, nullptr);

    sdpa_pd_ = *(++it);
    if (!sdpa_pd_) return status::unimplemented;

    return status::success;
}

} // namespace impl
} // namespace dnnl

#endif
