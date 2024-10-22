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

#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"
#include "sdpa_pd.hpp"
#include "sdpa_utils.hpp"

dnnl_status_t dnnl_sdpa_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc_iface, dnnl_engine_t engine,
        const_dnnl_memory_desc_t query_desc, const_dnnl_memory_desc_t key_desc,
        const_dnnl_memory_desc_t value_desc, const_dnnl_memory_desc_t dst_desc,
        const_dnnl_memory_desc_t mask_desc, dnnl_data_type_t scale_dt,
        bool invert_scale, dnnl_dim_t kv_head_number,
        const_dnnl_primitive_attr_t attr) {
    dnnl::impl::sdpa_desc_t sdpa_desc = dnnl::impl::create_sdpa_desc(query_desc,
            key_desc, value_desc, dst_desc, mask_desc,
            (dnnl::impl::data_type_t)scale_dt, 1, invert_scale);
    return dnnl::impl::primitive_desc_create(primitive_desc_iface, engine,
            (const dnnl::impl::op_desc_t *)&sdpa_desc, nullptr, attr);
}
