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

#include <oneapi/dnnl/experimental/dnnl_experimental.h>

namespace dnnl {
namespace experimental {

/// Scaled Dot Product Attention (sdpa) primitive.
struct sdpa : public dnnl::primitive {
    /// Primitive descriptor for a sdpa primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a sdpa primitive
        ///
        /// @param aengine Engine to use.
        /// @param query_desc Memory descriptor for query tensor.
        /// @param key_desc Memory descriptor for key tensor.
        /// @param value_desc Memory descriptor for value tensor.
        /// @param output_desc Memory descriptor for output tensor.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, const memory::desc &query_desc,
                const memory::desc &key_desc, const memory::desc &value_desc,
                const memory::desc &output_desc,
                const primitive_attr &attr = default_attr())
            : primitive_desc(aengine, query_desc, key_desc, value_desc, nullptr,
                    memory::data_type::undef, output_desc, false, 1, attr) {}

        /// Constructs a primitive descriptor for a sdpa primitive
        ///
        /// @param aengine Engine to use.
        /// @param query_desc Memory descriptor for query tensor.
        /// @param key_desc Memory descriptor for key tensor.
        /// @param value_desc Memory descriptor for value tensor.
        /// @param output_desc Memory descriptor for output tensor.
        /// @param attn_mask_desc Memory descriptor for attention mask tensor.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, const memory::desc &query_desc,
                const memory::desc &key_desc, const memory::desc &value_desc,
                const memory::desc &attn_mask_desc,
                const memory::desc &output_desc,
                const primitive_attr &attr = default_attr())
            : primitive_desc(aengine, query_desc, key_desc, value_desc, &attn_mask_desc,
                    memory::data_type::undef, output_desc, false, 1, attr) {}


        /// Constructs a primitive descriptor for a sdpa primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a sdpa primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::undef) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc query_desc() const { return query_md(query::src_md, 0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc key_desc() const { return query_md(query::src_md, 1); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc value_desc() const {return query_md(query::src_md, 2);}

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc attn_mask_desc() const {return query_md(query::src_md, 3);}


        /// @copydoc dnnl::convolution_forward::primitive_desc::bias_desc()const
        memory::desc bias_desc() const {
            return query_md(query::weights_md, 1);
        }

        ///// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return query_md(query::dst_md, 0); }

        primitive_desc(const engine &aengine, const memory::desc &query_desc,
                const memory::desc &key_desc, const memory::desc &value_desc,
                const memory::desc *attn_mask_desc, memory::data_type scale_dt,
                const memory::desc &output_desc, bool invert_scale,
                dnnl_dim_t kv_head_number, const primitive_attr &attr) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_sdpa_primitive_desc_create(&pd, aengine.get(),
                            query_desc.get(), key_desc.get(), value_desc.get(),
                            output_desc.get(), optional_arg(attn_mask_desc),
                            (dnnl::impl::data_type_t)scale_dt, invert_scale,
                            kv_head_number, attr.get());

            dnnl::error::wrap_c_api(status,
                    "could not create a primitive descriptor for a sdpa "
                    "primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    sdpa() = default;

    /// Constructs a sdpa primitive.
    /// @param pd Primitive descriptor for a sdpa primitive.
    sdpa(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a sdpa primitive from a cache blob.
    /// @param pd Primitive descriptor for a sdpa primitive.
    /// @param cache_blob Cache blob.
    sdpa(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};
} // namespace experimental
} // namespace dnnl
