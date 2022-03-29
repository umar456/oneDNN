/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#include "ssa_visitor.hpp"
#include <atomic>
#include <functional>
#include <list>
#include <string>
#include <utility>
#include <vector>
#include "ssa_data.hpp"
#include <compiler/ir/builder.hpp>
#include <util/array_ref.hpp>

namespace sc {

expr ssa_data_t::get_value_of_var() const {
    assert(!utils::is_uninitialized_weakptr(owner_));
    auto owner = owner_.lock();
    assert(owner);
    COMPILE_ASSERT(owner->node_type_ == sc_stmt_type::define,
            "Expecting define_node for get_value_of_var");
    return static_cast<define_node_t *>(owner.get())->init_;
}

void get_direct_dependency_of_expr(
        const expr &v, const std::function<void(array_ref<expr>)> &callback) {
    switch (v->node_type_) {
        case sc_expr_type::undef: assert(0 && "Unreachable"); break;
        case sc_expr_type::constant:
        case sc_expr_type::func_addr: break;
        case sc_expr_type::tensor:
        case sc_expr_type::var: {
            if (utils::is_uninitialized_weakptr(v->ssa_data_->owner_)) {
                // a var/tensor can be attached none of the define_node, which
                // means it is a parameter
                break;
            }
            stmt owner {v->ssa_data_->owner_.lock()};
            assert(owner.defined());
            auto init = owner.static_as<define>()->init_;
            if (owner.isa<define>() && init.defined()) {
                if (init.isa<var>()) {
                    callback({init});
                } else {
                    get_direct_dependency_of_expr(init, callback);
                }
            }
        } break;
        case sc_expr_type::cast: callback({v.static_as<cast>()->in_}); break;
        case sc_expr_type::add:
        case sc_expr_type::sub:
        case sc_expr_type::mul:
        case sc_expr_type::div:
        case sc_expr_type::mod: {
            auto val = v.static_as<binary>();
            callback({val->l_, val->r_});
        } break;
        case sc_expr_type::cmp_eq:
        case sc_expr_type::cmp_ne:
        case sc_expr_type::cmp_lt:
        case sc_expr_type::cmp_le:
        case sc_expr_type::cmp_gt:
        case sc_expr_type::cmp_ge: {
            auto val = v.static_as<cmp>();
            callback({val->l_, val->r_});
        } break;
        case sc_expr_type::logic_and:
        case sc_expr_type::logic_or: {
            auto val = v.static_as<logic>();
            callback({val->l_, val->r_});
        } break;
        case sc_expr_type::logic_not:
            callback(&v.static_as<logic_not>()->in_);
            break;
        case sc_expr_type::select: {
            auto val = v.static_as<select>();
            callback({val->cond_, val->l_, val->r_});
        } break;
        case sc_expr_type::indexing: {
            auto val = v.static_as<indexing>();
            callback(&val->ptr_);
            callback(val->idx_);
            if (val->mask_.defined()) { callback(&val->mask_); }
        } break;
        case sc_expr_type::call: callback(v.static_as<call>()->args_); break;
        case sc_expr_type::tensorptr: {
            auto val = v.static_as<tensorptr>();
            callback(&val->base_->ptr_);
            callback(val->base_->idx_);
        } break;
        case sc_expr_type::intrin_call:
            callback(v.static_as<intrin_call>()->args_);
            break;
        case sc_expr_type::low_level_intrin:
            callback(v.static_as<low_level_intrin>()->args_);
            break;
        case sc_expr_type::ssa_phi:
            callback(v.static_as<ssa_phi>()->values_);
            break;
    }
}

expr_c ssa_visitor_t::dispatch(expr_c e) {
    auto ret = ir_visitor_t::dispatch(e);
    if (!ret->ssa_data_) {
        ret.remove_const()->ssa_data_ = utils::make_unique<ssa_data_t>();
    }
    return ret;
}

static void process_define_node_after_visit(const stmt_c &ret) {
    auto def_node = ret.static_as<define_c>();
    auto &lhs = def_node->var_;
    assert(lhs->ssa_data_);
    lhs->ssa_data_->owner_ = ret.weak();
    auto &rhs = def_node->init_;
    if (!lhs->ssa_data_->is_global_) {
        if (rhs.defined()) {
            assert(rhs->ssa_data_);
            rhs->ssa_data_->referenced_ = false;
        } else {
            assert(lhs.isa<tensor>());
        }
        lhs->ssa_data_->referenced_ = false;
    }
}

// after visiting an stmt, we need to: 1. set "referenced" bit to the directly
// referenced exprs 2. Set the owners of the vars for define_node and for_node
stmt_c ssa_visitor_t::dispatch(stmt_c e) {
    auto ret = ir_visitor_t::dispatch(std::move(e));
    auto ths = this;
    auto set_referenced = [ths](const expr &d, bool check) {
        d->ssa_data_->referenced_ = true;
        ths->gc_roots_.emplace_back(d);
        if (check) { assert(d.isa<var>()); }
    };
    if (!ret.defined()) { return ret; }
    switch (ret->node_type_) {
        case sc_stmt_type::undef: assert(0 && "Unreachable"); break;
        case sc_stmt_type::assign:
            // the LHS of assign can be indexing
            set_referenced(ret.static_as<assign>()->var_, false);
            set_referenced(ret.static_as<assign>()->value_, true);
            break;
        case sc_stmt_type::if_else:
            set_referenced(ret.static_as<if_else>()->condition_, true);
            break;
        case sc_stmt_type::evaluate:
            // the value of evaluate can be call_node
            set_referenced(ret.static_as<evaluate>()->value_, false);
            break;
        case sc_stmt_type::for_loop: {
            auto the_for = ret.static_as<for_loop_c>();
            set_referenced(the_for->iter_begin_, true);
            set_referenced(the_for->iter_end_, true);
            set_referenced(the_for->step_, true);
            set_referenced(the_for->var_, true);
            the_for->var_->ssa_data_->owner_ = ret.weak();
        } break;
        case sc_stmt_type::returns: {
            auto the_ret = ret.static_as<returns_c>();
            if (the_ret->value_.defined()) {
                set_referenced(the_ret->value_, true);
            }
        } break;
        case sc_stmt_type::define: process_define_node_after_visit(ret); break;
        case sc_stmt_type::stmts: break;
    }

    return ret;
}

stmt_c ssa_visitor_t::top_level_dispatch(stmt_c e) {
    gc_roots_.clear();
    auto ret = dispatch(std::move(e));

    mark_garbage();
    gc_roots_.clear();

    return ret;
}

func_c ssa_visitor_t::top_level_dispatch(func_c e) {
    gc_roots_.clear();
    auto ret = dispatch(std::move(e));

    mark_garbage();
    gc_roots_.clear();

    return ret;
}

static void do_mark(const expr &cur) {
    cur->ssa_data_->referenced_ = true;

    get_direct_dependency_of_expr(cur, [](array_ref<expr> ref) {
        for (auto &v : ref) {
            if (!v->ssa_data_->referenced_) { do_mark(v); }
        }
    });
}

void ssa_visitor_t::mark_garbage() {
    for (auto &v : gc_roots_) {
        do_mark(v);
    }
}

static var_node *get_var_if_is_define(const stmt_c &s) {
    if (s.isa<define>()) {
        auto def = s.static_as<define>();
        if (def->var_.isa<var>() && def->var_->ssa_data_) {
            // assert(def->var_->ssa_data_);
            return def->var_.static_as<var>().get();
        }
    }
    return nullptr;
}

stmt_c ssa_visitor_t::visit(stmts_c v) {
    std::vector<stmt_c> ret_vec;
    auto old_scope = current_scope_;
    current_scope_ = &ret_vec;
    bool changed = false;
    for (auto &s : v->seq_) {
        if (auto var_p = get_var_if_is_define(s)) {
            if (var_p->ssa_data_->is_garbage()) {
                changed = true;
                continue;
            }
        }
        auto sz_before = ret_vec.size();
        auto ret = dispatch(s);

        // we allow the return value to be empty, which means we need to remove
        // the stmt from parent
        if (!ret.defined()) {
            changed = true;
            continue;
        }
        // if a SSA definition is inserted, the IR is changed
        changed |= sz_before != ret_vec.size();
        changed |= !ret.ptr_same(s);
        ret_vec.emplace_back(std::move(ret));
        // insert the stmts_to_insert_after
        for (auto &st : stmts_to_insert_after) {
            if (st.isa<define>()) { process_define_node_after_visit(st); }
            ret_vec.emplace_back(std::move(st));
        }
        stmts_to_insert_after.clear();
    }
    current_scope_ = old_scope;
    if (!changed) {
        return v;
    } else {
        return copy_attr(*v, builder::make_stmts_unattached(ret_vec));
    }
}

static std::atomic<uint64_t> var_def_idx = {0};

static define make_def(const expr_c &v) {
    auto ret = builder::make_var(
            v->dtype_, std::string("__tmp") + std::to_string(var_def_idx++));
    ret->ssa_data_ = utils::make_unique<ssa_data_t>();
    if (!v->ssa_data_) {
        v.remove_const()->ssa_data_ = utils::make_unique<ssa_data_t>();
    }
    return builder::make_var_tensor_def_unattached(ret, linkage::local, v)
            .static_as<define>();
}

expr ssa_visitor_t::add_def(const expr_c &v) {
    assert(current_scope_);
    auto ret = make_def(v);
    current_scope_->emplace_back(ret);
    process_define_node_after_visit(current_scope_->back());
    return ret->var_;
}

expr ssa_visitor_t::add_def_after_current_stmt(const expr_c &v) {
    assert(current_scope_);
    auto ret = make_def(v);
    stmts_to_insert_after.emplace_back(ret);
    return ret->var_;
}

expr_c ssa_visitor_t::visit(ssa_phi_c v) {
    std::vector<expr> newv;
    bool changed = dispatch_expr_vector(v->values_, newv);
    if (changed) {
        return copy_attr(*v, make_expr<ssa_phi_node>(newv, v->is_loop_phi_));
    } else {
        return v;
    }
}
expr_c ssa_visitor_t::visit(tensor_c v) {
    return v;
}

void ssa_viewer_t::view(ssa_phi_c v) {
    for (auto &val : v->values_) {
        dispatch(val);
    }
}
void ssa_viewer_t::view(tensor_c v) {}

void ssa_viewer_t::view(stmts_c v) {
    for (auto &s : v->seq_) {
        if (auto var_p = get_var_if_is_define(s)) {
            if (var_p->ssa_data_->is_garbage()) { continue; }
        }
        dispatch(s);
    }
}

} // namespace sc
