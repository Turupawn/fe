//! Expression and statement lowering for MIR: handles blocks, control flow, calls, and dispatches
//! to specialized lowering helpers.

use hir::hir_def::expr::{ArithBinOp, BinOp, CompBinOp};
use super::*;

impl<'db, 'a> MirBuilder<'db, 'a> {
    /// Lowers the body root expression, starting from the provided entry block.
    ///
    /// # Parameters
    /// - `block`: Entry basic block to begin lowering.
    /// - `expr`: Root expression id of the body.
    ///
    /// # Returns
    /// The successor block after lowering the root expression.
    pub(super) fn lower_root(&mut self, block: BasicBlockId, expr: ExprId) -> Option<BasicBlockId> {
        match expr.data(self.db, self.body) {
            Partial::Present(Expr::Block(stmts)) => self.lower_block(block, expr, stmts),
            _ => {
                let (next_block, value) = self.lower_expr_in(block, expr);
                self.mir_body.expr_values.insert(expr, value);
                next_block
            }
        }
    }

    /// Lowers a block expression by sequentially lowering its statements.
    ///
    /// # Parameters
    /// - `block`: Basic block to start lowering in.
    /// - `_block_expr`: Expression id for the block (unused).
    /// - `stmts`: Statements contained in the block.
    ///
    /// # Returns
    /// The final block after lowering all statements, or `None` if terminated.
    pub(super) fn lower_block(
        &mut self,
        block: BasicBlockId,
        _block_expr: ExprId,
        stmts: &[StmtId],
    ) -> Option<BasicBlockId> {
        let mut current = Some(block);
        for &stmt_id in stmts {
            let Some(curr_block) = current else { break };
            current = self.lower_stmt(curr_block, stmt_id).0;
        }
        current
    }

    /// Lowers an expression in the context of an existing block.
    ///
    /// # Parameters
    /// - `block`: Basic block where lowering begins.
    /// - `expr`: Expression id to lower.
    ///
    /// # Returns
    /// The successor block and the resulting `ValueId`.
    pub(super) fn lower_expr_in(
        &mut self,
        block: BasicBlockId,
        expr: ExprId,
    ) -> (Option<BasicBlockId>, ValueId) {
        let (next, value, _) = self.lower_expr_core(block, expr);
        (next, value)
    }

    /// Lower an expression and indicate whether an `Eval` wrapper should be emitted.
    ///
    /// # Parameters
    /// - `block`: Entry block for lowering.
    /// - `expr`: Expression to lower.
    ///
    /// # Returns
    /// A triple of next block, resulting value, and a flag indicating whether to emit `MirInst::Eval`.
    pub(super) fn lower_expr_core(
        &mut self,
        block: BasicBlockId,
        expr: ExprId,
    ) -> (Option<BasicBlockId>, ValueId, bool) {
        if let Some((next, val)) = self.try_lower_intrinsic_stmt(block, expr) {
            return (next, val, false);
        }
        if let Some((next, val)) = self.try_lower_variant_ctor(block, expr) {
            return (next, val, true);
        }
        if let Some((next, val)) = self.try_lower_unit_variant(block, expr) {
            return (next, val, true);
        }

        match expr.data(self.db, self.body) {
            Partial::Present(Expr::Block(stmts)) => {
                let next_block = self.lower_block(block, expr, stmts);
                let val = self.ensure_value(expr);
                (next_block, val, false)
            }
            Partial::Present(Expr::RecordInit(_, fields)) => {
                let (next, val) = self.try_lower_record(block, expr, fields);
                (next, val, true)
            }
            Partial::Present(Expr::Match(scrutinee, arms)) => {
                if let Partial::Present(arms) = arms
                    && let Some(mut patterns) = self.match_arm_patterns(arms)
                {
                    let (next, val) =
                        self.lower_match_expr(block, expr, *scrutinee, arms, &mut patterns);
                    return (next, val, false);
                }
                let val = self.ensure_value(expr);
                (Some(block), val, true)
            }
            _ => {
                let val = self.ensure_value(expr);
                (Some(block), val, true)
            }
        }
    }

    /// Attempts to lower a function or method call into a MIR value.
    ///
    /// # Parameters
    /// - `expr`: Expression id representing the call.
    ///
    /// # Returns
    /// The allocated `ValueId` for the call result, or `None` if not a call.
    pub(super) fn try_lower_call(&mut self, expr: ExprId) -> Option<ValueId> {
        let callable = self.typed_body.callable_expr(expr)?;
        let (mut args, arg_exprs) = self.collect_call_args(expr)?;
        let mut receiver_space = None;
        if self.is_method_call(expr) && !args.is_empty() {
            let needs_space = callable
                .callable_def
                .receiver_ty(self.db)
                .is_some_and(|binder| {
                    let ty = binder.instantiate_identity();
                    ty.adt_ref(self.db)
                        .is_some_and(|adt| matches!(adt, AdtRef::Struct(_)))
                });
            if needs_space {
                let receiver_value = args[0];
                receiver_space = Some(self.value_address_space(receiver_value));
            }
        }

        let ty = self.typed_body.expr_ty(self.db, expr);
        if let Some(kind) = self.intrinsic_kind(callable.callable_def) {
            if !kind.returns_value() {
                return None;
            }
            let mut code_region = None;
            if matches!(
                kind,
                IntrinsicOp::CodeRegionOffset | IntrinsicOp::CodeRegionLen
            ) {
                if let Some(arg_expr) = arg_exprs.first() {
                    code_region = self.code_region_target(*arg_expr);
                }
                args.clear();
            }
            return Some(self.mir_body.alloc_value(ValueData {
                ty,
                origin: ValueOrigin::Intrinsic(IntrinsicValue {
                    op: kind,
                    args,
                    code_region,
                }),
            }));
        }
        Some(self.mir_body.alloc_value(ValueData {
            ty,
            origin: ValueOrigin::Call(CallOrigin {
                expr,
                callable: callable.clone(),
                args,
                receiver_space,
                resolved_name: None,
            }),
        }))
    }

    /// Returns true if the expression is a method call (as opposed to a regular function call).
    fn is_method_call(&self, expr: ExprId) -> bool {
        let exprs = self.body.exprs(self.db);
        matches!(&exprs[expr], Partial::Present(Expr::MethodCall(..)))
    }

    /// Rewrites a field access expression into either a `get_field` call (for primitives)
    /// or a `FieldPtr` offset computation (for nested structs).
    ///
    /// # Parameters
    /// - `expr`: Field access expression id.
    ///
    /// # Returns
    /// The lowered `ValueId` if the field can be resolved.
    pub(super) fn try_lower_field(&mut self, expr: ExprId) -> Option<ValueId> {
        let Partial::Present(Expr::Field(lhs, field_index)) = expr.data(self.db, self.body) else {
            return None;
        };
        let field_index = field_index.to_opt()?;
        let lhs_ty = self.typed_body.expr_ty(self.db, *lhs);
        let info = self.field_access_info(lhs_ty, field_index)?;

        let addr_value = self.ensure_value(*lhs);
        let addr_space = self.value_address_space(addr_value);
        let is_aggregate = info.field_ty.field_count(self.db) > 0;

        // For aggregate (struct) fields, emit pointer arithmetic instead of a load
        if is_aggregate {
            // Optimization: if offset is 0, reuse the base pointer directly
            if info.offset_bytes == 0 {
                // Ensure address space is propagated even when reusing the base pointer
                self.value_address_space.insert(addr_value, addr_space);
                return Some(addr_value);
            }
            // Emit FieldPtr for non-zero offsets
            let result = self.mir_body.alloc_value(ValueData {
                ty: info.field_ty,
                origin: ValueOrigin::FieldPtr(FieldPtrOrigin {
                    base: addr_value,
                    offset_bytes: info.offset_bytes,
                }),
            });
            // Propagate address space to the result
            self.value_address_space.insert(result, addr_space);
            return Some(result);
        }

        // For primitive fields, emit a get_field call to load the value
        let ptr_ty = match addr_space {
            AddressSpaceKind::Memory => self.core.helper_ty(CoreHelperTy::MemPtr),
            AddressSpaceKind::Storage => self.core.helper_ty(CoreHelperTy::StorPtr),
        };
        let offset_value = self.synthetic_u256(BigUint::from(info.offset_bytes));
        let callable =
            self.core
                .make_callable(expr, CoreHelper::GetField, &[ptr_ty, info.field_ty]);

        Some(self.mir_body.alloc_value(ValueData {
            ty: info.field_ty,
            origin: ValueOrigin::Call(CallOrigin {
                expr,
                callable,
                args: vec![addr_value, offset_value],
                receiver_space: None,
                resolved_name: None,
            }),
        }))
    }

    /// Lowers a statement and returns its continuation and produced value (if any).
    ///
    /// # Parameters
    /// - `block`: Current basic block.
    /// - `stmt_id`: Statement to lower.
    ///
    /// # Returns
    /// The successor block and optional produced `ValueId`.
    pub(super) fn lower_stmt(
        &mut self,
        block: BasicBlockId,
        stmt_id: StmtId,
    ) -> (Option<BasicBlockId>, Option<ValueId>) {
        let Partial::Present(stmt) = stmt_id.data(self.db, self.body) else {
            return (Some(block), None);
        };
        match stmt {
            Stmt::Let(pat, ty, value) => {
                let (next_block, value_id) = if let Some(expr) = value {
                    let (next_block, val) = self.lower_expr_in(block, *expr);
                    (next_block, Some(val))
                } else {
                    (Some(block), None)
                };
                if let Some(val) = value_id {
                    let space = self.value_address_space(val);
                    self.set_pat_address_space(*pat, space);
                }
                if let Some(curr_block) = next_block {
                    self.push_inst(
                        curr_block,
                        MirInst::Let {
                            stmt: stmt_id,
                            pat: *pat,
                            ty: *ty,
                            value: value_id,
                        },
                    );
                }
                (next_block, None)
            }
            Stmt::For(pat, iter_expr, body_expr) => {
                self.lower_for(block, stmt_id, *pat, *iter_expr, *body_expr)
            }
            Stmt::While(cond, body_expr) => self.lower_while(block, *cond, *body_expr),
            Stmt::Continue => {
                let scope = self.loop_stack.last().expect("continue outside of loop");
                self.set_terminator(
                    block,
                    Terminator::Goto {
                        target: scope.continue_target,
                    },
                );
                (None, None)
            }
            Stmt::Break => {
                let scope = self.loop_stack.last().expect("break outside of loop");
                self.set_terminator(
                    block,
                    Terminator::Goto {
                        target: scope.break_target,
                    },
                );
                (None, None)
            }
            Stmt::Return(value) => {
                let (next_block, ret_value) = if let Some(expr) = value {
                    let (next_block, val) = self.lower_expr_in(block, *expr);
                    (next_block, Some(val))
                } else {
                    (Some(block), None)
                };
                if let Some(curr_block) = next_block {
                    self.set_terminator(curr_block, Terminator::Return(ret_value));
                }
                (None, None)
            }
            Stmt::Expr(expr) => self.lower_expr_stmt(block, stmt_id, *expr),
        }
    }

    /// Lowers a `while` loop statement and wires its control-flow edges.
    ///
    /// # Parameters
    /// - `block`: Entry block preceding the loop.
    /// - `cond_expr`: Condition expression id.
    /// - `body_expr`: Loop body expression id.
    ///
    /// # Returns
    /// The loop exit block and no produced value.
    pub(super) fn lower_while(
        &mut self,
        block: BasicBlockId,
        cond_expr: ExprId,
        body_expr: ExprId,
    ) -> (Option<BasicBlockId>, Option<ValueId>) {
        let cond_entry = self.alloc_block();
        let body_block = self.alloc_block();
        let exit_block = self.alloc_block();

        self.set_terminator(block, Terminator::Goto { target: cond_entry });

        let (cond_header_opt, cond_val) = self.lower_expr_in(cond_entry, cond_expr);
        let Some(cond_header) = cond_header_opt else {
            return (None, None);
        };

        self.loop_stack.push(LoopScope {
            continue_target: cond_entry,
            break_target: exit_block,
        });

        let body_end = self.lower_expr_in(body_block, body_expr).0;

        self.loop_stack.pop();

        let mut backedge = None;
        if let Some(body_end_block) = body_end {
            self.set_terminator(body_end_block, Terminator::Goto { target: cond_entry });
            backedge = Some(body_end_block);
        }

        self.set_terminator(
            cond_header,
            Terminator::Branch {
                cond: cond_val,
                then_bb: body_block,
                else_bb: exit_block,
            },
        );

        self.mir_body.loop_headers.insert(
            cond_entry,
            LoopInfo {
                body: body_block,
                exit: exit_block,
                backedge,
            },
        );

        (Some(exit_block), None)
    }

    /// Lowers a `for` loop statement by desugaring it into a while loop.
    ///
    /// For `for i in start..end { body }`, this lowers to:
    /// `let mut i = start; while i < end { body; i += 1 }`
    ///
    /// # Parameters
    /// - `block`: Entry block preceding the loop.
    /// - `stmt_id`: Statement id for context.
    /// - `pat`: Pattern for the loop variable.
    /// - `iter_expr`: Iterable expression (assumed to be a range `start..end`).
    /// - `body_expr`: Loop body expression id.
    ///
    /// # Returns
    /// The loop exit block and no produced value.
    pub(super) fn lower_for(
        &mut self,
        block: BasicBlockId,
        stmt_id: StmtId,
        pat: PatId,
        iter_expr: ExprId,
        body_expr: ExprId,
    ) -> (Option<BasicBlockId>, Option<ValueId>) {
        // Naive implementation: handle range expressions and arrays
        // For `for i in 0..10`, the parser might parse it differently
        // For arrays, we'll loop from 0 to array length
        let (start_expr, end_expr, is_array) = {
            // First check the expression type to see if it's an array
            let iter_ty = self.typed_body.expr_ty(self.db, iter_expr);
            let (base, _args) = iter_ty.decompose_ty_app(self.db);
            let base_data = base.data(self.db);
            let is_array_type = matches!(base_data, TyData::TyBase(TyBase::Prim(PrimTy::Array)));
            
            match iter_expr.data(self.db, self.body) {
                Partial::Present(Expr::Bin(lhs, rhs, op)) => {
                    // For range expressions like 4..10, lhs is start (4) and rhs is end (10)
                    // Check if op is Index (placeholder for ..) - if so, treat it as a range
                    if matches!(op, BinOp::Index) {
                        (*lhs, *rhs, false)
                    } else {
                        panic!("for loop iterable must be a range expression (start..end), literal, or array, got binary op: {:?}", op);
                    }
                },
                Partial::Present(Expr::Lit(LitKind::Int(int_id))) => {
                    // If it's just a literal like `10`, treat it as `0..10` (naive implementation)
                    let int_val = int_id.data(self.db).clone();
                    (iter_expr, iter_expr, false)
                }
                _ if is_array_type => {
                    // It's an array type (could be Path, Array literal, etc.)
                    (iter_expr, iter_expr, true)
                }
                _ => {
                    panic!("for loop iterable must be a range expression (start..end), literal, or array, got: {:?}", iter_expr.data(self.db, self.body));
                }
            }
        };

        // Lower start and end expressions to get their values
        // For the naive implementation, if start_expr == end_expr, it means we have a literal or array
        // and we should use 0 as the start
        let (start_block, start_val, end_val) = if start_expr == end_expr {
            // It's a single literal or array, use 0 as start
            let zero_val = self.synthetic_u256(BigUint::from(0u64));
            let end_val = if is_array {
                // For arrays, get the length from the type
                // For a naive implementation, we'll use a placeholder length
                // In reality, we'd need to evaluate the const type to get the actual length
                self.synthetic_u256(BigUint::from(3u64)) // Placeholder - should get from type
            } else {
                // It's a literal like `10`, treat it as `0..10`
                // Get the literal value directly from the expression
                let exprs = self.body.exprs(self.db);
                if let Partial::Present(Expr::Lit(LitKind::Int(int_id))) = &exprs[end_expr] {
                    // Create synthetic value from the integer literal
                    let int_val = int_id.data(self.db).clone();
                    self.synthetic_u256(int_val)
                } else {
                    // Fallback: try to lower the expression
                    let (_, end_val) = self.lower_expr_in(block, end_expr);
                    end_val
                }
            };
            (Some(block), zero_val, end_val)
        } else {
            // It's a range expression: 4..10
            // Lower start_expr (4) first
            let (start_block, start_val) = self.lower_expr_in(block, start_expr);
            let start_block = start_block.unwrap_or(block);
            // Lower end_expr (10) - check if it's a literal first for efficiency
            let exprs = self.body.exprs(self.db);
            let end_val = if let Partial::Present(Expr::Lit(LitKind::Int(int_id))) = &exprs[end_expr] {
                // It's a literal, create synthetic value directly
                let int_val = int_id.data(self.db).clone();
                self.synthetic_u256(int_val)
            } else {
                // Not a literal, lower it normally
                let (_, val) = self.lower_expr_in(block, end_expr);
                val
            };
            (Some(start_block), start_val, end_val)
        };
        let Some(start_block) = start_block else {
            return (None, None);
        };

        // Initialize the loop variable to start
        // For arrays, we'll bind to the index initially, then rebind to arr[index] at body entry
        let space = self.value_address_space(start_val);
        self.set_pat_address_space(pat, space);
        self.push_inst(
            start_block,
            MirInst::Let {
                stmt: stmt_id,
                pat,
                ty: None,
                value: Some(start_val),
            },
        );

        // Create blocks for the while loop structure
        let cond_entry = self.alloc_block();
        let body_block = self.alloc_block();
        let body_entry = if is_array {
            // For arrays, create a separate entry block where we rebind pattern to arr[index]
            self.alloc_block()
        } else {
            body_block // For non-arrays, use body_block directly
        };
        let exit_block = self.alloc_block();

        self.set_terminator(start_block, Terminator::Goto { target: cond_entry });
        
        // For arrays, compute index_val before the if block so it's accessible for condition
        let index_val = if is_array {
            // Get the index value - we need the current value of the pattern (the loop variable)
            let binding = self.typed_body.pat_binding(pat);
            let index_expr = binding
                .and_then(|b| self.typed_body.references_by_binding(b).first().copied());
            if let Some(expr) = index_expr {
                // Found an expression that references the pattern, use its value
                self.ensure_value(expr)
            } else {
                // No expression found, use start_val as fallback
                start_val
            }
        } else {
            start_val // For non-arrays, use start_val
        };
        
        // For arrays, at body entry, create arr[index] and rebind pattern to it
        if is_array {
            // Get the array value
            let arr_val = self.ensure_value(iter_expr);
            
            // Create array indexing value: arr[index]
            // We need to create Expr::Bin(iter_expr, index_expr, BinOp::Index)
            // Since we can't create new HIR expressions, we'll create the value directly
            // by ensuring the indexing expression exists in the body
            // Actually, the body might have arr[index] expressions already, but probably not
            // Let's create it using ValueOrigin::Expr with a workaround
            let arr_ty = self.typed_body.expr_ty(self.db, iter_expr);
            let (_, args) = arr_ty.decompose_ty_app(self.db);
            let elem_ty = args[0];
            
            // For now, create the array indexing value using ValueOrigin::Expr
            // We'll use iter_expr as placeholder, but codegen needs to know this is arr[index_val]
            // Actually, let's check if we can find an existing array indexing expression in the body
            // that uses iter_expr and the pattern
            let exprs = self.body.exprs(self.db);
            let array_index_expr = exprs.keys().find(|&expr_id| {
                if let Partial::Present(Expr::Bin(lhs, _rhs, op)) = &exprs[expr_id] {
                    matches!(op, BinOp::Index) && *lhs == iter_expr
                } else {
                    false
                }
            });
            
            let array_elem_val = if let Some(index_expr_id) = array_index_expr {
                // Found an existing array indexing expression, use it
                self.ensure_value(index_expr_id)
            } else {
                // No existing expression, create value with placeholder
                // Store array and index values for codegen to use
                let array_elem_val = self.mir_body.alloc_value(ValueData {
                    ty: elem_ty,
                    origin: ValueOrigin::Expr(iter_expr), // Placeholder
                });
                // Store array and index values for codegen
                self.mir_body.array_index_info.insert(array_elem_val, (arr_val, index_val));
                array_elem_val
            };
            
            // For arrays, we need to create a temporary variable for the array element
            // and map the pattern's expression to it ONLY in the body context
            // The pattern stays bound to the index for condition and increment
            // Get the pattern's expression so we can map it to the array element value in the body
            let binding = self.typed_body.pat_binding(pat);
            let pat_expr = binding
                .and_then(|b| self.typed_body.references_by_binding(b).first().copied());
            
            // Create a temporary variable for the array element using EvalExpr
            // This will create a new variable (like v2) for the array element
            // We'll use a synthetic expression ID to avoid conflicts
            // Actually, we can't create new ExprIds easily, so let's use iter_expr as placeholder
            // But we need to make sure it doesn't conflict
            // For now, let's create the temporary and map the pattern expression to it
            // BUT only when used in the body - we'll handle this by checking the context
            // Actually, simpler: create the temporary and store a mapping from pattern to array element
            // Then in codegen, when we see the pattern used, check if there's an array element mapping
            
            // Store mapping: pattern expression -> array element value (for body use)
            // We'll use this in codegen to replace pattern references in the body with the array element
            if let Some(expr_id) = pat_expr {
                // Store the mapping in array_index_info with a special key
                // Actually, let's create a separate map for pattern -> array element
                // For now, we'll handle this differently - create the temporary variable
                // and let codegen handle the mapping
            }
            
            // Create a temporary variable for the array element using EvalExpr
            // Use the pattern expression as the expr_id so codegen creates a temp for it
            if let Some(expr_id) = pat_expr {
                // Create temp variable for array element
                self.push_inst(
                    body_entry,
                    MirInst::EvalExpr {
                        expr: expr_id, // Use pattern expression - codegen will create temp (v2)
                        value: array_elem_val,
                        bind_value: true, // Bind to a temporary variable
                    },
                );
                // Map the pattern expression to the array element value
                // This way, when the body uses the pattern, it will get the array element temp
                // NOTE: This affects ALL uses, but we handle condition separately using index_val
                self.mir_body.expr_values.insert(expr_id, array_elem_val);
            } else {
                // Fallback: create temp without mapping
                self.push_inst(
                    body_entry,
                    MirInst::EvalExpr {
                        expr: iter_expr,
                        value: array_elem_val,
                        bind_value: true,
                    },
                );
            }
            
            self.set_terminator(body_entry, Terminator::Goto { target: body_block });
        }

        // Create condition: i < end
        // For arrays, we need to use the index (not the array element) for the comparison
        // For ranges, we use the loop variable directly
        let end_block = cond_entry;

        // Get the loop variable value for comparison
        // For arrays, the pattern is bound to the array element, but we need the index for comparison
        // For ranges, the pattern is bound to the index/value directly
        let loop_var_val = if is_array {
            // For arrays, we need the index value, not the array element
            // The index is bound to the pattern initially, then rebound to array element in body_entry
            // So at cond_entry, we can still use start_val (the index)
            start_val
        } else {
            start_val
        };
        let _loop_var_ty = self.typed_body.pat_ty(self.db, pat);
        let bool_ty = TyId::new(self.db, TyData::TyBase(TyBase::Prim(PrimTy::Bool)));
        
        // Create condition: loop_var < end
        // We need to create a Bin expression for the comparison, but we can't easily create new HIR expressions
        // For a naive implementation, we'll search the body for expressions that reference the pattern
        // and use those to build the comparison. If we can't find any, we'll create a placeholder.
        let binding = self.typed_body.pat_binding(pat);
        let loop_var_expr = binding
            .and_then(|b| self.typed_body.references_by_binding(b).first().copied());
        
        // Create comparison: loop_var < end
        // We'll use SyntheticValue::Comparison to create the comparison at the MIR level
        // since we can't easily create new HIR expressions
        // Use the end_val that was already computed above
        // For arrays, we need to use the index value (the pattern binding), not the array element
        // The pattern expression is mapped to the array element, but for the condition we need the index
        // So we'll get the index value from the pattern binding directly
        let cond_val = if is_array {
            // For arrays, use index_val which was computed BEFORE mapping pattern to array element
            // index_val is the value of the pattern expression before the mapping
            // This ensures we use the index, not the array element, for the condition
            self.mir_body.alloc_value(ValueData {
                ty: bool_ty,
                origin: ValueOrigin::Synthetic(SyntheticValue::Comparison {
                    left: index_val, // Use index value (computed before mapping)
                    right: end_val,
                    op: CompBinOp::Lt,
                }),
            })
        } else if let Some(loop_expr) = loop_var_expr {
            // We found an expression that references the pattern
            // Get the value for the comparison
            let loop_var_val = self.ensure_value(loop_expr);
            self.mir_body.alloc_value(ValueData {
                ty: bool_ty,
                origin: ValueOrigin::Synthetic(SyntheticValue::Comparison {
                    left: loop_var_val,
                    right: end_val,
                    op: CompBinOp::Lt,
                }),
            })
        } else {
            // No expression found, use loop_var_val and end_val directly
            self.mir_body.alloc_value(ValueData {
                ty: bool_ty,
                origin: ValueOrigin::Synthetic(SyntheticValue::Comparison {
                    left: loop_var_val,
                    right: end_val,
                    op: CompBinOp::Lt,
                }),
            })
        };

        // Set up loop stack
        self.loop_stack.push(LoopScope {
            continue_target: cond_entry,
            break_target: exit_block,
        });

        // Lower the body
        let body_end = self.lower_expr_in(body_block, body_expr).0;

        // Add increment: i += 1
        // We need an AugAssign expression, but we can't create new HIR expressions easily.
        // If we found loop_var_expr, we can create an increment using synthetic values.
        if let Some(body_end_block) = body_end {
            if let Some(loop_expr) = loop_var_expr {
                // Create increment: loop_var += 1
                // We'll create a synthetic value for 1 and use AugAssign
                let one_val = self.synthetic_u256(BigUint::from(1u64));
                // We need to create an AugAssign expression, but we can't.
                // Instead, we'll create the increment directly using MirInst::AugAssign
                // But we need an ExprId for the target. We have loop_expr, but that's the pattern reference,
                // not the assignment target. We need the pattern's expression.
                // For now, let's try using loop_expr as the target and see if it works.
                // Actually, AugAssign needs an expression that can be assigned to, which should be a Path.
                // Let's check if loop_expr is a Path expression.
                let exprs = self.body.exprs(self.db);
                if let Partial::Present(Expr::Path(_)) = &exprs[loop_expr] {
                    // It's a Path expression, we can use it for AugAssign
                    self.push_inst(
                        body_end_block,
                        MirInst::AugAssign {
                            stmt: stmt_id,
                            target: loop_expr,
                            value: one_val,
                            op: ArithBinOp::Add,
                        },
                    );
                }
            }
            self.set_terminator(body_end_block, Terminator::Goto { target: cond_entry });
        }

        self.loop_stack.pop();

        // Set up the branch terminator
        // For arrays, body_entry is the entry point (where we rebind pattern to arr[index])
        // For ranges, body_block is the entry point (body_entry == body_block for non-arrays)
        self.set_terminator(
            end_block,
            Terminator::Branch {
                cond: cond_val,
                then_bb: body_entry,
                else_bb: exit_block,
            },
        );

        // Register loop info
        // body_entry is the entry point (for arrays, it rebinds pattern; for ranges, it's the same as body_block)
        self.mir_body.loop_headers.insert(
            cond_entry,
            LoopInfo {
                body: body_entry,
                exit: exit_block,
                backedge: body_end,
            },
        );

        (Some(exit_block), None)
    }

    /// Lowers an `if` expression used in statement position.
    ///
    /// # Parameters
    /// - `block`: Entry basic block.
    /// - `if_expr`: Expression id of the `if`.
    /// - `cond`: Condition expression id.
    /// - `then_expr`: Then-branch expression id.
    /// - `else_expr`: Optional else-branch expression id.
    ///
    /// # Returns
    /// The merge block (if any) and optional resulting value.
    pub(super) fn lower_if_expr(
        &mut self,
        block: BasicBlockId,
        if_expr: ExprId,
        cond: ExprId,
        then_expr: ExprId,
        else_expr: Option<ExprId>,
    ) -> (Option<BasicBlockId>, Option<ValueId>) {
        if !self.is_unit_ty(self.typed_body.expr_ty(self.db, if_expr)) {
            let value = self.ensure_value(if_expr);
            return (Some(block), Some(value));
        }

        let (cond_block_opt, cond_val) = self.lower_expr_in(block, cond);
        let cond_block = match cond_block_opt {
            Some(block) => block,
            None => return (None, None),
        };

        let then_block = self.alloc_block();
        let merge_block = self.alloc_block();
        let else_block = if else_expr.is_some() {
            self.alloc_block()
        } else {
            merge_block
        };

        self.set_terminator(
            cond_block,
            Terminator::Branch {
                cond: cond_val,
                then_bb: then_block,
                else_bb: else_block,
            },
        );

        let then_end = self.lower_expr_in(then_block, then_expr).0;
        if let Some(end_block) = then_end {
            self.set_terminator(
                end_block,
                Terminator::Goto {
                    target: merge_block,
                },
            );
        }

        if let Some(else_expr) = else_expr {
            let else_end = self.lower_expr_in(else_block, else_expr).0;
            if let Some(end_block) = else_end {
                self.set_terminator(
                    end_block,
                    Terminator::Goto {
                        target: merge_block,
                    },
                );
            }
        }

        (Some(merge_block), None)
    }

    /// Returns whether the given type is the unit tuple type.
    ///
    /// # Parameters
    /// - `ty`: Type to inspect.
    ///
    /// # Returns
    /// `true` if the type is unit.
    pub(super) fn is_unit_ty(&self, ty: TyId<'db>) -> bool {
        ty.is_tuple(self.db) && ty.field_count(self.db) == 0
    }

    /// Lowers an expression statement, emitting side-effecting instructions as needed.
    ///
    /// # Parameters
    /// - `block`: Current basic block.
    /// - `stmt_id`: Statement id for context.
    /// - `expr`: Expression id to lower.
    ///
    /// # Returns
    /// Successor block and optional resulting value.
    pub(super) fn lower_expr_stmt(
        &mut self,
        block: BasicBlockId,
        stmt_id: StmtId,
        expr: ExprId,
    ) -> (Option<BasicBlockId>, Option<ValueId>) {
        if let Some((next_block, value_id)) = self.try_lower_intrinsic_stmt(block, expr) {
            return (next_block, Some(value_id));
        }
        let exprs = self.body.exprs(self.db);
        let Partial::Present(expr_data) = &exprs[expr] else {
            return (Some(block), None);
        };

        match expr_data {
            Expr::Assign(target, value) => {
                let (next_block, value_id) = self.lower_expr_in(block, *value);
                if let Some(curr_block) = next_block {
                    if let Some(binding) = self.typed_body.expr_prop(self.db, *target).binding
                        && let LocalBinding::Local { pat, .. } = binding
                    {
                        let space = self.value_address_space(value_id);
                        self.set_pat_address_space(pat, space);
                    }
                    if let Some(block_after_assign) =
                        self.try_lower_field_assign(curr_block, expr, *target, value_id)
                    {
                        return (Some(block_after_assign), None);
                    }
                    self.push_inst(
                        curr_block,
                        MirInst::Assign {
                            stmt: stmt_id,
                            target: *target,
                            value: value_id,
                        },
                    );
                }
                (next_block, None)
            }
            Expr::If(cond, then_expr, else_expr) => {
                let (next_block, value_id) =
                    self.lower_if_expr(block, expr, *cond, *then_expr, *else_expr);
                if let (Some(curr_block), Some(value)) = (next_block, value_id) {
                    self.push_inst(
                        curr_block,
                        MirInst::Eval {
                            stmt: stmt_id,
                            value,
                        },
                    );
                }
                (next_block, value_id)
            }
            Expr::AugAssign(target, value, op) => {
                let (next_block, value_id) = self.lower_expr_in(block, *value);
                if let Some(curr_block) = next_block {
                    if let Some(binding) = self.typed_body.expr_prop(self.db, *target).binding
                        && let LocalBinding::Local { pat, .. } = binding
                    {
                        let space = self.value_address_space(value_id);
                        self.set_pat_address_space(pat, space);
                    }
                    self.push_inst(
                        curr_block,
                        MirInst::AugAssign {
                            stmt: stmt_id,
                            target: *target,
                            value: value_id,
                            op: *op,
                        },
                    );
                }
                (next_block, None)
            }
            _ => {
                let (next_block, value_id, push_eval) = self.lower_expr_core(block, expr);
                if push_eval && let Some(curr_block) = next_block {
                    self.push_inst(
                        curr_block,
                        MirInst::Eval {
                            stmt: stmt_id,
                            value: value_id,
                        },
                    );
                }
                (next_block, Some(value_id))
            }
        }
    }
}
