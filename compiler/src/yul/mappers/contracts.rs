use crate::errors::CompileError;
use crate::yul::mappers::{constructor, functions, types};
use crate::yul::namespace::scopes::{ContractScope, ModuleScope, Scope, Shared};
use crate::yul::namespace::types::Type;
use crate::yul::runtime::abi as runtime_abi;
use crate::yul::runtime::functions as runtime_functions;

use std::rc::Rc;
use vyper_parser::ast as vyp;
use vyper_parser::span::Spanned;
use yultsur::*;

/// Builds a Yul object from a Vyper contract.
pub fn contract_def(
    module_scope: Shared<ModuleScope>,
    name: String,
    body: &Vec<Spanned<vyp::ContractStmt>>,
) -> Result<yul::Statement, CompileError> {
    let contract_scope = ContractScope::new(Rc::clone(&module_scope));

    let mut statements = body
        .iter()
        .map(|stmt| contract_stmt(Rc::clone(&contract_scope), &stmt.node))
        .collect::<Result<Vec<Option<yul::Statement>>, CompileError>>()?
        .into_iter()
        .filter_map(|stmt| stmt)
        .collect::<Vec<yul::Statement>>();

    statements.append(&mut runtime_functions::all());
    statements.push(runtime_abi::switch(
        &contract_scope.borrow().interface,
        &contract_scope.borrow().defs,
    )?);

    Ok(yul::Statement::Object(yul::Object {
        // TODO: use actual name
        name: identifier! { Contract },
        code: constructor::code(),
        objects: vec![yul::Object {
            name: identifier! { runtime },
            code: yul::Code {
                block: yul::Block { statements },
            },
            objects: vec![],
        }],
    }))
}

fn contract_stmt(
    scope: Shared<ContractScope>,
    stmt: &vyp::ContractStmt,
) -> Result<Option<yul::Statement>, CompileError> {
    match stmt {
        vyp::ContractStmt::ContractField { qual, name, typ } => {
            contract_field(scope, qual, name.node.to_string(), &typ.node)?;
            Ok(None)
        }
        vyp::ContractStmt::FuncDef {
            qual,
            name,
            args,
            return_type,
            body,
        } => {
            let function =
                functions::func_def(scope, qual, name.node.to_string(), args, return_type, body)?;
            Ok(Some(function))
        }
        _ => Err(CompileError::static_str(
            "Unable to translate module statement.",
        )),
    }
}

fn contract_field(
    scope: Shared<ContractScope>,
    qual: &Option<Spanned<vyp::ContractFieldQual>>,
    name: String,
    typ: &vyp::TypeDesc,
) -> Result<(), CompileError> {
    match types::type_desc(Scope::Contract(Rc::clone(&scope)), typ)? {
        Type::Map(map) => scope.borrow_mut().add_map(name, map),
        _ => {
            return Err(CompileError::static_str(
                "Contract field type not supported",
            ))
        }
    };

    Ok(())
}
