use fe_parser2::ast;

use crate::span::HirOrigin;

use super::{
    AttrListId, Body, FnParamListId, GenericParamListId, IdentId, TypeId, WherePredicateId,
};

#[salsa::tracked]
pub struct Fn {
    #[id]
    pub name: super::IdentId,
    pub generic_params: GenericParamListId,
    pub where_predicate: WherePredicateId,
    pub params: FnParamListId,
    pub ret_ty: Option<TypeId>,
    pub modifier: ItemModifier,
    pub attributes: AttrListId,
    pub body: Option<Body>,

    pub(crate) origin: HirOrigin<ast::AstPtr<ast::Fn>>,
}

#[salsa::tracked]
pub struct Struct {
    #[id]
    pub name: super::IdentId,

    pub is_pub: bool,
    pub generic_params: GenericParamListId,
    pub where_predicate: WherePredicateId,
    pub attributes: AttrListId,
    pub fields: RecordFieldListId,

    pub(crate) origin: HirOrigin<ast::AstPtr<ast::Struct>>,
}

#[salsa::tracked]
pub struct Contract {
    #[id]
    pub name: super::IdentId,

    pub is_pub: bool,
    pub attributes: AttrListId,
    pub fields: RecordFieldListId,

    pub(crate) origin: HirOrigin<ast::AstPtr<ast::Contract>>,
}

#[salsa::tracked]
pub struct Enum {
    #[id]
    pub name: super::IdentId,

    pub is_pub: bool,
    pub generic_params: GenericParamListId,
    pub attributes: AttrListId,
    pub where_predicate: WherePredicateId,
    pub variants: EnumVariantListId,

    pub(crate) origin: HirOrigin<ast::AstPtr<ast::Enum>>,
}

#[salsa::tracked]
pub struct TypeAlias {
    #[id]
    pub name: super::IdentId,

    pub is_pub: bool,
    pub generic_params: GenericParamListId,
    pub attributes: AttrListId,
    pub where_predicate: WherePredicateId,
    pub ty: TypeId,

    pub(crate) origin: HirOrigin<ast::AstPtr<ast::TypeAlias>>,
}

#[salsa::tracked]
pub struct Impl {
    #[id]
    pub ty: super::TypeId,

    pub generic_params: GenericParamListId,
    pub attributes: AttrListId,
    pub where_predicate: WherePredicateId,
    pub items: ImplItemListId,

    pub(crate) origin: HirOrigin<ast::AstPtr<ast::Impl>>,
}

#[salsa::tracked]
pub struct Trait {
    #[id]
    pub name: super::IdentId,

    pub generic_params: GenericParamListId,
    pub attributes: AttrListId,
    pub where_predicate: WherePredicateId,
    pub items: TraitItemListId,

    pub(crate) origin: HirOrigin<ast::AstPtr<ast::Trait>>,
}

#[salsa::tracked]
pub struct ImplTrait {
    #[id]
    pub trait_path: super::PathId,
    #[id]
    pub ty: TypeId,

    pub generic_params: GenericParamListId,
    pub attributes: AttrListId,
    pub where_predicate: WherePredicateId,
    pub items: ImplTraitItemListId,

    pub(crate) origin: HirOrigin<ast::AstPtr<ast::ImplTrait>>,
}

#[salsa::tracked]
pub struct Const {
    #[id]
    pub name: super::IdentId,
    pub body: Body,

    pub(crate) origin: HirOrigin<ast::AstPtr<ast::Const>>,
}

#[salsa::tracked]
pub struct Use {
    pub name: super::UseTreeId,

    pub(crate) origin: HirOrigin<ast::AstPtr<ast::Use>>,
}

#[salsa::tracked]
pub struct Extern {
    pub items: ExternItemListId,

    pub(crate) origin: HirOrigin<ast::AstPtr<ast::Extern>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::From)]
pub enum ItemKind {
    Fn(Fn),
    Struct(Struct),
    Contract(Contract),
    Enum(Enum),
    TypeAlias(TypeAlias),
    Impl(Impl),
    Trait(Trait),
    ImplTrait(ImplTrait),
    Const(Const),
    Use(Use),
    Extern(Extern),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemId {
    Ident(IdentId),
    Ty(TypeId),
    Ty2(TypeId, TypeId),
    Extern(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemModifier {
    Pub,
    Unsafe,
    PubAndUnsafe,
    None,
}

#[salsa::interned]
pub struct RecordFieldListId {
    #[return_ref]
    fields: Vec<RecordField>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordField {
    name: IdentId,
    ty: TypeId,
    is_pub: bool,
}

#[salsa::interned]
pub struct EnumVariantListId {
    #[return_ref]
    variants: Vec<EnumVariant>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnumVariant {
    name: IdentId,
    ty: TypeId,
}

#[salsa::interned]
pub struct ImplItemListId {
    #[return_ref]
    items: Vec<Fn>,
}

pub type TraitItemListId = ImplItemListId;
pub type ImplTraitItemListId = ImplItemListId;
pub type ExternItemListId = ImplItemListId;
