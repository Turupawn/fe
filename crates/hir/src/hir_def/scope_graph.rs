use std::io;

use common::indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{
    scope_graph_viz::ScopeGraphFormatter, Body, Enum, ExprId, Func, FuncParamName, IdentId,
    IngotId, ItemKind, TopLevelMod, Use, VariantKind, Visibility,
};
use crate::{
    hir_def::{BodyKind, GenericParamOwner},
    span::DynLazySpan,
    HirDb,
};

/// Represents a scope relation graph in a top-level module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeGraph<'db> {
    /// The top-level module containing the scope graph.
    pub top_mod: TopLevelMod<'db>,
    /// The scopes in the graph.
    pub scopes: FxHashMap<ScopeId<'db>, Scope<'db>>,
    /// The all unresolved uses in the graph, this is used in name resolution.
    pub unresolved_uses: FxHashSet<Use<'db>>,
}

impl<'db> ScopeGraph<'db> {
    /// Represents all item scopes in a top-level module in depth-first order.
    pub fn items_dfs<'a>(&'a self, db: &'a dyn HirDb) -> impl Iterator<Item = ItemKind<'a>> + 'a {
        ScopeGraphItemIterDfs {
            db,
            graph: self,
            visited: Default::default(),
            stack: vec![self.top_mod.scope()],
        }
    }

    /// Returns the direct child items of the given `scope`.
    pub fn child_items(&self, scope: ScopeId<'db>) -> impl Iterator<Item = ItemKind<'db>> + '_ {
        self.children(scope).filter_map(|child| child.to_item())
    }

    /// Returns the direct child scopes of the given `scope`
    pub fn children(&self, scope: ScopeId<'db>) -> impl Iterator<Item = ScopeId<'db>> + '_ {
        self.edges(scope).iter().filter_map(|edge| match edge.kind {
            EdgeKind::Lex(_)
            | EdgeKind::Super(_)
            | EdgeKind::Ingot(_)
            | EdgeKind::SelfTy(_)
            | EdgeKind::Self_(_) => None,

            _ => Some(edge.dest),
        })
    }

    /// Returns the all edges outgoing from the given `scope`.
    pub fn edges(&self, scope: ScopeId<'db>) -> &IndexSet<ScopeEdge<'db>> {
        &self.scopes[&scope].edges
    }

    /// Write a scope graph as a dot file format to given `w`.
    pub fn write_as_dot(&self, db: &dyn HirDb, w: &mut impl io::Write) -> io::Result<()> {
        ScopeGraphFormatter::new(db, self).render(w)
    }

    pub fn scope_data(&self, scope: &ScopeId<'db>) -> &Scope {
        &self.scopes[scope]
    }
}

/// An reference to a `[ScopeData]` in a `ScopeGraph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, salsa::Update)]
pub enum ScopeId<'db> {
    /// An item scope.
    Item(ItemKind<'db>),

    /// A generic parameter scope.
    GenericParam(ItemKind<'db>, usize),

    /// A function parameter scope.
    FuncParam(ItemKind<'db>, usize),

    /// A field scope.
    Field(FieldParent<'db>, usize),

    /// A variant scope.
    Variant(ItemKind<'db>, usize),

    /// A block scope.
    Block(Body<'db>, ExprId),
}
impl<'db> ScopeId<'db> {
    /// Returns the top level module containing this scope.
    pub fn top_mod(&self, db: &'db dyn HirDb) -> TopLevelMod<'db> {
        match self {
            ScopeId::Item(item) => item.top_mod(db),
            ScopeId::GenericParam(item, _) => item.top_mod(db),
            ScopeId::FuncParam(item, _) => item.top_mod(db),
            ScopeId::Field(FieldParent::Item(item), _) => item.top_mod(db),
            ScopeId::Field(FieldParent::Variant(item, _), _) => item.top_mod(db),
            ScopeId::Variant(item, _) => item.top_mod(db),
            ScopeId::Block(body, _) => body.top_mod(db),
        }
    }

    /// Convert an item to a scope id.
    pub fn from_item(item: ItemKind<'db>) -> Self {
        Self::Item(item)
    }

    /// Convert a scope id to an item if the scope is an item.
    pub fn to_item(self) -> Option<ItemKind<'db>> {
        match self {
            ScopeId::Item(item) => Some(item),
            _ => None,
        }
    }

    /// Returns the nearest item that contains this scope.
    /// If the scope is item itself, returns the item.
    pub fn item(self) -> ItemKind<'db> {
        match self {
            ScopeId::Item(item) => item,
            ScopeId::GenericParam(item, _) => item,
            ScopeId::FuncParam(item, _) => item,
            ScopeId::Field(FieldParent::Item(item), _) => item,
            ScopeId::Field(FieldParent::Variant(item, _), _) => item,
            ScopeId::Variant(item, _) => item,
            ScopeId::Block(body, _) => body.into(),
        }
    }

    /// Returns the scope graph containing this scope.
    pub fn scope_graph(self, db: &'db dyn HirDb) -> &'db ScopeGraph<'db> {
        self.top_mod(db).scope_graph(db)
    }

    pub fn edges(self, db: &'db dyn HirDb) -> &'db IndexSet<ScopeEdge<'db>> {
        self.scope_graph(db).edges(self)
    }

    /// Returns `true` if `scope` is reachable from `self` by following only
    /// lexical edges.
    pub fn is_lex_child(self, db: &dyn HirDb, scope: &ScopeId) -> bool {
        if self.top_mod(db) != scope.top_mod(db) {
            return false;
        }

        match self.lex_parent(db) {
            Some(lex_parent) => {
                if &lex_parent == scope {
                    return true;
                }
                lex_parent.is_lex_child(db, scope)
            }
            None => false,
        }
    }

    /// Returns true if `self` is a transitive reflexive child of `of`.
    pub fn is_transitive_child_of(self, db: &dyn HirDb, of: ScopeId) -> bool {
        let mut current = Some(self);

        while let Some(scope) = current {
            if scope == of {
                return true;
            }
            current = scope.parent(db);
        }

        false
    }

    /// Return the `IngotId` containing the scope.
    pub fn ingot(self, db: &'db dyn HirDb) -> IngotId<'db> {
        self.top_mod(db).ingot(db)
    }

    /// Returns the `Scope` data for this scope.
    pub fn data(self, db: &'db dyn HirDb) -> &'db Scope<'db> {
        self.top_mod(db).scope_graph(db).scope_data(&self)
    }

    /// Returns the parent scope of this scope.
    /// The parent scope is
    /// 1. the lexical parent if it exists
    /// 2. the parent module if 1. does not exist
    pub fn parent(self, db: &'db dyn HirDb) -> Option<Self> {
        let mut super_dest = None;
        for edge in self.edges(db) {
            if let EdgeKind::Lex(_) = edge.kind {
                return Some(edge.dest);
            }
            if let EdgeKind::Super(_) = edge.kind {
                super_dest = Some(edge.dest);
            }
        }
        super_dest
    }

    /// Returns the lexical parent scope of this scope.
    pub fn lex_parent(self, db: &'db dyn HirDb) -> Option<Self> {
        self.data(db)
            .edges
            .iter()
            .find(|e| matches!(e.kind, EdgeKind::Lex(_)))
            .map(|e| e.dest)
    }

    /// Returns the parent module of this scope.
    pub fn parent_module(self, db: &'db dyn HirDb) -> Option<Self> {
        let parent_item = self.parent_item(db)?;
        match parent_item {
            ItemKind::Mod(_) | ItemKind::TopMod(_) => Some(Self::Item(parent_item)),
            _ => {
                let parent_id = Self::from_item(parent_item);
                parent_id.parent_module(db)
            }
        }
    }

    /// Returns the direct child items of the given `scope`.
    pub fn child_items(self, db: &'db dyn HirDb) -> impl Iterator<Item = ItemKind<'db>> + 'db {
        self.scope_graph(db).child_items(self)
    }

    /// Returns the direct child scopes of the given `scope`
    pub fn children(self, db: &'db dyn HirDb) -> impl Iterator<Item = ScopeId<'db>> + 'db {
        self.scope_graph(db).children(self)
    }

    /// Returns `true` if the scope is a type.
    pub fn is_type(self) -> bool {
        match self {
            ScopeId::Item(item) => item.is_type(),
            ScopeId::GenericParam(..) => true,
            _ => false,
        }
    }

    pub fn is_enum(self) -> bool {
        matches!(self, ScopeId::Item(ItemKind::Enum(_)))
    }

    pub fn is_mod(self) -> bool {
        matches!(self, ScopeId::Item(ItemKind::Mod(_) | ItemKind::TopMod(_)))
    }

    /// Returns `true` if the scope is a trait definition.
    pub fn is_trait(self) -> bool {
        matches!(self, ScopeId::Item(ItemKind::Trait(_)))
    }

    /// Returns the item that contains this scope.
    pub fn parent_item(self, db: &'db dyn HirDb) -> Option<ItemKind<'db>> {
        let mut parent = self.parent(db)?;
        loop {
            match parent {
                ScopeId::Item(item) => return Some(item),
                _ => {
                    parent = parent.parent(db)?;
                }
            }
        }
    }

    pub fn name(self, db: &'db dyn HirDb) -> Option<IdentId<'db>> {
        match self.data(db).id {
            ScopeId::Item(item) => item.name(db),

            ScopeId::Variant(parent, idx) => {
                let enum_: Enum = parent.try_into().unwrap();
                enum_.variants(db).data(db)[idx].name.to_opt()
            }

            ScopeId::Field(FieldParent::Item(parent), idx) => match parent {
                ItemKind::Struct(s) => s.fields(db).data(db)[idx].name.to_opt(),
                ItemKind::Contract(c) => c.fields(db).data(db)[idx].name.to_opt(),
                _ => unreachable!(),
            },
            ScopeId::Field(FieldParent::Variant(parent, vidx), fidx) => {
                let enum_: Enum = parent.try_into().unwrap();
                match enum_.variants(db).data(db)[vidx].kind {
                    VariantKind::Record(fields) => fields.data(db)[fidx].name.to_opt(),
                    _ => unreachable!(),
                }
            }

            ScopeId::FuncParam(parent, idx) => {
                let func: Func = parent.try_into().unwrap();
                let param = &func.params(db).to_opt()?.data(db)[idx];
                if let Some(FuncParamName::Ident(ident)) = param.label {
                    Some(ident)
                } else {
                    param.name()
                }
            }

            ScopeId::GenericParam(parent, idx) => {
                let parent = GenericParamOwner::from_item_opt(parent).unwrap();

                let params = &parent.params(db).data(db)[idx];
                params.name().to_opt()
            }

            ScopeId::Block(..) => None,
        }
    }

    pub fn name_span(self, db: &'db dyn HirDb) -> Option<DynLazySpan<'db>> {
        match self.data(db).id {
            ScopeId::Item(item) => item.name_span(),

            ScopeId::Variant(parent, idx) => {
                let enum_: Enum = parent.try_into().unwrap();
                Some(enum_.lazy_span().variants().variant(idx).name().into())
            }

            ScopeId::Field(FieldParent::Item(parent), idx) => match parent {
                ItemKind::Struct(s) => Some(s.lazy_span().fields().field(idx).name().into()),
                ItemKind::Contract(c) => Some(c.lazy_span().fields().field(idx).name().into()),
                _ => unreachable!(),
            },
            ScopeId::Field(FieldParent::Variant(parent, vidx), fidx) => {
                let enum_: Enum = parent.try_into().unwrap();
                Some(
                    enum_
                        .lazy_span()
                        .variants()
                        .variant(vidx)
                        .fields()
                        .field(fidx)
                        .name()
                        .into(),
                )
            }

            ScopeId::FuncParam(parent, idx) => {
                let func: Func = parent.try_into().unwrap();
                let param = &func.params(db).to_opt()?.data(db)[idx];
                let param_span = func.lazy_span().params().param(idx);
                if let Some(FuncParamName::Ident(_)) = param.label {
                    Some(param_span.label().into())
                } else {
                    Some(param_span.name().into())
                }
            }

            ScopeId::GenericParam(parent, idx) => {
                let parent = GenericParamOwner::from_item_opt(parent).unwrap();

                Some(parent.params_span().param(idx).into())
            }

            ScopeId::Block(..) => None,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            ScopeId::Item(item) => item.kind_name(),
            ScopeId::GenericParam(_, _) => "type",
            ScopeId::FuncParam(_, _) => "value",
            ScopeId::Field(_, _) => "field",
            ScopeId::Variant(_, _) => "value",
            ScopeId::Block(_, _) => "block",
        }
    }
    pub fn pretty_path(self, db: &dyn HirDb) -> Option<String> {
        let name = match self {
            ScopeId::Block(body, expr) => format!("{{block{}}}", body.iter_block(db)[&expr]),
            ScopeId::Item(ItemKind::Body(body)) => match body.body_kind(db) {
                BodyKind::FuncBody => "{fn_body}".to_string(),
                BodyKind::Anonymous => "{anonymous_body}".to_string(),
            },
            _ => self.name(db)?.data(db).clone(),
        };

        if let Some(parent) = self.parent(db) {
            let parent_path = parent.pretty_path(db)?;
            Some(format!("{}::{}", parent_path, name))
        } else {
            Some(name)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, salsa::Update)]
pub enum FieldParent<'db> {
    Item(ItemKind<'db>),
    Variant(ItemKind<'db>, usize),
}

impl<'db> FieldParent<'db> {
    pub fn scope(self) -> ScopeId<'db> {
        match self {
            FieldParent::Item(item) => ScopeId::Item(item),
            FieldParent::Variant(variant, idx) => ScopeId::Variant(variant, idx),
        }
    }
}

struct ScopeGraphItemIterDfs<'db, 'a> {
    db: &'db dyn HirDb,
    graph: &'a ScopeGraph<'db>,
    visited: FxHashSet<ScopeId<'db>>,
    stack: Vec<ScopeId<'db>>,
}

impl<'db> std::iter::Iterator for ScopeGraphItemIterDfs<'db, '_> {
    type Item = ItemKind<'db>;

    fn next(&mut self) -> Option<ItemKind<'db>> {
        while let Some(scope) = self.stack.pop() {
            self.visited.insert(scope);
            for edge in self.graph.edges(scope).iter().rev() {
                let dest = edge.dest;
                let top_mod = dest.top_mod(self.db);
                if top_mod != self.graph.top_mod || self.visited.contains(&dest) {
                    continue;
                }

                match dest {
                    ScopeId::Item(_) | ScopeId::Block(..) => {
                        self.stack.push(dest);
                    }

                    _ => {}
                }
            }

            if let Some(item) = scope.to_item() {
                return Some(item);
            }
        }

        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope<'db> {
    pub id: ScopeId<'db>,
    pub edges: IndexSet<ScopeEdge<'db>>,
    pub vis: Visibility,
}

impl<'db> Scope<'db> {
    pub fn new(kind: ScopeId<'db>, vis: Visibility) -> Self {
        Self {
            id: kind,
            edges: Default::default(),
            vis,
        }
    }
}

/// An edge of the scope graph.
/// The edge contains the destination of the edge and the kind of the edge.
/// [`EdgeKind`] is contains supplementary information about the destination
/// scope, which is used for name resolution.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScopeEdge<'db> {
    pub dest: ScopeId<'db>,
    pub kind: EdgeKind<'db>,
}

/// A specific edge property definitions.
///
/// NOTE: The internal types of each variants contains very small amount of
/// information, the reason why we need to prepare each internal types is to
/// allow us to implement traits to each edges directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub enum EdgeKind<'db> {
    /// An edge to a lexical parent scope.
    Lex(LexEdge),
    /// An edge to a module.
    Mod(ModEdge<'db>),
    /// An edge to a type.
    Type(TypeEdge<'db>),
    /// An edge to a trait.
    Trait(TraitEdge<'db>),
    /// An edge from a scope to a generic parameter.
    GenericParam(GenericParamEdge<'db>),
    /// An edge to a value. The value is either a function or a
    /// constant.
    Value(ValueEdge<'db>),
    /// An edge to a field definition scope.
    Field(FieldEdge<'db>),
    /// An edge to a enum variant definition scope.
    Variant(VariantEdge<'db>),
    /// An edge to a module that is referenced by a `super` keyword.
    Super(SuperEdge),
    /// An edge to an ingot that is referenced by a `ingot` keyword.
    Ingot(IngotEdge),
    /// An edge to a scope that is referenced by a `self` keyword.
    Self_(SelfEdge),
    /// An edge to a scope that is referenced by a `Self` keyword.
    SelfTy(SelfTyEdge),
    /// An edge to an anonymous scope, e.g., `impl` or function body.
    Anon(AnonEdge),
}

impl<'db> EdgeKind<'db> {
    pub fn lex() -> Self {
        EdgeKind::Lex(LexEdge())
    }

    pub fn mod_(ident: IdentId<'db>) -> Self {
        EdgeKind::Mod(ident.into())
    }

    pub fn type_(ident: IdentId<'db>) -> Self {
        EdgeKind::Type(ident.into())
    }

    pub fn trait_(ident: IdentId<'db>) -> Self {
        EdgeKind::Trait(ident.into())
    }

    pub fn generic_param(ident: IdentId<'db>) -> Self {
        EdgeKind::GenericParam(ident.into())
    }

    pub fn value(ident: IdentId<'db>) -> Self {
        EdgeKind::Value(ident.into())
    }

    pub fn field(ident: IdentId<'db>) -> Self {
        EdgeKind::Field(ident.into())
    }

    pub fn variant(ident: IdentId<'db>) -> Self {
        EdgeKind::Variant(ident.into())
    }

    pub fn super_() -> Self {
        EdgeKind::Super(SuperEdge())
    }

    pub fn ingot() -> Self {
        EdgeKind::Ingot(IngotEdge())
    }

    pub fn self_ty() -> Self {
        EdgeKind::SelfTy(SelfTyEdge())
    }

    pub fn self_() -> Self {
        EdgeKind::Self_(SelfEdge())
    }

    pub fn anon() -> Self {
        EdgeKind::Anon(AnonEdge())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LexEdge();

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub struct ModEdge<'db>(pub IdentId<'db>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub struct TypeEdge<'db>(pub IdentId<'db>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub struct TraitEdge<'db>(pub IdentId<'db>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub struct ValueEdge<'db>(pub IdentId<'db>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub struct GenericParamEdge<'db>(pub IdentId<'db>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub struct FieldEdge<'db>(pub IdentId<'db>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub struct VariantEdge<'db>(pub IdentId<'db>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SuperEdge();

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IngotEdge();

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub struct SelfTyEdge();

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub struct SelfEdge();

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AnonEdge();

#[cfg(test)]
mod tests {

    use crate::{
        hir_def::{
            scope_graph::{FieldParent, ScopeId},
            ItemKind,
        },
        test_db::TestDb,
    };

    #[test]
    fn item_tree() {
        let mut db = TestDb::default();

        let text = r#"
            mod foo {
                fn bar() {}
                extern {
                    fn baz()
                }
            }

            enum MyEnum {}

            mod baz {
                struct MyS {}
            }
        "#;

        let file = db.standalone_file(text);
        let scope_graph = db.parse_source(file);
        assert_eq!(scope_graph.items_dfs(&db).count(), 8);

        for (i, item) in scope_graph.items_dfs(&db).enumerate() {
            match i {
                0 => assert!(matches!(item, ItemKind::TopMod(_))),
                1 => assert!(matches!(item, ItemKind::Mod(_))),
                2 => assert!(matches!(item, ItemKind::Func(_))),
                3 => assert!(matches!(item, ItemKind::Body(_))),
                4 => assert!(matches!(item, ItemKind::Func(_))),
                5 => assert!(matches!(item, ItemKind::Enum(_))),
                6 => assert!(matches!(item, ItemKind::Mod(_))),
                7 => assert!(matches!(item, ItemKind::Struct(_))),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn enum_record_fields() {
        let mut db = TestDb::default();

        let text = r#"
            enum Foo {
                X { a: i8, b: i8 },
            }
        "#;

        let file = db.standalone_file(text);
        let scope_graph = db.parse_source(file);
        let root = scope_graph.top_mod.scope();
        let enum_ = scope_graph.children(root).next().unwrap();
        assert!(matches!(enum_.item(), ItemKind::Enum(_)));

        let variant = scope_graph.children(enum_).next().unwrap();
        assert!(matches!(variant, ScopeId::Variant(_, _)));

        let field = scope_graph.children(variant).next().unwrap();
        assert!(matches!(
            field,
            ScopeId::Field(FieldParent::Variant(_, _), _)
        ));
    }
}
