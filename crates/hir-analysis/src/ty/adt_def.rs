use hir::{
    hir_def::{
        scope_graph::ScopeId, Contract, Enum, FieldDefListId, GenericParamOwner, IdentId, IngotId,
        ItemKind, Partial, Struct, TypeId as HirTyId, VariantDefListId, VariantKind,
    },
    span::DynLazySpan,
};
use salsa::Update;

use super::{
    binder::Binder,
    ty_def::{InvalidCause, TyId},
    ty_lower::{collect_generic_params, lower_hir_ty, GenericParamTypeSet},
};
use crate::HirAnalysisDb;

/// Lower HIR ADT definition(`struct/enum/contract`) to [`AdtDef`].
#[salsa::tracked]
pub fn lower_adt<'db>(db: &'db dyn HirAnalysisDb, adt: AdtRef<'db>) -> AdtDef<'db> {
    AdtTyBuilder::new(db, adt).build()
}

/// Represents a ADT type definition.
#[salsa::tracked]
#[derive(Debug)]
pub struct AdtDef<'db> {
    pub adt_ref: AdtRef<'db>,

    /// Type parameters of the ADT.
    #[return_ref]
    pub param_set: GenericParamTypeSet<'db>,

    /// Fields of the ADT, if the ADT is an enum, this represents variants.
    /// Otherwise, `fields[0]` represents all fields of the struct.
    #[return_ref]
    pub fields: Vec<AdtField<'db>>,
}

impl<'db> AdtDef<'db> {
    pub(crate) fn name(self, db: &'db dyn HirAnalysisDb) -> IdentId<'db> {
        self.adt_ref(db).name(db)
    }

    pub fn name_span(self, db: &'db dyn HirAnalysisDb) -> DynLazySpan<'db> {
        self.adt_ref(db).name_span(db)
    }

    pub(crate) fn params(self, db: &'db dyn HirAnalysisDb) -> &'db [TyId<'db>] {
        self.param_set(db).params(db)
    }

    pub(crate) fn original_params(self, db: &'db dyn HirAnalysisDb) -> &'db [TyId<'db>] {
        self.param_set(db).explicit_params(db)
    }

    pub(crate) fn is_struct(self, db: &dyn HirAnalysisDb) -> bool {
        matches!(self.adt_ref(db), AdtRef::Struct(_))
    }

    pub fn scope(self, db: &'db dyn HirAnalysisDb) -> ScopeId<'db> {
        self.adt_ref(db).scope()
    }

    #[allow(dead_code)] // xxx
    pub(crate) fn variant_ty_span(
        self,
        db: &'db dyn HirAnalysisDb,
        field_idx: usize,
        ty_idx: usize,
    ) -> DynLazySpan<'db> {
        match self.adt_ref(db) {
            AdtRef::Enum(e) => {
                let span = e.lazy_span().variants_moved().variant_moved(field_idx);
                match e.variants(db).data(db)[field_idx].kind {
                    VariantKind::Tuple(_) => span.tuple_type_moved().elem_ty_moved(ty_idx).into(),
                    VariantKind::Record(_) => {
                        span.fields_moved().field_moved(ty_idx).ty_moved().into()
                    }
                    VariantKind::Unit => unreachable!(),
                }
            }

            AdtRef::Struct(s) => s
                .lazy_span()
                .fields_moved()
                .field_moved(field_idx)
                .ty_moved()
                .into(),

            AdtRef::Contract(c) => c
                .lazy_span()
                .fields_moved()
                .field_moved(field_idx)
                .ty_moved()
                .into(),
        }
    }

    pub(crate) fn ingot(self, db: &'db dyn HirAnalysisDb) -> IngotId<'db> {
        match self.adt_ref(db) {
            AdtRef::Enum(e) => e.top_mod(db).ingot(db),
            AdtRef::Struct(s) => s.top_mod(db).ingot(db),
            AdtRef::Contract(c) => c.top_mod(db).ingot(db),
        }
    }

    pub(crate) fn as_generic_param_owner(
        self,
        db: &'db dyn HirAnalysisDb,
    ) -> Option<GenericParamOwner<'db>> {
        self.adt_ref(db).generic_owner()
    }
}

/// This struct represents a field of an ADT. If the ADT is an enum, this
/// represents a variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, salsa::Update)]
pub struct AdtField<'db> {
    /// Fields of the variant.
    /// If the adt is an struct or contract,
    /// the length of the vector is always 1.
    ///
    /// To allow recursive types, the type of the field is represented as a HIR
    /// type and.
    tys: Vec<Partial<HirTyId<'db>>>,

    scope: ScopeId<'db>,
}
impl<'db> AdtField<'db> {
    pub fn ty(&self, db: &'db dyn HirAnalysisDb, i: usize) -> Binder<TyId<'db>> {
        let ty = if let Some(ty) = self.tys[i].to_opt() {
            lower_hir_ty(db, ty, self.scope)
        } else {
            TyId::invalid(db, InvalidCause::Other)
        };

        Binder::bind(ty)
    }

    /// Iterates all fields types of the `field`.
    pub fn iter_types<'a>(
        &'a self,
        db: &'db dyn HirAnalysisDb,
    ) -> impl Iterator<Item = Binder<TyId<'db>>> + 'a {
        (0..self.num_types()).map(|i| self.ty(db, i))
    }

    pub fn num_types(&self) -> usize {
        self.tys.len()
    }

    pub(super) fn new(tys: Vec<Partial<HirTyId<'db>>>, scope: ScopeId<'db>) -> Self {
        Self { tys, scope }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::From, salsa::Supertype, Update)]
pub enum AdtRef<'db> {
    Enum(Enum<'db>),
    Struct(Struct<'db>),
    Contract(Contract<'db>),
}

impl<'db> AdtRef<'db> {
    pub fn try_from_item(item: ItemKind<'db>) -> Option<Self> {
        match item {
            ItemKind::Enum(x) => Some(x.into()),
            ItemKind::Struct(x) => Some(x.into()),
            ItemKind::Contract(x) => Some(x.into()),
            _ => None,
        }
    }

    pub fn scope(self) -> ScopeId<'db> {
        match self {
            Self::Enum(e) => e.scope(),
            Self::Struct(s) => s.scope(),
            Self::Contract(c) => c.scope(),
        }
    }

    pub fn as_item(self) -> ItemKind<'db> {
        match self {
            AdtRef::Enum(e) => e.into(),
            AdtRef::Struct(s) => s.into(),
            AdtRef::Contract(c) => c.into(),
        }
    }

    pub fn name(self, db: &'db dyn HirAnalysisDb) -> IdentId<'db> {
        match self {
            AdtRef::Enum(e) => e.name(db),
            AdtRef::Struct(s) => s.name(db),
            AdtRef::Contract(c) => c.name(db),
        }
        .to_opt()
        .unwrap_or_else(|| IdentId::new(db, "<unknown>".to_string()))
    }

    pub fn kind_name(self) -> &'static str {
        self.as_item().kind_name()
    }

    pub fn name_span(self, db: &'db dyn HirAnalysisDb) -> DynLazySpan<'db> {
        self.scope()
            .name_span(db)
            .unwrap_or_else(DynLazySpan::invalid)
    }

    pub(crate) fn generic_owner(self) -> Option<GenericParamOwner<'db>> {
        match self {
            AdtRef::Enum(e) => Some(e.into()),
            AdtRef::Struct(s) => Some(s.into()),
            AdtRef::Contract(_) => None,
        }
    }
}

struct AdtTyBuilder<'db> {
    db: &'db dyn HirAnalysisDb,
    adt: AdtRef<'db>,
    params: GenericParamTypeSet<'db>,
    variants: Vec<AdtField<'db>>,
}

impl<'db> AdtTyBuilder<'db> {
    fn new(db: &'db dyn HirAnalysisDb, adt: AdtRef<'db>) -> Self {
        Self {
            db,
            adt,
            params: GenericParamTypeSet::empty(db, adt.scope()),
            variants: Vec::new(),
        }
    }

    fn build(mut self) -> AdtDef<'db> {
        self.collect_generic_params();
        self.collect_variants();
        AdtDef::new(self.db, self.adt, self.params, self.variants)
    }

    fn collect_generic_params(&mut self) {
        let owner = match self.adt {
            AdtRef::Contract(_) => return,
            AdtRef::Enum(enum_) => enum_.into(),
            AdtRef::Struct(struct_) => struct_.into(),
        };
        self.params = collect_generic_params(self.db, owner);
    }

    fn collect_variants(&mut self) {
        match self.adt {
            AdtRef::Struct(struct_) => {
                self.collect_field_types(struct_.fields(self.db));
            }

            AdtRef::Contract(contract) => self.collect_field_types(contract.fields(self.db)),

            AdtRef::Enum(enum_) => self.collect_enum_variant_types(enum_.variants(self.db)),
        };
    }

    fn collect_field_types(&mut self, fields: FieldDefListId<'db>) {
        let scope = self.adt.scope();

        let fields = fields.data(self.db).iter().map(|field| field.ty).collect();

        self.variants.push(AdtField::new(fields, scope));
    }

    fn collect_enum_variant_types(&mut self, variants: VariantDefListId<'db>) {
        let scope = self.adt.scope();

        variants.data(self.db).iter().for_each(|variant| {
            // TODO: FIX here when record variant is introduced.
            let tys = match variant.kind {
                VariantKind::Tuple(tuple_id) => tuple_id.data(self.db).clone(),

                VariantKind::Record(fields) => {
                    fields.data(self.db).iter().map(|field| field.ty).collect()
                }

                VariantKind::Unit => vec![],
            };

            let variant = AdtField::new(tys, scope);
            self.variants.push(variant)
        })
    }
}
