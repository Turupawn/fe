use crate::diagnostics::CsDbWrapper;
use codespan_reporting::term::{
    self,
    termcolor::{BufferWriter, ColorChoice},
};
use common::{diagnostics::CompleteDiagnostic, InputFile, InputIngot};
use hir::{
    hir_def::TopLevelMod,
    lower::{map_file_to_mod, module_tree},
};
use hir_analysis::{
    analysis_pass::{AnalysisPassManager, ParsingPass},
    diagnostics::DiagnosticVoucher,
    name_resolution::{DefConflictAnalysisPass, ImportAnalysisPass, PathAnalysisPass},
    ty::{
        AdtDefAnalysisPass, BodyAnalysisPass, FuncAnalysisPass, ImplAnalysisPass,
        ImplTraitAnalysisPass, TraitAnalysisPass, TypeAliasAnalysisPass,
    },
};

use crate::diagnostics::ToCsDiag;

#[derive(Default, Clone)]
#[salsa::db]
pub struct DriverDataBase {
    storage: salsa::Storage<Self>,
}
#[salsa::db]
impl salsa::Database for DriverDataBase {
    fn salsa_event(&self, _event: &dyn Fn() -> salsa::Event) {}
}

impl DriverDataBase {
    // TODO: An temporary implementation for ui testing.
    pub fn run_on_top_mod<'db>(&'db self, top_mod: TopLevelMod<'db>) -> DiagnosticsCollection<'db> {
        self.run_on_file_with_pass_manager(top_mod, initialize_analysis_pass)
    }

    pub fn run_on_file_with_pass_manager<'db, F>(
        &'db self,
        top_mod: TopLevelMod<'db>,
        pm_builder: F,
    ) -> DiagnosticsCollection<'db>
    where
        F: FnOnce(&'db DriverDataBase) -> AnalysisPassManager<'db>,
    {
        let mut pass_manager = pm_builder(self);
        DiagnosticsCollection(pass_manager.run_on_module(top_mod))
    }

    pub fn run_on_ingot(&self, ingot: InputIngot) -> DiagnosticsCollection {
        self.run_on_ingot_with_pass_manager(ingot, initialize_analysis_pass)
    }

    pub fn run_on_ingot_with_pass_manager<'db, F>(
        &'db self,
        ingot: InputIngot,
        pm_builder: F,
    ) -> DiagnosticsCollection<'db>
    where
        F: FnOnce(&'db DriverDataBase) -> AnalysisPassManager<'db>,
    {
        let tree = module_tree(self, ingot);
        let mut pass_manager = pm_builder(self);
        DiagnosticsCollection(pass_manager.run_on_module_tree(tree))
    }

    pub fn top_mod(&self, ingot: InputIngot, input: InputFile) -> TopLevelMod {
        map_file_to_mod(self, ingot, input)
    }
}

pub struct DiagnosticsCollection<'db>(Vec<Box<dyn DiagnosticVoucher<'db> + 'db>>);
impl<'db> DiagnosticsCollection<'db> {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn emit(&self, db: &'db DriverDataBase) {
        let writer = BufferWriter::stderr(ColorChoice::Auto);
        let mut buffer = writer.buffer();
        let config = term::Config::default();

        for diag in self.finalize(db) {
            term::emit(&mut buffer, &config, &CsDbWrapper(db), &diag.to_cs(db)).unwrap();
        }

        eprintln!("{}", std::str::from_utf8(buffer.as_slice()).unwrap());
    }

    /// Format the accumulated diagnostics to a string.
    pub fn format_diags(&self, db: &'db DriverDataBase) -> String {
        let writer = BufferWriter::stderr(ColorChoice::Never);
        let mut buffer = writer.buffer();
        let config = term::Config::default();

        for diag in self.finalize(db) {
            term::emit(&mut buffer, &config, &CsDbWrapper(db), &diag.to_cs(db)).unwrap();
        }

        std::str::from_utf8(buffer.as_slice()).unwrap().to_string()
    }

    fn finalize(&self, db: &'db DriverDataBase) -> Vec<CompleteDiagnostic> {
        let mut diags: Vec<_> = self.0.iter().map(|d| d.to_complete(db)).collect();
        diags.sort_by(|lhs, rhs| match lhs.error_code.cmp(&rhs.error_code) {
            std::cmp::Ordering::Equal => lhs.primary_span().cmp(&rhs.primary_span()),
            ord => ord,
        });
        diags
    }
}

fn initialize_analysis_pass(db: &DriverDataBase) -> AnalysisPassManager<'_> {
    let mut pass_manager = AnalysisPassManager::new();
    pass_manager.add_module_pass(Box::new(ParsingPass::new(db)));
    pass_manager.add_module_pass(Box::new(DefConflictAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(ImportAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(PathAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(AdtDefAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(TypeAliasAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(TraitAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(ImplAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(ImplTraitAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(FuncAnalysisPass::new(db)));
    pass_manager.add_module_pass(Box::new(BodyAnalysisPass::new(db)));
    pass_manager
}
