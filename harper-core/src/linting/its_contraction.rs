use harper_brill::UPOS;

use crate::expr::All;
use crate::expr::Expr;
use crate::expr::SequenceExpr;
use crate::patterns::NominalPhrase;
use crate::patterns::Pattern;
use crate::patterns::UPOSSet;
use crate::patterns::WordSet;
use crate::{
    Token,
    linting::{ExprLinter, Lint, LintKind, Suggestion},
};

pub struct ItsContraction {
    expr: Box<dyn Expr>,
}

impl Default for ItsContraction {
    fn default() -> Self {
        let positive = SequenceExpr::default()
            .t_aco("its")
            .then_whitespace()
            .then(UPOSSet::new(&[UPOS::VERB, UPOS::AUX, UPOS::DET]));

        let exceptions = SequenceExpr::default()
            .then_anything()
            .then_anything()
            .then(WordSet::new(&["own", "intended"]));

        let inverted = SequenceExpr::default().if_not_then_step_one(exceptions);

        let expr = All::new(vec![Box::new(positive), Box::new(inverted)]);

        Self {
            expr: Box::new(expr),
        }
    }
}

impl ExprLinter for ItsContraction {
    fn expr(&self) -> &dyn Expr {
        self.expr.as_ref()
    }

    fn match_to_lint(&self, toks: &[Token], source: &[char]) -> Option<Lint> {
        let offender = toks.first()?;
        let offender_chars = offender.span.get_content(source);

        if !toks.get(2)?.kind.is_upos(UPOS::AUX)
            && NominalPhrase.matches(&toks[2..], source).is_some()
        {
            return None;
        }

        Some(Lint {
            span: offender.span,
            lint_kind: LintKind::WordChoice,
            suggestions: vec![
                Suggestion::replace_with_match_case_str("it's", offender_chars),
                Suggestion::replace_with_match_case_str("it has", offender_chars),
            ],
            message: "Use `it's` (short for `it has` or `it is`) here, not the possessive `its`."
                .to_owned(),
            priority: 54,
        })
    }

    fn description(&self) -> &str {
        "Detects the possessive `its` before `had`, `been`, or `got` and offers `it's` or `it has`."
    }
}

#[cfg(test)]
mod tests {
    use super::ItsContraction;
    use crate::linting::tests::{assert_lint_count, assert_suggestion_result};

    #[test]
    fn fix_had() {
        assert_suggestion_result(
            "Its had an enormous effect.",
            ItsContraction::default(),
            "It's had an enormous effect.",
        );
    }

    #[test]
    fn fix_been() {
        assert_suggestion_result(
            "Its been months since we spoke.",
            ItsContraction::default(),
            "It's been months since we spoke.",
        );
    }

    #[test]
    fn fix_got() {
        assert_suggestion_result(
            "I think its got nothing to do with us.",
            ItsContraction::default(),
            "I think it's got nothing to do with us.",
        );
    }

    #[test]
    fn ignore_correct_contraction() {
        assert_lint_count(
            "It's been a long year for everyone.",
            ItsContraction::default(),
            0,
        );
    }

    #[test]
    fn ignore_possessive() {
        assert_lint_count(
            "The company revised its policies last week.",
            ItsContraction::default(),
            0,
        );
    }

    #[test]
    fn ignore_coroutine() {
        assert_lint_count(
            "Launch each task within its own child coroutine.",
            ItsContraction::default(),
            0,
        );
    }

    #[test]
    fn issue_381() {
        assert_suggestion_result(
            "Its a nice day.",
            ItsContraction::default(),
            "It's a nice day.",
        );
    }
}
