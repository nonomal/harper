use crate::expr::Expr;
use crate::expr::SequenceExpr;
use crate::{CharStringExt, Token, patterns::WordSet};

use crate::Lint;
use crate::linting::{ExprLinter, LintKind, Suggestion};

/// See also:
/// harper-core/src/linting/compound_nouns/implied_ownership_compound_nouns.rs
/// harper-core/src/linting/lets_confusion/mod.rs
/// harper-core/src/linting/lets_confusion/let_us_redundancy.rs
/// harper-core/src/linting/lets_confusion/no_contraction_with_verb.rs
pub struct ShouldContract {
    expr: Box<dyn Expr>,
}

impl Default for ShouldContract {
    fn default() -> Self {
        Self {
            expr: Box::new(
                SequenceExpr::default()
                    .then(WordSet::new(&["your", "were"]))
                    .then_whitespace()
                    .then_determiner()
                    .then_whitespace()
                    .then_adjective(),
            ),
        }
    }
}

impl ShouldContract {
    fn mistake_to_correct(mistake: &str) -> impl Iterator<Item = Vec<char>> {
        match mistake.to_lowercase().as_str() {
            "your" => vec!["you're", "you are"],
            "were" => vec!["we're", "we are"],
            _ => panic!("The pattern in this linter should make a fall-through impossible."),
        }
        .into_iter()
        .map(|v| v.chars().collect())
    }
}

impl ExprLinter for ShouldContract {
    fn expr(&self) -> &dyn Expr {
        self.expr.as_ref()
    }

    fn match_to_lint(&self, matched_tokens: &[Token], source: &[char]) -> Option<Lint> {
        let mistake = matched_tokens[0].span.get_content(source);

        Some(Lint {
            span: matched_tokens[0].span,
            lint_kind: LintKind::WordChoice,
            suggestions: Self::mistake_to_correct(&mistake.to_lower().to_string())
                .map(|v| Suggestion::replace_with_match_case(v, mistake))
                .collect(),
            message: "Use the contraction or separate the words instead.".to_string(),
            priority: 31,
        })
    }

    fn description(&self) -> &'static str {
        "Neglecting the apostrophe when contracting pronouns with \"are\" (like \"your\" and \"you are\") is a fatal, but extremely common mistake to make."
    }
}

#[cfg(test)]
mod tests {
    use super::ShouldContract;
    use crate::linting::tests::{assert_lint_count, assert_suggestion_result};

    #[test]
    fn contracts_your_correctly() {
        assert_suggestion_result(
            "your the best",
            ShouldContract::default(),
            "you're the best",
        );
    }

    #[test]
    fn contracts_were_complex_correctly() {
        assert_suggestion_result(
            "were a good team",
            ShouldContract::default(),
            "we're a good team",
        );
    }

    #[test]
    fn case_insensitive_handling() {
        assert_suggestion_result(
            "Your the best",
            ShouldContract::default(),
            "You're the best",
        );
    }

    #[test]
    fn no_match_without_the() {
        assert_lint_count("your best", ShouldContract::default(), 0);
        assert_lint_count("were best", ShouldContract::default(), 0);
    }

    #[test]
    fn no_match_with_punctuation() {
        assert_lint_count("your, the best", ShouldContract::default(), 0);
    }

    #[test]
    fn allow_norm() {
        assert_lint_count(
            "Let's start this story by going back to the dark ages before internet applications were the norm.",
            ShouldContract::default(),
            0,
        );
    }
}
