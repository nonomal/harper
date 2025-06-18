use crate::expr::All;
use crate::expr::Expr;
use crate::expr::OwnedExprExt;
use crate::expr::SequenceExpr;
use crate::{
    Token, TokenStringExt,
    linting::{ExprLinter, Lint, LintKind, Suggestion},
    patterns::InflectionOfBe,
};

pub struct HowTo {
    expr: Box<dyn Expr>,
}

impl Default for HowTo {
    fn default() -> Self {
        let mut pattern = All::default();

        let pos_pattern = SequenceExpr::default()
            .t_aco("how")
            .then_whitespace()
            .then_verb();
        pattern.add(pos_pattern);

        let exceptions = SequenceExpr::default()
            .then_anything()
            .then_anything()
            .then(
                InflectionOfBe::new().or(Box::new(|tok: &Token, src: &[char]| {
                    if tok.kind.is_auxiliary_verb()
                        || tok.kind.is_adjective()
                        || tok.kind.is_verb_progressive_form()
                    {
                        true
                    } else {
                        let normed = tok.span.get_content_string(src).to_ascii_lowercase();
                        normed == "did" || normed == "come"
                    }
                })),
            );

        pattern.add(SequenceExpr::default().if_not_then_step_one(exceptions));

        Self {
            expr: Box::new(pattern),
        }
    }
}

impl ExprLinter for HowTo {
    fn expr(&self) -> &dyn Expr {
        self.expr.as_ref()
    }

    fn match_to_lint(&self, toks: &[Token], _src: &[char]) -> Option<Lint> {
        let span = toks[0..2].span()?;
        let fix: Vec<char> = "to ".chars().collect();

        Some(Lint {
            span,
            lint_kind: LintKind::WordChoice,
            suggestions: vec![Suggestion::InsertAfter(fix)],
            message: "Insert `to` after `how` (e.g., `how to clone`).".into(),
            priority: 63,
        })
    }

    fn description(&self) -> &str {
        "Detects the omission of `to` in constructions like `how clone / how install` and suggests `how to …`."
    }
}

#[cfg(test)]
mod tests {
    use super::HowTo;
    use crate::linting::tests::{assert_lint_count, assert_suggestion_result};

    #[test]
    fn flags_missing_to() {
        assert_suggestion_result(
            "Here's how clone the repository.",
            HowTo::default(),
            "Here's how to clone the repository.",
        );
    }

    #[test]
    fn ignores_correct_phrase() {
        assert_lint_count("Here's how to clone the repository.", HowTo::default(), 0);
    }

    #[test]
    fn flags_other_verbs() {
        assert_suggestion_result(
            "Learn how install Rust.",
            HowTo::default(),
            "Learn how to install Rust.",
        );
    }

    #[test]
    fn ros_package_install() {
        assert_suggestion_result(
            "Can someone explain how install this ROS package on Humble?",
            HowTo::default(),
            "Can someone explain how to install this ROS package on Humble?",
        );
    }

    #[test]
    fn extract_and_install_app() {
        assert_suggestion_result(
            "Here’s a quick guide on how install an app you’ve extracted from a tarball.",
            HowTo::default(),
            "Here’s a quick guide on how to install an app you’ve extracted from a tarball.",
        );
    }

    #[test]
    fn dll_files() {
        assert_suggestion_result(
            "This video shows how fix missing DLL files on Windows.",
            HowTo::default(),
            "This video shows how to fix missing DLL files on Windows.",
        );
    }

    #[test]
    fn dofus_on_ubuntu() {
        assert_suggestion_result(
            "Full tutorial on how install Dofus under Ubuntu.",
            HowTo::default(),
            "Full tutorial on how to install Dofus under Ubuntu.",
        );
    }

    #[test]
    fn tar_gz_install() {
        assert_suggestion_result(
            "Find out how install software shipped as a .tar.gz archive.",
            HowTo::default(),
            "Find out how to install software shipped as a .tar.gz archive.",
        );
    }

    #[test]
    fn thrift_libraries() {
        assert_suggestion_result(
            "Anyone know how install the Thrift libraries from source?",
            HowTo::default(),
            "Anyone know how to install the Thrift libraries from source?",
        );
    }

    #[test]
    fn windows_adk() {
        assert_suggestion_result(
            "Lost the Windows ADK again—remind me how install it?",
            HowTo::default(),
            "Lost the Windows ADK again—remind me how to install it?",
        );
    }

    #[test]
    fn accounting_errors() {
        assert_suggestion_result(
            "Eight common accounting errors and how fix them.",
            HowTo::default(),
            "Eight common accounting errors and how to fix them.",
        );
    }

    #[test]
    fn sentence_fragments() {
        assert_suggestion_result(
            "Here’s what sentence fragments are and how fix them.",
            HowTo::default(),
            "Here’s what sentence fragments are and how to fix them.",
        );
    }

    #[test]
    fn zipper_slider() {
        assert_suggestion_result(
            "Quick demo on how fix a broken zipper slider.",
            HowTo::default(),
            "Quick demo on how to fix a broken zipper slider.",
        );
    }

    #[test]
    fn door_lock() {
        assert_suggestion_result(
            "Tips on how fix a door that won’t lock.",
            HowTo::default(),
            "Tips on how to fix a door that won’t lock.",
        );
    }

    #[test]
    fn already_correct_install() {
        assert_lint_count(
            "See how to install the package with apt.",
            HowTo::default(),
            0,
        );
    }

    #[test]
    fn already_correct_fix() {
        assert_lint_count(
            "He showed me how to fix the zipper in ten minutes.",
            HowTo::default(),
            0,
        );
    }

    #[test]
    fn how_are_you() {
        assert_lint_count("How are you?", HowTo::default(), 0);
    }

    #[test]
    fn how_calm_you_are() {
        assert_lint_count("I like how calm you are.", HowTo::default(), 0);
    }

    #[test]
    fn how_will_you_make_up() {
        assert_lint_count(
            "How will you make up for your mistakes?",
            HowTo::default(),
            0,
        );
    }

    #[test]
    fn storytelling_clause() {
        assert_lint_count(
            "I will tell about how leaving my husband led to my dog winning a Nobel Prize.",
            HowTo::default(),
            0,
        );
    }

    #[test]
    fn dont_flag_how_did_you() {
        assert_lint_count("How did you get to school every day?", HowTo::default(), 0);
    }

    #[test]
    fn dont_flag_how_come() {
        assert_lint_count(
            "How come this has to be a special case?",
            HowTo::default(),
            0,
        );
    }
}
