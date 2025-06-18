use crate::{
    Token,
    linting::{Lint, LintKind, PatternLinter, Suggestion},
    patterns::{LongestMatchOf, SequencePattern, WordSet},
};

pub struct PronounKnew {
    pattern: Box<dyn crate::patterns::Pattern>,
}

trait PronounKnewExt {
    fn then_pronoun(self) -> Self;
}

impl Default for PronounKnew {
    fn default() -> Self {
        // The pronoun that would occur before a verb would be a subject pronoun.
        // But "its" commonly occurs before "new" and is a possessive pronoun. (Much more commonly a determiner)
        // Since "his" and "her" are possessive and object pronouns respectively, we ignore them too.
        let pronoun_pattern = |tok: &Token, source: &[char]| {
            if !tok.kind.is_pronoun() {
                return false;
            }

            let pronorm = tok.span.get_content_string(source).to_lowercase();
            let excluded = ["its", "his", "her", "every", "something", "nothing"];
            !excluded.contains(&&*pronorm)
        };

        let pronoun_then_new = SequencePattern::default()
            .then(pronoun_pattern)
            .then_whitespace()
            .then_any_capitalization_of("new");

        let pronoun_adverb_then_new = SequencePattern::default()
            .then(pronoun_pattern)
            .then_whitespace()
            .then(WordSet::new(&["always", "never", "also", "often"]))
            .then_whitespace()
            .then_any_capitalization_of("new");

        let combined_pattern = LongestMatchOf::new(vec![
            Box::new(pronoun_then_new),
            Box::new(pronoun_adverb_then_new),
        ]);

        Self {
            pattern: Box::new(combined_pattern),
        }
    }
}

impl PatternLinter for PronounKnew {
    fn pattern(&self) -> &dyn crate::patterns::Pattern {
        self.pattern.as_ref()
    }

    fn match_to_lint(&self, tokens: &[Token], source: &[char]) -> Option<Lint> {
        let typo_token = tokens.last()?;
        let typo_span = typo_token.span;
        let typo_text = typo_span.get_content(source);

        Some(Lint {
            span: typo_span,
            lint_kind: LintKind::WordChoice,
            suggestions: vec![Suggestion::replace_with_match_case(
                "knew".chars().collect(),
                typo_text,
            )],
            message: "Did you mean “knew” (the past tense of “know”)?".to_string(),
            priority: 31,
        })
    }

    fn description(&self) -> &str {
        "Detects when “new” following a pronoun (optionally with an adverb) is a typo for the past tense “knew.”"
    }
}

#[cfg(test)]
mod tests {
    use super::PronounKnew;
    use crate::linting::tests::{assert_lint_count, assert_suggestion_result};

    #[test]
    fn simple_pronoun_new() {
        assert_suggestion_result(
            "I new you would say that.",
            PronounKnew::default(),
            "I knew you would say that.",
        );
    }

    #[test]
    fn with_adverb() {
        assert_suggestion_result(
            "She often new the answer.",
            PronounKnew::default(),
            "She often knew the answer.",
        );
    }

    #[test]
    fn does_not_flag_without_pronoun() {
        assert_lint_count("The software is new.", PronounKnew::default(), 0);
    }

    #[test]
    fn does_not_flag_other_context() {
        assert_lint_count("They called it \"new\".", PronounKnew::default(), 0);
    }

    #[test]
    fn does_not_flag_with_its() {
        assert_lint_count(
            "In 2015, the US was paying on average around 2% for its new issuance bonds.",
            PronounKnew::default(),
            0,
        );
    }

    #[test]
    fn does_not_flag_with_his() {
        assert_lint_count("His new car is fast.", PronounKnew::default(), 0);
    }

    #[test]
    fn does_not_flag_with_her() {
        assert_lint_count("Her new car is fast.", PronounKnew::default(), 0);
    }

    #[test]
    fn does_not_flag_with_nothing_1298() {
        assert_lint_count("This is nothing new.", PronounKnew::default(), 0);
    }
}
