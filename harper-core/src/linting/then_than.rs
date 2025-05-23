use super::{Lint, LintKind, PatternLinter};
use crate::Token;
use crate::linting::Suggestion;
use crate::patterns::{
    All, EitherPattern, Invert, OwnedPatternExt, Pattern, SequencePattern, Word, WordSet,
};

#[doc = "Corrects the misuse of `then` to `than`."]
pub struct ThenThan {
    pattern: Box<dyn Pattern>,
}

impl ThenThan {
    pub fn new() -> Self {
        Self {
            pattern: Box::new(All::new(vec![
                Box::new(EitherPattern::new(vec![
                    // Comparative form of adjective
                    Box::new(
                        SequencePattern::default()
                            .then(Word::new("other").or(Box::new(
                                |tok: &Token, source: &[char]| {
                                    is_comparative_adjective(tok, source)
                                },
                            )))
                            .t_ws()
                            .t_aco("then")
                            .t_ws()
                            .then(Invert::new(Word::new("that"))),
                    ),
                    // Positive form of adjective following "more" or "less"
                    Box::new(
                        SequencePattern::default()
                            .then(WordSet::new(&["more", "less"]))
                            .t_ws()
                            .then(|tok: &Token, _source: &[char]| {
                                tok.kind.is_adjective() || tok.kind.is_adverb()
                            })
                            .t_ws()
                            .t_aco("then")
                            .t_ws()
                            .then(Invert::new(Word::new("that"))),
                    ),
                ])),
                // Exceptions to the rule.
                Box::new(Invert::new(WordSet::new(&["back", "this", "so", "but"]))),
            ])),
        }
    }
}

// TODO: This can be simplified or eliminated when the adjective improvements make it into the affix system.
fn is_comparative_adjective(tok: &Token, source: &[char]) -> bool {
    tok.kind
        .is_adjective()
        .then(|| tok.span.get_content(source))
        .is_some_and(|src| {
            // Regular comparative form?
            src.ends_with(&['e', 'r'])
                // Irregular comparatives.
                || src == ['l', 'e', 's', 's']
                || src == ['m', 'o', 'r', 'e']
                || src == ['w', 'o', 'r', 's', 'e']
        })
}

impl Default for ThenThan {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternLinter for ThenThan {
    fn pattern(&self) -> &dyn Pattern {
        self.pattern.as_ref()
    }
    fn match_to_lint(&self, matched_tokens: &[Token], source: &[char]) -> Option<Lint> {
        // For both "stupider then X" and "more stupid then X", "then" is 3rd last token.
        let span = matched_tokens[matched_tokens.len() - 3].span;
        let offending_text = span.get_content(source);

        Some(Lint {
            span,
            lint_kind: LintKind::Miscellaneous,
            suggestions: vec![Suggestion::replace_with_match_case(
                "than".chars().collect(),
                offending_text,
            )],
            message: "Did you mean `than`?".to_string(),
            priority: 31,
        })
    }
    fn description(&self) -> &'static str {
        "Corrects the misuse of `then` to `than`."
    }
}

#[cfg(test)]
mod tests {
    use super::ThenThan;
    use crate::linting::tests::{assert_lint_count, assert_suggestion_result};

    #[test]
    fn allows_back_then() {
        assert_lint_count("I was a gross kid back then.", ThenThan::default(), 0);
    }

    #[test]
    fn catches_shorter_then() {
        assert_suggestion_result(
            "One was shorter then the other.",
            ThenThan::default(),
            "One was shorter than the other.",
        );
    }

    #[test]
    fn catches_better_then() {
        assert_suggestion_result(
            "One was better then the other.",
            ThenThan::default(),
            "One was better than the other.",
        );
    }

    #[test]
    fn catches_longer_then() {
        assert_suggestion_result(
            "One was longer then the other.",
            ThenThan::default(),
            "One was longer than the other.",
        );
    }

    #[test]
    fn catches_less_then() {
        assert_suggestion_result(
            "I eat less then you.",
            ThenThan::default(),
            "I eat less than you.",
        );
    }

    #[test]
    fn catches_more_then() {
        assert_suggestion_result(
            "I eat more then you.",
            ThenThan::default(),
            "I eat more than you.",
        );
    }

    #[test]
    fn stronger_should_change() {
        assert_suggestion_result(
            "a chain is no stronger then its weakest link",
            ThenThan::default(),
            "a chain is no stronger than its weakest link",
        );
    }

    #[test]
    fn half_a_loaf_should_change() {
        assert_suggestion_result(
            "half a loaf is better then no bread",
            ThenThan::default(),
            "half a loaf is better than no bread",
        );
    }

    #[test]
    fn then_everyone_clapped_should_be_allowed() {
        assert_lint_count("and then everyone clapped", ThenThan::default(), 0);
    }

    #[test]
    fn crazier_than_rat_should_change() {
        assert_suggestion_result(
            "crazier then a shithouse rat",
            ThenThan::default(),
            "crazier than a shithouse rat",
        );
    }

    #[test]
    fn poke_in_eye_should_change() {
        assert_suggestion_result(
            "better then a poke in the eye with a sharp stick",
            ThenThan::default(),
            "better than a poke in the eye with a sharp stick",
        );
    }

    #[test]
    fn other_then_should_change() {
        assert_suggestion_result(
            "There was no one other then us at the campsite.",
            ThenThan::default(),
            "There was no one other than us at the campsite.",
        );
    }

    #[test]
    fn allows_and_then() {
        assert_lint_count("And then we left.", ThenThan::default(), 0);
    }

    #[test]
    fn allows_this_then() {
        assert_lint_count("Do this then that.", ThenThan::default(), 0);
    }

    #[test]
    fn allows_issue_720() {
        assert_lint_count(
            "And if just one of those is set incorrectly or it has the tiniest bit of dirt inside then that will wreak havoc with the engine's running ability.",
            ThenThan::default(),
            0,
        );
        assert_lint_count("So let's check it out then.", ThenThan::default(), 0);
        assert_lint_count(
            "And if just the tiniest bit of dirt gets inside then that will wreak havoc.",
            ThenThan::default(),
            0,
        );

        assert_lint_count(
            "He was always a top student in school but then his argument is that grades don't define intelligence.",
            ThenThan::default(),
            0,
        );
    }

    #[test]
    fn allows_issue_744() {
        assert_lint_count(
            "So then after talking about how he would, he didn't.",
            ThenThan::default(),
            0,
        );
    }

    #[test]
    fn issue_720_school_but_then_his() {
        assert_lint_count(
            "She loved the atmosphere of the school but then his argument is that it lacks proper resources for students.",
            ThenThan::default(),
            0,
        );
        assert_lint_count(
            "The teacher praised the efforts of the school but then his argument is that the curriculum needs to be updated.",
            ThenThan::default(),
            0,
        );
        assert_lint_count(
            "They were excited about the new program at school but then his argument is that it won't be effective without proper training.",
            ThenThan::default(),
            0,
        );
        assert_lint_count(
            "The community supported the school but then his argument is that funding is still a major issue.",
            ThenThan::default(),
            0,
        );
    }

    #[test]
    fn issue_720_so_then_these_resistors() {
        assert_lint_count(
            "So then these resistors are connected up in parallel to reduce the overall resistance.",
            ThenThan::default(),
            0,
        );
        assert_lint_count(
            "So then these resistors are connected up to ensure the current flows properly.",
            ThenThan::default(),
            0,
        );
        assert_lint_count(
            "So then these resistors are connected up to achieve the desired voltage drop.",
            ThenThan::default(),
            0,
        );
        assert_lint_count(
            "So then these resistors are connected up to demonstrate the principles of series and parallel circuits.",
            ThenThan::default(),
            0,
        );
        assert_lint_count(
            "So then these resistors are connected up to optimize the circuit's performance.",
            ThenThan::default(),
            0,
        );
    }

    #[test]
    fn issue_720_yes_so_then_sorry() {
        assert_lint_count(
            "Yes so then sorry you didn't receive the memo about the meeting changes.",
            ThenThan::default(),
            0,
        );
        assert_lint_count(
            "Yes so then sorry you had to wait so long for a response from our team.",
            ThenThan::default(),
            0,
        );
        assert_lint_count(
            "Yes so then sorry you felt left out during the discussion; we value your input.",
            ThenThan::default(),
            0,
        );
        assert_lint_count(
            "Yes so then sorry you missed the deadline; we can discuss an extension.",
            ThenThan::default(),
            0,
        );
        assert_lint_count(
            "Yes so then sorry you encountered issues with the software; let me help you troubleshoot.",
            ThenThan::default(),
            0,
        );
    }

    #[test]
    fn more_talented_then_her_issue_720() {
        assert_suggestion_result(
            "He was more talented then her at writing code.",
            ThenThan::default(),
            "He was more talented than her at writing code.",
        );
    }

    #[test]
    fn simpler_then_hers_issue_720() {
        assert_suggestion_result(
            "The design was simpler then hers in layout and color scheme.",
            ThenThan::default(),
            "The design was simpler than hers in layout and color scheme.",
        );
    }

    #[test]
    fn earlier_then_him_issue_720() {
        assert_suggestion_result(
            "We arrived earlier then him at the event.",
            ThenThan::default(),
            "We arrived earlier than him at the event.",
        );
    }

    #[test]
    fn more_robust_then_his_issue_720() {
        assert_suggestion_result(
            "This approach is more robust then his for handling edge cases.",
            ThenThan::default(),
            "This approach is more robust than his for handling edge cases.",
        );
    }

    #[test]
    fn patch_more_recently_then_last_week_issue_720() {
        assert_suggestion_result(
            "We submitted the patch more recently then last week, so they should have it already.",
            ThenThan::default(),
            "We submitted the patch more recently than last week, so they should have it already.",
        );
    }

    #[test]
    fn allows_well_then() {
        assert_lint_count(
            "Well then we're just going to raise all of these taxes",
            ThenThan::default(),
            0,
        );
    }

    #[test]
    fn allows_nervous_then() {
        assert_lint_count(
            "I think both of us were getting nervous then because the system would have automatically aborted.",
            ThenThan::default(),
            0,
        );
    }

    #[test]
    fn flags_stupider_then_and_more_and_less_stupid_then() {
        assert_lint_count(
            "He was stupider then her but she was more stupid then some. Then again he was less stupid then some too.",
            ThenThan::default(),
            3,
        );
    }

    #[test]
    fn patch_worse_then() {
        assert_suggestion_result(
            "He was worse then her at writing code.",
            ThenThan::default(),
            "He was worse than her at writing code.",
        );
    }
}
