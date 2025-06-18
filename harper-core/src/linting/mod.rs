//! Frameworks and rules that locate errors in text.
//!
//! See the [`Linter`] trait and the [documentation for authoring a rule](https://writewithharper.com/docs/contributors/author-a-rule) for more information.

mod a_part;
mod adjective_of_a;
mod amounts_for;
mod an_a;
mod ask_no_preposition;
mod avoid_curses;
mod back_in_the_day;
mod boring_words;
mod capitalize_personal_pronouns;
mod chock_full;
mod closed_compounds;
mod comma_fixes;
mod compound_nouns;
mod confident;
mod correct_number_suffix;
mod currency_placement;
mod dashes;
mod despite_of;
mod dot_initialisms;
mod ellipsis_length;
mod else_possessive;
mod everyday;
mod expand_time_shorthands;
mod few_units_of_time_ago;
mod first_aid_kit;
mod for_noun;
mod hedging;
mod hereby;
mod hop_hope;
mod how_to;
mod hyphenate_number_day;
mod inflected_verb_after_to;
mod initialism_linter;
mod initialisms;
mod it_is;
mod it_would_be;
mod its_contraction;
mod left_right_hand;
mod lets_confusion;
mod likewise;
mod linking_verbs;
mod lint;
mod lint_group;
mod lint_kind;
mod long_sentences;
mod map_phrase_linter;
mod merge_linters;
mod merge_words;
mod modal_of;
mod most_number;
mod multiple_sequential_pronouns;
mod nail_on_the_head;
mod no_oxford_comma;
mod nobody;
mod nominal_wants;
mod noun_instead_of_verb;
mod number_suffix_capitalization;
mod of_course;
mod one_and_the_same;
mod open_compounds;
mod open_the_light;
mod out_of_date;
mod oxford_comma;
mod oxymorons;
mod pattern_linter;
mod phrasal_verb_as_compound_noun;
mod phrase_corrections;
mod pique_interest;
mod possessive_your;
mod pronoun_contraction;
mod pronoun_knew;
mod proper_noun_capitalization_linters;
mod repeated_words;
mod save_to_safe;
mod sentence_capitalization;
mod since_duration;
mod somewhat_something;
mod spaces;
mod spell_check;
mod spelled_numbers;
mod suggestion;
mod that_which;
mod the_how_why;
mod the_my;
mod then_than;
mod throw_rubbish;
mod unclosed_quotes;
mod use_genitive;
mod was_aloud;
mod whereas;
mod widely_accepted;
mod win_prize;
mod wordpress_dotcom;

pub use a_part::APart;
pub use adjective_of_a::AdjectiveOfA;
pub use amounts_for::AmountsFor;
pub use an_a::AnA;
pub use ask_no_preposition::AskNoPreposition;
pub use avoid_curses::AvoidCurses;
pub use back_in_the_day::BackInTheDay;
pub use boring_words::BoringWords;
pub use capitalize_personal_pronouns::CapitalizePersonalPronouns;
pub use chock_full::ChockFull;
pub use comma_fixes::CommaFixes;
pub use compound_nouns::CompoundNouns;
pub use confident::Confident;
pub use correct_number_suffix::CorrectNumberSuffix;
pub use currency_placement::CurrencyPlacement;
pub use dashes::Dashes;
pub use despite_of::DespiteOf;
pub use dot_initialisms::DotInitialisms;
pub use ellipsis_length::EllipsisLength;
pub use everyday::Everyday;
pub use expand_time_shorthands::ExpandTimeShorthands;
pub use few_units_of_time_ago::FewUnitsOfTimeAgo;
pub use for_noun::ForNoun;
pub use hedging::Hedging;
pub use hereby::Hereby;
pub use hop_hope::HopHope;
pub use how_to::HowTo;
pub use hyphenate_number_day::HyphenateNumberDay;
pub use inflected_verb_after_to::InflectedVerbAfterTo;
pub use initialism_linter::InitialismLinter;
pub use its_contraction::ItsContraction;
pub use left_right_hand::LeftRightHand;
pub use lets_confusion::LetsConfusion;
pub use likewise::Likewise;
pub use linking_verbs::LinkingVerbs;
pub use lint::Lint;
pub use lint_group::{LintGroup, LintGroupConfig};
pub use lint_kind::LintKind;
pub use long_sentences::LongSentences;
pub use map_phrase_linter::MapPhraseLinter;
pub use merge_words::MergeWords;
pub use modal_of::ModalOf;
pub use most_number::MostNumber;
pub use multiple_sequential_pronouns::MultipleSequentialPronouns;
pub use nail_on_the_head::NailOnTheHead;
pub use no_oxford_comma::NoOxfordComma;
pub use nobody::Nobody;
pub use noun_instead_of_verb::NounInsteadOfVerb;
pub use number_suffix_capitalization::NumberSuffixCapitalization;
pub use of_course::OfCourse;
pub use one_and_the_same::OneAndTheSame;
pub use open_the_light::OpenTheLight;
pub use out_of_date::OutOfDate;
pub use oxford_comma::OxfordComma;
pub use oxymorons::Oxymorons;
pub use pattern_linter::PatternLinter;
pub use phrasal_verb_as_compound_noun::PhrasalVerbAsCompoundNoun;
pub use pique_interest::PiqueInterest;
pub use possessive_your::PossessiveYour;
pub use pronoun_contraction::PronounContraction;
pub use repeated_words::RepeatedWords;
pub use save_to_safe::SaveToSafe;
pub use sentence_capitalization::SentenceCapitalization;
pub use since_duration::SinceDuration;
pub use somewhat_something::SomewhatSomething;
pub use spaces::Spaces;
pub use spell_check::SpellCheck;
pub use spelled_numbers::SpelledNumbers;
pub use suggestion::Suggestion;
pub use that_which::ThatWhich;
pub use the_how_why::TheHowWhy;
pub use the_my::TheMy;
pub use then_than::ThenThan;
pub use throw_rubbish::ThrowRubbish;
pub use unclosed_quotes::UnclosedQuotes;
pub use use_genitive::UseGenitive;
pub use was_aloud::WasAloud;
pub use whereas::Whereas;
pub use widely_accepted::WidelyAccepted;
pub use win_prize::WinPrize;
pub use wordpress_dotcom::WordPressDotcom;

use crate::{Document, LSend, render_markdown};

/// A __stateless__ rule that searches documents for grammatical errors.
///
/// Commonly implemented via [`PatternLinter`].
///
/// See also: [`LintGroup`].
pub trait Linter: LSend {
    /// Analyzes a document and produces zero or more [`Lint`]s.
    /// We pass `self` mutably for caching purposes.
    fn lint(&mut self, document: &Document) -> Vec<Lint>;
    /// A user-facing description of what kinds of grammatical errors this rule looks for.
    /// It is usually shown in settings menus.
    fn description(&self) -> &str;
}

/// A blanket-implemented trait that renders the Markdown description field of a linter to HTML.
pub trait HtmlDescriptionLinter {
    fn description_html(&self) -> String;
}

impl<L: ?Sized> HtmlDescriptionLinter for L
where
    L: Linter,
{
    fn description_html(&self) -> String {
        let desc = self.description();
        render_markdown(desc)
    }
}

#[cfg(test)]
mod tests {
    use super::Linter;
    use crate::{Document, FstDictionary, parsers::PlainEnglish};

    #[track_caller]
    pub fn assert_lint_count(text: &str, mut linter: impl Linter, count: usize) {
        let test = Document::new_markdown_default_curated(text);
        let lints = linter.lint(&test);
        dbg!(&lints);
        if lints.len() != count {
            panic!(
                "Expected \"{text}\" to create {count} lints, but it created {}.",
                lints.len()
            );
        }
    }

    /// Assert the total number of suggestions produced by a [`Linter`], spread across all produced
    /// [`Lint`]s.
    #[track_caller]
    pub fn assert_suggestion_count(text: &str, mut linter: impl Linter, count: usize) {
        let test = Document::new_markdown_default_curated(text);
        let lints = linter.lint(&test);
        assert_eq!(
            lints.iter().map(|l| l.suggestions.len()).sum::<usize>(),
            count
        );
    }

    /// Runs a provided linter on text, applies the first suggestion from each lint
    /// and asserts whether the result is equal to a given value.
    #[track_caller]
    pub fn assert_suggestion_result(text: &str, linter: impl Linter, expected_result: &str) {
        assert_nth_suggestion_result(text, linter, expected_result, 0);
    }

    /// Runs a provided linter on text, applies the nth suggestion from each lint
    /// and asserts whether the result is equal to a given value.
    ///
    /// Note that `n` starts at zero.
    #[track_caller]
    pub fn assert_nth_suggestion_result(
        text: &str,
        mut linter: impl Linter,
        expected_result: &str,
        n: usize,
    ) {
        let transformed_str = transform_nth_str(text, &mut linter, n);

        if transformed_str.as_str() != expected_result {
            panic!(
                "Expected \"{transformed_str}\" to be \"{expected_result}\" after applying the computed suggestions."
            );
        }

        // Applying the suggestions should fix all the lints.
        assert_lint_count(&transformed_str, linter, 0);
    }

    pub fn assert_top3_suggestion_result(
        text: &str,
        mut linter: impl Linter,
        expected_result: &str,
    ) {
        let zeroth = transform_nth_str(text, &mut linter, 0);
        let first = transform_nth_str(text, &mut linter, 1);
        let second = transform_nth_str(text, &mut linter, 2);

        match (
            zeroth.as_str() == expected_result,
            first.as_str() == expected_result,
            second.as_str() == expected_result,
        ) {
            (true, false, false) => assert_lint_count(&zeroth, linter, 0),
            (false, true, false) => assert_lint_count(&first, linter, 0),
            (false, false, true) => assert_lint_count(&second, linter, 0),
            (false, false, false) => panic!(
                "None of the top 3 suggestions produced the expected result:\n\
                Expected: \"{expected_result}\"\n\
                Got:\n\
                [0]: \"{zeroth}\"\n\
                [1]: \"{first}\"\n\
                [2]: \"{second}\""
            ),
            // I think it's not possible for more than one suggestion to be correct
            (true, true, false) => unreachable!(),
            (true, false, true) => unreachable!(),
            (false, true, true) => unreachable!(),
            (true, true, true) => unreachable!(),
        }
    }

    fn transform_nth_str(text: &str, linter: &mut impl Linter, n: usize) -> String {
        let mut text_chars: Vec<char> = text.chars().collect();

        let mut iter_count = 0;

        loop {
            let test = Document::new_from_vec(
                text_chars.clone().into(),
                &PlainEnglish,
                &FstDictionary::curated(),
            );
            let lints = linter.lint(&test);

            if let Some(lint) = lints.first() {
                if let Some(sug) = lint.suggestions.get(n) {
                    sug.apply(lint.span, &mut text_chars);

                    let transformed_str: String = text_chars.iter().collect();
                    dbg!(transformed_str);
                } else {
                    break;
                }
            } else {
                break;
            }

            iter_count += 1;

            if iter_count == 100 {
                break;
            }
        }

        eprintln!("Corrected {} times.", iter_count);

        text_chars.iter().collect()
    }
}
