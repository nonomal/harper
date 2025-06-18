//! [`Pattern`]s are one of the more powerful ways to query text inside Harper, especially for beginners.
//!
//! Through the [`PatternLinter`](crate::linting::PatternLinter) trait, they make it much easier to
//! build Harper [rules](crate::linting::Linter).
//!
//! See the page about [`SequencePattern`] for a concrete example of their use.

use crate::{Document, LSend, Span, Token};

mod any_pattern;
mod implies_quantity;
mod indefinite_article;
mod inflection_of_be;
mod invert;
mod nominal_phrase;
mod whitespace_pattern;
mod within_edit_distance;
mod word;
mod word_set;

pub use any_pattern::AnyPattern;
pub use implies_quantity::ImpliesQuantity;
pub use indefinite_article::IndefiniteArticle;
pub use inflection_of_be::InflectionOfBe;
pub use invert::Invert;
pub use nominal_phrase::NominalPhrase;
pub use whitespace_pattern::WhitespacePattern;
pub use within_edit_distance::WithinEditDistance;
pub use word::Word;
pub use word_set::WordSet;

pub trait Pattern: LSend {
    /// Check if the pattern matches at the start of the given token slice.
    ///
    /// Returns the length of the match if successful, or `None` if not.
    fn matches(&self, tokens: &[Token], source: &[char]) -> Option<usize>;
}

pub trait PatternExt {
    fn iter_matches(&self, tokens: &[Token], source: &[char]) -> impl Iterator<Item = Span>;

    /// Search through all tokens to locate all non-overlapping pattern matches.
    fn find_all_matches(&self, tokens: &[Token], source: &[char]) -> Vec<Span> {
        self.iter_matches(tokens, source).collect()
    }
}

impl<P> PatternExt for P
where
    P: Pattern + ?Sized,
{
    fn iter_matches(&self, tokens: &[Token], source: &[char]) -> impl Iterator<Item = Span> {
        MatchIter::new(self, tokens, source)
    }
}

struct MatchIter<'a, 'b, 'c, P: ?Sized> {
    pattern: &'a P,
    tokens: &'b [Token],
    source: &'c [char],
    index: usize,
}
impl<'a, 'b, 'c, P> MatchIter<'a, 'b, 'c, P>
where
    P: Pattern + ?Sized,
{
    fn new(pattern: &'a P, tokens: &'b [Token], source: &'c [char]) -> Self {
        Self {
            pattern,
            tokens,
            source,
            index: 0,
        }
    }
}
impl<P> Iterator for MatchIter<'_, '_, '_, P>
where
    P: Pattern + ?Sized,
{
    type Item = Span;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.tokens.len() {
            if let Some(len) = self
                .pattern
                .matches(&self.tokens[self.index..], self.source)
            {
                let span = Span::new_with_len(self.index, len);
                self.index += len.max(1);
                return Some(span);
            } else {
                self.index += 1;
            }
        }

        None
    }
}

/// A simpler version of the [`Pattern`] trait that only matches a single
/// token.
pub trait SingleTokenPattern: LSend {
    fn matches_token(&self, token: &Token, source: &[char]) -> bool;
}

impl<S: SingleTokenPattern> Pattern for S {
    fn matches(&self, tokens: &[Token], source: &[char]) -> Option<usize> {
        if self.matches_token(tokens.first()?, source) {
            Some(1)
        } else {
            None
        }
    }
}

impl<F: LSend + Fn(&Token, &[char]) -> bool> SingleTokenPattern for F {
    fn matches_token(&self, token: &Token, source: &[char]) -> bool {
        self(token, source)
    }
}

pub trait DocPattern {
    fn find_all_matches_in_doc(&self, document: &Document) -> Vec<Span>;
}

impl<P: PatternExt> DocPattern for P {
    fn find_all_matches_in_doc(&self, document: &Document) -> Vec<Span> {
        self.find_all_matches(document.get_tokens(), document.get_source())
    }
}
