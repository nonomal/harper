use std::collections::VecDeque;
use std::sync::Arc;

use itertools::Itertools;

use super::Parser;
use crate::expr::{ExprExt, SequenceExpr};
use crate::{Dictionary, Lrc, Span, Token, TokenKind, VecExt};

/// A parser that wraps any other parser to collapse token strings that match
/// the pattern `word_word` or `word-word`.
pub struct CollapseIdentifiers {
    inner: Box<dyn Parser>,
    dict: Arc<dyn Dictionary>,
}

impl CollapseIdentifiers {
    pub fn new(inner: Box<dyn Parser>, dict: Box<Arc<dyn Dictionary>>) -> Self {
        Self {
            inner,
            dict: *dict.clone(),
        }
    }
}

thread_local! {
    static WORD_OR_NUMBER: Lrc<SequenceExpr> = Lrc::new(SequenceExpr::default()
                .then_any_word()
                .then_one_or_more(SequenceExpr::default()
        .then_case_separator()
        .then_any_word()));
}

impl Parser for CollapseIdentifiers {
    fn parse(&self, source: &[char]) -> Vec<Token> {
        let mut tokens = self.inner.parse(source);

        let mut to_remove = VecDeque::default();

        for tok_span in WORD_OR_NUMBER
            .with(|v| v.clone())
            .iter_matches(&tokens, source)
            .collect::<Vec<_>>()
        {
            let start_tok = &tokens[tok_span.start];
            let end_tok = &tokens[tok_span.end - 1];
            let char_span = Span::new(start_tok.span.start, end_tok.span.end);

            if self.dict.contains_word(char_span.get_content(source)) {
                tokens[tok_span.start] = Token::new(char_span, TokenKind::blank_word());
                to_remove.extend(tok_span.start + 1..tok_span.end);
            }
        }

        tokens.remove_indices(to_remove.into_iter().sorted().unique().collect());

        tokens
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        FstDictionary, MergedDictionary, MutableDictionary, WordMetadata,
        parsers::{PlainEnglish, StrParser},
    };

    use super::*;

    #[test]
    fn matches_kebab() {
        let source: Vec<_> = "kebab-case".chars().collect();

        assert_eq!(
            WORD_OR_NUMBER
                .with(|v| v.clone())
                .iter_matches(&PlainEnglish.parse(&source), &source)
                .count(),
            1
        );
    }

    #[test]
    fn no_collapse() {
        let dict = FstDictionary::curated();
        let source = "This is a test.";

        let tokens =
            CollapseIdentifiers::new(Box::new(PlainEnglish), Box::new(dict)).parse_str(source);
        assert_eq!(tokens.len(), 8);
    }

    #[test]
    fn one_collapse() {
        let source = "This is a separated_identifier, wow!";
        let curated_dictionary = FstDictionary::curated();

        let tokens =
            CollapseIdentifiers::new(Box::new(PlainEnglish), Box::new(curated_dictionary.clone()))
                .parse_str(source);
        assert_eq!(tokens.len(), 13);

        let mut dict = MutableDictionary::new();
        dict.append_word_str("separated_identifier", WordMetadata::default());

        let mut merged_dict = MergedDictionary::new();
        merged_dict.add_dictionary(curated_dictionary);
        merged_dict.add_dictionary(Arc::new(dict));

        let tokens =
            CollapseIdentifiers::new(Box::new(PlainEnglish), Box::new(Arc::new(merged_dict)))
                .parse_str(source);
        assert_eq!(tokens.len(), 11);
    }

    #[test]
    fn kebab_collapse() {
        let source = "This is a separated-identifier, wow!";
        let curated_dictionary = FstDictionary::curated();

        let tokens =
            CollapseIdentifiers::new(Box::new(PlainEnglish), Box::new(curated_dictionary.clone()))
                .parse_str(source);

        assert_eq!(tokens.len(), 13);

        let mut dict = MutableDictionary::new();
        dict.append_word_str("separated-identifier", WordMetadata::default());

        let mut merged_dict = MergedDictionary::new();
        merged_dict.add_dictionary(curated_dictionary);
        merged_dict.add_dictionary(Arc::new(dict));

        let tokens =
            CollapseIdentifiers::new(Box::new(PlainEnglish), Box::new(Arc::new(merged_dict)))
                .parse_str(source);

        assert_eq!(tokens.len(), 11);
    }

    #[test]
    fn double_collapse() {
        let source = "This is a separated_identifier_token, wow!";
        let curated_dictionary = FstDictionary::curated();

        let tokens =
            CollapseIdentifiers::new(Box::new(PlainEnglish), Box::new(curated_dictionary.clone()))
                .parse_str(source);
        assert_eq!(tokens.len(), 15);

        let mut dict = MutableDictionary::new();
        dict.append_word_str("separated_identifier_token", WordMetadata::default());

        let mut merged_dict = MergedDictionary::new();
        merged_dict.add_dictionary(curated_dictionary);
        merged_dict.add_dictionary(Arc::new(dict));

        let tokens =
            CollapseIdentifiers::new(Box::new(PlainEnglish), Box::new(Arc::new(merged_dict)))
                .parse_str(source);
        assert_eq!(tokens.len(), 11);
    }

    #[test]
    fn two_collapses() {
        let source = "This is a separated_identifier, wow! separated_identifier";
        let curated_dictionary = FstDictionary::curated();

        let tokens =
            CollapseIdentifiers::new(Box::new(PlainEnglish), Box::new(curated_dictionary.clone()))
                .parse_str(source);
        assert_eq!(tokens.len(), 17);

        let mut dict = MutableDictionary::new();
        dict.append_word_str("separated_identifier", WordMetadata::default());

        let mut merged_dict = MergedDictionary::new();
        merged_dict.add_dictionary(curated_dictionary);
        merged_dict.add_dictionary(Arc::new(dict));

        let tokens =
            CollapseIdentifiers::new(Box::new(PlainEnglish), Box::new(Arc::new(merged_dict)))
                .parse_str(source);
        assert_eq!(tokens.len(), 13);
    }

    #[test]
    fn overlapping_identifiers() {
        let source = "This is a separated_identifier_token, wow!";
        let curated_dictionary = FstDictionary::curated();

        let tokens =
            CollapseIdentifiers::new(Box::new(PlainEnglish), Box::new(curated_dictionary.clone()))
                .parse_str(source);
        assert_eq!(tokens.len(), 15);

        let mut dict = MutableDictionary::new();
        dict.append_word_str("separated_identifier", WordMetadata::default());
        dict.append_word_str("identifier_token", WordMetadata::default());

        let mut merged_dict = MergedDictionary::new();
        merged_dict.add_dictionary(curated_dictionary);
        merged_dict.add_dictionary(Arc::new(dict));

        let tokens =
            CollapseIdentifiers::new(Box::new(PlainEnglish), Box::new(Arc::new(merged_dict)))
                .parse_str(source);
        assert_eq!(tokens.len(), 15);
    }

    #[test]
    fn nested_identifiers() {
        let source = "This is a separated_identifier_token, wow!";
        let curated_dictionary = FstDictionary::curated();

        let tokens =
            CollapseIdentifiers::new(Box::new(PlainEnglish), Box::new(curated_dictionary.clone()))
                .parse_str(source);
        assert_eq!(tokens.len(), 15);

        let mut dict = MutableDictionary::new();
        dict.append_word_str("separated_identifier_token", WordMetadata::default());
        dict.append_word_str("separated_identifier", WordMetadata::default());

        let mut merged_dict = MergedDictionary::new();
        merged_dict.add_dictionary(curated_dictionary);
        merged_dict.add_dictionary(Arc::new(dict));

        let tokens =
            CollapseIdentifiers::new(Box::new(PlainEnglish), Box::new(Arc::new(merged_dict)))
                .parse_str(source);
        assert_eq!(tokens.len(), 11);
    }
}
