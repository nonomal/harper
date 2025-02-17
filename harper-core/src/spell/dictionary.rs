use blanket::blanket;

use super::FuzzyMatchResult;
use crate::WordMetadata;

/// An in-memory database that contains everything necessary to parse and analyze English text.
///
/// See also: [`super::FstDictionary`] and [`super::MutableDictionary`].
#[blanket(derive(Arc))]
pub trait Dictionary: Send + Sync {
    /// Check if the dictionary contains any capitalization of a given word.
    fn contains_word(&self, word: &[char]) -> bool;
    /// Check if the dictionary contains any capitalization of a given word.
    fn contains_word_str(&self, word: &str) -> bool;
    /// Check if the dictionary contains the exact capitalization of a given word.
    fn contains_exact_word(&self, word: &[char]) -> bool;
    /// Check if the dictionary contains the exact capitalization of a given word.
    fn contains_exact_word_str(&self, word: &str) -> bool;
    /// Gets best fuzzy match from dictionary
    fn fuzzy_match(
        &self,
        word: &[char],
        max_distance: u8,
        max_results: usize,
    ) -> Vec<FuzzyMatchResult>;
    /// Gets best fuzzy match from dictionary
    fn fuzzy_match_str(
        &self,
        word: &str,
        max_distance: u8,
        max_results: usize,
    ) -> Vec<FuzzyMatchResult>;
    fn get_correct_capitalization_of(&self, word: &[char]) -> Option<&'_ [char]>;
    /// Get the associated [`WordMetadata`] for any capitalization of a given word.
    /// If the word isn't in the dictionary, the resulting metadata will be
    /// empty.
    fn get_word_metadata(&self, word: &[char]) -> WordMetadata;
    /// Get the associated [`WordMetadata`] for any capitalization of a given word.
    /// If the word isn't in the dictionary, the resulting metadata will be
    /// empty.
    fn get_word_metadata_str(&self, word: &str) -> WordMetadata;
    /// Iterate over the words in the dictionary.
    fn words_iter(&self) -> Box<dyn Iterator<Item = &'_ [char]> + Send + '_>;

    /// Iterate over all the words in the dictionary of a given length
    fn words_with_len_iter(&self, len: usize) -> Box<dyn Iterator<Item = &'_ [char]> + Send + '_>;
}
