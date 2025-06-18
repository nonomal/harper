//! This test creates snapshots of the part-of-speech (POS) tags assigned by the
//! [`Document`] struct to the text files in the `tests/text` directory.
//!
//! # Usage
//!
//! To add a new snapshot, simply add the document to `tests/text` and run this
//! test. It will automatically create a new snapshot in `tests/text/tagged`.
//! To update an existing snapshot, also just run this test.
//!
//! Note: This test will fail if the snapshot files are not up to date. This
//! ensures that CI will fail if the POS tagger changes its behavior.
//!
//! # Snapshot format
//!
//! The snapshot files contain 2 lines for every line in the original text. The
//! first line contains the original text, and the second line contains the POS
//! tags. The text and tags are aligned so that the tags are directly below the
//! corresponding words in the text. Example:
//!
//! ```md
//! > I   told her   how   I   had stopped off       in          Chicago for a   day   on  my way East    .
//! # ISg V    I/J/D NSg/C ISg V   V/J     NSg/V/J/P NPrSg/V/J/P NPr     C/P D/P NPrSg J/P D  J   NPrSg/J .
//! ```
//!
//! ## Tags
//!
//! Tags are assigned based on the [`TokenKind`] and [`WordMetadata`] of a
//! token.
//!
//! - The tag of [`TokenKind::Word`] variants depends on their
//!   [`WordMetadata`]. If they don't have any metadata, they are denoted by `?`.
//!   Otherwise, the tag is constructed as follows:
//!
//!   - Nouns are denoted by `N`.
//!     - The `Pl` suffix means plural, and `Sg` means singular.
//!     - The `Pr` suffix means proper noun.
//!     - The `$` suffix means possessive.
//!   - Pronouns are denoted by `I`.
//!     - The `Pl` suffix means plural, and `Sg` means singular.
//!     - The `$` suffix means possessive.
//!   - Verbs are denoted by `V`.
//!     - The `L` suffix means linking verb.
//!     - The `X` suffix means auxiliary verb.
//!   - Adjectives are denoted by `J`.
//!   - Adverbs are denoted by `R`.
//!   - Conjunctions are denoted by `C`.
//!   - Determiners are denoted by `D`.
//!   - Prepositions are denoted by `P`.
//!   - Dialects are denoted by `Am`, `Br`, `Ca`, or `Au`.
//!
//!   The tagger supports uncertainty, so a single word can be e.g. both a
//!   noun and a verb. This is denoted by a `/` between the tags.
//!   For example, `N/V/J` means the word is a noun, verb, and/or adjective.
//!
//! - [`TokenKind::Punctuation`] are denoted by `.`.
//! - [`TokenKind::Number`] are denoted by `#`.
//! - [`TokenKind::Decade`] are denoted by `#d`.
//! - [`TokenKind::Space`], [`TokenKind::Newline`], and
//!   [`TokenKind::ParagraphBreak`] are ignored.
//! - All other token kinds are denoted by their variant name.
use std::borrow::Cow;

use harper_core::{Degree, Document, FstDictionary, TokenKind, WordMetadata};

mod snapshot;

fn format_word_tag(word: &WordMetadata) -> String {
    // These tags are inspired by the Penn Treebank POS tagset
    let mut tags = String::new();
    let mut add = |t: &str| {
        if !tags.is_empty() {
            tags.push('/');
        }
        tags.push_str(t);
    };

    fn add_bool(tag: &mut String, name: &str, value: Option<bool>) {
        if let Some(value) = value {
            if !value {
                tag.push('!');
            }
            tag.push_str(name);
        }
    }
    fn add_switch(tag: &mut String, value: Option<bool>, yes: &str, no: &str) {
        if let Some(value) = value {
            if value {
                tag.push_str(yes);
            } else {
                tag.push_str(no);
            }
        }
    }

    if let Some(noun) = word.noun {
        let mut tag = String::from("N");
        add_bool(&mut tag, "Pr", noun.is_proper);
        add_switch(&mut tag, noun.is_plural, "Pl", "Sg");
        add_bool(&mut tag, "$", noun.is_possessive);
        add(&tag);
    }
    if let Some(pronoun) = word.pronoun {
        let mut tag = String::from("I");
        add_switch(&mut tag, pronoun.is_plural, "Pl", "Sg");
        add_bool(&mut tag, "$", pronoun.is_possessive);
        add(&tag);
    }
    if let Some(verb) = word.verb {
        let mut tag = String::from("V");
        add_bool(&mut tag, "L", verb.is_linking);
        add_bool(&mut tag, "X", verb.is_auxiliary);
        add(&tag);
    }
    if let Some(adjective) = word.adjective {
        let mut tag = String::from("J");
        if let Some(dgree) = adjective.degree {
            tag.push_str(match dgree {
                Degree::Comparative => "C",
                Degree::Superlative => "S",
                Degree::Positive => "P",
            });
        }
        add(&tag);
    }
    if let Some(_adverb) = word.adverb {
        add("R");
    }
    if let Some(_conj) = word.conjunction {
        add("C");
    }
    if word.determiner {
        add("D");
    }
    if word.preposition {
        add("P");
    }

    word.dialects.iter().for_each(|d| {
        if let Ok(dialect) = harper_core::Dialect::try_from(d) {
            add(match dialect {
                harper_core::Dialect::American => "Am",
                harper_core::Dialect::British => "Br",
                harper_core::Dialect::Canadian => "Ca",
                harper_core::Dialect::Australian => "Au",
            });
        }
    });

    if tags.is_empty() {
        String::from("W?")
    } else {
        tags
    }
}
fn format_tag(kind: &TokenKind) -> Cow<'static, str> {
    match kind {
        TokenKind::Word(word) => {
            // These tags are inspired by the Penn Treebank POS tagset
            if let Some(word) = word {
                Cow::Owned(format_word_tag(word))
            } else {
                Cow::Borrowed("?")
            }
        }
        TokenKind::Punctuation(_) => Cow::Borrowed("."),
        TokenKind::Number(_) => Cow::Borrowed("#"),
        TokenKind::Decade => Cow::Borrowed("#d"),

        // The following variants just print their variant name
        TokenKind::Space(_) => Cow::Borrowed("Space"),
        TokenKind::Newline(_) => Cow::Borrowed("Newline"),
        TokenKind::EmailAddress => Cow::Borrowed("Email"),
        TokenKind::Url => Cow::Borrowed("Url"),
        TokenKind::Hostname => Cow::Borrowed("Hostname"),
        TokenKind::Unlintable => Cow::Borrowed("Unlintable"),
        TokenKind::Regexish => Cow::Borrowed("Regexish"),
        TokenKind::ParagraphBreak => Cow::Borrowed("ParagraphBreak"),
    }
}

struct Formatter {
    out: String,
    line1: String,
    line2: String,
}
impl Formatter {
    const LINE1_PREFIX: &'static str = "> ";
    const LINE2_PREFIX: &'static str = "# ";
    fn new() -> Self {
        Self {
            out: String::new(),
            line1: String::from(Self::LINE1_PREFIX),
            line2: String::from(Self::LINE2_PREFIX),
        }
    }

    fn add(&mut self, token: &str, tag: &str) {
        for (line_number, token_line) in token.split('\n').enumerate() {
            if line_number > 0 {
                self.new_line();
            }

            self.line1.push_str(token_line);
            self.line1.push(' ');
            self.line2.push_str(tag);
            self.line2.push(' ');
            let token_chars = token_line.chars().count();
            let tag_chars = tag.chars().count();
            for _ in token_chars..tag_chars {
                self.line1.push(' ');
            }
            for _ in tag_chars..token_chars {
                self.line2.push(' ');
            }
        }
    }

    fn new_line(&mut self) {
        self.out.push_str(self.line1.trim_end());
        self.out.push('\n');
        self.out.push_str(self.line2.trim_end());
        self.out.push('\n');

        self.line1.clear();
        self.line2.clear();

        self.line1.push_str(Self::LINE1_PREFIX);
        self.line2.push_str(Self::LINE2_PREFIX);
    }

    fn finish(mut self) -> String {
        self.new_line();
        self.out
    }
}

#[test]
fn test_pos_tagger() {
    snapshot::snapshot_all_text_files("tagged", ".md", |source| {
        let dict = FstDictionary::curated();
        let document = Document::new_markdown_default(source, &dict);

        let mut formatter = Formatter::new();
        for token in document.fat_string_tokens() {
            match token.kind {
                TokenKind::Space(_) => { /* ignore */ }
                TokenKind::ParagraphBreak => {
                    formatter.new_line();
                    formatter.new_line();
                }
                TokenKind::Newline(_) => {
                    formatter.new_line();
                }
                kind => {
                    let text = &token.content;
                    let tag = format_tag(&kind);
                    formatter.add(text, &tag);
                }
            }
        }

        formatter.finish()
    });
}
