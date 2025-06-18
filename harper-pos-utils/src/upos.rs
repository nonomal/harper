use is_macro::Is;
use serde::{Deserialize, Serialize};
use strum_macros::{AsRefStr, EnumIter};

/// Represents the universal parts of speech as outlined by [universaldependencies.org](https://universaldependencies.org/u/pos/index.html).
#[derive(
    Debug,
    Default,
    Hash,
    Eq,
    PartialEq,
    Clone,
    Copy,
    EnumIter,
    AsRefStr,
    Serialize,
    Deserialize,
    PartialOrd,
    Ord,
    Is,
)]
pub enum UPOS {
    ADJ,
    ADP,
    ADV,
    AUX,
    CCONJ,
    DET,
    INTJ,
    #[default]
    NOUN,
    NUM,
    PART,
    PRON,
    PROPN,
    PUNCT,
    SCONJ,
    SYM,
    VERB,
}

impl UPOS {
    pub fn from_conllu(other: rs_conllu::UPOS) -> Option<Self> {
        Some(match other {
            rs_conllu::UPOS::ADJ => UPOS::ADJ,
            rs_conllu::UPOS::ADP => UPOS::ADP,
            rs_conllu::UPOS::ADV => UPOS::ADV,
            rs_conllu::UPOS::AUX => UPOS::AUX,
            rs_conllu::UPOS::CCONJ => UPOS::CCONJ,
            rs_conllu::UPOS::DET => UPOS::DET,
            rs_conllu::UPOS::INTJ => UPOS::INTJ,
            rs_conllu::UPOS::NOUN => UPOS::NOUN,
            rs_conllu::UPOS::NUM => UPOS::NUM,
            rs_conllu::UPOS::PART => UPOS::PART,
            rs_conllu::UPOS::PRON => UPOS::PRON,
            rs_conllu::UPOS::PROPN => UPOS::PROPN,
            rs_conllu::UPOS::PUNCT => UPOS::PUNCT,
            rs_conllu::UPOS::SCONJ => UPOS::SCONJ,
            rs_conllu::UPOS::SYM => UPOS::SYM,
            rs_conllu::UPOS::VERB => UPOS::VERB,
            rs_conllu::UPOS::X => return None,
        })
    }

    pub fn is_nominal(&self) -> bool {
        matches!(self, Self::NOUN | Self::PROPN)
    }
}
