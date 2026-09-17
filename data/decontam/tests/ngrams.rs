//! Group: ngrams — 8-word grams after lowercase + whitespace collapse.

mod common;
mod reference;

use prometheus_decontam::{ngrams, NGRAM_N};

#[test]
fn ngram_n_is_eight() {
    // Combined with a real ngrams call so the stub cannot pass.
    assert_eq!(NGRAM_N, 8);
    let grams = ngrams(&common::words(8));
    assert_eq!(grams.len(), 1);
    assert_eq!(grams[0].split_whitespace().count(), 8);
}

#[test]
fn ngrams_emits_overlapping_eight_word_grams() {
    let text = "one two three four five six seven eight nine";
    let expected = vec![
        "one two three four five six seven eight".to_string(),
        "two three four five six seven eight nine".to_string(),
    ];
    assert_eq!(ngrams(text), expected);
    assert_eq!(ngrams(text), reference::ngrams(text));
}

#[test]
fn ngrams_lowercase_and_collapse_whitespace() {
    let messy = "  One\tTWO\n three   four\r\nfive\t\tsix seven   EIGHT nine  ";
    let clean = "one two three four five six seven eight nine";
    assert_eq!(ngrams(messy), ngrams(clean));
    assert_eq!(ngrams(messy), reference::ngrams(messy));
    assert_eq!(ngrams(messy), reference::ngrams(clean));
}

#[test]
fn ngrams_empty_and_short_texts_are_empty() {
    assert!(ngrams("").is_empty());
    assert!(ngrams("   \t\n  ").is_empty());
    assert!(ngrams(common::SEVEN_WORDS).is_empty());
    assert!(ngrams(&common::words(7)).is_empty());
    assert_eq!(ngrams(""), reference::ngrams(""));
    assert_eq!(
        ngrams(&common::words(7)),
        reference::ngrams(&common::words(7))
    );
}

#[test]
fn ngrams_exactly_eight_words_is_one_gram() {
    let text = common::words(8);
    let grams = ngrams(&text);
    assert_eq!(grams, vec![text.clone()]);
    assert_eq!(grams, reference::ngrams(&text));
}

#[test]
fn ngrams_property_each_gram_has_ngram_n_words() {
    for n in 0..=20 {
        let text = common::words(n);
        let grams = ngrams(&text);
        assert_eq!(grams, reference::ngrams(&text), "n={n}");
        let expected_len =
            n.saturating_sub(NGRAM_N)
                .saturating_add(if n >= NGRAM_N { 1 } else { 0 });
        assert_eq!(grams.len(), expected_len, "n={n}");
        for gram in &grams {
            assert_eq!(gram.split_whitespace().count(), NGRAM_N);
            assert_eq!(*gram, gram.to_lowercase());
        }
    }
}

#[test]
fn ngrams_punctuation_stays_attached_to_words() {
    let text = "alpha, bravo charlie delta echo foxtrot golf hotel";
    let grams = ngrams(text);
    assert_eq!(grams.len(), 1);
    assert!(grams[0].starts_with("alpha,"), "got {:?}", grams[0]);
    assert_eq!(grams, reference::ngrams(text));
}

#[test]
fn ngrams_is_deterministic_and_case_insensitive() {
    let text = common::PLANTED_TEXT;
    assert_eq!(ngrams(text), ngrams(&text.to_uppercase()));
    assert_eq!(ngrams(text), ngrams(text));
    assert_eq!(ngrams(text), reference::ngrams(text));
}
