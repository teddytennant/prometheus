//! Group: shingles of size n; short texts → empty.

mod common;
mod reference;

use prometheus_dedup::shingles;

#[test]
fn size_three_on_five_tokens() {
    let got = shingles("the cat sat on the", 3);
    assert_eq!(
        got,
        vec![
            "the cat sat".to_string(),
            "cat sat on".to_string(),
            "sat on the".to_string()
        ]
    );
    assert_eq!(got, reference::shingles("the cat sat on the", 3));
}

#[test]
fn size_one_is_the_word_list() {
    assert_eq!(
        shingles("Hello   THERE world", 1),
        vec!["hello", "there", "world"]
    );
}

#[test]
fn short_text_is_empty() {
    assert_eq!(shingles("one two three", 5), Vec::<String>::new());
    assert_eq!(shingles("hi", 5), Vec::<String>::new());
    assert_eq!(shingles("", 1), Vec::<String>::new());
}

#[test]
fn exact_n_tokens_is_one_shingle() {
    assert_eq!(
        shingles("one two three four five", 5),
        vec!["one two three four five"]
    );
}

#[test]
fn n_zero_is_empty() {
    assert_eq!(shingles("one two three", 0), Vec::<String>::new());
}

#[test]
fn uses_normalize_so_case_and_spacing_do_not_matter() {
    let a = shingles("The CAT sat", 2);
    let b = shingles("the   cat SAT", 2);
    assert_eq!(a, b);
    assert_eq!(a, vec!["the cat", "cat sat"]);
}

#[test]
fn matches_reference_on_fixture_corpus() {
    for d in common::fixture_corpus() {
        for n in [1, 2, 5, 8] {
            assert_eq!(
                shingles(&d.text, n),
                reference::shingles(&d.text, n),
                "shingles n={n} mismatch on {}",
                d.id
            );
        }
    }
}
