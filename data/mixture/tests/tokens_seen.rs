//! Group: tokens_seen — unique * epochs, None if unique is None.

mod common;
mod reference;

use prometheus_mixture::{flagship_catalog, tokens_seen, Source};

#[test]
fn none_unique_is_none() {
    assert_eq!(tokens_seen(None, 4), None);
    assert_eq!(reference::tokens_seen(None, 4), None);
}

#[test]
fn product_matches_reference() {
    for &(unique, epochs) in &[
        (0_u64, 5_u32),
        (1, 1),
        (10, 2),
        (20_000_000_000_000, 3),
        (4_000_000_000_000, 4),
        (u64::MAX / 2, 1),
    ] {
        let got = tokens_seen(Some(unique), epochs);
        let want = reference::tokens_seen(Some(unique), epochs);
        assert_eq!(got, want, "unique={unique} epochs={epochs}");
        assert_eq!(got, Some(unique * u64::from(epochs)));
    }
}

#[test]
fn zero_epochs_is_zero_tokens() {
    assert_eq!(tokens_seen(Some(123), 0), Some(0));
    assert_eq!(reference::tokens_seen(Some(123), 0), Some(0));
}

#[test]
fn catalog_web_three_epochs() {
    let web = flagship_catalog()
        .into_iter()
        .find(|e| e.source == Source::Web)
        .expect("web");
    assert_eq!(
        tokens_seen(web.unique_tokens, 3),
        Some(60_000_000_000_000)
    );
}

#[test]
fn catalog_books_stays_none() {
    let books = flagship_catalog()
        .into_iter()
        .find(|e| e.source == Source::BooksPapers)
        .expect("books");
    assert_eq!(tokens_seen(books.unique_tokens, books.epochs_max), None);
    assert_eq!(
        reference::tokens_seen(books.unique_tokens, books.epochs_max),
        None
    );
}
