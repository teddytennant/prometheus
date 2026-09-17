//! Group: flagship_catalog / Source::all — 7.1 table (already implemented).

use prometheus_mixture::{flagship_catalog, Source, SourceSpec};

#[test]
fn source_all_is_the_seven_one_table_order() {
    assert_eq!(
        Source::all(),
        [
            Source::Web,
            Source::Code,
            Source::MathScienceArxiv,
            Source::BooksPapers,
            Source::SyntheticRewrites,
            Source::SyntheticReasoning,
            Source::ProceduralArc,
            Source::AgenticTrajectories,
        ]
    );
}

#[test]
fn catalog_has_one_row_per_source_in_table_order() {
    let catalog = flagship_catalog();
    let sources: Vec<Source> = catalog.iter().map(|e| e.source).collect();
    assert_eq!(sources, Source::all().to_vec());
    assert_eq!(catalog.len(), 8);
}

#[test]
fn catalog_unique_tokens_and_epoch_bounds() {
    let catalog = flagship_catalog();
    let want: [(Source, Option<u64>, u32, u32); 8] = [
        (Source::Web, Some(20_000_000_000_000), 2, 3),
        (Source::Code, Some(4_000_000_000_000), 4, 4),
        (Source::MathScienceArxiv, Some(1_500_000_000_000), 4, 4),
        (Source::BooksPapers, None, 2, 4),
        (Source::SyntheticRewrites, Some(40_000_000_000_000), 1, 1),
        (Source::SyntheticReasoning, Some(10_000_000_000_000), 1, 1),
        (Source::ProceduralArc, Some(2_000_000_000_000), 1, 1),
        (Source::AgenticTrajectories, Some(1_000_000_000_000), 1, 1),
    ];
    for (got, (source, unique, lo, hi)) in catalog.iter().zip(want) {
        assert_eq!(
            *got,
            SourceSpec {
                source,
                unique_tokens: unique,
                epochs_min: lo,
                epochs_max: hi,
            }
        );
        assert!(got.epochs_min >= 1);
        assert!(got.epochs_max >= got.epochs_min);
    }
}

#[test]
fn books_papers_unique_tokens_are_none() {
    let books = flagship_catalog()
        .into_iter()
        .find(|e| e.source == Source::BooksPapers)
        .expect("books");
    assert_eq!(books.unique_tokens, None);
}
