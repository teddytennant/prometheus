//! Implementer-facing tests for `prometheus-synth`.
//!
//! Every test that calls extract_claims / check_claim / fact_check / accept /
//! token_count / precision / Orchestrator hits `unimplemented!` today, so
//! `cargo test -p prometheus-synth` must be red. After E1 these must match
//! `common::reference`.

mod common;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use prometheus_synth::{
    accept, check_claim, extract_claims, fact_check, generate_prompt, precision, token_count,
    Claim, Error, FactCheck, Generator, LabeledExample, Orchestrator, OrchestratorConfig,
    SourceDocument, Style, TokenCounter, Verdict, DEFAULT_MAX_BATCH, DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE, SCHEMA_VERSION,
};

use common::reference;

const GOLDEN_PRECISION: f64 = 0.8;
const LABELED_SAMPLE_SIZE: usize = 12;

const WATER_TEXT: &str = include_str!("../../tests/fixtures/synth/water.txt");
const PARIS_TEXT: &str = include_str!("../../tests/fixtures/synth/paris.txt");
const LABELED_JSONL: &str = include_str!("../../tests/fixtures/synth/labeled.jsonl");

fn water_text() -> &'static str {
    WATER_TEXT.trim()
}

fn paris_text() -> &'static str {
    PARIS_TEXT.trim()
}

fn water_doc() -> SourceDocument {
    SourceDocument {
        source_id: "water".into(),
        text: water_text().to_string(),
    }
}

fn paris_doc() -> SourceDocument {
    SourceDocument {
        source_id: "paris".into(),
        text: paris_text().to_string(),
    }
}

fn labeled_examples() -> Vec<LabeledExample> {
    LABELED_JSONL
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("labeled.jsonl row"))
        .collect()
}

const EXTRACT_GOLDENS: &[(&str, &[&str])] = &[
    ("", &[]),
    ("   \t\n", &[]),
    ("Hello. World!", &["Hello.", "World!"]),
    ("One sentence", &["One sentence"]),
    ("First. Second! Third?", &["First.", "Second!", "Third?"]),
    ("Hello.World!", &["Hello.World!"]),
    ("  Hello.   World!  ", &["Hello.", "World!"]),
    ("Dr. Smith went home.", &["Dr.", "Smith went home."]),
    ("What?Yes.", &["What?Yes."]),
    ("Done.", &["Done."]),
    ("A. B. C.", &["A.", "B.", "C."]),
    ("Hello.\nWorld!", &["Hello.", "World!"]),
    (
        "Café is open. Naïve approach.",
        &["Café is open.", "Naïve approach."],
    ),
    ("What??", &["What??"]),
    ("Hello. ", &["Hello."]),
];

struct CheckGolden {
    source: &'static str,
    claim: &'static str,
    expected: Verdict,
}

const CHECK_GOLDENS: &[CheckGolden] = &[
    CheckGolden {
        source: "Water boils at 100 degrees Celsius at sea level",
        claim: "Water boils at 100 degrees",
        expected: Verdict::Supported,
    },
    CheckGolden {
        source: "Water boils at 100 degrees Celsius at sea level",
        claim: "Water boils at 90 degrees",
        expected: Verdict::Contradicted,
    },
    CheckGolden {
        source: "The solution is not acidic and the mixture contains salt",
        claim: "The solution is acidic",
        expected: Verdict::Contradicted,
    },
    CheckGolden {
        source: "There are no cats in the library",
        claim: "There are cats in the library",
        expected: Verdict::Contradicted,
    },
    CheckGolden {
        source: "Paris is the capital of France",
        claim: "Berlin is in Germany",
        expected: Verdict::NotInSource,
    },
    CheckGolden {
        source: "Paris is the capital of France",
        claim: "Paris is the capital of France",
        expected: Verdict::Supported,
    },
    CheckGolden {
        source: "hello world",
        claim: "   ",
        expected: Verdict::NotInSource,
    },
    CheckGolden {
        source: "hello world",
        claim: "",
        expected: Verdict::NotInSource,
    },
    CheckGolden {
        source: "Water is a liquid at room temperature",
        claim: "is a",
        expected: Verdict::Supported,
    },
    CheckGolden {
        source: "Water is a liquid at room temperature",
        claim: "is an",
        expected: Verdict::NotInSource,
    },
    CheckGolden {
        source: "The mass is 3.14 kilograms",
        claim: "The mass is 2.71 kilograms",
        expected: Verdict::Contradicted,
    },
    CheckGolden {
        source: "Paris is the capital of France and the city has many museums",
        claim: "Paris has 12 museums",
        expected: Verdict::Supported,
    },
    CheckGolden {
        source: "that this with from they",
        claim: "that this",
        expected: Verdict::Supported,
    },
    CheckGolden {
        source: "hello world",
        claim: "that this",
        expected: Verdict::NotInSource,
    },
    CheckGolden {
        source: "The cats are here",
        claim: "The cats are here",
        expected: Verdict::Supported,
    },
    CheckGolden {
        source: "The cats are here",
        claim: "The dogs are here",
        expected: Verdict::NotInSource,
    },
    CheckGolden {
        source: "Mass is 3.14",
        claim: "Volume is 3.14",
        expected: Verdict::NotInSource,
    },
    CheckGolden {
        source: "The value is 3.14",
        claim: "The value is 14",
        expected: Verdict::Contradicted,
    },
    CheckGolden {
        source: "page 14",
        claim: "page 3.14",
        expected: Verdict::Contradicted,
    },
    CheckGolden {
        source: "not available here",
        claim: "available",
        expected: Verdict::Contradicted,
    },
    CheckGolden {
        source: "note available here",
        claim: "available",
        expected: Verdict::Supported,
    },
    CheckGolden {
        source: "no cats here",
        claim: "cats",
        expected: Verdict::Contradicted,
    },
    CheckGolden {
        source: "nobody cats here",
        claim: "cats",
        expected: Verdict::Supported,
    },
    CheckGolden {
        source: "The mixture contains salt",
        claim: "The mixture contains salt",
        expected: Verdict::Supported,
    },
    CheckGolden {
        source: "Café serves pastry",
        claim: "Café serves pastry",
        expected: Verdict::Supported,
    },
    CheckGolden {
        source: "The solution is not acidic",
        claim: "The solution is not acidic",
        expected: Verdict::Contradicted,
    },
    CheckGolden {
        source: "catscradle sits here",
        claim: "cats",
        expected: Verdict::NotInSource,
    },
    CheckGolden {
        source: "WATER boils",
        claim: "water BOILS",
        expected: Verdict::Supported,
    },
    CheckGolden {
        source: "water    boils",
        claim: "water boils",
        expected: Verdict::Supported,
    },
    CheckGolden {
        source: "hello world",
        claim: "hello.",
        expected: Verdict::NotInSource,
    },
    CheckGolden {
        source: "The cats are here",
        claim: "The cats are here.",
        expected: Verdict::NotInSource,
    },
    CheckGolden {
        source: "The mass is 3.14 kilograms",
        claim: "The mass is 3.14 kilograms",
        expected: Verdict::Supported,
    },
    CheckGolden {
        source: "hello",
        claim: "3.14",
        expected: Verdict::NotInSource,
    },
];

const PROPERTY_TEXTS: &[&str] = &[
    "",
    " ",
    "\t\n",
    "Hello.",
    "Hello. World!",
    "Dr. Smith went home.",
    "Water boils at 100 degrees.",
    "The solution is not acidic.",
    "Café is open. Naïve? Yes!",
    "No cats here.",
    "3.14 is pi.",
    "Hello.World!",
    "What? Yes. No!",
    "a",
    "that this with from",
    "The mass is 3.14 kilograms",
    "There are no cats in the library",
    "Paris is the capital of France",
    "note available here",
    "not available here",
    "The cats are here.",
    "page 14",
];

#[derive(Clone, Default)]
struct CallLog {
    calls: Rc<RefCell<Vec<(Vec<String>, u32, f32)>>>,
}

struct ScriptedGenerator {
    map: HashMap<String, String>,
    log: CallLog,
}

impl Generator for ScriptedGenerator {
    fn generate(
        &mut self,
        prompts: &[String],
        max_tokens: u32,
        temperature: f32,
    ) -> prometheus_synth::Result<Vec<String>> {
        self.log
            .calls
            .borrow_mut()
            .push((prompts.to_vec(), max_tokens, temperature));
        Ok(prompts
            .iter()
            .map(|p| {
                self.map
                    .get(p)
                    .cloned()
                    .unwrap_or_else(|| panic!("no scripted reply for {p}"))
            })
            .collect())
    }
}

struct MismatchGenerator {
    n: usize,
    log: CallLog,
}

impl Generator for MismatchGenerator {
    fn generate(
        &mut self,
        prompts: &[String],
        max_tokens: u32,
        temperature: f32,
    ) -> prometheus_synth::Result<Vec<String>> {
        self.log
            .calls
            .borrow_mut()
            .push((prompts.to_vec(), max_tokens, temperature));
        Ok(vec!["x".to_string(); self.n])
    }
}

struct BoomGenerator;

impl Generator for BoomGenerator {
    fn generate(
        &mut self,
        _prompts: &[String],
        _max_tokens: u32,
        _temperature: f32,
    ) -> prometheus_synth::Result<Vec<String>> {
        panic!("generator must not be called");
    }
}

struct FakeCounter {
    n: u32,
    seen: Rc<RefCell<Vec<String>>>,
}

impl TokenCounter for FakeCounter {
    fn token_count(&self, text: &str) -> u32 {
        self.seen.borrow_mut().push(text.to_string());
        self.n
    }
}

struct LenCounter;

impl TokenCounter for LenCounter {
    fn token_count(&self, text: &str) -> u32 {
        text.len() as u32
    }
}

struct BoomCounter;

impl TokenCounter for BoomCounter {
    fn token_count(&self, text: &str) -> u32 {
        panic!("token_count must not be called for {text:?}");
    }
}

// ---------------------------------------------------------------------------
// Constants / Style / FactCheck counts / generate_prompt (may pass)
// ---------------------------------------------------------------------------

#[test]
fn schema_and_defaults() {
    assert_eq!(SCHEMA_VERSION, 1);
    assert_eq!(DEFAULT_MAX_TOKENS, 512);
    assert!((DEFAULT_TEMPERATURE - 0.7).abs() < 1e-6);
    assert_eq!(DEFAULT_MAX_BATCH, 8);
    assert_eq!(Style::ALL.len(), 6);
    assert_eq!(
        Style::ALL,
        [
            Style::Encyclopedia,
            Style::Textbook,
            Style::Conversational,
            Style::News,
            Style::Technical,
            Style::Simplified,
        ]
    );
}

#[test]
fn style_and_verdict_as_str() {
    assert_eq!(Style::Encyclopedia.as_str(), "encyclopedia");
    assert_eq!(Style::Textbook.as_str(), "textbook");
    assert_eq!(Style::Conversational.as_str(), "conversational");
    assert_eq!(Style::News.as_str(), "news");
    assert_eq!(Style::Technical.as_str(), "technical");
    assert_eq!(Style::Simplified.as_str(), "simplified");
    assert_eq!(Verdict::Supported.as_str(), "supported");
    assert_eq!(Verdict::Contradicted.as_str(), "contradicted");
    assert_eq!(Verdict::NotInSource.as_str(), "not_in_source");
    assert_eq!(Style::parse("news").unwrap(), Style::News);
    assert!(Style::parse("nope").is_err());
}

#[test]
fn factcheck_counts_are_derived() {
    let fc = FactCheck {
        claims: vec![
            Claim {
                text: "a".into(),
                verdict: Verdict::Supported,
            },
            Claim {
                text: "b".into(),
                verdict: Verdict::Contradicted,
            },
            Claim {
                text: "c".into(),
                verdict: Verdict::NotInSource,
            },
            Claim {
                text: "d".into(),
                verdict: Verdict::Supported,
            },
        ],
    };
    assert_eq!(fc.n_supported(), 2);
    assert_eq!(fc.n_contradicted(), 1);
    assert_eq!(fc.n_not_in_source(), 1);
    let empty = FactCheck { claims: vec![] };
    assert_eq!(empty.n_supported(), 0);
    assert_eq!(empty.n_contradicted(), 0);
    assert_eq!(empty.n_not_in_source(), 0);
}

#[test]
fn generate_prompt_matches_template() {
    let doc = SourceDocument {
        source_id: "w".into(),
        text: "Water boils.".into(),
    };
    let prompt = generate_prompt(&doc, Style::News).unwrap();
    assert_eq!(
        prompt,
        "Rephrase the document in news style. Preserve every fact. Do not add facts.\n\nDocument:\nWater boils."
    );
}

#[test]
fn generate_prompt_embeds_every_style() {
    let doc = SourceDocument {
        source_id: "id".into(),
        text: "Body text.".into(),
    };
    for style in Style::ALL {
        let prompt = generate_prompt(&doc, style).unwrap();
        let head = format!(
            "Rephrase the document in {} style. Preserve every fact. Do not add facts.",
            style.as_str()
        );
        assert!(prompt.starts_with(&head), "{prompt}");
        assert!(prompt.ends_with("Document:\nBody text."), "{prompt}");
    }
}

#[test]
fn generate_prompt_preserves_utf8() {
    let doc = SourceDocument {
        source_id: "cafe".into(),
        text: "Café is open.".into(),
    };
    let prompt = generate_prompt(&doc, Style::Encyclopedia).unwrap();
    assert!(prompt.contains("Café is open."));
    assert!(prompt.contains("encyclopedia"));
}

#[test]
fn generate_prompt_empty_source_raises() {
    let err = generate_prompt(
        &SourceDocument {
            source_id: "w".into(),
            text: String::new(),
        },
        Style::News,
    )
    .unwrap_err();
    assert!(matches!(err, Error::EmptySource));
}

// ---------------------------------------------------------------------------
// extract_claims
// ---------------------------------------------------------------------------

#[test]
fn extract_claims_goldens() {
    for (text, expected) in EXTRACT_GOLDENS {
        let got = extract_claims(text);
        let want: Vec<String> = expected.iter().map(|s| (*s).to_string()).collect();
        assert_eq!(got, want, "extract_claims({text:?})");
    }
}

#[test]
fn extract_claims_preserves_casing() {
    assert_eq!(
        extract_claims("Hello. World!"),
        vec!["Hello.".to_string(), "World!".to_string()]
    );
}

// ---------------------------------------------------------------------------
// check_claim
// ---------------------------------------------------------------------------

#[test]
fn check_claim_goldens() {
    for row in CHECK_GOLDENS {
        let got = check_claim(row.source, row.claim).unwrap();
        assert_eq!(
            got, row.expected,
            "check_claim({:?}, {:?})",
            row.source, row.claim
        );
    }
}

#[test]
fn check_claim_empty_source_errors() {
    assert!(matches!(
        check_claim("", "Water boils.").unwrap_err(),
        Error::EmptySource
    ));
    assert!(matches!(
        check_claim("", "").unwrap_err(),
        Error::EmptySource
    ));
}

#[test]
fn check_claim_whitespace_source_is_not_empty_error() {
    assert_eq!(check_claim("   ", "hello").unwrap(), Verdict::NotInSource);
}

// ---------------------------------------------------------------------------
// fact_check
// ---------------------------------------------------------------------------

#[test]
fn fact_check_empty_source_errors() {
    assert!(matches!(
        fact_check("", "Water is a liquid.").unwrap_err(),
        Error::EmptySource
    ));
    assert!(matches!(
        fact_check("", "").unwrap_err(),
        Error::EmptySource
    ));
}

#[test]
fn fact_check_empty_rewrite_is_zero_claims() {
    let fc = fact_check("Water is a liquid.", "").unwrap();
    assert!(fc.claims.is_empty());
    assert_eq!(fc.n_supported(), 0);
    assert_eq!(fc.n_contradicted(), 0);
    assert_eq!(fc.n_not_in_source(), 0);
}

#[test]
fn fact_check_whitespace_rewrite_is_zero_claims() {
    let fc = fact_check("Water is a liquid.", "   \n").unwrap();
    assert!(fc.claims.is_empty());
}

#[test]
fn fact_check_composes_extract_and_check() {
    let source = "Water boils at 100 degrees Celsius at sea level and water is a liquid";
    let rewrite = "Water is a liquid. Helium is a liquid. Water boils at 90 degrees.";
    let fc = fact_check(source, rewrite).unwrap();
    let texts = extract_claims(rewrite);
    let got_texts: Vec<String> = fc.claims.iter().map(|c| c.text.clone()).collect();
    assert_eq!(got_texts, texts);
    let got_verdicts: Vec<Verdict> = fc.claims.iter().map(|c| c.verdict).collect();
    let want_verdicts: Vec<Verdict> = texts
        .iter()
        .map(|t| check_claim(source, t).unwrap())
        .collect();
    assert_eq!(got_verdicts, want_verdicts);
    // Trailing periods stay on the last token, so "liquid." does not match "liquid".
    assert_eq!(
        got_verdicts,
        [
            Verdict::NotInSource,
            Verdict::NotInSource,
            Verdict::Contradicted
        ]
    );
    assert_eq!(fc.n_supported(), 0);
    assert_eq!(fc.n_not_in_source(), 2);
    assert_eq!(fc.n_contradicted(), 1);
    assert_eq!(fc, reference::fact_check(source, rewrite).unwrap());
}

// ---------------------------------------------------------------------------
// accept
// ---------------------------------------------------------------------------

#[test]
fn accept_contradicted_rejects() {
    let fc = FactCheck {
        claims: vec![Claim {
            text: "x".into(),
            verdict: Verdict::Contradicted,
        }],
    };
    assert!(!accept(&fc));
    let mixed = FactCheck {
        claims: vec![
            Claim {
                text: "ok".into(),
                verdict: Verdict::Supported,
            },
            Claim {
                text: "bad".into(),
                verdict: Verdict::Contradicted,
            },
            Claim {
                text: "miss".into(),
                verdict: Verdict::NotInSource,
            },
        ],
    };
    assert!(!accept(&mixed));
}

#[test]
fn accept_not_in_source_alone_accepts() {
    let fc = FactCheck {
        claims: vec![Claim {
            text: "x".into(),
            verdict: Verdict::NotInSource,
        }],
    };
    assert!(accept(&fc));
    let mixed = FactCheck {
        claims: vec![
            Claim {
                text: "ok".into(),
                verdict: Verdict::Supported,
            },
            Claim {
                text: "miss".into(),
                verdict: Verdict::NotInSource,
            },
        ],
    };
    assert!(accept(&mixed));
}

#[test]
fn accept_empty_claims_accepts() {
    assert!(accept(&FactCheck { claims: vec![] }));
}

// ---------------------------------------------------------------------------
// token_count
// ---------------------------------------------------------------------------

#[test]
fn token_count_empty_is_zero_without_calling_tokenizer() {
    assert_eq!(token_count("", None), 0);
    assert_eq!(token_count("", Some(&BoomCounter)), 0);
}

#[test]
fn token_count_whitespace_words() {
    assert_eq!(token_count("hello  world\t\nfoo", None), 3);
    assert_eq!(token_count("  hello  ", None), 1);
    assert_eq!(token_count("   \t", None), 0);
    assert_eq!(token_count("hello\u{00a0}world", None), 2);
    assert_eq!(token_count("hello.", None), 1);
}

#[test]
fn token_count_uses_counter_when_provided() {
    let seen = Rc::new(RefCell::new(Vec::new()));
    let tok = FakeCounter {
        n: 4,
        seen: seen.clone(),
    };
    assert_eq!(token_count("hello world", Some(&tok)), 4);
    assert_eq!(seen.borrow().as_slice(), ["hello world"]);
    assert_eq!(token_count("abcd", Some(&LenCounter)), 4);
}

// ---------------------------------------------------------------------------
// precision (E1 gate sample)
// ---------------------------------------------------------------------------

#[test]
fn precision_on_labeled_gate_sample() {
    let rows = labeled_examples();
    assert_eq!(rows.len(), LABELED_SAMPLE_SIZE);
    let mut golds = Vec::new();
    for row in &rows {
        if !golds.contains(&row.gold) {
            golds.push(row.gold);
        }
    }
    assert!(golds.contains(&Verdict::Supported));
    assert!(golds.contains(&Verdict::Contradicted));
    assert!(golds.contains(&Verdict::NotInSource));
    let got = precision(&rows).unwrap();
    assert!((got - GOLDEN_PRECISION).abs() < 1e-12);
    let want = reference::precision(&rows).unwrap();
    assert!((got - want).abs() < 1e-12);
}

#[test]
fn precision_fail_closed_on_empty_or_no_supported_preds() {
    assert_eq!(precision(&[]).unwrap(), 0.0);
    let only_neg = vec![LabeledExample {
        source: "hello world".into(),
        claim: "zzzz missing claim".into(),
        gold: Verdict::NotInSource,
    }];
    assert_eq!(precision(&only_neg).unwrap(), 0.0);
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

#[test]
fn rephrase_empty_source_errors_before_generate() {
    let mut orch = Orchestrator::new(BoomGenerator, OrchestratorConfig::default()).unwrap();
    let err = orch
        .rephrase(
            &SourceDocument {
                source_id: "w".into(),
                text: String::new(),
            },
            Style::News,
        )
        .unwrap_err();
    assert!(matches!(err, Error::EmptySource));
}

#[test]
fn rephrase_scripted_generator_matches_reference() {
    let doc = water_doc();
    let style = Style::News;
    let prompt = generate_prompt(&doc, style).unwrap();
    let completion = "Water is a liquid at room temperature.";
    let log = CallLog::default();
    let mut map = HashMap::new();
    map.insert(prompt.clone(), completion.to_string());
    let gen = ScriptedGenerator {
        map,
        log: log.clone(),
    };
    let cfg = OrchestratorConfig {
        max_tokens: 64,
        temperature: 0.25,
        max_batch: 2,
        styles: vec![style],
    };
    let mut orch = Orchestrator::new(gen, cfg).unwrap();
    let got = orch.rephrase(&doc, style).unwrap();
    let expected = reference::rewrite_from_completion(&doc, style, completion, None).unwrap();
    assert_eq!(got, expected);
    let calls = log.calls.borrow();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, vec![prompt]);
    assert_eq!(calls[0].1, 64);
    assert!((calls[0].2 - 0.25).abs() < 1e-6);
    assert!(got.accepted);
    assert_eq!(got.source_id, "water");
    assert_eq!(got.style, Style::News);
    assert_eq!(got.text, completion);
}

#[test]
fn rephrase_uses_tokenizer_when_provided() {
    let doc = water_doc();
    let style = Style::Textbook;
    let prompt = generate_prompt(&doc, style).unwrap();
    let completion = "Water is a liquid.";
    let mut map = HashMap::new();
    map.insert(prompt, completion.to_string());
    let seen = Rc::new(RefCell::new(Vec::new()));
    let gen = ScriptedGenerator {
        map,
        log: CallLog::default(),
    };
    let mut orch = Orchestrator::new(gen, OrchestratorConfig::default())
        .unwrap()
        .with_tokenizer(FakeCounter {
            n: 7,
            seen: seen.clone(),
        });
    let got = orch.rephrase(&doc, style).unwrap();
    assert_eq!(got.token_count, 7);
    assert_eq!(seen.borrow().as_slice(), [completion]);
}

#[test]
fn rephrase_contradicted_completion_is_not_accepted() {
    let doc = water_doc();
    let style = Style::Technical;
    let prompt = generate_prompt(&doc, style).unwrap();
    let completion = "Water boils at 90 degrees.";
    let mut map = HashMap::new();
    map.insert(prompt, completion.to_string());
    let gen = ScriptedGenerator {
        map,
        log: CallLog::default(),
    };
    let mut orch = Orchestrator::new(gen, OrchestratorConfig::default()).unwrap();
    let got = orch.rephrase(&doc, style).unwrap();
    assert!(!got.accepted);
    assert!(got.fact_check.n_contradicted() >= 1);
    let expected = reference::rewrite_from_completion(&doc, style, completion, None).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn rephrase_generator_length_mismatch_errors() {
    let doc = water_doc();
    let mut orch = Orchestrator::new(
        MismatchGenerator {
            n: 0,
            log: CallLog::default(),
        },
        OrchestratorConfig::default(),
    )
    .unwrap();
    let err = orch.rephrase(&doc, Style::News).unwrap_err();
    assert!(matches!(err, Error::LengthMismatch { want: 1, got: 0 }));
    let mut orch = Orchestrator::new(
        MismatchGenerator {
            n: 2,
            log: CallLog::default(),
        },
        OrchestratorConfig::default(),
    )
    .unwrap();
    let err = orch.rephrase(&doc, Style::News).unwrap_err();
    assert!(matches!(err, Error::LengthMismatch { want: 1, got: 2 }));
}

#[test]
fn rephrase_many_empty_docs_errors_before_generate() {
    let mut orch = Orchestrator::new(BoomGenerator, OrchestratorConfig::default()).unwrap();
    let err = orch.rephrase_many(&[], None).unwrap_err();
    assert!(matches!(err, Error::EmptyBatch));
    let err = orch.rephrase_many(&[], Some(&[Style::News])).unwrap_err();
    assert!(matches!(err, Error::EmptyBatch));
}

#[test]
fn rephrase_many_empty_source_in_batch_errors_before_generate() {
    let docs = vec![
        water_doc(),
        SourceDocument {
            source_id: "empty".into(),
            text: String::new(),
        },
    ];
    let mut orch = Orchestrator::new(BoomGenerator, OrchestratorConfig::default()).unwrap();
    let err = orch.rephrase_many(&docs, Some(&[Style::News])).unwrap_err();
    assert!(matches!(err, Error::EmptySource));
}

#[test]
fn rephrase_many_cartesian_docs_major_and_max_batch_chunking() {
    let docs = vec![water_doc(), paris_doc()];
    let styles = [Style::News, Style::Technical];
    let cfg = OrchestratorConfig {
        max_tokens: 32,
        temperature: 0.5,
        max_batch: 3,
        styles: styles.to_vec(),
    };
    let completions = [
        (("water", Style::News), "Water is a liquid."),
        (("water", Style::Technical), "Water boils at 90 degrees."),
        (("paris", Style::News), "Paris is the capital of France."),
        (
            ("paris", Style::Technical),
            "Berlin is the capital of Germany.",
        ),
    ];
    let mut map = HashMap::new();
    for doc in &docs {
        for &style in &styles {
            let text = completions
                .iter()
                .find(|((id, s), _)| *id == doc.source_id && *s == style)
                .unwrap()
                .1;
            map.insert(generate_prompt(doc, style).unwrap(), text.to_string());
        }
    }
    let log = CallLog::default();
    let gen = ScriptedGenerator {
        map,
        log: log.clone(),
    };
    let mut orch = Orchestrator::new(gen, cfg.clone()).unwrap();
    let got = orch.rephrase_many(&docs, Some(&styles)).unwrap();
    assert_eq!(
        got.iter().map(|r| r.source_id.as_str()).collect::<Vec<_>>(),
        ["water", "water", "paris", "paris"]
    );
    assert_eq!(
        got.iter().map(|r| r.style).collect::<Vec<_>>(),
        [Style::News, Style::Technical, Style::News, Style::Technical]
    );
    let expected: Vec<_> = docs
        .iter()
        .flat_map(|doc| {
            styles.iter().map(move |&style| {
                let text = completions
                    .iter()
                    .find(|((id, s), _)| *id == doc.source_id && *s == style)
                    .unwrap()
                    .1;
                reference::rewrite_from_completion(doc, style, text, None).unwrap()
            })
        })
        .collect();
    assert_eq!(got, expected);
    let calls = log.calls.borrow();
    let sizes: Vec<usize> = calls.iter().map(|c| c.0.len()).collect();
    assert_eq!(sizes, vec![3, 1]);
    assert!(calls
        .iter()
        .all(|c| c.1 == 32 && (c.2 - 0.5).abs() < 1e-6 && c.0.len() <= cfg.max_batch as usize));
    let concat: Vec<String> = calls.iter().flat_map(|c| c.0.clone()).collect();
    let want_prompts: Vec<String> = docs
        .iter()
        .flat_map(|doc| {
            styles
                .iter()
                .map(move |&style| generate_prompt(doc, style).unwrap())
        })
        .collect();
    assert_eq!(concat, want_prompts);
}

#[test]
fn rephrase_many_default_styles_come_from_config() {
    let doc = water_doc();
    let cfg = OrchestratorConfig {
        max_tokens: 16,
        temperature: 0.1,
        max_batch: 8,
        styles: vec![Style::Simplified],
    };
    let prompt = generate_prompt(&doc, Style::Simplified).unwrap();
    let mut map = HashMap::new();
    map.insert(prompt.clone(), "Water is a liquid.".to_string());
    let log = CallLog::default();
    let gen = ScriptedGenerator {
        map,
        log: log.clone(),
    };
    let mut orch = Orchestrator::new(gen, cfg).unwrap();
    let got = orch.rephrase_many(&[doc], None).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].style, Style::Simplified);
    assert_eq!(log.calls.borrow()[0].0, vec![prompt]);
}

#[test]
fn rephrase_many_generator_length_mismatch_errors() {
    let docs = [water_doc()];
    let styles = [Style::News, Style::Simplified];
    let cfg = OrchestratorConfig {
        max_batch: 8,
        styles: styles.to_vec(),
        ..OrchestratorConfig::default()
    };
    let mut orch = Orchestrator::new(
        MismatchGenerator {
            n: 1,
            log: CallLog::default(),
        },
        cfg,
    )
    .unwrap();
    let err = orch.rephrase_many(&docs, Some(&styles)).unwrap_err();
    assert!(matches!(err, Error::LengthMismatch { want: 2, got: 1 }));
}

#[test]
fn rephrase_defaults_max_tokens_and_temperature() {
    let doc = water_doc();
    let prompt = generate_prompt(&doc, Style::News).unwrap();
    let mut map = HashMap::new();
    map.insert(prompt, "Water is a liquid.".to_string());
    let log = CallLog::default();
    let gen = ScriptedGenerator {
        map,
        log: log.clone(),
    };
    let mut orch = Orchestrator::new(gen, OrchestratorConfig::default()).unwrap();
    orch.rephrase(&doc, Style::News).unwrap();
    let calls = log.calls.borrow();
    assert_eq!(calls[0].1, DEFAULT_MAX_TOKENS);
    assert!((calls[0].2 - DEFAULT_TEMPERATURE).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Property: production matches the independent reference
// ---------------------------------------------------------------------------

#[test]
fn production_matches_reference_on_hand_rolled_strings() {
    for text in PROPERTY_TEXTS {
        assert_eq!(
            extract_claims(text),
            reference::extract_claims(text),
            "extract_claims({text:?})"
        );
        assert_eq!(
            token_count(text, None),
            reference::token_count(text, None),
            "token_count({text:?})"
        );
        if text.is_empty() {
            assert_eq!(token_count(text, Some(&LenCounter)), 0);
        } else {
            assert_eq!(
                token_count(text, Some(&LenCounter)),
                reference::token_count(text, Some(&LenCounter))
            );
        }
    }
    let sources: Vec<&&str> = PROPERTY_TEXTS.iter().filter(|t| !t.is_empty()).collect();
    for source in &sources {
        for claim in PROPERTY_TEXTS {
            assert_eq!(
                check_claim(source, claim).unwrap(),
                reference::check_claim(source, claim).unwrap(),
                "check_claim({source:?}, {claim:?})"
            );
        }
        for rewrite in PROPERTY_TEXTS {
            let prod = fact_check(source, rewrite).unwrap();
            let refer = reference::fact_check(source, rewrite).unwrap();
            assert_eq!(prod, refer, "fact_check({source:?}, {rewrite:?})");
            assert_eq!(accept(&prod), reference::accept(&refer));
        }
    }
    let mut rows = Vec::new();
    for source in sources.iter().take(6) {
        for claim in PROPERTY_TEXTS.iter().skip(3).take(6) {
            rows.push(LabeledExample {
                source: (*source).to_string(),
                claim: (*claim).to_string(),
                gold: reference::check_claim(source, claim).unwrap(),
            });
        }
    }
    let prod_p = precision(&rows).unwrap();
    let ref_p = reference::precision(&rows).unwrap();
    assert!((prod_p - ref_p).abs() < 1e-12);
}
