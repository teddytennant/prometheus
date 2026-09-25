# Working on Prometheus

Read this before changing anything. It applies to every contributor, human or
otherwise, and to every worktree.

## Start here

1. Read [PROGRESS.md](PROGRESS.md). It says what is on main, what is broken, what
   has run on an H200 and what comes next.
2. Read the spec section for whatever you're about to touch (`spec/`, or
   [MONOLITH.md](MONOLITH.md) for all of it). Build order and scope are in
   `spec/15-build-order.md`; S3 (15.6) is out of scope. Layout is
   `spec/17-repo-layout.md` rooted at the repo root.

## Keep PROGRESS.md current

PROGRESS.md is the shared record of where the build stands. It goes stale the
moment someone merges without touching it, so updating it is part of the change,
not a follow-up.

- **Same commit.** Any commit that changes what PROGRESS.md says (a feature lands,
  a defect is fixed or found, a V stage runs, LOC moves a lot) updates PROGRESS.md
  in that commit. For a merge, update it on the merge commit.
- **What to touch.** The coverage row for the step, the Known defects list (delete
  a line when its fix is on main, with the commit in the message), the H200 stages
  table for any job, the Tests table after a full gate run, In flight, and the date
  and commit at the top.
- **Only numbers you saw.** Test counts, job ids, LOC and timings come from output
  you ran this session. If you didn't run it, don't change the number; say it's
  unchecked.
- **Merged is not done.** A row is done when the code does what its spec section
  says at S1/S2 scale and a test would fail if it didn't. Say what's missing.
- **A V stage is verified only** when its runner drives production code on the
  hardware and scale spec 16.2 sets, and the exit check read the job's own output.
  Anything else is written down as not verified.
- **Keep it short.** Current state, not history. History is `git log`. Keep the
  file under about 400 lines; cut finished items instead of piling on.

## Rules

- Production code never imports from `tests/`. CI checks this
  (`.github/ci/checks.py tree`).
- Oracles in `tests/reference/` are written from the spec or the source paper, not
  from the production code. A reference that restates production's formulas proves
  nothing; the audit behind PROGRESS.md found a dozen bugs hiding that way.
- A test must be able to fail. If it checks a function against itself, a value
  the test injected, or a hash of its input, it isn't a test.
- No stubs, placeholder math, weakened tests, `except: pass` or silent fallbacks on
  main. A pytest skip on a CPU runner is only allowed for a missing GPU.
- Don't edit `spec/`, `MONOLITH.md` or `README.md` except to fix something the
  build proved wrong. Edit `spec/`, then rebuild MONOLITH.md with the command in
  README.md.

## Gate

Run the same checks CI runs before every push. Red means no push.

```
python3 .github/ci/checks.py tree
uv run ruff check .
uv run pytest --strict-markers --strict-config -o xfail_strict=true --junitxml=pytest.xml
python3 .github/ci/checks.py skips pytest.xml
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

CI pins Rust 1.95.0.

## Git

- One logical change per commit, present tense, imperative, specific.
- Author and committer: Teddy Tennant <teddytennant@icloud.com>.
- No co-author or tool attribution lines. No em dashes in prose.
- Push to `origin main` after a green gate. Never force push or rewrite history.
