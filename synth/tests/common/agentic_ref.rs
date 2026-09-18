//! Slow, obvious E4 agentic-trajectory reference.
//!
//! Production `prometheus-synth` must match this module. Production must never
//! import `tests/`. This file must not call production `generate_one`,
//! `generate_batch`, `outcome_verified_rate`, or `keep_verified`.
#![allow(dead_code)]

use prometheus_synth::{AgenticTask, Error, Generator, Outcome, Result, Sandbox, Step, Trajectory};

/// Terminal tool name. Case-sensitive. Any other tool is non-terminal.
pub const SUBMIT_TOOL: &str = "submit";

/// Sandbox result that, together with [`SUBMIT_TOOL`], marks [`Outcome::Verified`].
/// Compared exactly; no trim.
pub const VERIFIED_RESULT: &str = "verified";

/// Prompt sent to the generator at a given step.
///
/// The first call is exactly `task.prompt`. After each executed step the
/// prompt grows by `\n{tool}\n{args}\n{result}` in step order.
pub fn step_prompt(task: &AgenticTask, steps: &[Step]) -> String {
    let mut out = task.prompt.clone();
    for step in steps {
        out.push('\n');
        out.push_str(&step.tool);
        out.push('\n');
        out.push_str(&step.args);
        out.push('\n');
        out.push_str(&step.result);
    }
    out
}

/// Split a generator completion into `(tool, args)`.
///
/// 1. Trim Unicode whitespace from both ends.
/// 2. Empty remainder is not a tool call (`None`).
/// 3. The tool is the first Unicode-whitespace-separated token.
/// 4. `args` is the remainder with leading whitespace stripped.
pub fn parse_tool_call(text: &str) -> Option<(String, String)> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let mut parts = text.splitn(2, char::is_whitespace);
    let tool = parts.next().unwrap();
    if tool.is_empty() {
        return None;
    }
    let args = parts.next().unwrap_or("").trim_start().to_string();
    Some((tool.to_string(), args))
}

/// One trajectory. Sequential generator calls of a single prompt each.
///
/// Loop `max_steps` times (zero means no calls, [`Outcome::Truncated`]):
/// - `generate(&[step_prompt], max_tokens, temperature)`
/// - wrong completion count is [`Error::LengthMismatch`] with `want == 1`
/// - unparseable completion: [`Outcome::Failed`], keep steps so far, stop
/// - `sandbox.call(tool, args)`: `Err` surfaces unchanged
/// - record the [`Step`], then:
///   - tool `submit` and result exactly `verified`: [`Outcome::Verified`], stop
///   - tool `submit` otherwise: [`Outcome::Failed`], stop
///   - any other tool: continue
/// - loop exhausted without a terminal submit: [`Outcome::Truncated`]
fn run_one<G: Generator, S: Sandbox>(
    gen: &mut G,
    sandbox: &mut S,
    task: &AgenticTask,
    max_steps: u32,
    max_tokens: u32,
    temperature: f32,
) -> Result<Trajectory> {
    let mut steps = Vec::new();
    let mut outcome = Outcome::Truncated;
    for _ in 0..max_steps {
        let prompt = step_prompt(task, &steps);
        let completions = gen.generate(&[prompt], max_tokens, temperature)?;
        if completions.len() != 1 {
            return Err(Error::LengthMismatch {
                want: 1,
                got: completions.len(),
            });
        }
        let Some((tool, args)) = parse_tool_call(&completions[0]) else {
            outcome = Outcome::Failed;
            break;
        };
        let result = sandbox.call(&tool, &args)?;
        let is_submit = tool == SUBMIT_TOOL;
        let verified = is_submit && result == VERIFIED_RESULT;
        steps.push(Step { tool, args, result });
        if verified {
            outcome = Outcome::Verified;
            break;
        }
        if is_submit {
            outcome = Outcome::Failed;
            break;
        }
    }
    Ok(Trajectory {
        task_id: task.task_id.clone(),
        steps,
        outcome,
    })
}

/// Empty `task.prompt` is [`Error::EmptyPrompt`]. `n == 0` is
/// [`Error::ZeroSamples`] and wins over an empty prompt.
///
/// Returns exactly `n` trajectories, in order. Each trajectory starts with
/// empty history. The same generator and sandbox are reused (no reset).
pub fn generate_one<G: Generator, S: Sandbox>(
    gen: &mut G,
    sandbox: &mut S,
    task: &AgenticTask,
    n: u32,
    max_steps: u32,
    max_tokens: u32,
    temperature: f32,
) -> Result<Vec<Trajectory>> {
    if n == 0 {
        return Err(Error::ZeroSamples);
    }
    if task.prompt.is_empty() {
        return Err(Error::EmptyPrompt);
    }
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        out.push(run_one(
            gen,
            sandbox,
            task,
            max_steps,
            max_tokens,
            temperature,
        )?);
    }
    Ok(out)
}

/// Empty `tasks` is [`Error::EmptyBatch`] and wins over `n == 0`.
/// Then `n == 0` is [`Error::ZeroSamples`]. Then any empty prompt is
/// [`Error::EmptyPrompt`] (first in order). Checks run before any generate.
///
/// One inner vec per task, same order, each of length `n`.
pub fn generate_batch<G: Generator, S: Sandbox>(
    gen: &mut G,
    sandbox: &mut S,
    tasks: &[AgenticTask],
    n: u32,
    max_steps: u32,
    max_tokens: u32,
    temperature: f32,
) -> Result<Vec<Vec<Trajectory>>> {
    if tasks.is_empty() {
        return Err(Error::EmptyBatch);
    }
    if n == 0 {
        return Err(Error::ZeroSamples);
    }
    for task in tasks {
        if task.prompt.is_empty() {
            return Err(Error::EmptyPrompt);
        }
    }
    let mut out = Vec::with_capacity(tasks.len());
    for task in tasks {
        out.push(generate_one(
            gen,
            sandbox,
            task,
            n,
            max_steps,
            max_tokens,
            temperature,
        )?);
    }
    Ok(out)
}

/// Fraction of trajectories with [`Outcome::Verified`]. Empty is 0.0.
pub fn outcome_verified_rate(trajs: &[Trajectory]) -> f64 {
    if trajs.is_empty() {
        return 0.0;
    }
    let n_ok = trajs
        .iter()
        .filter(|t| t.outcome == Outcome::Verified)
        .count();
    n_ok as f64 / trajs.len() as f64
}

/// Keep only [`Outcome::Verified`], original order. Drops Failed and Truncated.
pub fn keep_verified(trajs: Vec<Trajectory>) -> Vec<Trajectory> {
    trajs
        .into_iter()
        .filter(|t| t.outcome == Outcome::Verified)
        .collect()
}
