//! Agentic trajectories in D1 (spec 8, 15.5 E4).
//!
//! A generator proposes tool calls. A [`Sandbox`] executes them. Traces
//! are kept only when [`Outcome::Verified`]. The sandbox is a trait so
//! this crate does not import `prometheus-envs`.

use serde::{Deserialize, Serialize};

use crate::{Generator, Result};

/// One tool invocation inside a D1 sandbox.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub tool: String,
    pub args: String,
    pub result: String,
}

/// End state of a trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Verified,
    Failed,
    Truncated,
}

/// One agent run on one task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trajectory {
    pub task_id: String,
    pub steps: Vec<Step>,
    pub outcome: Outcome,
}

/// A task the agent should solve in the sandbox.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgenticTask {
    pub task_id: String,
    pub prompt: String,
}

/// D1 sandbox. One call is one tool invocation.
pub trait Sandbox {
    fn call(&mut self, tool: &str, args: &str) -> Result<String>;
}

/// Empty `task.prompt` is [`Error::EmptyPrompt`]. `n == 0` is
/// [`Error::ZeroSamples`].
pub fn generate_one<G: Generator, S: Sandbox>(
    gen: &mut G,
    sandbox: &mut S,
    task: &AgenticTask,
    n: u32,
    max_steps: u32,
    max_tokens: u32,
    temperature: f32,
) -> Result<Vec<Trajectory>> {
    let _ = (gen, sandbox, task, n, max_steps, max_tokens, temperature);
    unimplemented!("E4 generate_one")
}

/// Batch. Empty `tasks` is [`Error::EmptyBatch`].
pub fn generate_batch<G: Generator, S: Sandbox>(
    gen: &mut G,
    sandbox: &mut S,
    tasks: &[AgenticTask],
    n: u32,
    max_steps: u32,
    max_tokens: u32,
    temperature: f32,
) -> Result<Vec<Vec<Trajectory>>> {
    let _ = (gen, sandbox, tasks, n, max_steps, max_tokens, temperature);
    unimplemented!("E4 generate_batch")
}

/// Fraction of trajectories with [`Outcome::Verified`]. Empty is 0.0.
pub fn outcome_verified_rate(trajs: &[Trajectory]) -> f64 {
    let _ = trajs;
    unimplemented!("E4 outcome_verified_rate")
}

/// Keep only [`Outcome::Verified`].
pub fn keep_verified(trajs: Vec<Trajectory>) -> Vec<Trajectory> {
    let _ = trajs;
    unimplemented!("E4 keep_verified")
}
