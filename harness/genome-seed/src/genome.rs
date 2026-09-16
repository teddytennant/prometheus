//! Seed genome: roles, programs, tools, skills (spec 14.3).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Role {
    pub name: String,
    pub instructions: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Program {
    pub role: String,
    pub loop_src: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Tool {
    pub name: String,
    pub schema: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub body: String,
    pub test: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Genome {
    pub roles: Vec<Role>,
    pub programs: Vec<Program>,
    pub tools: Vec<Tool>,
    pub skills: Vec<Skill>,
}

impl Genome {
    pub fn seed() -> Self {
        Self {
            roles: vec![
                Role {
                    name: "planner".into(),
                    instructions: "decompose the mission".into(),
                },
                Role {
                    name: "worker".into(),
                    instructions: "execute one task".into(),
                },
                Role {
                    name: "reviewer".into(),
                    instructions: "veto or accept".into(),
                },
            ],
            programs: vec![Program {
                role: "worker".into(),
                loop_src: "claim; act; finish".into(),
            }],
            tools: vec![
                Tool {
                    name: "read".into(),
                    schema: r#"{"path":"string"}"#.into(),
                },
                Tool {
                    name: "edit".into(),
                    schema: r#"{"path":"string","old":"string","new":"string"}"#.into(),
                },
            ],
            skills: vec![Skill {
                name: "cargo-test".into(),
                body: "cargo test --workspace".into(),
                test: "exit 0".into(),
            }],
        }
    }

    pub fn serialize(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| e.to_string())
    }

    pub fn deserialize(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_seed() {
        let g = Genome::seed();
        let s = g.serialize().unwrap();
        let back = Genome::deserialize(&s).unwrap();
        assert_eq!(g, back);
        assert!(!g.roles.is_empty());
        assert!(!g.programs.is_empty());
        assert!(!g.tools.is_empty());
        assert!(!g.skills.is_empty());
    }
}
