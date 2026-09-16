//! CLI parse/dispatch: init, join, mission, status, logs, pause, freeze, update.

use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Init,
    NodeJoin { addr: String },
    MissionAdd { spec: String },
    Status { json: bool },
    Logs { task: String },
    Replay { task: String },
    Pause,
    Resume,
    Freeze { signature: String },
    SelfUpdate { version: String, signatures: Vec<String> },
}

pub fn parse(args: &[&str]) -> Result<Command, String> {
    let mut it = args.iter().copied().peekable();
    let cmd = it.next().ok_or("empty")?;
    match cmd {
        "init" => Ok(Command::Init),
        "node" => match it.next() {
            Some("join") => Ok(Command::NodeJoin {
                addr: it.next().ok_or("addr")?.to_string(),
            }),
            _ => Err("node subcommand".into()),
        },
        "mission" => match it.next() {
            Some("add") => Ok(Command::MissionAdd {
                spec: it.next().ok_or("spec")?.to_string(),
            }),
            _ => Err("mission subcommand".into()),
        },
        "status" => {
            let json = it.next() == Some("--json");
            Ok(Command::Status { json })
        }
        "logs" => Ok(Command::Logs {
            task: it.next().unwrap_or("").to_string(),
        }),
        "replay" => Ok(Command::Replay {
            task: it.next().ok_or("task")?.to_string(),
        }),
        "pause" => Ok(Command::Pause),
        "resume" => Ok(Command::Resume),
        "freeze" => Ok(Command::Freeze {
            signature: it.next().ok_or("signature")?.to_string(),
        }),
        "self-update" => {
            let version = it.next().ok_or("version")?.to_string();
            let mut signatures = Vec::new();
            for a in it {
                signatures.push(a.to_string());
            }
            Ok(Command::SelfUpdate {
                version,
                signatures,
            })
        }
        _ => Err(format!("unknown {cmd}")),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpsState {
    pub inited: bool,
    pub nodes: Vec<String>,
    pub missions: Vec<String>,
    pub paused: bool,
    pub frozen: bool,
    pub version: u32,
    pub prev_version: u32,
    pub log: Vec<String>,
    pub freeze_key: String,
    pub update_keys: Vec<String>,
}

impl Default for OpsState {
    fn default() -> Self {
        Self {
            inited: false,
            nodes: vec![],
            missions: vec![],
            paused: false,
            frozen: false,
            version: 1,
            prev_version: 1,
            log: vec![],
            freeze_key: "human-freeze".into(),
            update_keys: vec!["k0".into(), "k1".into(), "k2".into()],
        }
    }
}

pub fn sign(key: &str, msg: &str) -> String {
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    h.update(msg.as_bytes());
    format!("{:x}", h.finalize())
}

pub fn verify_sig(key: &str, msg: &str, sig: &str) -> bool {
    sign(key, msg) == sig
}

impl OpsState {
    pub fn dispatch(&mut self, cmd: &Command) -> Result<String, String> {
        if self.frozen && !matches!(cmd, Command::Status { .. } | Command::Logs { .. }) {
            return Err("frozen".into());
        }
        match cmd {
            Command::Init => {
                self.inited = true;
                self.log.push("init".into());
                Ok("inited".into())
            }
            Command::NodeJoin { addr } => {
                if !self.inited {
                    return Err("not inited".into());
                }
                self.nodes.push(addr.clone());
                self.log.push(format!("join:{addr}"));
                Ok(addr.clone())
            }
            Command::MissionAdd { spec } => {
                self.missions.push(spec.clone());
                self.log.push(format!("mission:{spec}"));
                Ok(spec.clone())
            }
            Command::Status { json } => {
                if *json {
                    serde_json::to_string(self).map_err(|e| e.to_string())
                } else {
                    Ok(format!(
                        "nodes={} missions={} paused={} frozen={} ver={}",
                        self.nodes.len(),
                        self.missions.len(),
                        self.paused,
                        self.frozen,
                        self.version
                    ))
                }
            }
            Command::Logs { task } => {
                let lines: Vec<&String> = if task.is_empty() {
                    self.log.iter().collect()
                } else {
                    self.log.iter().filter(|l| l.contains(task.as_str())).collect()
                };
                Ok(lines
                    .into_iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
            Command::Replay { task } => {
                let n = self.log.iter().filter(|l| l.contains(task.as_str())).count();
                Ok(format!("replayed {n}"))
            }
            Command::Pause => {
                self.paused = true;
                Ok("paused".into())
            }
            Command::Resume => {
                self.paused = false;
                Ok("resumed".into())
            }
            Command::Freeze { signature } => {
                if !verify_sig(&self.freeze_key, "freeze", signature) {
                    return Err("bad freeze signature".into());
                }
                self.frozen = true;
                self.log.push("freeze".into());
                Ok("frozen".into())
            }
            Command::SelfUpdate {
                version,
                signatures,
            } => {
                let ver: u32 = version.parse().map_err(|_| "bad version")?;
                let quorum = (self.update_keys.len() * 2 + 2) / 3;
                let mut ok = 0usize;
                let msg = format!("update:{ver}");
                for k in &self.update_keys {
                    if signatures.iter().any(|s| verify_sig(k, &msg, s)) {
                        ok += 1;
                    }
                }
                if ok < quorum {
                    return Err("quorum failed".into());
                }
                self.prev_version = self.version;
                self.version = ver;
                self.log.push(format!("update:{ver}"));
                Ok(format!("updated {ver}"))
            }
        }
    }

    pub fn rollback_update(&mut self) {
        self.version = self.prev_version;
        self.log.push(format!("rollback:{}", self.version));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_freeze() {
        assert_eq!(parse(&["init"]).unwrap(), Command::Init);
        assert_eq!(
            parse(&["node", "join", "10.0.0.1"]).unwrap(),
            Command::NodeJoin {
                addr: "10.0.0.1".into()
            }
        );
        let mut st = OpsState::default();
        st.dispatch(&Command::Init).unwrap();
        st.dispatch(&Command::NodeJoin {
            addr: "n1".into(),
        })
        .unwrap();
        let sig = sign(&st.freeze_key, "freeze");
        st.dispatch(&Command::Freeze { signature: sig }).unwrap();
        assert!(st.frozen);
        assert!(st
            .dispatch(&Command::MissionAdd { spec: "x".into() })
            .is_err());
        assert!(st
            .dispatch(&Command::Freeze {
                signature: "nope".into()
            })
            .is_err());
    }

    #[test]
    fn quorum_update_and_rollback() {
        let mut st = OpsState::default();
        st.dispatch(&Command::Init).unwrap();
        let msg = "update:9";
        let sigs: Vec<String> = st.update_keys.iter().take(2).map(|k| sign(k, msg)).collect();
        st.dispatch(&Command::SelfUpdate {
            version: "9".into(),
            signatures: sigs,
        })
        .unwrap();
        assert_eq!(st.version, 9);
        st.rollback_update();
        assert_eq!(st.version, 1);
        let one = vec![sign(&st.update_keys[0], "update:3")];
        assert!(st
            .dispatch(&Command::SelfUpdate {
                version: "3".into(),
                signatures: one,
            })
            .is_err());
    }
}
