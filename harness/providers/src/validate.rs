//! Transport validation: rate-limit text, empty completions, truncated tools, JSON.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValidateError {
    #[error("rate-limit-as-text")]
    RateLimitText,
    #[error("empty completion")]
    EmptyCompletion,
    #[error("truncated tool call")]
    TruncatedToolCall,
    #[error("malformed json")]
    MalformedJson,
}

pub fn validate_response(body: &str) -> Result<serde_json::Value, ValidateError> {
    let trimmed = body.trim();
    let lower = trimmed.to_ascii_lowercase();
    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
        if lower.contains("rate limit") || lower.contains("too many requests") || lower.contains("429")
        {
            return Err(ValidateError::RateLimitText);
        }
        return Err(ValidateError::MalformedJson);
    }
    let v: serde_json::Value =
        serde_json::from_str(trimmed).map_err(|_| ValidateError::MalformedJson)?;
    if is_empty_completion(&v) {
        return Err(ValidateError::EmptyCompletion);
    }
    if is_truncated_tool(&v) {
        return Err(ValidateError::TruncatedToolCall);
    }
    Ok(v)
}

fn is_empty_completion(v: &serde_json::Value) -> bool {
    if let Some(choices) = v.get("choices").and_then(|c| c.as_array()) {
        if choices.is_empty() {
            return true;
        }
        return choices.iter().all(|c| {
            let msg = c.get("message").unwrap_or(c);
            let content = msg.get("content").and_then(|x| x.as_str()).unwrap_or("");
            let tools = msg
                .get("tool_calls")
                .and_then(|x| x.as_array())
                .map(|a| !a.is_empty())
                .unwrap_or(false);
            content.trim().is_empty() && !tools
        });
    }
    false
}

fn is_truncated_tool(v: &serde_json::Value) -> bool {
    let Some(choices) = v.get("choices").and_then(|c| c.as_array()) else {
        return false;
    };
    for c in choices {
        let msg = c.get("message").unwrap_or(c);
        let Some(tools) = msg.get("tool_calls").and_then(|x| x.as_array()) else {
            continue;
        };
        for t in tools {
            let args = t
                .pointer("/function/arguments")
                .or_else(|| t.get("arguments"))
                .and_then(|a| a.as_str())
                .unwrap_or("");
            if args.is_empty() {
                return true;
            }
            let open = args.chars().filter(|&ch| ch == '{' || ch == '[').count();
            let close = args.chars().filter(|&ch| ch == '}' || ch == ']').count();
            if open != close {
                return true;
            }
            if serde_json::from_str::<serde_json::Value>(args).is_err() && args.contains('{') {
                return true;
            }
        }
        if c.get("finish_reason").and_then(|f| f.as_str()) == Some("length")
            && msg.get("tool_calls").is_some()
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catches_bad_transport() {
        assert_eq!(
            validate_response("Rate limit exceeded, retry later").unwrap_err(),
            ValidateError::RateLimitText
        );
        assert_eq!(
            validate_response("{not json").unwrap_err(),
            ValidateError::MalformedJson
        );
        let empty = r#"{"choices":[{"message":{"content":""}}]}"#;
        assert_eq!(
            validate_response(empty).unwrap_err(),
            ValidateError::EmptyCompletion
        );
        let trunc = r#"{"choices":[{"message":{"tool_calls":[{"function":{"arguments":"{\"a\":"}}]}}]}"#;
        assert_eq!(
            validate_response(trunc).unwrap_err(),
            ValidateError::TruncatedToolCall
        );
        let ok = r#"{"choices":[{"message":{"content":"hi"}}]}"#;
        assert!(validate_response(ok).is_ok());
    }
}
