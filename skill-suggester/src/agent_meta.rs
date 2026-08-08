//! Agent (subagent `.md`) frontmatter + body key-phrase extraction.
//!
//! PSS is pivoting from suggesting SKILLS to suggesting AGENTS (see
//! `suggest_mode.rs`). That only works if agents are indexed as richly as
//! skills are — and until now they were not: the discovery pass
//! (`scripts/pss_discover.py`) keeps `name/path/source/type/description/
//! preview/use_context` for an agent and DISCARDS the rest of its frontmatter.
//! For a skill that loses little, but an agent's frontmatter is exactly where
//! its capability surface is declared:
//!
//! | field | why it matters for matching |
//! |---|---|
//! | `skills` | the skills the harness preloads into the agent — an agent that preloads `playwright` should be reachable from "browser test" even though its own prose never says it |
//! | `tools` / `disallowedTools` | what it can actually DO (`Bash`, `Edit`, `WebFetch`) |
//! | `mcpServers` | which integrations it can reach |
//! | `model` / `effort` | cost tier, for tie-breaking between otherwise equal agents |
//!
//! Extraction lives HERE in Rust rather than being widened in the Python
//! discoverer because enrichment already runs in this binary (`--pass1-batch`),
//! reads are `rayon`-parallel, and keeping one implementation avoids the
//! Python/Rust drift that has bitten this project before.
//!
//! Field set tracks the documented subagent frontmatter (name, description,
//! tools, disallowedTools, model, permissionMode, maxTurns, skills, mcpServers,
//! hooks, memory, background, effort, isolation, color, initialPrompt); only
//! `name` and `description` are required.

use std::collections::HashSet;

use serde::Serialize;

/// Everything PSS indexes about one agent definition.
///
/// `Serialize` is derived so the enrichment pass can emit this straight into
/// the JSONL entry. It is built here rather than as a nested `json!` literal
/// at the call site: the enriched entry is already a large `json!` block, and
/// nesting another one there exceeds `serde_json`'s macro recursion limit.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct AgentMeta {
    // Skipped in the serialized block: both are already top-level fields of
    // the enriched entry, and duplicating them would bloat every agent row.
    #[serde(skip_serializing)]
    pub name: Option<String>,
    #[serde(skip_serializing)]
    pub description: Option<String>,
    /// Allowed tools. Empty means "inherits all" — NOT "has none"; the
    /// distinction matters, so callers must not read empty as a restriction.
    pub tools: Vec<String>,
    pub disallowed_tools: Vec<String>,
    pub model: Option<String>,
    pub permission_mode: Option<String>,
    pub max_turns: Option<u32>,
    /// Skills preloaded into the agent's context at startup — the field that
    /// makes "this agent already knows how to do X" indexable.
    pub skills: Vec<String>,
    pub mcp_servers: Vec<String>,
    pub memory: Option<String>,
    pub background: Option<bool>,
    pub effort: Option<String>,
    pub isolation: Option<String>,
    pub color: Option<String>,
    /// True when the description asks Claude to reach for this agent on its
    /// own ("use proactively", "MUST BE USED"). A strong ranking signal: such
    /// agents are written to be auto-delegated.
    pub proactive: bool,
    /// Key phrases mined from the body (the system prompt).
    pub key_phrases: Vec<String>,
}

impl AgentMeta {
    /// Every token that should be searchable for this agent, deduped and
    /// lowercased: the declared capability surface (skills, tools, MCP
    /// servers) plus the mined body phrases.
    ///
    /// Folding `skills` in here is the whole point of the agent pivot — it is
    /// how a prompt about a *skill's* subject reaches the *agent* that carries
    /// that skill.
    pub fn search_terms(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        let sources = self
            .skills
            .iter()
            .chain(self.tools.iter())
            .chain(self.mcp_servers.iter())
            .chain(self.key_phrases.iter());
        for raw in sources {
            let term = raw.trim().to_ascii_lowercase();
            // Single characters and empties are noise, never a useful match.
            if term.chars().count() < 2 {
                continue;
            }
            if seen.insert(term.clone()) {
                out.push(term);
            }
        }
        out
    }
}

/// Split a frontmatter value that may be either an inline comma list
/// (`tools: Read, Glob, Grep`) or a YAML block sequence, which
/// `crate::parse_frontmatter` flattens into newline-joined `- item` lines.
///
/// Both spellings appear in real agent files, so handling only one silently
/// indexes half the corpus with empty tool/skill lists.
fn split_list(raw: &str) -> Vec<String> {
    raw.lines()
        .flat_map(|line| {
            let line = line.trim().trim_start_matches("- ").trim();
            line.split(',')
        })
        .map(|item| item.trim().trim_matches(['"', '\'']).trim().to_string())
        .filter(|item| !item.is_empty() && item != "[]")
        // A YAML inline list still carries its brackets after the flat parse.
        .map(|item| item.trim_matches(['[', ']']).trim().to_string())
        .filter(|item| !item.is_empty())
        .collect()
}

/// True when the description marks the agent for automatic delegation.
fn detect_proactive(description: &str) -> bool {
    let lowered = description.to_ascii_lowercase();
    [
        "use proactively",
        "proactively use",
        "must be used",
        "use this agent when",
        "automatically",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
}

/// Mine key phrases from an agent's system-prompt body.
///
/// Deliberately cheap and structural rather than linguistic: headings, bolded
/// lead-ins, and backticked identifiers are where an agent states what it
/// handles. Prose sentences are already covered by the description-based
/// keyword pipeline, so mining them again only adds noise.
pub fn key_phrases_from_body(body: &str, limit: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // 1. Markdown headings — an agent's section titles name its duties.
        if let Some(heading) = trimmed.strip_prefix('#') {
            let heading = heading.trim_start_matches('#').trim();
            push_phrase(heading, &mut seen, &mut out);
        }

        // 2. Backticked identifiers — tools, commands, file types, APIs.
        let mut rest = trimmed;
        while let Some(open) = rest.find('`') {
            let after = &rest[open + 1..];
            match after.find('`') {
                Some(close) => {
                    push_phrase(&after[..close], &mut seen, &mut out);
                    rest = &after[close + 1..];
                }
                None => break,
            }
        }

        if out.len() >= limit {
            break;
        }
    }

    out.truncate(limit);
    out
}

fn push_phrase(candidate: &str, seen: &mut HashSet<String>, out: &mut Vec<String>) {
    let phrase = candidate
        .trim()
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_' && c != ' ')
        .trim()
        .to_ascii_lowercase();
    // Cap the length: a whole backticked shell pipeline is not a key phrase,
    // and letting one through would poison the keyword index with a string
    // nothing will ever match.
    if phrase.chars().count() < 2 || phrase.chars().count() > 60 {
        return;
    }
    if seen.insert(phrase.clone()) {
        out.push(phrase);
    }
}

/// Parse a full agent definition (frontmatter + body).
///
/// Reuses `crate::parse_frontmatter` / `crate::extract_md_body` so agents are
/// parsed by exactly the same code as the agent profiler — one parser, one set
/// of quirks, no second dialect to keep in sync.
pub fn parse_agent_definition(content: &str) -> AgentMeta {
    let fm = crate::parse_frontmatter(content);
    let body = crate::extract_md_body(content);

    let get = |key: &str| fm.get(key).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    let list = |key: &str| fm.get(key).map(|v| split_list(v)).unwrap_or_default();

    let description = get("description");

    AgentMeta {
        name: get("name"),
        proactive: description.as_deref().map(detect_proactive).unwrap_or(false),
        description,
        tools: list("tools"),
        disallowed_tools: list("disallowedTools"),
        model: get("model"),
        permission_mode: get("permissionMode"),
        // A non-numeric maxTurns is a malformed file, not a reason to drop the
        // agent from the index — keep None and index everything else.
        max_turns: get("maxTurns").and_then(|v| v.parse::<u32>().ok()),
        skills: list("skills"),
        mcp_servers: list("mcpServers"),
        memory: get("memory"),
        // Goes through the one shared coercion. An inline `match` here is exactly
        // how this drifted from `agent_archetypes::frontmatter_bool` and ended up
        // accepting a different set of spellings than the sibling parser.
        background: get("background").as_deref().and_then(parse_frontmatter_bool),
        effort: get("effort"),
        isolation: get("isolation"),
        color: get("color"),
        key_phrases: key_phrases_from_body(body, 40),
    }
}

/// Coerce one frontmatter scalar to a boolean — the single source of truth for
/// truthiness anywhere PSS reads authored YAML frontmatter.
///
/// Claude Code 2.1.218 widened skill and plugin frontmatter booleans to accept
/// `yes`/`no`/`on`/`off`/`1`/`0` case-insensitively alongside `true`/`false`. All
/// eight spellings are therefore legal input that authors will actually write, and
/// every one of them has to land on the same answer — which is why this lives in
/// one function instead of being re-matched at each call site.
///
/// An unrecognized value is `None`, deliberately **not** `false`: that is a
/// malformed field, and collapsing it to `false` would invent a setting the author
/// never wrote and hide the typo forever.
///
/// Hand-rolled rather than pulling in a YAML crate because callers read files
/// authored by third parties — an unrelated YAML quirk elsewhere in the block must
/// not be able to fail the read of the one field we came for.
pub fn parse_frontmatter_bool(value: &str) -> Option<bool> {
    match value
        .trim()
        .trim_matches(['"', '\''])
        .to_ascii_lowercase()
        .as_str()
    {
        "true" | "yes" | "on" | "1" => Some(true),
        "false" | "no" | "off" | "0" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_frontmatter_bool_accepts_every_spelling_cc_2_1_218_legalized() {
        for truthy in ["true", "yes", "on", "1", "TRUE", "Yes", "ON", "\"yes\"", " on "] {
            assert_eq!(
                parse_frontmatter_bool(truthy),
                Some(true),
                "{truthy:?} should parse as true"
            );
        }
        for falsy in ["false", "no", "off", "0", "FALSE", "No", "OFF", "'off'"] {
            assert_eq!(
                parse_frontmatter_bool(falsy),
                Some(false),
                "{falsy:?} should parse as false"
            );
        }
    }

    #[test]
    fn parse_frontmatter_bool_returns_none_not_false_for_garbage() {
        // None and Some(false) are different answers: None means "the author wrote
        // something we do not understand", and must never be silently downgraded
        // into a real `false` setting.
        for junk in ["", "maybe", "2", "truthy", "y", "n"] {
            assert_eq!(parse_frontmatter_bool(junk), None, "{junk:?} should be None");
        }
    }

    #[test]
    fn background_field_reads_the_widened_spellings() {
        for (raw, want) in [
            ("on", Some(true)),
            ("OFF", Some(false)),
            ("1", Some(true)),
            ("0", Some(false)),
            ("Yes", Some(true)),
            ("banana", None),
        ] {
            let doc = format!("---\nname: a\ndescription: d\nbackground: {raw}\n---\nbody\n");
            assert_eq!(
                parse_agent_definition(&doc).background,
                want,
                "background: {raw} misparsed"
            );
        }
    }

    const SAMPLE: &str = r#"---
name: code-reviewer
description: Use proactively to review code for quality and best practices
tools: Read, Glob, Grep
skills:
  - playwright
  - typescript
model: sonnet
maxTurns: 12
memory: project
isolation: worktree
---

# Review duties

Check `tsc --noEmit` and `eslint` output before approving.

## Security
Look for injection in `Bash` calls.
"#;

    #[test]
    fn parses_every_scalar_field() {
        let m = parse_agent_definition(SAMPLE);
        assert_eq!(m.name.as_deref(), Some("code-reviewer"));
        assert_eq!(m.model.as_deref(), Some("sonnet"));
        assert_eq!(m.max_turns, Some(12));
        assert_eq!(m.memory.as_deref(), Some("project"));
        assert_eq!(m.isolation.as_deref(), Some("worktree"));
    }

    #[test]
    fn parses_inline_comma_list_and_yaml_block_sequence() {
        let m = parse_agent_definition(SAMPLE);
        assert_eq!(m.tools, vec!["Read", "Glob", "Grep"]);
        assert_eq!(m.skills, vec!["playwright", "typescript"]);
    }

    #[test]
    fn detects_the_proactive_delegation_marker() {
        assert!(parse_agent_definition(SAMPLE).proactive);
        let passive = "---\nname: x\ndescription: A quiet helper\n---\nbody";
        assert!(!parse_agent_definition(passive).proactive);
    }

    #[test]
    fn preloaded_skills_become_searchable_terms() {
        // The core of the agent pivot: a prompt about the SKILL's subject must
        // be able to reach the AGENT that preloads it.
        let terms = parse_agent_definition(SAMPLE).search_terms();
        assert!(terms.contains(&"playwright".to_string()));
        assert!(terms.contains(&"typescript".to_string()));
    }

    #[test]
    fn body_key_phrases_capture_headings_and_backticked_identifiers() {
        let m = parse_agent_definition(SAMPLE);
        assert!(m.key_phrases.contains(&"review duties".to_string()));
        assert!(m.key_phrases.contains(&"security".to_string()));
        assert!(m.key_phrases.iter().any(|p| p.contains("eslint")));
    }

    #[test]
    fn search_terms_are_deduped_and_lowercased() {
        let dup = "---\nname: a\ndescription: d\ntools: Bash, bash, BASH\n---\nbody";
        let terms = parse_agent_definition(dup).search_terms();
        assert_eq!(terms.iter().filter(|t| *t == "bash").count(), 1);
    }

    #[test]
    fn missing_optional_fields_are_none_not_a_parse_failure() {
        let minimal = "---\nname: tiny\ndescription: Does one thing\n---\nBody text.";
        let m = parse_agent_definition(minimal);
        assert_eq!(m.name.as_deref(), Some("tiny"));
        assert!(m.tools.is_empty());
        assert_eq!(m.model, None);
        assert_eq!(m.max_turns, None);
    }

    #[test]
    fn malformed_max_turns_does_not_drop_the_rest_of_the_agent() {
        let bad = "---\nname: n\ndescription: d\nmaxTurns: soon\nmodel: opus\n---\nbody";
        let m = parse_agent_definition(bad);
        assert_eq!(m.max_turns, None, "unparseable maxTurns must degrade to None");
        assert_eq!(m.model.as_deref(), Some("opus"), "and must not lose sibling fields");
    }

    #[test]
    fn content_without_frontmatter_yields_empty_meta_not_a_panic() {
        let m = parse_agent_definition("# Just a heading\n\nSome prose.");
        assert_eq!(m.name, None);
        assert!(!m.proactive);
    }

    #[test]
    fn key_phrase_extraction_rejects_oversized_backticked_blobs() {
        let long = "x".repeat(200);
        let body = format!("Run `{long}` now.");
        assert!(key_phrases_from_body(&body, 10).is_empty());
    }
}
