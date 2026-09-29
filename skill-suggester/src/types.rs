//! Shared types: `SuggesterError`, PSS per-skill matcher file structs, Claude
//! Code hook input payloads, agent-profile input/output, and `ProjectContext`
//! (XUD7YUZH modularization step 4). Re-exported at the crate root via
//! `pub(crate) use types::*;` so every existing call site keeps compiling
//! unedited.

use serde::{Deserialize, Serialize};
use std::io;
use std::path::PathBuf;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum SuggesterError {
    #[error("Failed to read stdin: {0}")]
    StdinRead(#[from] io::Error),

    #[error("Failed to parse input JSON: {0}")]
    InputParse(#[from] serde_json::Error),

    #[error("Failed to read skill index from {path}: {source}")]
    IndexRead { path: PathBuf, source: io::Error },

    #[error("Failed to parse skill index: {0}")]
    IndexParse(String),

    #[error("Home directory not found")]
    NoHomeDir,

    #[error("Skill index not found at {0}")]
    IndexNotFound(PathBuf),
}

// ============================================================================
// PSS File Types (per-skill matcher files)
// ============================================================================

/// PSS file format v1.0 - Per-skill matcher file
#[derive(Debug, Deserialize)]
pub struct PssFile {
    /// PSS format version (must be "1.0")
    pub version: String,

    /// Skill identification
    pub skill: PssSkill,

    /// Matcher keywords and patterns
    pub matchers: PssMatchers,

    /// Scoring hints
    #[serde(default)]
    pub scoring: PssScoring,

    /// Generation metadata
    pub metadata: PssMetadata,
}

/// Skill identification in PSS file
#[derive(Debug, Deserialize)]
pub struct PssSkill {
    /// Skill name (kebab-case)
    pub name: String,

    /// Type: skill, agent, or command
    #[serde(rename = "type")]
    pub skill_type: String,

    /// Source: user, project, or plugin
    #[serde(default)]
    pub source: String,

    /// Relative path to SKILL.md
    #[serde(default)]
    pub path: String,
}

/// Matcher keywords and patterns in PSS file
#[derive(Debug, Deserialize)]
pub struct PssMatchers {
    /// Primary trigger keywords (lowercase)
    pub keywords: Vec<String>,

    /// Intent phrases for matching
    #[serde(default)]
    pub intents: Vec<String>,

    /// Regex patterns for complex matching
    #[serde(default)]
    pub patterns: Vec<String>,

    /// Directory names that suggest this skill
    #[serde(default)]
    pub directories: Vec<String>,

    /// Keywords that should NOT trigger this skill
    #[serde(default)]
    pub negative_keywords: Vec<String>,
}

/// Scoring hints in PSS file
#[derive(Debug, Deserialize, Default)]
pub struct PssScoring {
    /// Element importance tier: primary, secondary, specialized
    #[serde(default)]
    pub tier: String,

    /// Skill category for grouping
    #[serde(default)]
    pub category: String,

    /// Score boost (-10 to +10)
    #[serde(default)]
    pub boost: i32,
}

/// Generation metadata in PSS file
#[derive(Debug, Deserialize)]
pub struct PssMetadata {
    /// How the matchers were generated: ai, manual, hybrid
    pub generated_by: String,

    /// ISO-8601 timestamp of generation
    pub generated_at: String,

    /// Version of the generator tool
    #[serde(default)]
    pub generator_version: String,

    /// SHA-256 hash of SKILL.md for staleness detection
    #[serde(default)]
    pub skill_hash: String,
}

// ============================================================================
// Input Types (from Claude Code hook)
// ============================================================================

/// Input payload from Claude Code UserPromptSubmit hook.
/// Field names match CC's snake_case hook input schema (hooks.md "Common input fields").
///
/// IMPORTANT: DO NOT add `#[serde(rename_all = "camelCase")]` to this struct.
/// CC sends snake_case in hook INPUT (`transcript_path`, `session_id`,
/// `permission_mode`) but expects camelCase in hook OUTPUT
/// (`hookSpecificOutput`, `hookEventName`, `additionalContext`). This
/// asymmetry is intentional and required by the CC hook protocol:
///   - `HookInput` below: snake_case (default serde naming)
///   - `HookOutput` at line ~2979: camelCase (via `rename_all`)
///   - `HookSpecificOutput` at line ~2987: camelCase (via `rename_all`)
/// A "consistency cleanup" that unified these would break the hook boundary.
#[derive(Debug, Deserialize)]
pub struct HookInput {
    /// The user's prompt text
    pub prompt: String,

    /// Current working directory
    #[serde(default)]
    pub cwd: String,

    /// Session ID
    #[serde(default)]
    pub session_id: String,

    /// Path to conversation transcript
    #[serde(default)]
    pub transcript_path: String,

    /// Permission mode (ask, auto, etc.)
    #[serde(default)]
    pub permission_mode: String,

    // Context metadata detected by Python hook

    /// Detected platforms from project context (e.g., ["ios", "macos"])
    #[serde(default)]
    pub context_platforms: Vec<String>,

    /// Detected frameworks from project context (e.g., ["swiftui", "react"])
    #[serde(default)]
    pub context_frameworks: Vec<String>,

    /// Detected languages from project context (e.g., ["swift", "rust"])
    #[serde(default)]
    pub context_languages: Vec<String>,

    /// Detected domains from conversation context (e.g., ["writing", "graphics"])
    #[serde(default)]
    pub context_domains: Vec<String>,

    /// Detected tools from conversation context (e.g., ["ffmpeg", "pandoc"])
    #[serde(default)]
    pub context_tools: Vec<String>,

    /// Detected file types from conversation context (e.g., ["pdf", "xlsx"])
    #[serde(default)]
    pub context_file_types: Vec<String>,
}

/// Input for --agent-profile mode: describes an agent to profile against the skill index.
/// The profiler agent writes this JSON file, then invokes the binary with --agent-profile <path>.
#[derive(Debug, Deserialize)]
pub struct AgentProfileInput {
    /// Agent name (e.g., "security-auditor")
    pub name: String,

    /// Full agent description — what the agent does, its specialization
    #[serde(default)]
    pub description: String,

    /// Agent's primary role (e.g., "developer", "tester", "reviewer")
    #[serde(default)]
    pub role: String,

    /// List of duties/responsibilities extracted from the agent definition
    #[serde(default)]
    pub duties: Vec<String>,

    /// Tools the agent uses (e.g., ["grep", "semgrep", "bandit"])
    #[serde(default)]
    pub tools: Vec<String>,

    /// Domain tags (e.g., ["security", "testing"])
    #[serde(default)]
    pub domains: Vec<String>,

    /// Condensed summary of all requirements/design documents
    #[serde(default)]
    pub requirements_summary: String,

    /// Current working directory for project context scanning
    #[serde(default)]
    pub cwd: String,

    /// Auto-skills declared in frontmatter (must be pinned to primary tier)
    #[serde(default)]
    pub auto_skills: Vec<String>,

    /// Whether agent is a non-coding orchestrator (detected from role/description)
    #[serde(default)]
    pub is_orchestrator: bool,

    /// Absolute path to the agent's .md definition file (set by resolve_agent_input)
    #[serde(skip)]
    pub source_path: String,
}

/// Output for --agent mode: tiered skill recommendations written as .agent.toml
#[derive(Debug, Serialize)]
pub struct AgentProfileOutput {
    /// Agent name
    pub agent: String,

    /// Tiered skill recommendations
    pub skills: AgentProfileSkills,

    /// Complementary agents found via scoring and co_usage data
    pub complementary_agents: Vec<AgentProfileCandidate>,

    /// Recommended slash commands for this agent
    pub commands: Vec<AgentProfileCandidate>,

    /// Rules that should be active when this agent runs
    pub rules: Vec<AgentProfileCandidate>,

    /// MCP servers that enhance this agent's capabilities
    pub mcp: Vec<AgentProfileCandidate>,

    /// LSP servers relevant to this agent
    pub lsp: Vec<AgentProfileCandidate>,

    /// Output styles relevant to this agent
    pub output_styles: Vec<AgentProfileCandidate>,
}

/// Tiered skill lists for agent profile output
#[derive(Debug, Serialize)]
pub struct AgentProfileSkills {
    /// Core skills (score >= 60% of max)
    pub primary: Vec<AgentProfileCandidate>,

    /// Useful skills (score 30-59% of max)
    pub secondary: Vec<AgentProfileCandidate>,

    /// Niche skills (score 15-29% of max)
    pub specialized: Vec<AgentProfileCandidate>,
}

/// A single skill candidate in the agent profile output
#[derive(Debug, Serialize)]
pub struct AgentProfileCandidate {
    pub name: String,
    pub path: String,
    pub score: f64,
    pub confidence: String,
    pub evidence: Vec<String>,
    pub description: String,
}

/// Typed candidate tuple for skill entries: (name, score, evidence, path, confidence, description, entry_type)
pub type SkillCandidate = (String, i32, Vec<String>, String, String, String, String);

/// Typed candidate tuple for non-skill entries: (name, score, evidence, path, confidence, description)
pub type TypedCandidate = (String, i32, Vec<String>, String, String, String);

/// Project context for filtering skills by platform/framework/language/domain/tools/file-types
#[derive(Debug, Clone, Default)]
pub struct ProjectContext {
    /// Detected platforms from project (e.g., ["ios", "macos"])
    pub platforms: Vec<String>,
    /// Detected frameworks from project (e.g., ["swiftui", "react"])
    pub frameworks: Vec<String>,
    /// Detected languages from project (e.g., ["swift", "rust"])
    pub languages: Vec<String>,
    /// Detected domains from conversation (e.g., ["writing", "graphics", "media"])
    pub domains: Vec<String>,
    /// Detected tools from conversation (e.g., ["ffmpeg", "pandoc"])
    pub tools: Vec<String>,
    /// Detected file types from conversation (e.g., ["pdf", "xlsx"])
    pub file_types: Vec<String>,
}

impl ProjectContext {
    /// Create context from HookInput fields
    pub fn from_hook_input(input: &HookInput) -> Self {
        ProjectContext {
            platforms: input.context_platforms.clone(),
            frameworks: input.context_frameworks.clone(),
            languages: input.context_languages.clone(),
            domains: input.context_domains.clone(),
            tools: input.context_tools.clone(),
            file_types: input.context_file_types.clone(),
        }
    }

    /// Merge Rust project scan results into this context, adding items that
    /// are not already present (case-insensitive dedup). This ensures the
    /// scoring boosts in match_skill() benefit from fresh on-disk project data,
    /// not just the hook-provided metadata.
    pub fn merge_scan(&mut self, scan: &crate::ProjectScanResult) {
        for item in &scan.languages {
            if !self.languages.iter().any(|l| l.eq_ignore_ascii_case(item)) {
                self.languages.push(item.clone());
            }
        }
        for item in &scan.frameworks {
            if !self.frameworks.iter().any(|f| f.eq_ignore_ascii_case(item)) {
                self.frameworks.push(item.clone());
            }
        }
        for item in &scan.platforms {
            if !self.platforms.iter().any(|p| p.eq_ignore_ascii_case(item)) {
                self.platforms.push(item.clone());
            }
        }
        for item in &scan.tools {
            if !self.tools.iter().any(|t| t.eq_ignore_ascii_case(item)) {
                self.tools.push(item.clone());
            }
        }
        for item in &scan.file_types {
            if !self.file_types.iter().any(|ft| ft.eq_ignore_ascii_case(item)) {
                self.file_types.push(item.clone());
            }
        }
    }

    /// Check if context is empty (no filtering)
    pub fn is_empty(&self) -> bool {
        self.platforms.is_empty()
            && self.frameworks.is_empty()
            && self.languages.is_empty()
            && self.domains.is_empty()
            && self.tools.is_empty()
            && self.file_types.is_empty()
    }

    /// Calculate context match score for a skill entry
    /// Returns (score_boost, should_filter_out)
    /// - score_boost: +10 for platform match, +8 for framework match, +6 for language match
    /// - should_filter_out: true if skill is platform-specific but context doesn't match
    pub fn match_skill(&self, skill: &crate::SkillEntry) -> (i32, bool) {
        let mut boost = 0i32;
        let mut should_filter = false;

        // Platform matching
        if !skill.platforms.is_empty() && !skill.platforms.contains(&"universal".to_string()) {
            // Skill is platform-specific
            if !self.platforms.is_empty() {
                // We have context - check for match
                let has_platform_match = skill.platforms.iter().any(|p| {
                    self.platforms.iter().any(|cp| cp.to_lowercase() == p.to_lowercase())
                });
                if has_platform_match {
                    boost += 10; // Strong boost for matching platform
                } else {
                    should_filter = true; // Filter out non-matching platform-specific skills
                }
            }
            // If no context, don't filter but don't boost either
        }

        // Framework matching (less strict - don't filter, just boost)
        if !skill.frameworks.is_empty() && !self.frameworks.is_empty() {
            let has_framework_match = skill.frameworks.iter().any(|f| {
                self.frameworks.iter().any(|cf| cf.to_lowercase() == f.to_lowercase())
            });
            if has_framework_match {
                boost += 8; // Good boost for matching framework
            }
        }

        // Language matching (less strict - don't filter, just boost)
        if !skill.languages.is_empty()
            && !skill.languages.contains(&"any".to_string())
            && !self.languages.is_empty()
        {
            let has_lang_match = skill.languages.iter().any(|l| {
                self.languages.iter().any(|cl| cl.to_lowercase() == l.to_lowercase())
            });
            if has_lang_match {
                boost += 6; // Moderate boost for matching language
            }
        }

        // Domain matching (boost for matching domain expertise)
        if !skill.domains.is_empty() && !self.domains.is_empty() {
            let has_domain_match = skill.domains.iter().any(|d| {
                self.domains.iter().any(|cd| cd.to_lowercase() == d.to_lowercase())
            });
            if has_domain_match {
                boost += 8; // Good boost for matching domain
            }
        }

        // Tool matching (strong boost for matching specific tools)
        if !skill.tools.is_empty() && !self.tools.is_empty() {
            let has_tool_match = skill.tools.iter().any(|t| {
                self.tools.iter().any(|ct| ct.to_lowercase() == t.to_lowercase())
            });
            if has_tool_match {
                boost += 12; // Very strong boost for matching tools (specific expertise)
            }
        }

        // File type matching (boost for matching file formats)
        if !skill.file_types.is_empty() && !self.file_types.is_empty() {
            let has_file_type_match = skill.file_types.iter().any(|ft| {
                self.file_types.iter().any(|cft| cft.to_lowercase() == ft.to_lowercase())
            });
            if has_file_type_match {
                boost += 10; // Strong boost for matching file types
            }
        }

        (boost, should_filter)
    }
}
