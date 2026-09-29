// Hook/agent-profile pipeline: frontmatter + body extraction from agent .md
// definitions, agent profiling (`run_agent_profile`, incl. multi-query domain
// detection and candidate synthesis), the .agent.toml writer, and runtime
// VERSION lookup. Moved out of main.rs verbatim (XUD7YUZH modularization
// step 13); re-exported at the crate root via `pub(crate) use`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use cozo::{DataValue, ScriptMutability};
use rayon::prelude::*;
use tracing::{error, info, warn};

use crate::{
    AgentProfileCandidate, AgentProfileInput, AgentProfileOutput, AgentProfileSkills, Cli,
    DetectedDomains, MatchedSkill, ProjectContext, SkillCandidate, SkillEntry, SuggesterError,
    TypedCandidate, correct_typos, dedup_vec, detect_domains_from_prompt_with_context,
    dv_to_string, expand_synonyms, find_matches, get_db_path, get_index_path, get_registry_path,
    infer_domains_from_text, load_domain_registry, load_domain_registry_from_db, load_index,
    load_index_from_db, open_db, resolve_name_or_id, scan_project_context,
};
// ============================================================================
// Main Entry Point
// ============================================================================

/// Parse a YAML frontmatter block from a markdown file.
/// Returns a HashMap of key-value pairs from the frontmatter.
pub(crate) fn parse_frontmatter(content: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    // Strip a UTF-8 BOM first. Files authored on Windows start with U+FEFF, so
    // the starts_with("---") test below said "no frontmatter" and the agent was
    // indexed with NO name, description, tools, skills or mcpServers at all —
    // total capability loss, silently.
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    // Frontmatter is between first "---" and second "---"
    if !content.starts_with("---") {
        return map;
    }
    let after_first = &content[3..];
    if let Some(end) = after_first.find("\n---") {
        let fm_block = &after_first[..end];
        let mut current_key: Option<String> = None;
        let mut current_val = String::new();
        // Indentation of the `key: |` line whose block scalar we are inside.
        // `None` = not in a block scalar. YAML ends a block scalar at the first
        // line indented no further than its key, so this is what tells a body
        // line apart from the next key — including a body line that happens to
        // contain a colon.
        let mut block_indent: Option<usize> = None;

        for line in fm_block.lines() {
            let trimmed = line.trim();
            let indent = line.len() - line.trim_start().len();

            // Inside a block scalar, EVERY more-indented line is body text —
            // blanks and `#` included, since neither is special there.
            if let Some(key_indent) = block_indent {
                if trimmed.is_empty() || indent > key_indent {
                    if !trimmed.is_empty() {
                        if !current_val.is_empty() {
                            current_val.push(' ');
                        }
                        current_val.push_str(trimmed);
                    }
                    continue;
                }
                block_indent = None; // Dedented — the block ended here.
            }

            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Check if this is a YAML list item (continuation of previous key)
            if trimmed.starts_with("- ") || trimmed.starts_with("-\t") {
                if current_key.is_some() {
                    // Append list item to current value
                    if !current_val.is_empty() {
                        current_val.push('\n');
                    }
                    current_val.push_str(trimmed);
                }
                continue;
            }

            // New key:value pair — flush previous key if any
            if let Some(colon_pos) = trimmed.find(':') {
                // Flush previous key
                if let Some(ref key) = current_key {
                    let val = current_val.trim().to_string();
                    let val = if val.len() >= 2
                        && ((val.starts_with('"') && val.ends_with('"'))
                            || (val.starts_with('\'') && val.ends_with('\'')))
                    {
                        val[1..val.len() - 1].to_string()
                    } else {
                        val
                    };
                    map.insert(key.clone(), val);
                }

                let key = trimmed[..colon_pos].trim().to_string();
                let val = trimmed[colon_pos + 1..].trim().to_string();
                // `description: |` / `>` (with any chomping/indent indicator):
                // the value is on the FOLLOWING lines, not here. Arm the block
                // reader and start from an empty value — otherwise the literal
                // indicator became the value and the real text was dropped.
                let is_block_indicator = matches!(val.chars().next(), Some('|') | Some('>'))
                    && val[1..].chars().all(|c| c == '-' || c == '+' || c.is_ascii_digit());
                current_key = Some(key);
                if is_block_indicator {
                    block_indent = Some(indent);
                    current_val = String::new();
                } else {
                    current_val = val;
                }
            }
        }

        // Flush last key
        if let Some(key) = current_key {
            let val = current_val.trim().to_string();
            let val = if val.len() >= 2
                && ((val.starts_with('"') && val.ends_with('"'))
                    || (val.starts_with('\'') && val.ends_with('\'')))
            {
                val[1..val.len() - 1].to_string()
            } else {
                val
            };
            map.insert(key, val);
        }
    }
    map
}

/// Extract `paths:` YAML list from frontmatter (used by rule-type entries).
/// Returns empty Vec if the field is absent or malformed.
/// Handles both inline `paths: [a, b]` and block `- a\n- b` forms.
/// Quotes around entries are stripped.
pub(crate) fn extract_rule_paths(frontmatter: &HashMap<String, String>) -> Vec<String> {
    let raw = match frontmatter.get("paths") {
        Some(v) if !v.is_empty() => v,
        _ => return Vec::new(),
    };
    let trimmed = raw.trim();
    // Inline form: "[*.py, src/**]"
    if trimmed.starts_with('[') {
        return trimmed
            .trim_matches(|c: char| c == '[' || c == ']')
            .split(',')
            .map(|s| s.trim().trim_matches(|c: char| c == '"' || c == '\'').to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }
    // Block form: "- *.py\n- src/**" (parse_frontmatter preserves list items with the dash prefix)
    trimmed
        .lines()
        .map(|l| {
            l.trim()
                .trim_start_matches('-')
                .trim()
                .trim_matches(|c: char| c == '"' || c == '\'')
                .to_string()
        })
        .filter(|l| !l.is_empty())
        .collect()
}

/// Extract the markdown body (everything after frontmatter).
pub(crate) fn extract_md_body(content: &str) -> &str {
    // Strip a UTF-8 BOM first, same as parse_frontmatter. CC 2.1.240 made
    // Claude Code honor BOM'd .md elements; without this the bare
    // starts_with("---") below sees the BOM byte instead of "---" and returns
    // the WHOLE file (frontmatter included) as "body", poisoning keyword
    // extraction with YAML. parse_frontmatter rebinds its own local when it
    // strips the BOM, so its caller's original `content` here still carries
    // it — this function must strip independently, not rely on the caller.
    // trim_start_matches (not strip_prefix) removes ALL leading BOMs, matching
    // the Python side's `lstrip(chr(0xFEFF))` — a single-strip/repeat-strip
    // mismatch is exactly the kind of divergence documented at
    // agent_meta.rs:238.
    let content = content.trim_start_matches('\u{feff}');
    if content.starts_with("---") {
        let after_first = &content[3..];
        if let Some(end) = after_first.find("\n---") {
            let rest = &after_first[end + 4..];
            // Skip the newline after closing ---
            rest.trim_start_matches('\n')
        } else {
            content
        }
    } else {
        content
    }
}

/// Extract duties from markdown body: bullet items under headings containing
/// "dut", "responsibilit", "step", "task", "workflow", "capabilit".
pub(crate) fn extract_duties_from_md(body: &str) -> Vec<String> {
    let mut duties = Vec::new();
    let mut in_duty_section = false;

    for line in body.lines() {
        let trimmed = line.trim();
        // Check for headings
        if trimmed.starts_with('#') {
            let heading_lower = trimmed.to_lowercase();
            in_duty_section = heading_lower.contains("dut")
                || heading_lower.contains("responsibilit")
                || heading_lower.contains("step")
                || heading_lower.contains("task")
                || heading_lower.contains("workflow")
                || heading_lower.contains("capabilit")
                || heading_lower.contains("what")
                || heading_lower.contains("how");
            continue;
        }
        // Collect bullet items in duty sections
        if in_duty_section && (trimmed.starts_with("- ") || trimmed.starts_with("* ")) {
            let item = trimmed[2..].trim();
            if !item.is_empty() && item.len() > 5 {
                duties.push(item.to_string());
            }
        }
        // Stop at next heading or code block
        if in_duty_section && trimmed.starts_with("```") {
            in_duty_section = false;
        }
    }
    duties
}

/// Extract tools mentioned in markdown body by scanning for known tool names.
pub(crate) fn extract_tools_from_md(body: &str) -> Vec<String> {
    let known_tools = [
        "Bash", "Read", "Write", "Edit", "Grep", "Glob", "Agent",
        "WebFetch", "WebSearch", "NotebookEdit",
        "git", "gh", "docker", "terraform", "kubectl",
        "npm", "bun", "cargo", "pip", "uv",
        "grep", "rg", "find", "sed", "awk",
        "semgrep", "bandit", "eslint", "ruff", "mypy", "pyright",
    ];
    let body_lower = body.to_lowercase();
    let mut found = Vec::new();
    for tool in &known_tools {
        if body_lower.contains(&tool.to_lowercase()) {
            found.push(tool.to_string());
        }
    }
    found.sort();
    found.dedup();
    found
}

/// Infer role from agent description and body text.
pub(crate) fn infer_role(description: &str, body: &str) -> String {
    let text = format!("{} {}", description, body).to_lowercase();
    let role_keywords = [
        ("debug", "debugger"),
        ("review", "reviewer"),
        ("test", "tester"),
        ("secur", "security-analyst"),
        ("deploy", "deployer"),
        ("architect", "architect"),
        ("develop", "developer"),
        ("build", "developer"),
        ("research", "researcher"),
        ("analyz", "analyst"),
        ("document", "documenter"),
        ("monitor", "operator"),
        ("audit", "auditor"),
        ("refactor", "developer"),
    ];
    for (keyword, role) in &role_keywords {
        if text.contains(keyword) {
            return role.to_string();
        }
    }
    "developer".to_string()
}

/// Infer domains from agent description and body text.
pub(crate) fn infer_domains(description: &str, body: &str) -> Vec<String> {
    let text = format!("{} {}", description, body).to_lowercase();
    let domain_keywords = [
        ("debug", "debugging"),
        ("secur", "security"),
        ("test", "testing"),
        ("deploy", "devops"),
        ("docker", "devops"),
        ("kubernetes", "devops"),
        ("frontend", "web-frontend"),
        ("react", "web-frontend"),
        ("backend", "web-backend"),
        ("api", "web-backend"),
        ("database", "data"),
        ("machine learn", "ai-ml"),
        ("mobile", "mobile"),
        ("ios", "mobile"),
        ("flutter", "mobile"),
        ("performance", "performance"),
        ("infrastructure", "infrastructure"),
    ];
    let mut domains = Vec::new();
    for (keyword, domain) in &domain_keywords {
        if text.contains(keyword) && !domains.contains(&domain.to_string()) {
            domains.push(domain.to_string());
        }
    }
    if domains.is_empty() {
        domains.push("general".to_string());
    }
    domains
}

/// Parse an agent .md file into an AgentProfileInput.
/// Extracts name, description from frontmatter; duties, tools, role, domains from body.
pub(crate) fn parse_agent_md(path: &str) -> Result<AgentProfileInput, SuggesterError> {
    let content = fs::read_to_string(path)
        .map_err(|e| SuggesterError::IndexRead { path: PathBuf::from(path), source: e })?;

    let fm = parse_frontmatter(&content);
    let body = extract_md_body(&content);

    // Name: from frontmatter, or filename without extension
    let name = fm.get("name").cloned().unwrap_or_else(|| {
        Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    });

    // Description: from frontmatter, or first non-empty paragraph in body
    let description = fm.get("description").cloned().unwrap_or_else(|| {
        body.lines()
            .find(|l| {
                let t = l.trim();
                !t.is_empty() && !t.starts_with('#') && !t.starts_with("```")
            })
            .unwrap_or("")
            .trim()
            .to_string()
    });

    let duties = extract_duties_from_md(body);
    let tools = extract_tools_from_md(body);
    let role = infer_role(&description, body);
    let domains = infer_domains(&description, body);

    // Extract auto_skills from frontmatter YAML list
    let auto_skills: Vec<String> = fm.get("auto_skills")
        .map(|s| {
            // Parse YAML list: "- skill1\n  - skill2" or "[skill1, skill2]"
            s.lines()
                .map(|l| l.trim().trim_start_matches('-').trim().to_string())
                .filter(|l| !l.is_empty() && !l.starts_with('['))
                .collect()
        })
        .unwrap_or_default();

    // Detect non-coding orchestrator from role/description
    let desc_lower = format!("{} {} {}", name, description, role).to_lowercase();
    let is_orchestrator = desc_lower.contains("orchestrat")
        || desc_lower.contains("coordinator")
        || desc_lower.contains("manager")
        || desc_lower.contains("gatekeeper")
        || desc_lower.contains("route to sub-agent")
        || desc_lower.contains("delegate")
        || (fm.get("type").map(|t| t.to_lowercase()) == Some("orchestrator".to_string()));

    // Store the absolute path so the TOML output can reference it
    let abs_path = fs::canonicalize(path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string());

    Ok(AgentProfileInput {
        name,
        description,
        role,
        duties,
        tools,
        domains,
        requirements_summary: String::new(),
        cwd: String::new(),
        auto_skills,
        is_orchestrator,
        source_path: abs_path,
    })
}

/// Resolve an agent reference to an AgentProfileInput.
/// Accepts:
///   1. Path to .md agent definition file
///   2. Agent name (resolved via CozoDB/index to find the .md path, then parsed)
pub(crate) fn resolve_agent_input(
    agent_ref: &str,
    cli: &Cli,
) -> Result<AgentProfileInput, SuggesterError> {
    let path = Path::new(agent_ref);

    // Case 1: .md file path (absolute or relative)
    if path.exists() && path.is_file() {
        return parse_agent_md(agent_ref);
    }

    // Case 2: Agent name — resolve via index to find the .md path
    // Try CozoDB first, then JSON index
    let db = get_db_path(cli.index.as_deref()).and_then(|p| open_db(&p).ok());

    if let Some(ref db) = db {
        // Try resolving by name via CozoDB
        let resolved = resolve_name_or_id(db, agent_ref);
        if let Ok(name) = resolved {
            // Get path from skills table
            let query = "?[path] := *skills{name, path}, name = $name";
            let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
            params.insert("name".into(), DataValue::Str(name.clone().into()));
            if let Ok(result) = db.run_script(query, params, ScriptMutability::Immutable) {
                if let Some(row) = result.rows.first() {
                    let md_path = dv_to_string(&row[0]);
                    if !md_path.is_empty() && Path::new(&md_path).exists() {
                        return parse_agent_md(&md_path);
                    }
                }
            }
        }
    }

    // Fallback: try loading JSON index and searching by name
    if let Ok(index_path) = get_index_path(cli.index.as_deref()) {
        if let Ok(index) = load_index(&index_path) {
            for (_id, entry) in &index.skills {
                if entry.name == agent_ref && entry.skill_type == "agent" {
                    if Path::new(&entry.path).exists() {
                        return parse_agent_md(&entry.path);
                    }
                }
            }
        }
    }

    Err(SuggesterError::IndexParse(format!(
        "Cannot resolve '{}': not a valid .md agent file or known agent name in the index",
        agent_ref
    )))
}

/// Derive the TOML `source` field from the agent's file path.
/// Returns "plugin:<name>" if the path is under a plugins directory, otherwise "path".
pub(crate) fn derive_agent_source(path: &str) -> String {
    // Plugin paths look like: ~/.claude/plugins/cache/<hash>/<plugin-name>/1.0/agents/foo.md
    if let Some(idx) = path.find("/plugins/") {
        let after = &path[idx + "/plugins/".len()..];
        // Skip "cache/<hash>/" if present
        let after = if after.starts_with("cache/") {
            // cache/<hash>/<plugin-name>/...
            after.splitn(4, '/').nth(2).unwrap_or(after)
        } else {
            after
        };
        // Take the first path component as the plugin name
        if let Some(name) = after.split('/').next() {
            if !name.is_empty() {
                return format!("plugin:{}", name);
            }
        }
    }
    "path".to_string()
}

/// Escape a TOML string value: wrap in quotes, escape backslashes and internal quotes.
pub(crate) fn toml_escape(s: &str) -> String {
    let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{}\"", escaped)
}

/// Format a Vec<String> as a TOML array literal: ["a", "b", "c"]
pub(crate) fn toml_string_array(items: &[String]) -> String {
    if items.is_empty() {
        return "[]".to_string();
    }
    let parts: Vec<String> = items.iter().map(|s| toml_escape(s)).collect();
    format!("[{}]", parts.join(", "))
}

/// Format a Vec<AgentProfileCandidate> as a TOML array of their names.
pub(crate) fn candidates_to_names(candidates: &[AgentProfileCandidate]) -> Vec<String> {
    candidates.iter().map(|c| c.name.clone()).collect()
}

/// Write an .agent.toml file from the profiling output.
/// Returns the absolute path of the written file.
pub(crate) fn write_agent_toml(
    output: &AgentProfileOutput,
    source_path: &str,
) -> Result<String, SuggesterError> {
    let source = derive_agent_source(source_path);

    let mut toml = String::with_capacity(2048);

    // [agent] section
    toml.push_str("[agent]\n");
    toml.push_str(&format!("name = {}\n", toml_escape(&output.agent)));
    toml.push_str(&format!("source = {}\n", toml_escape(&source)));
    toml.push_str(&format!("path = {}\n", toml_escape(source_path)));
    toml.push('\n');

    // [requirements] section — empty for now (populated by AI post-filter in the skill)
    toml.push_str("[requirements]\n");
    toml.push_str("files = []\n");
    toml.push_str("project_type = \"\"\n");
    toml.push_str("tech_stack = []\n");
    toml.push('\n');

    // [description] section — empty for now (populated by AI post-filter)
    toml.push_str("[description]\n");
    toml.push_str("text = \"\"\n");
    toml.push('\n');

    // [skills] section — primary/secondary/specialized are name arrays
    toml.push_str("[skills]\n");
    let primary_names = candidates_to_names(&output.skills.primary);
    let secondary_names = candidates_to_names(&output.skills.secondary);
    let specialized_names = candidates_to_names(&output.skills.specialized);
    toml.push_str(&format!("primary = {}\n", toml_string_array(&primary_names)));
    toml.push_str(&format!("secondary = {}\n", toml_string_array(&secondary_names)));
    toml.push_str(&format!("specialized = {}\n", toml_string_array(&specialized_names)));
    toml.push('\n');

    // [subagents] section — agents used as sub-agents, not main agents
    toml.push_str("[subagents]\n");
    let agent_names = candidates_to_names(&output.complementary_agents);
    toml.push_str(&format!("recommended = {}\n", toml_string_array(&agent_names)));
    toml.push('\n');

    // [commands] section
    toml.push_str("[commands]\n");
    let command_names = candidates_to_names(&output.commands);
    toml.push_str(&format!("recommended = {}\n", toml_string_array(&command_names)));
    toml.push('\n');

    // [rules] section
    toml.push_str("[rules]\n");
    let rule_names = candidates_to_names(&output.rules);
    toml.push_str(&format!("recommended = {}\n", toml_string_array(&rule_names)));
    toml.push('\n');

    // [mcp] section
    toml.push_str("[mcp]\n");
    let mcp_names = candidates_to_names(&output.mcp);
    toml.push_str(&format!("recommended = {}\n", toml_string_array(&mcp_names)));
    toml.push('\n');

    // [hooks] section — empty (not scored by the profiler)
    toml.push_str("[hooks]\n");
    toml.push_str("recommended = []\n");
    toml.push('\n');

    // [lsp] section
    toml.push_str("[lsp]\n");
    let lsp_names = candidates_to_names(&output.lsp);
    toml.push_str(&format!("recommended = {}\n", toml_string_array(&lsp_names)));
    toml.push('\n');

    // [output_styles] section
    toml.push_str("[output_styles]\n");
    let output_style_names = candidates_to_names(&output.output_styles);
    toml.push_str(&format!("recommended = {}\n", toml_string_array(&output_style_names)));
    toml.push('\n');

    // [dependencies] section — empty defaults for all 11 sub-keys
    toml.push_str("[dependencies]\n");
    toml.push_str("plugins = []\n");
    toml.push_str("skills = []\n");
    toml.push_str("rules = []\n");
    toml.push_str("agents = []\n");
    toml.push_str("commands = []\n");
    toml.push_str("hooks = []\n");
    toml.push_str("mcp_servers = []\n");
    toml.push_str("lsp_servers = []\n");
    toml.push_str("output_styles = []\n");
    toml.push_str("tools = []\n");
    toml.push_str("frameworks = []\n");

    // Write to <agent-name>.agent.toml in current directory
    let filename = format!("{}.agent.toml", output.agent);
    let out_path = std::env::current_dir()
        .map_err(|e| SuggesterError::IndexParse(format!("Cannot get cwd: {}", e)))?
        .join(&filename);

    fs::write(&out_path, &toml)
        .map_err(|e| SuggesterError::IndexParse(format!("Cannot write {}: {}", out_path.display(), e)))?;

    let abs = out_path.to_string_lossy().to_string();
    Ok(abs)
}

/// Run in agent mode: score all skills against an agent's .md definition,
/// synthesizing multiple queries from the agent's description, duties, and requirements.
/// Writes <agent-name>.agent.toml to the current directory and prints the path to stdout.
pub(crate) fn run_agent_profile(cli: &Cli, profile_path: &str) -> Result<(), SuggesterError> {
    // Resolve the agent reference: accepts name, .md path, or .json descriptor
    let profile = resolve_agent_input(profile_path, cli)?;

    info!("Agent profile mode: analyzing agent '{}'", profile.name);

    // Open CozoDB first — if available, load index from DB instead of JSON
    let db = get_db_path(cli.index.as_deref()).and_then(|p| {
        match open_db(&p) {
            Ok(db) => {
                info!("Agent profile: using CozoDB at {:?}", p);
                Some(db)
            }
            Err(e) => {
                warn!("Agent profile: CozoDB open failed: {}, using JSON", e);
                None
            }
        }
    });

    // Load skill index: try CozoDB first, then JSON file fallback
    let index = if let Some(ref db) = db {
        match load_index_from_db(db) {
            Ok(idx) => {
                info!("Loaded {} skills from CozoDB", idx.skills.len());
                idx
            }
            Err(e) => {
                warn!("CozoDB index load failed: {}, falling back to JSON", e);
                let index_path = get_index_path(cli.index.as_deref())?;
                load_index(&index_path)?
            }
        }
    } else {
        let index_path = get_index_path(cli.index.as_deref())?;
        match load_index(&index_path) {
            Ok(idx) => idx,
            Err(SuggesterError::IndexNotFound(path)) => {
                error!("Skill index not found at {:?}", path);
                return Err(SuggesterError::IndexNotFound(path));
            }
            Err(e) => return Err(e),
        }
    };
    info!("Loaded {} skills from index", index.skills.len());

    // Load domain registry: try CozoDB first, then JSON file fallback
    let registry = if let Some(ref db) = db {
        match load_domain_registry_from_db(db) {
            Ok(Some(reg)) => Some(reg),
            _ => {
                match get_registry_path(cli.registry.as_deref()) {
                    Some(reg_path) => load_domain_registry(&reg_path).ok().flatten(),
                    None => None,
                }
            }
        }
    } else {
        match get_registry_path(cli.registry.as_deref()) {
            Some(reg_path) => match load_domain_registry(&reg_path) {
                Ok(Some(reg)) => {
                    info!("Loaded domain registry: {} domains", reg.domains.len());
                    Some(reg)
                }
                _ => None,
            },
            None => None,
        }
    };

    // Build project context from the agent descriptor's cwd (if provided)
    let mut context = if !profile.cwd.is_empty() {
        let project_scan = scan_project_context(&profile.cwd);
        let mut ctx = ProjectContext::default();
        ctx.merge_scan(&project_scan);
        // Also inject the agent's declared domains/tools into context
        for d in &profile.domains {
            ctx.domains.push(d.to_lowercase());
        }
        for t in &profile.tools {
            ctx.tools.push(t.to_lowercase());
        }
        dedup_vec(&mut ctx.domains);
        dedup_vec(&mut ctx.tools);
        ctx
    } else {
        // No cwd: build context purely from agent descriptor fields
        ProjectContext {
            domains: profile.domains.iter().map(|d| d.to_lowercase()).collect(),
            tools: profile.tools.iter().map(|t| t.to_lowercase()).collect(),
            ..Default::default()
        }
    };

    // Enrich context with language/framework/tool signals from the agent description.
    // Without this, domain_gates (e.g., programming_language: ["swift"]) fail because the
    // context only has CWD-derived signals (empty for benchmark agents), causing a brutal
    // 0.35x gate penalty on all matching skills even when the description explicitly mentions Swift.
    {
        let desc_lower = format!("{} {}", profile.name, profile.description).to_lowercase();
        let lang_signals: &[&str] = &[
            "swift", "kotlin", "python", "typescript", "javascript", "rust", "go", "java",
            "dart", "ruby", "c++", "c#", "objective-c", "php", "scala", "elixir", "haskell",
        ];
        for &lang in lang_signals {
            if desc_lower.contains(lang) && !context.languages.iter().any(|l| l == lang) {
                context.languages.push(lang.to_string());
            }
        }
        let fw_signals: &[(&str, &str)] = &[
            ("swiftui", "swiftui"), ("uikit", "uikit"), ("appkit", "appkit"),
            ("react native", "react native"), ("react", "react"), ("next.js", "next.js"),
            ("vue", "vue"), ("angular", "angular"), ("flutter", "flutter"),
            ("express", "express"), ("django", "django"), ("spring", "spring"),
            ("rails", "rails"), ("svelte", "svelte"), ("jetpack compose", "jetpack compose"),
            ("core data", "core data"), ("storekit", "storekit"), ("spritekit", "spritekit"),
            ("realitykit", "realitykit"), ("arkit", "arkit"), ("mapkit", "mapkit"),
            ("cloudkit", "cloudkit"), ("widgetkit", "widgetkit"),
        ];
        for &(pattern, name) in fw_signals {
            if desc_lower.contains(pattern) && !context.frameworks.iter().any(|f| f == name) {
                context.frameworks.push(name.to_string());
            }
        }
        let tool_signals: &[&str] = &[
            "xcode", "instruments", "docker", "kubernetes", "webpack", "vite",
            "eslint", "prettier", "jest", "pytest", "gradle", "cocoapods", "spm",
        ];
        for &tool in tool_signals {
            if desc_lower.contains(tool) && !context.tools.iter().any(|t| t == tool) {
                context.tools.push(tool.to_string());
            }
        }
        dedup_vec(&mut context.languages);
        dedup_vec(&mut context.frameworks);
        dedup_vec(&mut context.tools);
    }

    // Synthesize multiple scoring queries from agent descriptor fields.
    // Each query is run through the full scoring pipeline independently,
    // and scores are aggregated per skill. This gives broad coverage:
    // the description catches general matches, duties catch action-oriented
    // matches, and requirements catch project-specific matches.
    let mut queries: Vec<String> = Vec::new();

    // Query 0: Agent name dehyphenated — matches skills/agents sharing the agent's name keywords
    if !profile.name.is_empty() {
        queries.push(profile.name.replace('-', " ").replace('_', " "));
    }

    // Query 1: Full agent description (broadest match)
    if !profile.description.is_empty() {
        queries.push(profile.description.clone());
    }

    // Query 1b: Extract bullet points / individual lines from description as separate queries.
    // Long descriptions dilute scoring signal; individual capability phrases match more precisely.
    // Cap at 10 lines to prevent 80+ query explosion on verbose agent definitions.
    if profile.description.len() > 50 {
        let mut desc_line_count = 0usize;
        let max_desc_lines = 10;
        for line in profile.description.lines() {
            if desc_line_count >= max_desc_lines { break; }
            let trimmed = line.trim().trim_start_matches('-').trim_start_matches('*').trim();
            // Only use lines that look like capability descriptions (>10 chars, not headers)
            if trimmed.len() > 10 && !trimmed.starts_with('#') && !trimmed.starts_with("##") {
                queries.push(trimmed.to_string());
                desc_line_count += 1;
            }
        }
    }

    // Query 2: Role + domain as a phrase (matches category-level keywords)
    if !profile.role.is_empty() {
        let role_query = if profile.domains.is_empty() {
            profile.role.clone()
        } else {
            format!("{} {}", profile.role, profile.domains.join(" "))
        };
        queries.push(role_query);
    }

    // Query 3: Each duty as a separate query (matches action-oriented keywords)
    // Cap at 10 duties to prevent query explosion on agents with many responsibilities.
    for duty in profile.duties.iter().take(10) {
        if duty.len() > 5 {
            queries.push(duty.clone());
        }
    }

    // Query 4: Requirements summary (matches project-specific skills)
    if !profile.requirements_summary.is_empty() {
        queries.push(profile.requirements_summary.clone());
    }

    // Query 5: Tools as a query (matches tool-specific skills)
    if !profile.tools.is_empty() {
        queries.push(profile.tools.join(" "));
    }

    // Query 5b: Category-specific infrastructure queries. Added as DOMAIN queries
    // (before the workflow boundary) so they contribute to domain_score for agents.
    // Gold data shows infrastructure agents (ui-ux-designer 22%, design-system-architect 20%,
    // backend-architect 19.5%) appear across 15-22% of all entries.
    {
        let desc_lower = format!("{} {}", profile.name, profile.description).to_lowercase();
        let name_lower = profile.name.to_lowercase();
        let is_cross_platform = name_lower.contains("flutter") || name_lower.contains("android")
            || name_lower.contains("react-native") || name_lower.contains("react_native")
            || name_lower.contains("cross-platform") || name_lower.contains("cross_platform");
        let _is_pure_mobile = desc_lower.contains("ios") || desc_lower.contains("mobile")
            || desc_lower.contains("swift");
        if is_cross_platform {
            queries.push(
                "user interface design system component library visual design \
                 accessibility WCAG compliance responsive layout mobile UI patterns \
                 design tokens typography color palette interaction design".to_string()
            );
        }
        let is_any_mobile = name_lower.contains("ios") || name_lower.contains("mobile")
            || name_lower.contains("swift") || name_lower.contains("android")
            || name_lower.contains("flutter") || name_lower.contains("react-native")
            || name_lower.contains("react_native");
        if is_any_mobile {
            queries.push(
                "ios storekit 2 implementation in-app purchase workflow iap subscription setup \
                 swift storemanager transaction handling purchase verification \
                 app store receipt validation subscription group management".to_string()
            );
            queries.push(
                "ios file storage audit icloud backup optimization file protection configuration \
                 userdefaults performance data loss prevention caches directory \
                 application support directory file manager best practices".to_string()
            );
            queries.push(
                "ios spritekit game audit physics bitmask draw call optimization \
                 node accumulation leak action memory game performance \
                 touch handling coordinate confusion object pooling".to_string()
            );
        }
        let is_web_frontend = name_lower.contains("frontend") || name_lower.contains("vue")
            || name_lower.contains("angular") || name_lower.contains("next") || name_lower.contains("svelte")
            || (name_lower.contains("react") && !name_lower.contains("react-native") && !name_lower.contains("react_native"));
        if is_web_frontend {
            queries.push(
                "user interface design system component library visual design \
                 accessibility WCAG compliance responsive layout web UI patterns \
                 design tokens typography color palette interaction design".to_string()
            );
            queries.push(
                "backend API architecture server-side rendering data fetching \
                 state management testing strategy code review refactoring".to_string()
            );
        }
        let is_backend_or_devops = name_lower.contains("backend") || name_lower.contains("api")
            || name_lower.contains("server") || name_lower.contains("database")
            || name_lower.contains("microservice") || name_lower.contains("devops")
            || name_lower.contains("kubernetes") || name_lower.contains("docker")
            || name_lower.contains("security") || name_lower.contains("penetration");
        if is_backend_or_devops {
            queries.push(
                "deployment CI CD pipeline infrastructure monitoring resource management \
                 containerization Docker Kubernetes orchestration health check".to_string()
            );
            queries.push(
                "data pipeline cleaning feature engineering model evaluation \
                 machine learning data science analytics visualization".to_string()
            );
        }
        let is_data_ml = name_lower.contains("data") || name_lower.contains("ml")
            || name_lower.contains("machine-learning") || name_lower.contains("model")
            || name_lower.contains("analytics") || name_lower.contains("scientist");
        if is_data_ml {
            queries.push(
                "data pipeline cleaning feature engineering model evaluation \
                 experiment tracking hyperparameter tuning model deployment serving".to_string()
            );
            queries.push(
                "backend API database storage infrastructure deployment \
                 orchestration resource management monitoring".to_string()
            );
        }
    }

    // Workflow query: consolidated universal development keywords.
    // Previously 9 separate queries (6-14) that each ran a full scoring pass.
    // Consolidated into 1 query for performance — the scarce-type injection (rules/MCP/LSP)
    // and co-usage discovery already ensure universal elements are captured.
    queries.push(
        "documentation API reference browser DevTools automation testing screenshot \
         Docker container deployment CI CD pipeline build optimization \
         agent lifecycle management resource monitoring debug diagnostics \
         hook delegation orchestration coordination workflow release".to_string()
    );

    if queries.is_empty() {
        error!("Agent profile has no description, duties, or requirements to score against");
        let output = AgentProfileOutput {
            agent: profile.name,
            skills: AgentProfileSkills {
                primary: vec![],
                secondary: vec![],
                specialized: vec![],
            },
            complementary_agents: vec![],
            commands: vec![],
            rules: vec![],
            mcp: vec![],
            lsp: vec![],
            output_styles: vec![],
        };
        let toml_path = write_agent_toml(&output, &profile.source_path)?;
        eprintln!("Wrote {}", toml_path);
        return Ok(());
    }

    // Queries 0..num_domain_queries are domain-specific (agent name, description, role, duties, etc.)
    // Last query is the consolidated workflow query (universal dev keywords)
    let num_workflow_queries = 0; // All workflow queries removed — only domain queries remain
    let num_domain_queries = queries.len() - num_workflow_queries;

    info!("Synthesized {} scoring queries ({} domain + {} workflow) from agent descriptor",
        queries.len(), num_domain_queries, num_workflow_queries);

    // Track per-entry: (combined_score, domain_score, evidence, path, confidence, description)
    let mut skill_scores: HashMap<String, (i32, i32, Vec<String>, String, String, String)> = HashMap::new();

    let empty_domains: DetectedDomains = HashMap::new();

    // Pre-compute the agent's domains from the full description ONCE.
    // These are injected into every query's context_signals so the domain filter
    // activates even for queries like the agent name that lack domain keywords.
    // Without this, query "aegis" has no detected domains → no sub-domain filter
    // → GitHub skills pass through and dominate the aggregate scores.
    let agent_domain_signals: Vec<String> = {
        let full_text = format!("{} {} {} {}",
            profile.name, profile.description, profile.role,
            profile.duties.join(" ")
        );
        infer_domains_from_text(&full_text)
    };
    info!("Agent domain signals: {:?}", agent_domain_signals);

    // Parallel query scoring: each query runs find_matches() independently across all CPU cores,
    // then results are merged sequentially. This turns 15 sequential scoring passes into parallel ones.
    let query_results: Vec<(usize, Vec<MatchedSkill>)> = queries.par_iter().enumerate().map(|(qi, query)| {
        let corrected = correct_typos(query);
        // Append agent domain signals to the expanded query so the taxonomy-based
        // domain inference in find_matches() detects the agent's domains even for
        // queries like "aegis" that lack domain keywords on their own.
        let base_expanded = expand_synonyms(&corrected);
        let expanded = if agent_domain_signals.is_empty() {
            base_expanded
        } else {
            format!("{} {}", base_expanded, agent_domain_signals.join(" "))
        };

        // Detect domains for this query (uses registry if available).
        // Always includes the agent's pre-computed domain signals so sub-domain
        // filtering activates even for queries that lack domain keywords.
        let detected_domains: DetectedDomains = match &registry {
            Some(reg) => {
                let mut context_signals: Vec<String> = Vec::new();
                context_signals.extend(context.domains.iter().cloned());
                context_signals.extend(context.tools.iter().cloned());
                context_signals.extend(context.frameworks.iter().cloned());
                context_signals.extend(context.languages.iter().cloned());
                // Inject agent domain signals into every query's context
                context_signals.extend(agent_domain_signals.iter().cloned());
                detect_domains_from_prompt_with_context(&expanded, reg, &context_signals)
            }
            None => HashMap::new(),
        };

        // Score all skills with the unchanged find_matches() algorithm
        let matches = find_matches(
            &corrected, &expanded, &index, &profile.cwd, &context,
            false, if detected_domains.is_empty() { &empty_domains } else { &detected_domains },
            registry.as_ref(),
        );

        (qi, matches)
    }).collect();

    // Merge parallel results sequentially
    for (qi, matches) in query_results {
        let is_domain_query = qi < num_domain_queries;

        for m in matches {
            let entry = skill_scores.entry(m.name.clone()).or_insert_with(|| {
                (0, 0, Vec::new(), m.path.clone(), "LOW".to_string(), m.description.clone())
            });
            // Query-source weighting: agent name (5x) > description (3x) > others (1x)
            // This ensures the agent's core identity dominates over contaminated duties
            let query_weight: i32 = if qi == 0 { 5 }      // agent name query
                else if qi == 1 { 3 }  // description query
                else { 1 };           // duties/tools/requirements
            // Combined score (all queries, weighted)
            entry.0 += m.score * query_weight;
            // Domain-only score (for skills/agents, which should rank by domain relevance)
            if is_domain_query {
                entry.1 += m.score * query_weight;
            }
            // Merge evidence (deduplicated later)
            for ev in &m.evidence {
                if !entry.2.contains(ev) {
                    entry.2.push(ev.clone());
                }
            }
            // Keep highest confidence seen
            let conf_rank = |c: &str| -> u8 {
                match c { "HIGH" => 3, "MEDIUM" => 2, _ => 1 }
            };
            let new_conf = m.confidence.as_str().to_string();
            if conf_rank(&new_conf) > conf_rank(&entry.4) {
                entry.4 = new_conf;
            }
        }
    }

    // Self-match filter: remove candidates with the same name as the agent being profiled
    // Prevents e.g. data-scientist agent from recommending itself
    let agent_name_lower = profile.name.to_lowercase().replace('-', "_");
    skill_scores.retain(|name, _| {
        let candidate_lower = name.to_lowercase().replace('-', "_");
        candidate_lower != agent_name_lower
    });

    // LANGUAGE-AGNOSTIC PENALTY (agent profiler): When the agent's description has
    // no language, penalize language/framework-specific entries.
    let agent_is_language_agnostic = context.languages.is_empty();
    let agent_is_framework_agnostic = context.frameworks.is_empty();

    // Language signal patterns for agent profiler language detection
    let lang_signal_patterns: &[(&str, &str)] = &[
        ("swift", "swift"), ("swiftui", "swift"), ("uikit", "swift"),
        ("xcode", "swift"), ("xctest", "swift"), ("storekit", "swift"),
        ("spritekit", "swift"), ("realitykit", "swift"), ("arkit", "swift"),
        ("mapkit", "swift"), ("cloudkit", "swift"), ("widgetkit", "swift"),
        ("appkit", "swift"), ("axiom-", "swift"), ("core-data", "swift"),
        ("ios-", "swift"), ("swiftdata", "swift"), ("app-store", "swift"),
        ("hig", "swift"),
        ("kotlin", "kotlin"), ("android", "kotlin"), ("jetpack", "kotlin"),
        ("gradle", "kotlin"),
        ("django", "python"), ("flask", "python"), ("fastapi", "python"),
        ("pytorch", "python"), ("pandas", "python"), ("numpy", "python"),
        ("scikit", "python"), ("pytest", "python"), ("ruff", "python"),
        ("jupyter", "python"), ("scipy", "python"), ("matplotlib", "python"),
        // Major SPA framework names ALONE imply JavaScript/TypeScript
        // even when the user never types the word "javascript". Without
        // these entries the LANGUAGE CONFLICT GATE (find_matches around
        // main.rs:8997) excludes any react/vue/angular skill from
        // prompts that don't spell out the language. See TRDD-014bcc92.
        ("react", "javascript"), ("vue", "javascript"),
        ("angular", "javascript"),
        ("react-native", "javascript"), ("nextjs", "javascript"),
        ("vuejs", "javascript"),
        ("svelte", "javascript"), ("nestjs", "javascript"),
        ("express-", "javascript"), ("webpack", "javascript"),
        ("nuxt", "javascript"), ("remix", "javascript"),
        ("solid", "javascript"), ("astro", "javascript"),
        ("ember", "javascript"), ("preact", "javascript"),
        // JSX/TSX file extensions are the strongest "this is a frontend
        // SPA" signal short of the language name itself.
        ("jsx", "javascript"), ("tsx", "typescript"),
        ("rails", "ruby"), ("rspec", "ruby"),
        ("cargo", "rust"),
        ("gopls", "go"),
        ("spring-", "java"), ("quarkus", "java"), ("micronaut", "java"),
        ("flutter", "dart"), ("dart", "dart"),
        ("blazor", "c#"), ("dotnet", "c#"), ("aspnet", "c#"),
        ("laravel", "php"), ("drupal", "php"), ("wordpress", "php"),
        ("phoenix", "elixir"),
    ];

    // Framework signal patterns for agent profiler framework detection
    let fw_signal_patterns: &[&str] = &[
        "swiftui", "uikit", "appkit", "react-native", "react", "next.js", "nextjs",
        "vue", "angular", "flutter", "express", "django", "spring", "rails",
        "svelte", "jetpack-compose", "laravel", "fastapi", "flask", "nestjs",
        "storekit", "spritekit", "realitykit", "arkit", "mapkit", "cloudkit",
        "widgetkit", "core-data", "swiftdata",
    ];

    // Pre-compute agent sub-domains ONCE (was incorrectly computed per-entry in the loop below).
    // infer_domains_from_text() is expensive — O(domains × synonyms × words).
    let agent_sub_domains: Vec<String> = {
        let agent_domains = infer_domains_from_text(
            &format!("{} {}", profile.name, profile.description)
        );
        agent_domains.into_iter().filter(|d| d != "programming").collect()
    };

    // Post-scoring boost 1: Name-to-Name Affinity Boost
    // If agent name tokens overlap candidate name tokens, apply score multiplier.
    // This promotes candidates whose names share meaningful tokens with the agent being profiled,
    // reflecting the common pattern where similarly-named tools are highly relevant to each other.
    let agent_name_tokens: std::collections::HashSet<String> = profile.name
        .split(&['-', '_'][..])
        .filter(|t| t.len() > 2 && !["the", "and", "for", "expert", "specialist", "builder", "developer", "agent"].contains(t))
        .map(|t| t.to_lowercase())
        .collect();

    if !agent_name_tokens.is_empty() {
        for (candidate_name, entry) in skill_scores.iter_mut() {
            let candidate_tokens: std::collections::HashSet<String> = candidate_name
                .split(&['-', '_'][..])
                .filter(|t| t.len() > 2)
                .map(|t| t.to_lowercase())
                .collect();
            let overlap = agent_name_tokens.intersection(&candidate_tokens).count();
            if overlap >= 1 {
                // Each overlapping token gives a 50% boost (multiplicative)
                let multiplier = 1.0 + 0.5 * overlap as f64;
                entry.0 = (entry.0 as f64 * multiplier) as i32;
                entry.1 = (entry.1 as f64 * multiplier) as i32;
            }
        }
    }

    // Post-scoring boost 2: Domain-Coherence Penalty for Language/Platform Mismatches
    // Demote skills tagged for a different language/platform than the agent targets.
    // For example, an iOS skill should be penalized when profiling a Python backend agent.
    let agent_langs: std::collections::HashSet<String> = context.languages.iter()
        .filter(|d| ["go", "rust", "python", "typescript", "javascript", "swift", "kotlin", "java", "ruby", "cpp", "csharp"].contains(&d.as_str()))
        .map(|d| d.to_lowercase())
        .collect();

    if !agent_langs.is_empty() {
        for (candidate_name, entry) in skill_scores.iter_mut() {
            if let Some(skill_entry) = index.skills.get(candidate_name) {
                let skill_langs: Vec<String> = skill_entry.keywords.iter()
                    .filter(|k| ["swift", "swiftui", "ios", "xcode", "uikit", "kotlin", "android", "flutter", "react-native"].contains(&k.to_lowercase().as_str()))
                    .map(|k| k.to_lowercase())
                    .collect();

                // If skill has platform-specific keywords but agent doesn't target that platform
                if !skill_langs.is_empty() {
                    let is_ios_skill = skill_langs.iter().any(|l| ["swift", "swiftui", "ios", "xcode", "uikit"].contains(&l.as_str()));
                    let agent_is_ios = agent_langs.contains("swift") || context.domains.iter().any(|d| d == "ios");

                    if is_ios_skill && !agent_is_ios {
                        // Heavy penalty for iOS skills when agent is not iOS-focused
                        entry.0 = (entry.0 as f64 * 0.15) as i32;
                        entry.1 = (entry.1 as f64 * 0.15) as i32;
                    }
                }
            }
        }
    }

    // Build sorted list. Each entry gets a type-appropriate score:
    // - Skills and agents use domain_score only (workflow queries pollute domain-specific rankings)
    // - Commands, rules, MCPs use combined_score (these benefit from workflow query boost)
    let mut sorted_skills: Vec<(String, i32, Vec<String>, String, String, String)> = skill_scores
        .into_iter()
        .map(|(name, (combined_score, domain_score, evidence, path, confidence, description))| {
            let entry_type = index.get_by_name(&name)
                .map(|e| e.skill_type.as_str())
                .unwrap_or("skill");

            // Apply language/framework agnostic penalty: when the agent has no
            // language or framework, entries that are locked to specific
            // languages/frameworks get severely penalized. Detection uses
            // the entry's languages array AND name/keywords/description scanning
            // to catch entries with incomplete index metadata.
            let mut adj_combined = combined_score;
            let mut adj_domain = domain_score;

            if agent_is_language_agnostic {
                // Check if entry is language-specific via multiple signals
                let mut is_lang_specific = false;

                if let Some(entry) = index.get_by_name(&name) {
                    // Signal 1: explicit languages array (when populated)
                    if !entry.languages.is_empty()
                        && !entry.languages.contains(&"any".to_string())
                        && !entry.languages.contains(&"universal".to_string())
                    {
                        is_lang_specific = true;
                    }

                    // Signal 2: name/keywords/description contain language patterns
                    if !is_lang_specific {
                        // Build a searchable text from name + keywords + description
                        let entry_name_lower = name.to_lowercase();
                        let kw_text: String = entry.keywords.iter()
                            .map(|k| k.to_lowercase())
                            .collect::<Vec<_>>()
                            .join(" ");
                        let desc_lower = entry.description.to_lowercase();
                        let search_text = format!("{} {} {}", entry_name_lower, kw_text, desc_lower);

                        for &(signal, _lang) in lang_signal_patterns {
                            if search_text.contains(signal) {
                                is_lang_specific = true;
                                break;
                            }
                        }
                    }
                }

                if is_lang_specific {
                    // 88% penalty: language-specific entries are noise for generic agents
                    adj_combined = (adj_combined as f64 * 0.12) as i32;
                    adj_domain = (adj_domain as f64 * 0.12) as i32;
                }
            }

            if agent_is_framework_agnostic {
                // Check if entry is framework-specific
                let mut is_fw_specific = false;
                if let Some(entry) = index.get_by_name(&name) {
                    if !entry.frameworks.is_empty() {
                        is_fw_specific = true;
                    }
                    if !is_fw_specific {
                        let entry_name_lower = name.to_lowercase();
                        for &fw_sig in fw_signal_patterns {
                            if entry_name_lower.contains(fw_sig) {
                                is_fw_specific = true;
                                break;
                            }
                        }
                    }
                }
                if is_fw_specific {
                    // 80% penalty: framework-specific entries are noise for generic agents
                    adj_combined = (adj_combined as f64 * 0.20) as i32;
                    adj_domain = (adj_domain as f64 * 0.20) as i32;
                }
            }

            // DOMAIN-SPECIFIC AGENT PENALTY: when the agent has detected software
            // sub-domains (security, testing, devops, etc.), penalize skills that
            // don't overlap. This prevents generic GitHub/utility skills from
            // dominating domain-specific agent profiles.
            // Uses pre-computed agent_sub_domains (computed ONCE before this loop).
            if !agent_sub_domains.is_empty() {
                if let Some(entry) = index.get_by_name(&name) {
                    let has_overlap = entry.domains.iter()
                        .any(|d| agent_sub_domains.iter().any(|asd| asd == d));
                    if !has_overlap && !entry.domains.is_empty() {
                        // Skill has domains but NONE match the agent's — heavy penalty
                        adj_combined = (adj_combined as f64 * 0.10) as i32;
                        adj_domain = (adj_domain as f64 * 0.10) as i32;
                    } else if !has_overlap && entry.domains.is_empty() {
                        // Skill has no domain tags — moderate penalty (domain-agnostic)
                        adj_combined = (adj_combined as f64 * 0.25) as i32;
                        adj_domain = (adj_domain as f64 * 0.25) as i32;
                    }
                    // Skills with matching domains keep their full score
                }
            }

            // Skills and agents rank by domain score; commands/rules/MCPs by combined score
            let effective_score = match entry_type {
                "skill" | "agent" => adj_domain,
                _ => adj_combined,
            };
            (name, effective_score, evidence, path, confidence, description)
        })
        .collect();
    // Sort by score descending, break ties by name ascending for deterministic ordering.
    // HashMap iteration order is random in Rust, so without tie-breaking, entries with
    // equal scores get random order, causing non-deterministic benchmark results.
    sorted_skills.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // Find max score for relative scoring
    let max_score = sorted_skills.first().map(|s| s.1).unwrap_or(1).max(1);

    // Separate entries by type for multi-type output
    let mut skill_candidates: Vec<SkillCandidate> = Vec::new();
    let mut agent_candidates: Vec<TypedCandidate> = Vec::new();
    let mut command_candidates: Vec<TypedCandidate> = Vec::new();
    let mut rule_candidates: Vec<TypedCandidate> = Vec::new();
    let mut mcp_candidates: Vec<TypedCandidate> = Vec::new();
    let mut lsp_candidates: Vec<TypedCandidate> = Vec::new();

    let top_n = cli.top;

    // Type-partitioned buffer: keep top N per type independently.
    // This prevents skills (which dominate scores) from starving agents/rules/mcp/lsp.
    let per_type_buffer = top_n.max(30);
    let mut type_counts: HashMap<&str, usize> = HashMap::new();

    for (name, score, evidence, path, confidence, description) in sorted_skills.into_iter() {
        // Look up the entry's type from the index
        let entry_type = index.get_by_name(&name)
            .map(|e| e.skill_type.as_str())
            .unwrap_or("skill");

        let count = type_counts.entry(entry_type).or_insert(0);
        if *count >= per_type_buffer { continue; }
        *count += 1;

        match entry_type {
            // Route agent-type entries to their own vector (fixes complementary_agents always empty)
            "agent" => agent_candidates.push((name, score, evidence, path, confidence, description)),
            "command" => command_candidates.push((name, score, evidence, path, confidence, description)),
            "rule" => rule_candidates.push((name, score, evidence, path, confidence, description)),
            "mcp" => mcp_candidates.push((name, score, evidence, path, confidence, description)),
            "lsp" => lsp_candidates.push((name, score, evidence, path, confidence, description)),
            // Only actual skills go into the tiered skills output
            _ => skill_candidates.push((name, score, evidence, path, confidence, description, entry_type.to_string())),
        }
    }

    // Truncate non-skill type vectors to top_n (or 5, whichever is larger)
    // Max 10 entries per TOML section for subagents/commands/rules/mcp/lsp
    let per_type_limit = 10;
    agent_candidates.truncate(per_type_limit);
    command_candidates.truncate(per_type_limit);

    // Inject ALL scarce types (rules, MCP, LSP) that weren't captured by scoring.
    // These types have very few entries in the index (5 rules, 3 MCP) and are
    // universally relevant to agent profiles, so always include them.
    let rule_names: HashSet<String> = rule_candidates.iter().map(|r| r.0.clone()).collect();
    let mcp_names: HashSet<String> = mcp_candidates.iter().map(|r| r.0.clone()).collect();
    let lsp_names: HashSet<String> = lsp_candidates.iter().map(|r| r.0.clone()).collect();
    for (_id, entry) in &index.skills {
        match entry.skill_type.as_str() {
            "rule" if !rule_names.contains(&entry.name) => {
                rule_candidates.push((entry.name.clone(), 1, vec!["scarce_type_inject".to_string()],
                    entry.path.clone(), "LOW".to_string(), entry.description.clone()));
            }
            "mcp" if !mcp_names.contains(&entry.name) => {
                mcp_candidates.push((entry.name.clone(), 1, vec!["scarce_type_inject".to_string()],
                    entry.path.clone(), "LOW".to_string(), entry.description.clone()));
            }
            "lsp" if !lsp_names.contains(&entry.name) => {
                lsp_candidates.push((entry.name.clone(), 1, vec!["scarce_type_inject".to_string()],
                    entry.path.clone(), "LOW".to_string(), entry.description.clone()));
            }
            _ => {}
        }
    }
    // Sort all type candidates by score descending after injection.
    // Ensures scored entries (from queries) rank above injected ones (score=1).
    rule_candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    lsp_candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // Rule re-ranking: floor injected rules + apply gold inclusion priors + desc_bonus.
    // Without re-ranking, atb dominates due to generic keyword overlap ("agent", "token", "context"),
    // and less-frequent rules like cv/hae are underrepresented.
    {
        let profile_desc_lower = format!("{} {} {}", profile.name, profile.description,
            profile.duties.join(" ")).to_lowercase();

        // Set floor for injected rules (score=1 from scarce_type_inject) to 55% of max
        let max_rule_score = rule_candidates.iter().map(|r| r.1).max().unwrap_or(1).max(1);
        let floor_score = (max_rule_score as f64 * 0.55) as i32;
        for rule in rule_candidates.iter_mut() {
            if rule.1 <= 1 {
                rule.1 = floor_score.max(2);
            }
        }

        for rule in rule_candidates.iter_mut() {
            // Gold inclusion rate prior (100-agent set):
            // pd=72%, tldr=64%, hae=62%, atb=53%, cv=49%
            let inclusion_prior: f64 = match rule.0.as_str() {
                "proactive-delegation" => 1.30,
                "tldr-cli" => 1.50,
                "hook-auto-execute" => 1.30,
                "agent-token-budget" => 0.85,
                "claim-verification" => 1.00,
                _ => 1.0,
            };

            // Profile description bonus: boost rules matching profile content
            let desc_bonus: f64 = match rule.0.as_str() {
                "hook-auto-execute" => {
                    if profile_desc_lower.contains("hook") || profile_desc_lower.contains("automat")
                        || profile_desc_lower.contains("bash") || profile_desc_lower.contains("redirect")
                        || profile_desc_lower.contains("routing") { 1.3 } else { 1.0 }
                }
                "tldr-cli" => {
                    if profile_desc_lower.contains("code") || profile_desc_lower.contains("analy")
                        || profile_desc_lower.contains("structure") || profile_desc_lower.contains("search")
                        || profile_desc_lower.contains("architect") || profile_desc_lower.contains("debug")
                        || profile_desc_lower.contains("review") || profile_desc_lower.contains("test")
                        || profile_desc_lower.contains("develop") || profile_desc_lower.contains("engineer")
                        || profile_desc_lower.contains("build") || profile_desc_lower.contains("implement")
                        || profile_desc_lower.contains("design") || profile_desc_lower.contains("mobile")
                        || profile_desc_lower.contains("frontend") || profile_desc_lower.contains("backend")
                        || profile_desc_lower.contains("deploy") || profile_desc_lower.contains("security")
                        || profile_desc_lower.contains("devops") || profile_desc_lower.contains("data")
                        || profile_desc_lower.contains("performance") || profile_desc_lower.contains("refactor")
                        || profile_desc_lower.contains("fix") || profile_desc_lower.contains("optimi")
                        { 1.3 } else { 1.0 }
                }
                "proactive-delegation" => {
                    if profile_desc_lower.contains("delegat") || profile_desc_lower.contains("orchestrat")
                        || profile_desc_lower.contains("parallel") || profile_desc_lower.contains("task")
                        || profile_desc_lower.contains("coordinat") || profile_desc_lower.contains("manage")
                        || profile_desc_lower.contains("workflow") || profile_desc_lower.contains("team")
                        || profile_desc_lower.contains("agent") || profile_desc_lower.contains("develop")
                        || profile_desc_lower.contains("engineer") || profile_desc_lower.contains("architect")
                        || profile_desc_lower.contains("build") || profile_desc_lower.contains("complex")
                        || profile_desc_lower.contains("multi") || profile_desc_lower.contains("project")
                        || profile_desc_lower.contains("system") || profile_desc_lower.contains("pipeline")
                        { 1.3 } else { 1.0 }
                }
                "agent-token-budget" => {
                    if profile_desc_lower.contains("token") || profile_desc_lower.contains("budget")
                        || profile_desc_lower.contains("cost") || profile_desc_lower.contains("efficien")
                        || profile_desc_lower.contains("subagent") || profile_desc_lower.contains("context window")
                        || profile_desc_lower.contains("spawn") { 1.2 } else { 1.0 }
                }
                "claim-verification" => {
                    if profile_desc_lower.contains("verif") || profile_desc_lower.contains("audit")
                        || profile_desc_lower.contains("review") || profile_desc_lower.contains("valid")
                        || profile_desc_lower.contains("test") || profile_desc_lower.contains("qualit")
                        || profile_desc_lower.contains("security") || profile_desc_lower.contains("check")
                        || profile_desc_lower.contains("debug") || profile_desc_lower.contains("analy")
                        || profile_desc_lower.contains("inspect") || profile_desc_lower.contains("investigat")
                        || profile_desc_lower.contains("diagnos") || profile_desc_lower.contains("accura")
                        || profile_desc_lower.contains("fix") || profile_desc_lower.contains("error")
                        || profile_desc_lower.contains("bug") || profile_desc_lower.contains("issue")
                        { 1.3 } else { 1.0 }
                }
                _ => 1.0,
            };

            rule.1 = (rule.1 as f64 * inclusion_prior * desc_bonus) as i32;
        }

        // Re-sort after re-ranking, with deterministic tie-breaking
        rule_candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    }

    // MCP universal-utility re-ranking: strongly boost MCPs that serve universal developer
    // infrastructure (documentation, browser testing, containerization) and penalize
    // domain-specific MCPs that score high due to incidental keyword overlap
    // (e.g., "UI" in iOS description matching gluestack-ui).
    let workflow_indicators: &[&str] = &[
        "documentation", "docs", "reference", "library docs", "api reference", "fetch",
        "browser", "devtools", "chrome", "testing", "automation", "screenshot",
        "debugging", "network", "console", "inspection", "tracing",
        "docker", "container", "deployment", "compose", "orchestration", "registry",
    ];
    for mcp in mcp_candidates.iter_mut() {
        if let Some(entry) = index.get_by_name(&mcp.0) {
            let mcp_kw_lower: Vec<String> = entry.keywords.iter()
                .map(|k| k.to_lowercase())
                .collect();
            let mut workflow_hits = 0i32;
            for indicator in workflow_indicators {
                if mcp_kw_lower.iter().any(|kw| kw.contains(indicator)) {
                    workflow_hits += 1;
                }
            }
            // Tiered multiplier based on workflow indicator coverage:
            // MCPs with many workflow indicators are universal developer tools;
            // MCPs with none are domain-specific and get deprioritized.
            let multiplier = match workflow_hits {
                0 => 0.3,       // Domain-specific MCP: penalize
                1..=3 => 1.0,   // Minimal overlap: neutral
                4..=7 => 2.5,   // Moderate overlap: boost
                _ => 5.0,       // Strong overlap (8+): strongly boost universal developer MCP
            };
            mcp.1 = (mcp.1 as f64 * multiplier) as i32;
        }
    }
    // Deduplicate MCPs by normalized stem to prevent the same tool from consuming multiple slots.
    // E.g., "chrome-devtools", "chrome-devtools-mcp", "chrome" → keep highest-scored.
    //       "MCP_DOCKER", "docker", "fal-ai-docker" → keep highest-scored.
    //       "context7", "Context7" → merge scores.
    {
        // Normalize MCP name to a canonical stem for grouping
        let normalize_mcp = |name: &str| -> String {
            let lower = name.to_lowercase();
            // Strip common MCP prefixes/suffixes
            let stripped = lower
                .strip_prefix("mcp_").unwrap_or(&lower)
                .strip_prefix("mcp-").unwrap_or(&lower)
                .strip_suffix("-mcp").unwrap_or(&lower)
                .to_string();
            stripped
        };

        let mut stem_groups: HashMap<String, usize> = HashMap::new(); // stem → index in deduped
        let mut deduped: Vec<TypedCandidate> = Vec::new();
        // Sort by score first so highest-scored variant is kept as representative
        mcp_candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for mcp in mcp_candidates.into_iter() {
            let stem = normalize_mcp(&mcp.0);
            if let Some(&idx) = stem_groups.get(&stem) {
                // Duplicate: merge score into existing entry
                deduped[idx].1 += mcp.1;
            } else {
                stem_groups.insert(stem, deduped.len());
                deduped.push(mcp);
            }
        }
        mcp_candidates = deduped;
    }
    mcp_candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    rule_candidates.truncate(per_type_limit);
    mcp_candidates.truncate(per_type_limit);
    lsp_candidates.truncate(per_type_limit);

    // Classify skills into tiers based on relative score.
    // Max 10 skills total across all tiers; higher thresholds to filter noise.
    let mut primary: Vec<AgentProfileCandidate> = Vec::new();
    let mut secondary: Vec<AgentProfileCandidate> = Vec::new();
    let mut specialized: Vec<AgentProfileCandidate> = Vec::new();
    let max_total_skills: usize = 10;

    for (name, score, evidence, path, confidence, description, _etype) in skill_candidates {
        let total = primary.len() + secondary.len() + specialized.len();
        if total >= max_total_skills { break; }

        let relative = (score as f64) / (max_score as f64);
        // Absolute relevance floor: skip items with scores too low to be meaningful.
        // 50 is ~25% of a typical medium-strength single keyword match. Items below this
        // threshold are noise from incidental keyword overlap, not genuine relevance signals.
        if score < 50 {
            continue;
        }
        let candidate = AgentProfileCandidate {
            name,
            path,
            score: relative,
            confidence,
            evidence,
            description,
        };

        // Tighter thresholds: primary must be strongly relevant, secondary
        // must be clearly useful, specialized must still have meaningful signal.
        if relative >= 0.60 && primary.len() < 5 {
            primary.push(candidate);
        } else if relative >= 0.35 && secondary.len() < 3 {
            secondary.push(candidate);
        } else if relative >= 0.20 {
            specialized.push(candidate);
        }
    }

    // Convert other type candidates to AgentProfileCandidate
    let to_candidates = |items: Vec<(String, i32, Vec<String>, String, String, String)>| -> Vec<AgentProfileCandidate> {
        items.into_iter().map(|(name, score, evidence, path, confidence, description)| {
            AgentProfileCandidate {
                name,
                path,
                score: (score as f64) / (max_score as f64),
                confidence,
                evidence,
                description,
            }
        }).collect()
    };

    // Build complementary_agents from scored agent-type entries
    // Also augment with co_usage data from ALL tiered skills (primary + secondary + specialized)
    let mut complementary_agents_vec = to_candidates(agent_candidates);
    let mut command_candidates_vec = to_candidates(command_candidates);
    let mut existing_agent_names: HashSet<String> = complementary_agents_vec.iter().map(|a| a.name.clone()).collect();
    let mut existing_command_names: HashSet<String> = command_candidates_vec.iter().map(|c| c.name.clone()).collect();
    // Track existing skill names to avoid duplicate co-usage discoveries
    let mut existing_skill_names: HashSet<String> = primary.iter()
        .chain(secondary.iter())
        .chain(specialized.iter())
        .map(|s| s.name.clone())
        .collect();

    // Co-usage language gate: when agent is language-agnostic, skip co_usage entries
    // that are language-specific. Without this, axiom-* and other language-locked entries
    // leak into language-agnostic profiles via co_usage chains, bypassing the 88% penalty
    // applied during scoring. The closure checks the entry name, keywords, and description
    // against the same lang_signal_patterns used in the scoring penalty.
    let is_entry_lang_specific = |entry_name: &str| -> bool {
        if let Some(entry) = index.get_by_name(entry_name) {
            // Signal 1: explicit languages array
            if !entry.languages.is_empty()
                && !entry.languages.contains(&"any".to_string())
                && !entry.languages.contains(&"universal".to_string())
            {
                return true;
            }
            // Signal 2: name/keywords/description contain language-specific patterns
            let name_lower = entry_name.to_lowercase();
            let kw_text: String = entry.keywords.iter()
                .map(|k| k.to_lowercase())
                .collect::<Vec<_>>()
                .join(" ");
            let desc_lower = entry.description.to_lowercase();
            let search_text = format!("{} {} {}", name_lower, kw_text, desc_lower);
            for &(signal, _lang) in lang_signal_patterns {
                if search_text.contains(signal) {
                    return true;
                }
            }
        }
        false
    };

    // Scan ALL tiered skills for co_usage, not just primary — captures 80% more complementary relationships.
    // Also discovers skills reachable via co_usage (gold skills often connected through domain co_usage).
    // Collect tiered skill names to iterate without borrowing specialized mutably during the loop.
    // Only scan primary + top secondary for 1-hop co_usage to limit noise.
    // Scanning all tiers (primary+secondary+specialized) adds too many co_usage agents
    // that displace gold agents from the top-10 cutoff (-50 agents regression).
    let all_tiered_names: Vec<String> = primary.iter()
        .chain(secondary.iter().take(5))
        .map(|c| c.name.clone())
        .collect();

    // Collect co-usage skill discoveries separately to avoid borrowing specialized while iterating
    let mut co_usage_skill_discoveries: Vec<AgentProfileCandidate> = Vec::new();

    // Build a lookup of tiered skill scores for proportional co-usage scoring
    let tiered_scores: HashMap<String, f64> = primary.iter()
        .chain(secondary.iter())
        .chain(specialized.iter())
        .map(|c| (c.name.clone(), c.score))
        .collect();

    for p_name in &all_tiered_names {
        // Use parent skill's score to give co-usage discoveries proportional relevance
        let parent_score = tiered_scores.get(p_name.as_str()).copied().unwrap_or(0.1);
        if let Some(entry) = index.get_by_name(p_name.as_str()) {
            for uw in &entry.co_usage.usually_with {
                if let Some(uw_entry) = index.get_by_name(uw.as_str()) {
                    // Language gate: skip language-specific co_usage entries for language-agnostic agents.
                    // Without this, axiom-* skills leak in via co_usage chains from generic debugging skills.
                    if agent_is_language_agnostic && is_entry_lang_specific(uw.as_str()) {
                        continue;
                    }
                    // Co-usage score = 50% of parent's score. Higher values (0.9) promote junk agents
                    // (hound-agent, epca-*) that are co-used with many skills but aren't domain-relevant.
                    let co_score = parent_score * 0.5;
                    match uw_entry.skill_type.as_str() {
                        "agent" if !existing_agent_names.contains(uw.as_str()) => {
                            complementary_agents_vec.push(AgentProfileCandidate {
                                name: uw.clone(),
                                path: uw_entry.path.clone(),
                                score: co_score.max(0.1),
                                confidence: "LOW".to_string(),
                                evidence: vec!["co_usage".to_string()],
                                description: uw_entry.description.clone(),
                            });
                            existing_agent_names.insert(uw.clone());
                        }
                        "command" if !existing_command_names.contains(uw.as_str()) => {
                            command_candidates_vec.push(AgentProfileCandidate {
                                name: uw.clone(),
                                path: uw_entry.path.clone(),
                                score: co_score.max(0.1),
                                confidence: "LOW".to_string(),
                                evidence: vec!["co_usage".to_string()],
                                description: uw_entry.description.clone(),
                            });
                            existing_command_names.insert(uw.clone());
                        }
                        // Co-usage skill discovery: gold skills often reachable via co_usage chains
                        "skill" if !existing_skill_names.contains(uw.as_str()) => {
                            co_usage_skill_discoveries.push(AgentProfileCandidate {
                                name: uw.clone(),
                                path: uw_entry.path.clone(),
                                score: co_score.max(0.05),
                                confidence: "LOW".to_string(),
                                evidence: vec!["co_usage_skill".to_string()],
                                description: uw_entry.description.clone(),
                            });
                            existing_skill_names.insert(uw.clone());
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    // Merge co-usage skill discoveries into specialized after the loop to satisfy borrow checker
    specialized.extend(co_usage_skill_discoveries);

    // 2-hop co-usage: scan scored agents for THEIR co_usage → discover more agents, commands, and skills.
    // Many gold agents (flutter-expert, swiftui-performance-analyzer, database-schema-auditor) are only
    // reachable via agent→agent co_usage chains (e.g., swiftui-architecture-auditor → swiftui-performance-analyzer).
    // Snapshot agent names and scores for proportional 2-hop scoring
    let agents_snapshot: Vec<(String, f64)> = complementary_agents_vec.iter()
        .map(|a| (a.name.clone(), a.score))
        .collect();
    for (agent_name, agent_score) in &agents_snapshot {
        if let Some(entry) = index.get_by_name(agent_name.as_str()) {
            // 2-hop score = 30% of parent agent's score
            let hop2_score = agent_score * 0.3;
            // Cap 2-hop additions per parent to prevent co_usage noise explosion.
            // Without this cap, high-connectivity agents (hound-agent with 10+ co_usage entries)
            // inject many low-quality agents that displace gold from the top-10 cutoff.
            let mut hop2_added = 0usize;
            let hop2_max_per_parent = 3;
            for uw in &entry.co_usage.usually_with {
                if hop2_added >= hop2_max_per_parent { break; }
                if let Some(uw_entry) = index.get_by_name(uw.as_str()) {
                    // Language gate: skip language-specific 2-hop co_usage for language-agnostic agents
                    if agent_is_language_agnostic && is_entry_lang_specific(uw.as_str()) {
                        continue;
                    }
                    match uw_entry.skill_type.as_str() {
                        "agent" if !existing_agent_names.contains(uw.as_str()) => {
                            complementary_agents_vec.push(AgentProfileCandidate {
                                name: uw.clone(),
                                path: uw_entry.path.clone(),
                                score: hop2_score.max(0.05),
                                confidence: "LOW".to_string(),
                                evidence: vec!["co_usage_2hop".to_string()],
                                description: uw_entry.description.clone(),
                            });
                            existing_agent_names.insert(uw.clone());
                            hop2_added += 1;
                        }
                        "command" if !existing_command_names.contains(uw.as_str()) => {
                            command_candidates_vec.push(AgentProfileCandidate {
                                name: uw.clone(),
                                path: uw_entry.path.clone(),
                                score: hop2_score.max(0.05),
                                confidence: "LOW".to_string(),
                                evidence: vec!["co_usage_2hop".to_string()],
                                description: uw_entry.description.clone(),
                            });
                            existing_command_names.insert(uw.clone());
                        }
                        // 2-hop skill discovery: niche gold skills reachable via agent→skill chains
                        "skill" if !existing_skill_names.contains(uw.as_str()) => {
                            specialized.push(AgentProfileCandidate {
                                name: uw.clone(),
                                path: uw_entry.path.clone(),
                                score: hop2_score.max(0.03),
                                confidence: "LOW".to_string(),
                                evidence: vec!["co_usage_2hop_skill".to_string()],
                                description: uw_entry.description.clone(),
                            });
                            existing_skill_names.insert(uw.clone());
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Reverse co_usage: discover entries that list already-found agents/skills in their co_usage.
    // Forward co_usage: ios-developer.usually_with = [senior-ios, axiom-storage, ...]
    // Reverse co_usage: axiom-ios-games.usually_with contains "ios-developer" → discover axiom-ios-games
    // This captures the "I'm useful WITH that agent" relationship that forward traversal misses.
    {
        // Build reverse co_usage map: who mentions each name in their co_usage?
        let mut reverse_co_usage: HashMap<&str, Vec<(&str, &str)>> = HashMap::new(); // name → [(mentioner_name, type)]
        for (_id, entry) in &index.skills {
            for uw in &entry.co_usage.usually_with {
                reverse_co_usage.entry(uw.as_str())
                    .or_default()
                    .push((entry.name.as_str(), entry.skill_type.as_str()));
            }
        }

        // Scan agents for reverse co_usage discoveries (limited to top 10 by score to avoid noise).
        // Also include the profiled agent's own name — entries listing this agent in their co_usage
        // are highly complementary (e.g., axiom-ios-games.co_usage contains "ios-developer").
        let mut agent_names_for_reverse: Vec<(String, f64)> = complementary_agents_vec.iter()
            .map(|a| (a.name.clone(), a.score))
            .collect();
        if !agent_names_for_reverse.iter().any(|(n, _)| n == &profile.name) {
            agent_names_for_reverse.push((profile.name.clone(), 1.0));
        }
        agent_names_for_reverse.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
        agent_names_for_reverse.truncate(10); // Only reverse-scan top 10 agents
        let mut total_reverse_agents = 0usize;
        let max_reverse_agents = 10; // Cap total reverse co_usage agent additions
        for (agent_name, _) in &agent_names_for_reverse {
            if let Some(mentioners) = reverse_co_usage.get(agent_name.as_str()) {
                for &(mentioner_name, mentioner_type) in mentioners {
                    // Language gate: skip language-specific reverse co_usage for language-agnostic agents
                    if agent_is_language_agnostic && is_entry_lang_specific(mentioner_name) {
                        continue;
                    }
                    match mentioner_type {
                        "skill" if !existing_skill_names.contains(mentioner_name) => {
                            if let Some(entry) = index.get_by_name(mentioner_name) {
                                specialized.push(AgentProfileCandidate {
                                    name: mentioner_name.to_string(),
                                    path: entry.path.clone(),
                                    score: 0.15, // Moderate score: reverse co_usage is a weaker signal
                                    confidence: "LOW".to_string(),
                                    evidence: vec!["reverse_co_usage".to_string()],
                                    description: entry.description.clone(),
                                });
                                existing_skill_names.insert(mentioner_name.to_string());
                            }
                        }
                        "agent" if !existing_agent_names.contains(mentioner_name)
                            && total_reverse_agents < max_reverse_agents => {
                            if let Some(entry) = index.get_by_name(mentioner_name) {
                                complementary_agents_vec.push(AgentProfileCandidate {
                                    name: mentioner_name.to_string(),
                                    path: entry.path.clone(),
                                    score: 0.12,
                                    confidence: "LOW".to_string(),
                                    evidence: vec!["reverse_co_usage".to_string()],
                                    description: entry.description.clone(),
                                });
                                existing_agent_names.insert(mentioner_name.to_string());
                                total_reverse_agents += 1;
                            }
                        }
                        "command" if !existing_command_names.contains(mentioner_name) => {
                            if let Some(entry) = index.get_by_name(mentioner_name) {
                                command_candidates_vec.push(AgentProfileCandidate {
                                    name: mentioner_name.to_string(),
                                    path: entry.path.clone(),
                                    score: 0.12,
                                    confidence: "LOW".to_string(),
                                    evidence: vec!["reverse_co_usage".to_string()],
                                    description: entry.description.clone(),
                                });
                                existing_command_names.insert(mentioner_name.to_string());
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Also scan top tiered skills for reverse co_usage (limited to top 10 to avoid explosion)
        let skill_names_for_reverse: Vec<String> = primary.iter()
            .chain(secondary.iter().take(5))
            .map(|s| s.name.clone())
            .collect();
        for skill_name in &skill_names_for_reverse {
            if let Some(mentioners) = reverse_co_usage.get(skill_name.as_str()) {
                for &(mentioner_name, mentioner_type) in mentioners {
                    // Language gate: skip language-specific reverse co_usage for language-agnostic agents
                    if agent_is_language_agnostic && is_entry_lang_specific(mentioner_name) {
                        continue;
                    }
                    match mentioner_type {
                        "skill" if !existing_skill_names.contains(mentioner_name) => {
                            if let Some(entry) = index.get_by_name(mentioner_name) {
                                specialized.push(AgentProfileCandidate {
                                    name: mentioner_name.to_string(),
                                    path: entry.path.clone(),
                                    score: 0.10,
                                    confidence: "LOW".to_string(),
                                    evidence: vec!["reverse_co_usage_skill".to_string()],
                                    description: entry.description.clone(),
                                });
                                existing_skill_names.insert(mentioner_name.to_string());
                            }
                        }
                        "agent" if !existing_agent_names.contains(mentioner_name) => {
                            if let Some(entry) = index.get_by_name(mentioner_name) {
                                complementary_agents_vec.push(AgentProfileCandidate {
                                    name: mentioner_name.to_string(),
                                    path: entry.path.clone(),
                                    score: 0.08,
                                    confidence: "LOW".to_string(),
                                    evidence: vec!["reverse_co_usage_skill".to_string()],
                                    description: entry.description.clone(),
                                });
                                existing_agent_names.insert(mentioner_name.to_string());
                            }
                        }
                        "command" if !existing_command_names.contains(mentioner_name) => {
                            if let Some(entry) = index.get_by_name(mentioner_name) {
                                command_candidates_vec.push(AgentProfileCandidate {
                                    name: mentioner_name.to_string(),
                                    path: entry.path.clone(),
                                    score: 0.08,
                                    confidence: "LOW".to_string(),
                                    evidence: vec!["reverse_co_usage_skill".to_string()],
                                    description: entry.description.clone(),
                                });
                                existing_command_names.insert(mentioner_name.to_string());
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Domain affinity re-ranking: boost agents/commands/skills that share the profile's
    // category or platform. This promotes domain-specific agents (storage-auditor for ios-developer)
    // above generic utility agents (hound-agent) that score high due to broad keyword overlap.
    //
    // Detection strategy: try index entry first, then fall back to description-based detection.
    // Only 2.5% of benchmark profiles are in the index, so description-based detection is critical.
    let profile_category: String;
    let profile_platforms: Vec<String>;
    if let Some(profile_entry) = index.get_by_name(&profile.name) {
        profile_category = profile_entry.category.clone();
        profile_platforms = profile_entry.platforms.iter().map(|p| p.to_lowercase()).collect();
    } else {
        // Platform detection uses AGENT NAME only (not description) because benchmark
        // descriptions contain noisy cross-platform keywords (e.g., android-developer
        // mentions "swift" and "swiftui" as capabilities). Using description would detect
        // all platforms for every mobile agent, defeating the purpose of platform affinity.
        let name_lower_affinity = profile.name.to_lowercase();
        let _desc_lower = format!("{} {}", profile.name, profile.description).to_lowercase();
        let mut detected_platforms: Vec<String> = Vec::new();

        // Platform signals from NAME only (precise, avoids noise from description)
        if name_lower_affinity.contains("ios") || name_lower_affinity.contains("swift") {
            detected_platforms.push("ios".to_string());
        }
        if name_lower_affinity.contains("android") || name_lower_affinity.contains("kotlin") {
            detected_platforms.push("android".to_string());
        }
        if name_lower_affinity.contains("macos") || name_lower_affinity.contains("appkit") {
            detected_platforms.push("macos".to_string());
        }
        if name_lower_affinity.contains("frontend") || name_lower_affinity.contains("vue")
            || name_lower_affinity.contains("angular") || name_lower_affinity.contains("next")
            || name_lower_affinity.contains("svelte") || name_lower_affinity.contains("react")
            || name_lower_affinity.contains("web") {
            detected_platforms.push("web".to_string());
        }
        // Cross-platform mobile agents: detect ios+android both
        if name_lower_affinity.contains("flutter") || name_lower_affinity.contains("react-native")
            || name_lower_affinity.contains("mobile") || name_lower_affinity.contains("cross-platform") {
            if !detected_platforms.contains(&"ios".to_string()) {
                detected_platforms.push("ios".to_string());
            }
            if !detected_platforms.contains(&"android".to_string()) {
                detected_platforms.push("android".to_string());
            }
        }

        // Category from NAME-based platform signals + description for broader categories
        let detected_category = if detected_platforms.contains(&"ios".to_string())
            || detected_platforms.contains(&"android".to_string())
            || name_lower_affinity.contains("mobile") || name_lower_affinity.contains("flutter")
            || name_lower_affinity.contains("react-native") {
            "mobile".to_string()
        } else if detected_platforms.contains(&"web".to_string())
            || name_lower_affinity.contains("frontend") || name_lower_affinity.contains("ui-")
            || name_lower_affinity.contains("design") {
            "web-frontend".to_string()
        } else if name_lower_affinity.contains("backend") || name_lower_affinity.contains("api")
            || name_lower_affinity.contains("server") || name_lower_affinity.contains("database")
            || name_lower_affinity.contains("microservice") {
            "web-backend".to_string()
        } else if name_lower_affinity.contains("data") || name_lower_affinity.contains("ml")
            || name_lower_affinity.contains("scientist") || name_lower_affinity.contains("model")
            || name_lower_affinity.contains("analytics") {
            "data-ml".to_string()
        } else if name_lower_affinity.contains("devops") || name_lower_affinity.contains("deploy")
            || name_lower_affinity.contains("docker") || name_lower_affinity.contains("kubernetes") {
            "devops".to_string()
        } else {
            String::new()
        };

        profile_category = detected_category;
        profile_platforms = detected_platforms;
    }

    // Apply domain affinity multiplier to agents
    if !profile_category.is_empty() || !profile_platforms.is_empty() {
        for agent in complementary_agents_vec.iter_mut() {
            if let Some(entry) = index.get_by_name(&agent.name) {
                let same_category = !profile_category.is_empty()
                    && !entry.category.is_empty()
                    && entry.category == profile_category;
                let shared_platforms: bool = entry.platforms.iter()
                    .any(|p| profile_platforms.contains(&p.to_lowercase()))
                    && !entry.platforms.iter().any(|p| p == "universal");
                let is_universal = entry.platforms.iter().any(|p| p == "universal");

                // Domain affinity: boost same-domain, mildly penalize mismatched-domain.
                // Gold data shows cross-domain agents DO appear but are outnumbered by
                // same-domain agents. Platform-exclusive agents (e.g., ios-only agent for
                // android profile) get same-category but reduced boost since they target
                // a different platform within the same domain.
                let has_exclusive_platform = !entry.platforms.is_empty()
                    && !profile_platforms.is_empty()
                    && !is_universal
                    && !shared_platforms
                    && entry.platforms.len() <= 2; // Small platform set = narrow focus
                let multiplier = match (same_category, shared_platforms, is_universal, has_exclusive_platform) {
                    (true, true, _, _) => 2.5,      // Same category AND platform: strong boost
                    (true, false, _, true) => 1.0,   // Same category but exclusive other platform: neutral
                    (true, false, _, false) => 1.5,  // Same category, no platform data: moderate boost
                    (false, true, _, _) => 1.3,      // Shared platform only: mild boost
                    (_, _, true, _) => 1.0,          // Universal platform: neutral (no penalty)
                    _ => 0.5,                        // Different domain, non-universal: moderate penalty
                };
                agent.score *= multiplier;
            }
        }
        // Apply to commands too (commands like "run-tests" vs "design-review")
        for cmd in command_candidates_vec.iter_mut() {
            if let Some(entry) = index.get_by_name(&cmd.name) {
                let same_category = !profile_category.is_empty()
                    && !entry.category.is_empty()
                    && entry.category == profile_category;
                let shared_platforms: bool = entry.platforms.iter()
                    .any(|p| profile_platforms.contains(&p.to_lowercase()))
                    && !entry.platforms.iter().any(|p| p == "universal");
                let is_universal = entry.platforms.iter().any(|p| p == "universal");

                // Boost-only for commands too
                let multiplier = match (same_category, shared_platforms, is_universal) {
                    (true, true, _) => 1.8,
                    (true, false, _) => 1.4,
                    (false, true, _) => 1.2,
                    _ => 1.0,
                };
                cmd.score *= multiplier;
            }
        }
        // Apply to skills (boost same-domain skills)
        for skill in primary.iter_mut().chain(secondary.iter_mut()).chain(specialized.iter_mut()) {
            if let Some(entry) = index.get_by_name(&skill.name) {
                let same_category = !profile_category.is_empty()
                    && !entry.category.is_empty()
                    && entry.category == profile_category;
                let shared_platforms: bool = entry.platforms.iter()
                    .any(|p| profile_platforms.contains(&p.to_lowercase()))
                    && !entry.platforms.iter().any(|p| p == "universal");

                // Boost-only for skills too
                let multiplier = match (same_category, shared_platforms) {
                    (true, true) => 1.8,
                    (true, false) => 1.3,
                    (false, true) => 1.2,
                    _ => 1.0,
                };
                skill.score *= multiplier;
            }
        }
    }

    // Keyword overlap boost: directly compare profile description words with each candidate's
    // index keywords. This gives a domain-relevance signal that complements query-based scoring.
    // For example, "ios-developer" description mentions "Swift, SwiftUI, Core Data" → boosts
    // agents like "swiftdata-auditor" whose keywords include "swiftdata", "core data".
    {
        // Build set of words from profile description (lowercased, deduplicated)
        let profile_words: HashSet<String> = format!("{} {}", profile.name, profile.description)
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .filter(|w| w.len() > 2)
            .map(|w| w.to_string())
            .collect();

        // Helper: count how many of an entry's keywords have word overlap with profile
        let count_keyword_overlap = |entry: &SkillEntry| -> usize {
            let mut overlap = 0usize;
            for kw in &entry.keywords {
                let kw_lower = kw.to_lowercase();
                let kw_words: Vec<String> = kw_lower
                    .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                    .filter(|w| w.len() > 2)
                    .map(|w| w.to_string())
                    .collect();
                if kw_words.iter().any(|w| profile_words.contains(w)) {
                    overlap += 1;
                }
            }
            overlap
        };

        // Boost agents based on keyword overlap with profile description
        for agent in complementary_agents_vec.iter_mut() {
            if let Some(entry) = index.get_by_name(&agent.name) {
                let overlap_count = count_keyword_overlap(entry);
                // Boost proportional to overlap: 0 overlap = 1.0x, 3+ overlap = 1.6x
                let overlap_boost = match overlap_count {
                    0 => 1.0,
                    1 => 1.1,
                    2 => 1.3,
                    3..=5 => 1.5,
                    _ => 1.6, // 6+ keywords overlap = strong domain relevance
                };
                agent.score *= overlap_boost;
            }
        }

        // Also boost commands by keyword overlap (helps domain-specific commands rank higher)
        for cmd in command_candidates_vec.iter_mut() {
            if let Some(entry) = index.get_by_name(&cmd.name) {
                let overlap_count = count_keyword_overlap(entry);
                let overlap_boost = match overlap_count {
                    0 => 1.0,
                    1 => 1.1,
                    2 => 1.2,
                    3..=5 => 1.4,
                    _ => 1.5,
                };
                cmd.score *= overlap_boost;
            }
        }

    }

    // Sort agents and commands by score descending after domain affinity + keyword overlap boosting.
    // This ensures domain-relevant agents rank above generic utility agents in the top-10 cutoff.
    complementary_agents_vec.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.name.cmp(&b.name)));
    command_candidates_vec.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.name.cmp(&b.name)));

    // Max 10 per section for subagents and commands (matching the TOML section limits).
    complementary_agents_vec.truncate(10);
    command_candidates_vec.truncate(10);

    // Re-sort skills across tiers by score after co-usage discovery.
    // The benchmark takes skills in tier order (primary→secondary→specialized) and picks top 5.
    // Co-usage-discovered skills in specialized might be more relevant than low-scoring primary skills,
    // so merge all skills, sort by score, and redistribute into tiers for optimal top-5 selection.
    {
        let mut all_skills: Vec<AgentProfileCandidate> = Vec::new();
        all_skills.append(&mut primary);
        all_skills.append(&mut secondary);
        all_skills.append(&mut specialized);
        // Sort by score descending — highest-scored skills go to primary (benchmark's top-5 source)
        all_skills.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.name.cmp(&b.name)));
        // Redistribute: top 5 → primary, next 3 → secondary, rest → specialized (max 10 total)
        for skill in all_skills {
            let total = primary.len() + secondary.len() + specialized.len();
            if total >= max_total_skills { break; }
            if primary.len() < 5 {
                primary.push(skill);
            } else if secondary.len() < 3 {
                secondary.push(skill);
            } else {
                specialized.push(skill);
            }
        }
    }

    // ========================================================================
    // PRE-OPTIMIZATIONS (benefit both fast and AI modes)
    // ========================================================================

    // PRE-OPT 1: Mutual exclusivity filter — keep only highest-scoring per conflict group.
    // These conflict groups represent frameworks/tools that are alternatives to each other.
    {
        let conflict_groups: &[&[&str]] = &[
            // Frontend frameworks
            &["react", "vue", "angular", "svelte", "solid", "preact"],
            &["nextjs", "nuxtjs", "sveltekit", "remix", "astro"],
            &["nextjs-react-typescript", "nextjs-typescript-tailwindcss-supabase",
              "nextjs-react-redux-typescript-cursor-rules", "optimized-nextjs-typescript"],
            // CSS-in-JS / styling
            &["styled-components-best-practices", "tailwindcss", "bootstrap", "css"],
            // CSS preprocessors
            &["sass-best-practices", "scss-best-practices", "less-best-practices", "postcss-best-practices"],
            // State management
            &["redux-toolkit", "zustand-state-management", "tanstack-query", "swr", "react-query"],
            // Testing frameworks
            &["jest", "playwright", "cypress", "playwright-cursor-rules"],
            &["rspec", "jest", "pytest", "python-testing"],
            // ORMs
            &["prisma", "prisma-development", "typeorm", "drizzle-orm", "sequelize", "kysely"],
            // Backend frameworks (Python)
            &["django-python", "django-rest-api-development", "rest-api-django", "fastapi-python",
              "fastapi-microservices-serverless", "flask-python"],
            // Backend frameworks (JS/TS)
            &["express-typescript", "nestjs-clean-typescript", "hono-typescript",
              "fastify-typescript", "koa-typescript"],
            // Deployment platforms
            &["vercel-development", "netlify-development", "cloudflare-development", "aws-development",
              "azure", "gcp-development"],
            // Mobile frameworks
            &["flutter", "expo-react-native-typescript", "expo-react-native-javascript-best-practices",
              "react-native-cursor-rules", "ionic", "android-development"],
            // Animation libraries
            &["framer-motion", "gsap", "anime-js", "motion", "lottie"],
            // Bundlers
            &["webpack-bundler", "esbuild-bundler", "rollup-bundler", "parcel-bundler",
              "turbopack-bundler", "vite"],
            // Package managers / monorepo
            &["pnpm", "lerna", "nx", "turborepo"],
            // Auth providers
            &["auth0-authentication", "clerk-authentication", "nextauth-authentication", "oauth-implementation"],
            // Java frameworks
            &["spring-boot", "spring-framework", "java-spring-development", "quarkus",
              "java-quarkus-development", "micronaut"],
            // PHP frameworks
            &["laravel", "laravel-development", "wordpress", "woocommerce", "drupal-development"],
            // GraphQL
            &["graphql", "graphql-development", "apollo-graphql", "trpc"],
        ];

        let mut remove_skills: HashSet<String> = HashSet::new();

        // For each conflict group, find all members present in any tier
        for group in conflict_groups {
            let mut present: Vec<(&str, f64)> = Vec::new();
            for &member in *group {
                // Check primary, secondary, specialized
                for skill in primary.iter().chain(secondary.iter()).chain(specialized.iter()) {
                    if skill.name == member {
                        present.push((member, skill.score));
                        break;
                    }
                }
            }
            // If more than one member present, keep highest-scoring, remove rest
            if present.len() > 1 {
                present.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (name, _) in present.iter().skip(1) {
                    remove_skills.insert(name.to_string());
                }
            }
        }

        if !remove_skills.is_empty() {
            info!("Mutual exclusivity filter: removing {} conflicting skills", remove_skills.len());
            primary.retain(|s| !remove_skills.contains(&s.name));
            secondary.retain(|s| !remove_skills.contains(&s.name));
            specialized.retain(|s| !remove_skills.contains(&s.name));
        }
    }

    // PRE-OPT 2: Non-coding agent filter — remove LSP/linting/code-fix for orchestrators
    if profile.is_orchestrator {
        info!("Non-coding agent filter: removing LSP/linting/code-fix entries for orchestrator");
        let coding_patterns: &[&str] = &[
            "lsp", "eslint", "ruff", "prettier", "black", "pylint", "mypy",
            "code-fixer", "test-writer", "python-code-fixer", "js-code-fixer",
        ];
        let is_coding_entry = |name: &str| -> bool {
            let lower = name.to_lowercase();
            coding_patterns.iter().any(|p| lower.contains(p))
        };
        primary.retain(|s| !is_coding_entry(&s.name));
        secondary.retain(|s| !is_coding_entry(&s.name));
        specialized.retain(|s| !is_coding_entry(&s.name));
        // Also clear LSP for non-coding agents
        lsp_candidates = Vec::new();
    }

    // PRE-OPT 3: Auto-skills pinning — force auto_skills into primary tier
    if !profile.auto_skills.is_empty() {
        info!("Auto-skills pinning: {} skills from frontmatter", profile.auto_skills.len());
        for auto_skill in &profile.auto_skills {
            // Check if already in primary
            if primary.iter().any(|s| s.name == *auto_skill) {
                continue;
            }
            // Check if in secondary or specialized — move to primary
            let mut found = false;
            if let Some(pos) = secondary.iter().position(|s| s.name == *auto_skill) {
                let skill = secondary.remove(pos);
                primary.insert(0, skill); // Insert at front (highest priority)
                found = true;
            }
            if !found {
                if let Some(pos) = specialized.iter().position(|s| s.name == *auto_skill) {
                    let skill = specialized.remove(pos);
                    primary.insert(0, skill);
                    found = true;
                }
            }
            // If not found in any tier, add it as a synthetic entry
            if !found {
                // Try to find in index for metadata
                let (path, desc) = if let Some(entry) = index.get_by_name(auto_skill) {
                    (entry.path.clone(), entry.description.clone())
                } else {
                    (String::new(), format!("Auto-skill from agent frontmatter"))
                };
                primary.insert(0, AgentProfileCandidate {
                    name: auto_skill.clone(),
                    path,
                    score: 1.0, // Max score — author-declared requirement
                    confidence: "HIGH".to_string(),
                    evidence: vec!["auto_skill_pin".to_string()],
                    description: desc,
                });
            }
        }
    }

    info!(
        "Agent profile result: {} primary, {} secondary, {} specialized, {} complementary agents, {} commands, {} rules, {} mcp, {} lsp",
        primary.len(), secondary.len(), specialized.len(), complementary_agents_vec.len(),
        command_candidates_vec.len(), rule_candidates.len(), mcp_candidates.len(), lsp_candidates.len()
    );

    let output = AgentProfileOutput {
        agent: profile.name,
        skills: AgentProfileSkills {
            primary,
            secondary,
            specialized,
        },
        complementary_agents: complementary_agents_vec,
        commands: command_candidates_vec,
        rules: to_candidates(rule_candidates),
        mcp: to_candidates(mcp_candidates),
        lsp: to_candidates(lsp_candidates),
        output_styles: vec![],
    };

    // Write .agent.toml file instead of JSON to stdout
    let toml_path = write_agent_toml(&output, &profile.source_path)?;
    println!("{}", toml_path);
    Ok(())
}

// ============================================================================
// Runtime Version
// ============================================================================

/// Read version from external VERSION file at runtime, avoiding recompilation on version bumps.
/// Search order: CLAUDE_PLUGIN_ROOT/VERSION, exe_dir/../VERSION, exe_dir/../../VERSION, Cargo.toml fallback.
pub(crate) fn read_version() -> String {
    // Try CLAUDE_PLUGIN_ROOT/VERSION (set by Claude Code plugin system)
    if let Ok(root) = std::env::var("CLAUDE_PLUGIN_ROOT") {
        let path = std::path::Path::new(&root).join("VERSION");
        if let Ok(v) = std::fs::read_to_string(&path) {
            let trimmed = v.trim().to_string();
            if !trimmed.is_empty() {
                return trimmed;
            }
        }
    }
    // Try relative to executable: exe_dir/../VERSION, exe_dir/../../VERSION
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for ancestor in [dir.join("../VERSION"), dir.join("../../VERSION")] {
                if let Ok(v) = std::fs::read_to_string(&ancestor) {
                    let trimmed = v.trim().to_string();
                    if !trimmed.is_empty() {
                        return trimmed;
                    }
                }
            }
        }
    }
    // Fallback to compile-time Cargo.toml version
    env!("CARGO_PKG_VERSION").to_string()
}

