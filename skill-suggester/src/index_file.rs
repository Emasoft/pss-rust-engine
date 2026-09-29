// Single-file indexing (--index-file) + CozoDB index path resolution
// (XUD7YUZH modularization step 9). Moved out of main.rs verbatim;
// re-exported at the crate root via `pub(crate) use`.

use std::fs;
use std::path::PathBuf;

use crate::consts::{CACHE_DIR, DB_FILE};
use crate::{
    SuggesterError, classify_entry_activities, extract_md_body, extract_negation_keywords,
    extract_pass1_frameworks, extract_pass1_languages, extract_pass1_platforms,
    extract_pass1_services, extract_pass1_tools, extract_rule_paths,
    generate_pass1_keywords, infer_domains_from_text, infer_pass1_intents,
    is_pass1_stopword, parse_frontmatter, stem_word,
};

// ============================================================================
// Single-file indexing (--index-file)
// ============================================================================

/// Read a single element .md file, parse frontmatter+body, run Pass 1
/// enrichment pipeline, and output enriched JSON to stdout.
pub(crate) fn run_index_file(path: &str) -> Result<(), SuggesterError> {
    let file_path = std::path::Path::new(path);

    // (a) Read the file
    let content = fs::read_to_string(file_path).map_err(|e| SuggesterError::IndexRead {
        path: PathBuf::from(path),
        source: e,
    })?;

    // (b) Parse frontmatter
    let frontmatter = parse_frontmatter(&content);

    // (c) Extract body (everything after frontmatter)
    let body = extract_md_body(&content);

    // (d) Extract name: from frontmatter, fallback to filename stem
    // For SKILL.md files, use parent directory name as the skill name
    let name = frontmatter
        .get("name")
        .cloned()
        .or_else(|| frontmatter.get("title").cloned())
        .unwrap_or_else(|| {
            let fname = file_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            // SKILL.md uses parent dir as name (e.g., skills/my-skill/SKILL.md → "my-skill")
            if fname.eq_ignore_ascii_case("SKILL.md") || fname.eq_ignore_ascii_case("skill.md") {
                file_path
                    .parent()
                    .and_then(|p| p.file_name())
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            } else {
                file_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            }
        });

    // (e) Extract description: from frontmatter, fallback to first non-empty non-heading paragraph
    let description = frontmatter
        .get("description")
        .cloned()
        .unwrap_or_else(|| {
            body.lines()
                .map(|l| l.trim())
                .find(|l| !l.is_empty() && !l.starts_with('#'))
                .unwrap_or("")
                .to_string()
        });

    // (f) Determine type from frontmatter or infer from path
    // Also check for frontmatter keys that imply command type (argument-hint, allowed-tools)
    let elem_type = frontmatter
        .get("type")
        .cloned()
        .unwrap_or_else(|| {
            // Canonicalize path separators for reliable matching
            let p = path.replace('\\', "/").to_lowercase();
            let fname = file_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            // Check both /skills/ and leading skills/ (relative paths)
            if p.contains("/skills/") || p.starts_with("skills/") || fname == "SKILL.md" {
                "skill".to_string()
            } else if p.contains("/agents/") || p.starts_with("agents/") {
                "agent".to_string()
            } else if p.contains("/commands/") || p.starts_with("commands/")
                || frontmatter.contains_key("argument-hint")
                || frontmatter.contains_key("allowed-tools")
            {
                "command".to_string()
            } else if p.contains("/rules/") || p.starts_with("rules/") {
                "rule".to_string()
            } else {
                "skill".to_string()
            }
        });

    // (g) Determine source from path using canonicalized path for reliable matching
    let canonical_path = fs::canonicalize(file_path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string());
    let source = {
        let p = canonical_path.to_lowercase();
        if p.contains("plugins/cache") {
            "marketplace:unknown"
        } else if p.contains(".claude/") {
            "user"
        } else {
            "project"
        }
    };

    // (h) Extract use_context: sections headed "When to Use", "Use Cases", etc.
    let use_context = extract_use_context_from_body(body);

    // (i) Run the EXACT same enrichment pipeline as run_pass1_batch
    let combined_text = if use_context.is_empty() {
        description.clone()
    } else {
        format!("{} {}", description, use_context)
    };

    let keywords = generate_pass1_keywords(&name, &combined_text);
    let activities = classify_entry_activities(&name, &description, &use_context);
    let category = activities
        .first()
        .map(|(a, _)| a.as_str())
        .unwrap_or("general-development");
    let intents = infer_pass1_intents(category);
    let languages = extract_pass1_languages(&keywords);
    let frameworks = extract_pass1_frameworks(&keywords);
    let platforms = extract_pass1_platforms(&keywords);
    let tools = extract_pass1_tools(&keywords);
    let services = extract_pass1_services(&keywords);

    let (extra_positive, negative_keywords) = extract_negation_keywords(&combined_text);

    // Merge extra positive words into keywords (dedup via seen set)
    let mut all_keywords = keywords;
    let kw_set: std::collections::HashSet<String> = all_keywords.iter().cloned().collect();
    for w in extra_positive {
        let stemmed = stem_word(&w);
        if !kw_set.contains(&w) && !is_pass1_stopword(&w) && w.len() > 2 {
            all_keywords.push(w);
        }
        if !kw_set.contains(&stemmed) && !is_pass1_stopword(&stemmed) && stemmed.len() > 2 {
            all_keywords.push(stemmed);
        }
    }
    all_keywords.truncate(20);

    // Dedup negative keywords
    let neg_set: std::collections::HashSet<String> = negative_keywords.into_iter().collect();
    let negative_kw: Vec<String> = neg_set.iter().cloned().collect();

    // Remove negative keywords AND their stems from positive keywords
    if !neg_set.is_empty() {
        let neg_stems: std::collections::HashSet<String> =
            neg_set.iter().map(|w| stem_word(w)).collect();
        all_keywords.retain(|kw| !neg_set.contains(kw) && !neg_stems.contains(kw));
    }

    // Extract use_cases from use_context bullet points
    let use_cases: Vec<String> = if use_context.is_empty() {
        vec![]
    } else {
        use_context
            .lines()
            .filter(|line| {
                let t = line.trim();
                t.starts_with("- ") || t.starts_with("* ") || t.starts_with("• ")
            })
            .map(|line| {
                line.trim()
                    .trim_start_matches(|c: char| c == '-' || c == '*' || c == '•')
                    .trim()
                    .to_string()
            })
            .filter(|s| !s.is_empty())
            .take(5)
            .collect()
    };

    // Determine tier from source
    let tier = if source.starts_with("marketplace:") {
        "community"
    } else if source == "project" || source.starts_with("project:") {
        "project"
    } else {
        "built-in"
    };

    // Generate domain_gates from extracted fields (same logic as run_pass1_batch)
    let mut domain_gates: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    if !languages.is_empty() {
        domain_gates.insert("programming_language".to_string(), serde_json::json!(languages));
    }
    if !platforms.is_empty() {
        domain_gates.insert("target_platform".to_string(), serde_json::json!(platforms));
    }
    if !frameworks.is_empty() {
        domain_gates.insert("framework".to_string(), serde_json::json!(frameworks));
    }

    // Infer domains from name + description using the shared synonym taxonomy
    let domains = infer_domains_from_text(&format!("{} {} {}", name, description, use_context));

    // Extract rule path gates from frontmatter `paths:` field (rules only).
    // Empty for non-rule entries; non-empty for rules that declare path-scoped activation.
    let path_gates: Vec<String> = if elem_type == "rule" {
        extract_rule_paths(&frontmatter)
    } else {
        Vec::new()
    };

    // (j) Build enriched output JSON (same fields as run_pass1_batch)
    let output = serde_json::json!({
        "name": name,
        "type": elem_type,
        "source": source,
        "path": path,
        "description": description,
        "keywords": all_keywords,
        "negative_keywords": negative_kw,
        "category": category,
        "activities": activities.iter().map(|(name, score)| serde_json::json!({"name": name, "score": score})).collect::<Vec<_>>(),
        "intents": intents,
        "tier": tier,
        "boost": 0,
        "platforms": platforms,
        "frameworks": frameworks,
        "languages": languages,
        "domains": domains,
        "tools": tools,
        "services": services,
        "file_types": [],
        "patterns": [],
        "directories": [],
        "path_patterns": [],
        "use_cases": use_cases,
        "secondary_categories": [],
        "domain_gates": domain_gates,
        "path_gates": path_gates,
    });

    // (k) Print to stdout
    println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());

    eprintln!("Index file: enriched '{}' from {}", name, path);
    Ok(())
}

/// Extract "when to use" / "use cases" sections from a markdown body.
/// Looks for headings containing: "When to Use", "Use Cases", "Usage",
/// "Use this when", "Triggers" (case-insensitive). Collects all text
/// (bullet points and paragraphs) under those sections until the next
/// heading or EOF.
pub(crate) fn extract_use_context_from_body(body: &str) -> String {
    let trigger_patterns = [
        "when to use",
        "use cases",
        "usage",
        "use this when",
        "triggers",
    ];

    let mut collecting = false;
    let mut lines_collected: Vec<&str> = Vec::new();

    for line in body.lines() {
        let trimmed = line.trim();

        // Check if this is a heading line
        if trimmed.starts_with('#') {
            let heading_text = trimmed.trim_start_matches('#').trim().to_lowercase();
            // Check if heading matches any trigger pattern
            let matches = trigger_patterns
                .iter()
                .any(|pat| heading_text.contains(pat));
            if matches {
                collecting = true;
                continue; // skip the heading line itself
            } else if collecting {
                // Hit a new non-matching heading — stop collecting
                break;
            }
        } else if collecting {
            // Collect body text under matching heading
            if !trimmed.is_empty() {
                lines_collected.push(trimmed);
            }
        }
    }

    lines_collected.join("\n")
}

// ============================================================================
// CozoDB Index (SQLite-backed pre-filtering for fast skill scoring)
// ============================================================================

/// Resolve the CozoDB index path. Returns None if the DB file does not exist.
/// Derives from a JSON index path by taking the sibling `DB_FILE` in its
/// directory (NOT by swapping the `.json` extension — the DB filename is fixed).
///
/// Resolution order: `--index` → `PSS_INDEX_PATH` → `~/.claude/cache/<DB_FILE>`.
///
/// An explicit override (`--index` / `PSS_INDEX_PATH`) is AUTHORITATIVE, never a
/// hint: when one is set this resolves ONLY from it and returns None if that does
/// not land on an existing file. It must NEVER fall through to a lower-priority
/// source. This is F12 of TRDD-1Z8SGQ7N, and the "why" is not theoretical: the
/// old code had no `else` on either override branch, so during F7 development an
/// agent pointed `PSS_INDEX_PATH` at a scratch dir whose DB had a different
/// filename — the sibling `pss-skill-index.db` did not exist, the override was
/// silently discarded, resolution fell through to the home default, and
/// `merge-events` wrote 1368 events (800 of them removals) into the USER'S REAL
/// INDEX, which had to be restored from a snapshot. A silently-ignored override
/// aims a destructive writer at whatever the fallback happens to be. Returning
/// None instead is safe at every call site: the writers (merge-events,
/// prune-history, migrate-element-ids) eprintln + exit, and the readers either
/// raise IndexNotFound or fall back to the JSON index — which `get_index_path`
/// resolves from the SAME override, so they can never silently read the real
/// index either.
pub(crate) fn get_db_path(cli_index: Option<&str>) -> Option<PathBuf> {
    let env_index = std::env::var("PSS_INDEX_PATH").ok();
    resolve_db_path_gated(cli_index, env_index.as_deref())
}

/// The pure decision behind [`get_db_path`], with the env value passed IN rather
/// than read from the process environment, so it is testable without
/// `std::env::set_var` (process-global ⇒ racy under cargo's threaded harness).
///
/// The two override branches look near-identical but are ASYMMETRIC ON PURPOSE
/// and must not be merged into one shared helper:
///
///   * `--index x.db`        → `x.db` ITSELF
///   * `PSS_INDEX_PATH=x.db` → the SIBLING `<dir>/pss-skill-index.db`
///
/// `resolve_db_path_canonical` and the Python twin (`scripts/pss_cozodb.py`
/// L150-153) implement the same asymmetry, and `tests/unit/test_pss_db_path_parity.py`
/// enforces the Rust↔Python parity. Unifying the branches would silently break it.
pub(crate) fn resolve_db_path_gated(cli_index: Option<&str>, env_index: Option<&str>) -> Option<PathBuf> {
    // 1. --index — authoritative when present: resolve from it or give up (None).
    if let Some(path) = cli_index {
        // An --index that already names a .db file IS the DB (no sibling lookup).
        if path.ends_with(".db") {
            let p = PathBuf::from(path);
            return if p.exists() { Some(p) } else { None };
        }
        // Otherwise it is the JSON index: take the sibling DB in its directory.
        // `parent()` is None only for the degenerate empty string; that is a
        // malformed override, and per the authority rule it yields None rather
        // than deferring to the env var or the home default.
        let db_path = PathBuf::from(path).parent()?.join(DB_FILE);
        return if db_path.exists() { Some(db_path) } else { None };
    }

    // 2. PSS_INDEX_PATH — likewise authoritative. An empty value means "unset"
    //    (`std::env::var` yields Ok("") for an exported-but-empty var), so only a
    //    non-empty value engages this branch.
    if let Some(path) = env_index {
        if !path.is_empty() {
            // NOTE: no `.ends_with(".db")` shortcut here — see the asymmetry note
            //       on this function. The env branch always takes the sibling.
            let db_path = PathBuf::from(path).parent()?.join(DB_FILE);
            return if db_path.exists() { Some(db_path) } else { None };
        }
    }

    // 3. Default: ~/.claude/cache/pss-skill-index.db — reached only when NO
    //    override was supplied.
    let home = dirs::home_dir()?;
    let db_path = home.join(".claude").join(CACHE_DIR).join(DB_FILE);
    if db_path.exists() { Some(db_path) } else { None }
}
