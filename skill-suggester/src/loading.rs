//! Index/registry/PSS-file loading + activation logging (XUD7YUZH
//! modularization step 7 — pure move, zero behavior change). Banner sections
//! "Index Loading", "Domain Registry Loading", "PSS File Loading" and
//! "Activation Logging" moved here byte-for-byte; every moved top-level
//! item's visibility widened to `pub(crate)` so it stays reachable from
//! `main.rs` via the crate-root re-export:
//!
//! ```ignore
//! mod loading;
//! pub(crate) use loading::*;
//! ```
//!
//! Referenced items from sibling modules are imported explicitly because
//! child modules do not see the crate root's glob re-exports.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use chrono::Utc;
use tracing::{debug, info, warn};

use crate::consts::{
    ACTIVATION_LOG_FILE, CACHE_DIR, INDEX_FILE, LOG_DIR, MAX_LOG_ENTRIES,
    MAX_LOG_PROMPT_LENGTH, REGISTRY_FILE,
};
use crate::matching::MatchedSkill;
use crate::scoring::make_entry_id;
use crate::types::{PssFile, SuggesterError};
use crate::{
    ActivationLogEntry, ActivationMatch, CoUsageData, DomainRegistry, SkillEntry, SkillIndex,
};

// ============================================================================
// Index Loading
// ============================================================================

/// Get the path to the skill index file.
/// Resolution order: --index CLI flag > PSS_INDEX_PATH env var > ~/.claude/cache/skill-index.json
pub(crate) fn get_index_path(cli_index: Option<&str>) -> Result<PathBuf, SuggesterError> {
    // 1. CLI flag takes priority (required on WASM targets)
    if let Some(path) = cli_index {
        return Ok(PathBuf::from(path));
    }

    // 2. Environment variable fallback
    if let Ok(path) = std::env::var("PSS_INDEX_PATH") {
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    // 3. Default: ~/.claude/cache/skill-index.json
    let home = dirs::home_dir().ok_or(SuggesterError::NoHomeDir)?;
    Ok(home.join(".claude").join(CACHE_DIR).join(INDEX_FILE))
}

/// Load and parse the skill index.
/// After JSON deserialization, re-keys the HashMap from name-based to ID-based
/// to prevent collisions when different sources provide same-named elements.
pub(crate) fn load_index(path: &PathBuf) -> Result<SkillIndex, SuggesterError> {
    if !path.exists() {
        return Err(SuggesterError::IndexNotFound(path.clone()));
    }

    // Bincode cache: skip JSON parsing when a fresh binary cache exists.
    // The cache is invalidated automatically when the JSON file is newer.
    let cache_path = path.with_extension("bin");
    if let Some(index) = load_bincode_cache(&cache_path, path) {
        return Ok(index);
    }

    // Cache miss — parse JSON (slow path)
    info!("Bincode cache miss, parsing JSON index: {:?}", path);
    let content = fs::read_to_string(path).map_err(|e| SuggesterError::IndexRead {
        path: path.clone(),
        source: e,
    })?;

    let mut index: SkillIndex =
        serde_json::from_str(&content).map_err(|e| SuggesterError::IndexParse(e.to_string()))?;

    // Re-key HashMap from name-based (JSON format) to ID-based (collision-safe).
    // JSON key formats: "name" (legacy) or "source::name" (new composite format).
    let old_skills = std::mem::take(&mut index.skills);
    let mut rekeyed = HashMap::with_capacity(old_skills.len());
    for (key, mut entry) in old_skills {
        // Extract element name from key: new "source::name" format or legacy "name" format
        let element_name = if let Some(pos) = key.find("::") {
            key[pos + 2..].to_string()
        } else {
            key
        };
        // Set the name field if not already populated from JSON
        if entry.name.is_empty() {
            entry.name = element_name;
        }
        // Use entry ID as HashMap key (unique per name+source combination)
        let entry_id = make_entry_id(&entry.name, &entry.source);
        rekeyed.insert(entry_id, entry);
    }
    index.skills = rekeyed;
    index.skills_count = index.skills.len();
    index.build_name_index();

    // Write bincode cache for next invocation (best-effort, don't fail on write errors)
    write_bincode_cache(&cache_path, &index);

    Ok(index)
}

/// Try to load the post-processed SkillIndex from a bincode cache file.
/// Returns None if the cache is missing, stale (older than JSON source), or corrupt.
pub(crate) fn load_bincode_cache(cache_path: &Path, json_path: &Path) -> Option<SkillIndex> {
    let cache_meta = fs::metadata(cache_path).ok()?;
    let json_meta = fs::metadata(json_path).ok()?;

    // Stale check: cache must be newer than JSON source
    let cache_mtime = cache_meta.modified().ok()?;
    let json_mtime = json_meta.modified().ok()?;
    if cache_mtime <= json_mtime {
        info!("Bincode cache stale, will rebuild");
        return None;
    }

    let data = fs::read(cache_path).ok()?;
    let mut index: SkillIndex = bincode::deserialize(&data).ok()?;
    // Rebuild the skip-serialized secondary index
    index.build_name_index();
    info!("Loaded {} skills from bincode cache", index.skills.len());
    Some(index)
}

/// Write the post-processed SkillIndex to a bincode cache file (best-effort).
pub(crate) fn write_bincode_cache(cache_path: &Path, index: &SkillIndex) {
    match bincode::serialize(index) {
        Ok(data) => {
            // Atomic write: write to .tmp then rename
            let tmp_path = cache_path.with_extension("bin.tmp");
            if fs::write(&tmp_path, &data).is_ok() {
                let _ = fs::rename(&tmp_path, cache_path);
                info!("Wrote bincode cache ({} bytes)", data.len());
            }
        }
        Err(e) => {
            warn!("Failed to serialize bincode cache: {}", e);
        }
    }
}

// ============================================================================
// Domain Registry Loading
// ============================================================================

/// Get the path to the domain registry file.
/// Resolution order: --registry CLI flag > PSS_REGISTRY_PATH env var > ~/.claude/cache/domain-registry.json
pub(crate) fn get_registry_path(cli_registry: Option<&str>) -> Option<PathBuf> {
    // 1. CLI flag takes priority
    if let Some(path) = cli_registry {
        return Some(PathBuf::from(path));
    }

    // 2. Environment variable fallback
    if let Ok(path) = std::env::var("PSS_REGISTRY_PATH") {
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }

    // 3. Default: ~/.claude/cache/domain-registry.json
    let home = dirs::home_dir()?;
    Some(home.join(".claude").join(CACHE_DIR).join(REGISTRY_FILE))
}

/// Load and parse the domain registry. Returns None if registry doesn't exist
/// (domain gates will not be enforced). Returns Err only on parse failure.
pub(crate) fn load_domain_registry(path: &PathBuf) -> Result<Option<DomainRegistry>, SuggesterError> {
    if !path.exists() {
        debug!("Domain registry not found at {:?}, domain gates will not be enforced", path);
        return Ok(None);
    }

    let content = fs::read_to_string(path).map_err(|e| SuggesterError::IndexRead {
        path: path.clone(),
        source: e,
    })?;

    let registry: DomainRegistry =
        serde_json::from_str(&content).map_err(|e| SuggesterError::IndexParse(
            format!("Failed to parse domain registry: {}", e)
        ))?;

    // A registry that parses but carries ZERO domains is the worst of both worlds and
    // must never be used silently. `detect_domains_from_prompt_with_context` iterates
    // `registry.domains`, so an empty map yields an empty `detected_domains`, and
    // `check_domain_gates` then takes its `if !domain_detected` branch for EVERY gated
    // entry — excluding all of them. PSS returns a near-empty suggestion block and
    // nothing anywhere says why.
    //
    // This is a real on-disk state, not a hypothetical: a stale 234-byte
    // `domain-registry.json` with `"domain_count": 0` sits in the plugin data dir
    // (2026-07-16). It is inert today only because `get_registry_path()` resolves to
    // ~/.claude/cache/ while the DB resolves via `get_data_dir()` — align those two
    // and every gated element vanishes from suggestions.
    //
    // Degrade the way the rest of the hot path does (warn loudly, keep working) rather
    // than exiting: this runs inside UserPromptSubmit, where a hard failure would take
    // the user's prompt down with it. Treating it as ABSENT restores the documented
    // "gates not enforced" behaviour, which is over-permissive but visible, instead of
    // silently suppressing every gated suggestion.
    if registry.domains.is_empty() {
        eprintln!(
            "PSS WARNING: domain registry at {:?} contains ZERO domains. \
             Treating it as absent — domain gates will NOT be enforced. \
             Re-run `/pss-reindex-skills` to regenerate it.",
            path
        );
        return Ok(None);
    }

    info!(
        "Loaded domain registry: {} domains from {:?}",
        registry.domains.len(),
        path
    );

    Ok(Some(registry))
}

// ============================================================================
// PSS File Loading
// ============================================================================

/// Load a single PSS file and merge it into the skill index
pub(crate) fn load_pss_file(pss_path: &PathBuf, index: &mut SkillIndex) -> Result<(), io::Error> {
    let content = fs::read_to_string(pss_path)?;
    let pss: PssFile = match serde_json::from_str(&content) {
        Ok(p) => p,
        Err(e) => {
            warn!("Failed to parse PSS file {:?}: {}", pss_path, e);
            return Ok(()); // Non-fatal, continue with other files
        }
    };

    // Check version
    if pss.version != "1.0" {
        warn!("Unsupported PSS version {} in {:?}", pss.version, pss_path);
        return Ok(());
    }

    let skill_name = &pss.skill.name;

    // Find entry by name using the secondary name index, then get mutable ref by ID
    let existing_id = index.name_to_ids.get(skill_name.as_str())
        .and_then(|ids| ids.first())
        .cloned();

    // If skill exists in index, merge PSS data
    if let Some(entry) = existing_id.as_ref().and_then(|id| index.skills.get_mut(id)) {
        // Merge keywords (add any not already present)
        for kw in &pss.matchers.keywords {
            if !entry.keywords.contains(kw) {
                entry.keywords.push(kw.clone());
            }
        }

        // Merge intents
        for intent in &pss.matchers.intents {
            if !entry.intents.contains(intent) {
                entry.intents.push(intent.clone());
            }
        }

        // Merge patterns
        for pattern in &pss.matchers.patterns {
            if !entry.patterns.contains(pattern) {
                entry.patterns.push(pattern.clone());
            }
        }

        // Merge directories
        for dir in &pss.matchers.directories {
            if !entry.directories.contains(dir) {
                entry.directories.push(dir.clone());
            }
        }

        // Set negative keywords (PSS takes precedence)
        if !pss.matchers.negative_keywords.is_empty() {
            entry.negative_keywords = pss.matchers.negative_keywords.clone();
        }

        // Set scoring hints (PSS takes precedence)
        if !pss.scoring.tier.is_empty() {
            entry.tier = pss.scoring.tier.clone();
        }
        if pss.scoring.boost != 0 {
            entry.boost = pss.scoring.boost;
        }
        if !pss.scoring.category.is_empty() {
            entry.category = pss.scoring.category.clone();
        }

        debug!(
            "Merged PSS data for skill '{}': {} keywords, tier={}, boost={}",
            skill_name,
            entry.keywords.len(),
            entry.tier,
            entry.boost
        );
    } else {
        // Skill not in index - create new entry from PSS
        let skill_md_path = pss_path.with_file_name("SKILL.md");
        let path = if skill_md_path.exists() {
            skill_md_path.to_string_lossy().to_string()
        } else if !pss.skill.path.is_empty() {
            pss.skill.path.clone()
        } else {
            pss_path.parent().unwrap_or(pss_path).to_string_lossy().to_string()
        };

        let source = pss.skill.source.clone();
        let entry_id = make_entry_id(skill_name, &source);
        let entry = SkillEntry {
            name: skill_name.clone(),
            source,
            path,
            // A hand-authored .pss file describes matchers, not provenance —
            // there is no plugin manifest behind it, so it renders bare.
            plugin: None,
            origin: None,
            skill_type: pss.skill.skill_type.clone(),
            keywords: pss.matchers.keywords.clone(),
            intents: pss.matchers.intents.clone(),
            patterns: pss.matchers.patterns.clone(),
            directories: pss.matchers.directories.clone(),
            path_patterns: vec![],
            description: String::new(),
            negative_keywords: pss.matchers.negative_keywords.clone(),
            tier: pss.scoring.tier.clone(),
            boost: pss.scoring.boost,
            category: pss.scoring.category.clone(),
            // Platform/Framework/Language metadata (empty for PSS files - populated by reindex)
            platforms: vec![],
            frameworks: vec![],
            languages: vec![],
            domains: vec![],
            tools: vec![],
            services: vec![],
            file_types: vec![],
            // Domain gates (empty for PSS files - populated by reindex)
            domain_gates: HashMap::new(),
            path_gates: Vec::new(),
            // MCP server metadata (empty for PSS files - populated by reindex)
            server_type: String::new(),
            server_command: String::new(),
            server_args: vec![],
            // LSP server metadata (empty for PSS files - populated by reindex)
            language_ids: vec![],
            // Co-usage fields (empty for PSS files - populated by reindex)
            co_usage: CoUsageData::default(),
            alternatives: vec![],
            use_cases: vec![],
            first_indexed_at: String::new(),
            last_updated_at: String::new(),
        };

        info!("Added skill '{}' from PSS file: {:?}", skill_name, pss_path);
        // Insert with entry ID as key; update secondary name index
        index.name_to_ids.entry(skill_name.clone()).or_default().push(entry_id.clone());
        index.skills.insert(entry_id, entry);
    }

    Ok(())
}

/// Discover and load all PSS files from skill directories.
pub(crate) fn load_pss_files(index: &mut SkillIndex) {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => {
            warn!("Could not get home directory for PSS file discovery");
            return;
        }
    };

    // Search locations for PSS files
    let search_paths = vec![
        home.join(".claude/skills"),
        home.join(".claude/agents"),
        home.join(".claude/commands"),
        PathBuf::from(".claude/skills"),
        PathBuf::from(".claude/agents"),
        PathBuf::from(".claude/commands"),
    ];

    let mut pss_count = 0;

    for search_path in search_paths {
        if !search_path.exists() {
            continue;
        }

        // Recursively find .pss files
        if let Ok(entries) = fs::read_dir(&search_path) {
            for entry in entries.flatten() {
                let path = entry.path();

                if path.is_dir() {
                    // Check for .pss file in skill directory
                    if let Ok(subentries) = fs::read_dir(&path) {
                        for subentry in subentries.flatten() {
                            let subpath = subentry.path();
                            // Load .pss files found in subdirectories
                            if subpath.extension().is_some_and(|e| e == "pss")
                                && load_pss_file(&subpath, index).is_ok()
                            {
                                pss_count += 1;
                            }
                        }
                    }
                } else if path.extension().is_some_and(|e| e == "pss")
                    && load_pss_file(&path, index).is_ok()
                {
                    pss_count += 1;
                }
            }
        }
    }

    if pss_count > 0 {
        info!("Loaded {} PSS files", pss_count);
    }
}


// ============================================================================
// Activation Logging
// ============================================================================

/// Get the path to the activation log file.
pub(crate) fn get_log_path() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let log_dir = home.join(".claude").join(LOG_DIR);

    // Create log directory if it doesn't exist
    if !log_dir.exists() {
        if let Err(e) = fs::create_dir_all(&log_dir) {
            warn!("Failed to create log directory {:?}: {}", log_dir, e);
            return None;
        }
    }

    Some(log_dir.join(ACTIVATION_LOG_FILE))
}


/// Calculate a simple hash of the prompt for deduplication
pub(crate) fn hash_prompt(prompt: &str) -> String {
    // FNV-1a 64-bit — deterministic across runs unlike DefaultHasher
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in prompt.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}

/// Truncate prompt for privacy while preserving meaning
pub(crate) fn truncate_prompt(prompt: &str, max_len: usize) -> String {
    if prompt.len() <= max_len {
        prompt.to_string()
    } else {
        // Find the largest valid char boundary at or before max_len
        // to avoid panicking on multi-byte UTF-8 characters (emoji, CJK, etc.)
        let mut end = max_len;
        while end > 0 && !prompt.is_char_boundary(end) {
            end -= 1;
        }
        let truncated = &prompt[..end];
        if let Some(last_space) = truncated.rfind(' ') {
            format!("{}...", &truncated[..last_space])
        } else {
            format!("{}...", truncated)
        }
    }
}

/// Log an activation event to the JSONL log file
pub(crate) fn log_activation(
    prompt: &str,
    session_id: Option<&str>,
    cwd: Option<&str>,
    subtask_count: usize,
    matches: &[MatchedSkill],
    processing_ms: Option<u64>,
) {
    // Get log path, skip logging if unavailable
    let log_path = match get_log_path() {
        Some(p) => p,
        None => {
            debug!("Activation logging disabled (no log path)");
            return;
        }
    };

    // Check if logging is disabled via environment variable
    if std::env::var("PSS_NO_LOGGING").is_ok() {
        debug!("Activation logging disabled via PSS_NO_LOGGING");
        return;
    }

    // Build log entry
    let entry = ActivationLogEntry {
        timestamp: Utc::now().to_rfc3339(),
        session_id: session_id.filter(|s| !s.is_empty()).map(String::from),
        prompt_preview: truncate_prompt(prompt, MAX_LOG_PROMPT_LENGTH),
        prompt_hash: hash_prompt(prompt),
        subtask_count,
        cwd: cwd.filter(|s| !s.is_empty()).map(String::from),
        matches: matches
            .iter()
            .map(|m| ActivationMatch {
                name: m.name.clone(),
                skill_type: m.skill_type.clone(),
                score: m.score,
                confidence: m.confidence.as_str().to_string(),
                evidence: m.evidence.clone(),
            })
            .collect(),
        processing_ms,
    };

    // Serialize to JSON line
    let json_line = match serde_json::to_string(&entry) {
        Ok(j) => j,
        Err(e) => {
            warn!("Failed to serialize activation log: {}", e);
            return;
        }
    };

    // Append to log file
    let result = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .and_then(|mut file| writeln!(file, "{}", json_line));

    match result {
        Ok(_) => debug!("Logged activation to {:?}", log_path),
        Err(e) => warn!("Failed to write activation log: {}", e),
    }

    // Check for log rotation (non-blocking, best effort)
    rotate_log_if_needed(&log_path);
}

/// Rotate log file if it exceeds MAX_LOG_ENTRIES
pub(crate) fn rotate_log_if_needed(log_path: &PathBuf) {
    // Count lines (fast approximation using file size)
    let metadata = match fs::metadata(log_path) {
        Ok(m) => m,
        Err(_) => return,
    };

    // Approximate: assume ~500 bytes per entry on average
    let estimated_entries = metadata.len() / 500;

    if estimated_entries < MAX_LOG_ENTRIES as u64 {
        return;
    }

    info!("Rotating activation log (estimated {} entries)", estimated_entries);

    // Create backup filename with timestamp
    let backup_name = format!(
        "pss-activations-{}.jsonl",
        Utc::now().format("%Y%m%d-%H%M%S")
    );
    let backup_path = log_path.with_file_name(&backup_name);

    // Rename current log to backup
    if let Err(e) = fs::rename(log_path, &backup_path) {
        warn!("Failed to rotate log file: {}", e);
        return;
    }

    info!("Rotated log to {:?}", backup_path);

    // Clean up old backups (keep last 5)
    if let Some(log_dir) = log_path.parent() {
        let mut backups: Vec<_> = fs::read_dir(log_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| {
                e.path()
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("pss-activations-") && n.ends_with(".jsonl"))
                    .unwrap_or(false)
            })
            .collect();

        if backups.len() > 5 {
            // Sort by name (which includes timestamp) and remove oldest
            backups.sort_by_key(|e| e.path());
            for old_backup in backups.iter().take(backups.len() - 5) {
                if let Err(e) = fs::remove_file(old_backup.path()) {
                    warn!("Failed to remove old backup {:?}: {}", old_backup.path(), e);
                } else {
                    debug!("Removed old backup {:?}", old_backup.path());
                }
            }
        }
    }
}
