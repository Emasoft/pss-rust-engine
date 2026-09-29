//! Domain gate detection + weighted-scoring matching engine (XUD7YUZH
//! modularization step 6 — pure move, zero behavior change). Banner sections
//! "Domain Gate Detection and Filtering" and "Matching Logic (Enhanced with
//! weighted scoring)" moved here byte-for-byte; every moved top-level item's
//! visibility widened to `pub(crate)` so it stays reachable from `main.rs`
//! via the crate-root re-export:
//!
//! ```ignore
//! mod matching;
//! pub(crate) use matching::*;
//! ```
//!
//! Referenced items from sibling modules are imported explicitly because
//! child modules do not see the crate root's glob re-exports.

use std::collections::HashSet;

use std::collections::HashMap;

use rayon::prelude::*;
use regex::Regex;
use tracing::debug;

use crate::consts::{ABSOLUTE_ANCHOR, MAX_SUGGESTIONS};
use crate::data::{DOMAIN_TAXONOMY, LOW_SIGNAL_WORDS};
use crate::scoring::{ALL_LOW_SIGNAL_CAP, ConfidenceThresholds, LOW_SIGNAL_DIVISOR, MatchWeights};
use crate::text::{
    is_abbreviation_match, is_fuzzy_match, normalize_separators, stem_word,
};
use crate::Confidence;
use crate::DomainRegistry;
use crate::SkillIndex;
use crate::ProjectContext;
use crate::marketplace_of;

// ============================================================================
// Domain Gate Detection and Filtering
// ============================================================================

/// Detected domains from the user prompt, mapped from canonical domain name
/// to the set of matching keywords found in the prompt.
pub type DetectedDomains = HashMap<String, Vec<String>>;

/// Detect which domains are relevant to the user prompt by scanning for
/// keywords from the domain registry.
///
/// Returns a map: canonical_domain_name -> [matched_keywords]
/// A domain is "detected" if at least one of its example_keywords appears in the prompt.
#[cfg(test)]
pub(crate) fn detect_domains_from_prompt(
    prompt: &str,
    registry: &DomainRegistry,
) -> DetectedDomains {
    detect_domains_from_prompt_with_context(prompt, registry, &[])
}

/// Detect domains from prompt text AND project context signals.
///
/// Two sources of domain detection:
/// 1. Keyword matches in the prompt text (primary)
/// 2. Project context signals from the hook (languages, frameworks, platforms, tools)
///    that match domain registry keywords. This ensures that e.g. a project with
///    Objective-C files triggers the "programming_language" or "target_language" domain
///    even if the user doesn't explicitly mention it in the prompt.
pub(crate) fn detect_domains_from_prompt_with_context(
    prompt: &str,
    registry: &DomainRegistry,
    context_signals: &[String],
) -> DetectedDomains {
    let prompt_lower = prompt.to_lowercase();
    let mut detected: DetectedDomains = HashMap::new();

    // Build a combined text for matching: prompt + context signals
    // Context signals are lowercased tokens from project analysis
    let context_lower: Vec<String> = context_signals.iter().map(|s| s.to_lowercase()).collect();

    for (canonical_name, domain_entry) in &registry.domains {
        let mut matched_keywords: Vec<String> = Vec::new();

        for keyword in &domain_entry.example_keywords {
            // Skip the "generic" meta-keyword — it's not a detection keyword
            if keyword == "generic" {
                continue;
            }

            let kw_lower = keyword.to_lowercase();

            // Source 1: keyword found in prompt text (substring match)
            if prompt_lower.contains(&kw_lower) {
                matched_keywords.push(keyword.clone());
                continue;
            }

            // Source 2: keyword matches a project context signal
            // This catches cases like project having objective-c files but prompt
            // not mentioning it — the domain is still relevant
            if context_lower.iter().any(|ctx| ctx.contains(&kw_lower) || kw_lower.contains(ctx.as_str())) {
                matched_keywords.push(format!("ctx:{}", keyword));
            }
        }

        if !matched_keywords.is_empty() {
            debug!(
                "Domain '{}' detected via keywords: {:?}",
                canonical_name, matched_keywords
            );
            detected.insert(canonical_name.clone(), matched_keywords);
        }
    }

    detected
}

/// Map a programming-language name to the file extensions it typically uses.
///
/// Used by `check_path_gates()` so a rule with `paths: ["**/*.py"]` matches a
/// Python project even when `context.file_types` is empty. PSS's project scanner
/// does NOT add source-language extensions to `file_types` — languages are
/// detected via manifest files (`pyproject.toml`, `Cargo.toml`, `package.json`),
/// so the project ends up with `languages: ["python"]` but `file_types: []`.
/// Without this mapping, path gates would silently exclude rules from their
/// intended target projects.
pub(crate) fn language_to_extensions(language: &str) -> &'static [&'static str] {
    match language.to_lowercase().as_str() {
        "python" => &["py", "pyi", "pyx", "pyw", "ipynb"],
        "rust" => &["rs"],
        "javascript" | "js" => &["js", "mjs", "cjs", "jsx"],
        "typescript" | "ts" => &["ts", "tsx", "mts", "cts"],
        "go" | "golang" => &["go"],
        "java" => &["java"],
        "kotlin" => &["kt", "kts"],
        "swift" => &["swift"],
        "objective-c" | "objc" => &["m", "mm"],
        "c" => &["c", "h"],
        "cpp" | "c++" => &["cpp", "cc", "cxx", "hpp", "hxx", "h"],
        "csharp" | "c#" | "cs" => &["cs"],
        "ruby" => &["rb", "erb"],
        "php" => &["php", "phtml"],
        "scala" => &["scala", "sc"],
        "elixir" => &["ex", "exs"],
        "erlang" => &["erl", "hrl"],
        "haskell" => &["hs", "lhs"],
        "ocaml" => &["ml", "mli"],
        "fsharp" | "f#" => &["fs", "fsi", "fsx"],
        "lua" => &["lua"],
        "perl" => &["pl", "pm", "t"],
        "r" => &["r", "rmd"],
        "julia" => &["jl"],
        "dart" => &["dart"],
        "clojure" => &["clj", "cljs", "cljc", "edn"],
        "shell" | "bash" | "sh" => &["sh", "bash", "zsh", "fish"],
        "powershell" => &["ps1", "psm1", "psd1"],
        "zig" => &["zig"],
        "nim" => &["nim", "nims"],
        "crystal" => &["cr"],
        "solidity" => &["sol"],
        "vyper" => &["vy"],
        "html" => &["html", "htm"],
        "css" => &["css"],
        "scss" | "sass" => &["scss", "sass"],
        "less" => &["less"],
        "svelte" => &["svelte"],
        "vue" => &["vue"],
        "astro" => &["astro"],
        "sql" => &["sql"],
        _ => &[],
    }
}

/// Check whether a rule's `paths:` activation globs align with the current project.
///
/// Returns `true` when the rule should be considered (pass), `false` when it
/// should be excluded. Semantics:
/// - **Empty `path_gates`**: always passes (rule has no path scope).
/// - **Extension-based globs** (e.g. `**/*.py`, `*.rs`): extract the trailing
///   extension and check whether the project's `file_types` OR the extensions
///   derived from the project's `languages` (via [`language_to_extensions`])
///   contain it. This fixes the common case where PSS detects a Python project
///   via `pyproject.toml` but doesn't add `py` to `file_types`.
/// - **Non-extension globs** (e.g. `src/**`, `Dockerfile*`): permissively pass,
///   because PSS currently doesn't do full cwd glob walking.
///
/// Matching is case-insensitive. Exact glob matching against cwd paths would
/// require a glob crate and is left as future work.
pub(crate) fn check_path_gates(
    path_gates: &[String],
    file_types: &[String],
    languages: &[String],
) -> bool {
    if path_gates.is_empty() {
        return true;
    }
    // Build the project's effective extension set: raw file_types plus all
    // extensions inferred from the project's detected languages.
    let mut ext_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for ft in file_types {
        ext_set.insert(ft.to_lowercase());
    }
    for lang in languages {
        for ext in language_to_extensions(lang) {
            ext_set.insert((*ext).to_string());
        }
    }
    let mut has_extractable_ext = false;
    for glob in path_gates {
        if let Some(dot) = glob.rfind('.') {
            let ext = glob[dot + 1..]
                .trim_matches(|c: char| !c.is_ascii_alphanumeric())
                .to_lowercase();
            if !ext.is_empty() && ext.chars().all(|c| c.is_ascii_alphanumeric()) {
                has_extractable_ext = true;
                if ext_set.contains(&ext) {
                    return true;
                }
            }
        }
    }
    // If no gate was extension-parseable, pass permissively — the glob is
    // path-based (e.g. `src/**`) and we don't do full cwd globbing yet.
    !has_extractable_ext
}

/// Check whether a single skill passes ALL its domain gates.
///
/// Gate logic:
/// - For each gate in the skill's domain_gates:
///   1. Normalize the gate name to canonical form (already done at index time)
///   2. Check if the canonical domain was detected from the prompt
///   3. If the gate contains "generic": passes if domain is detected (any keyword)
///   4. Otherwise: passes if at least one gate keyword appears in the prompt
/// - ALL gates must pass. If any gate fails, the skill is filtered out.
///
/// Returns (passes, failed_gate_name) — passes=true means all gates OK.
pub(crate) fn check_domain_gates(
    skill_name: &str,
    domain_gates: &HashMap<String, Vec<String>>,
    detected_domains: &DetectedDomains,
    prompt_lower: &str,
    registry: &DomainRegistry,
) -> (bool, Option<String>) {
    // Skills with no domain gates always pass
    if domain_gates.is_empty() {
        return (true, None);
    }

    for (gate_name, gate_keywords) in domain_gates {
        // The gate_name in skill-index.json should already be canonical
        // (set by haiku during indexing), but we do a lookup in the registry
        // to handle any aliases
        let canonical_name = find_canonical_domain(gate_name, registry);

        // Check if this domain was detected from the prompt
        let domain_detected = detected_domains.contains_key(&canonical_name);

        if !domain_detected {
            // Domain not detected in prompt at all — gate fails
            debug!(
                "Skill '{}' gate '{}' (canonical: '{}'): domain NOT detected in prompt → FAIL",
                skill_name, gate_name, canonical_name
            );
            return (false, Some(gate_name.clone()));
        }

        // Domain was detected. Now check if the specific gate keywords match.
        let has_generic = gate_keywords.iter().any(|kw| kw.to_lowercase() == "generic");

        if has_generic {
            // "generic" wildcard: domain detected = gate passes
            debug!(
                "Skill '{}' gate '{}': domain detected + generic wildcard → PASS",
                skill_name, gate_name
            );
            continue;
        }

        // Check if any gate keyword appears in the prompt as a WHOLE TOKEN.
        //
        // Substring matching is wrong here, and silently so: synonym expansion
        // injects index element NAMES into the prompt text, so "write rust tests"
        // expands to a string containing the literal "python-test-writer" — which
        // CONTAINS "python" and therefore satisfied that agent's
        // {programming_language: [python, shell]} gate on a Rust prompt. The gate
        // was admitting the very element whose name it was matching against.
        //
        // Splitting on whitespace ONLY (never on '-') keeps "python-test-writer" a
        // single token that cannot equal "python". Synonym expansion still satisfies
        // gates, because every expansion is appended as its own whitespace-separated
        // token. Inner '-', '+' and '#' are preserved so "c++", "c#" and
        // "objective-c" remain matchable.
        //
        // NB: the W5 comment at the call site claims "ts" expands to "typescript";
        // measured 2026-08-02, it does not — "write ts tests" expands without any
        // "typescript" token, so typescript-gated entries are excluded under BOTH
        // the old substring and this token match. That is a separate recall gap in
        // the synonym table, not something this comparison can fix.
        let gate_passes = gate_keywords.iter().any(|kw| {
            let kw_l = kw.to_lowercase();
            if kw_l.contains(' ') {
                // Multi-word keyword ("react native") has no single-token form.
                prompt_lower.contains(&kw_l)
            } else {
                prompt_lower.split_whitespace().any(|w| {
                    w.trim_matches(|c: char| {
                        !c.is_alphanumeric() && c != '-' && c != '+' && c != '#'
                    }) == kw_l
                })
            }
        });

        if !gate_passes {
            debug!(
                "Skill '{}' gate '{}': domain detected but no gate keyword matched ({:?}) → FAIL",
                skill_name, gate_name, gate_keywords
            );
            return (false, Some(gate_name.clone()));
        }

        debug!(
            "Skill '{}' gate '{}': gate keyword matched → PASS",
            skill_name, gate_name
        );
    }

    (true, None)
}

/// Find the canonical domain name for a gate name.
/// Checks the registry domains and their aliases.
/// Falls back to the gate name itself if no match found.
pub(crate) fn find_canonical_domain(gate_name: &str, registry: &DomainRegistry) -> String {
    let gate_lower = gate_name.to_lowercase();

    // Direct match on canonical name
    if registry.domains.contains_key(&gate_lower) {
        return gate_lower;
    }

    // Check aliases in each domain
    for (canonical_name, entry) in &registry.domains {
        for alias in &entry.aliases {
            if alias.to_lowercase() == gate_lower {
                return canonical_name.clone();
            }
        }
    }

    // No match found — return gate name as-is (will likely fail detection)
    gate_lower
}

// ============================================================================
// Matching Logic (Enhanced with weighted scoring)
// ============================================================================

/// A matched skill with scoring details
#[derive(Debug)]
pub(crate) struct MatchedSkill {
    pub(crate) name: String,
    pub(crate) path: String,
    pub(crate) skill_type: String,
    pub(crate) description: String,
    pub(crate) score: i32,
    pub(crate) confidence: Confidence,
    pub(crate) evidence: Vec<String>,
    /// Ownership, carried through so the emitted name can be namespaced.
    /// Without these the identity is lost at this hop and the hook can only
    /// print a bare, unattributable name.
    pub(crate) plugin: Option<String>,
    pub(crate) marketplace: Option<String>,
    pub(crate) origin: Option<String>,
}

/// Find matching skills with weighted scoring (combines reliable + LimorAI approaches)
///
/// # Arguments
/// * `original_prompt` - The original user prompt
/// Calculate position-based multiplier for a term in a skill's metadata.
/// Scores by WHERE the term appears: name (2x), description (1.5x), body/keywords (1x, capped 4x).
/// Returns total multiplier; defaults to 1.0 if term not found anywhere.
pub(crate) fn position_multiplier(term: &str, skill_name: &str, skill_desc: &str, skill_keywords: &[String]) -> f64 {
    let term_lower = term.to_lowercase();
    // Count occurrences in name (each worth 2x)
    let name_lower = skill_name.to_lowercase();
    let name_count = name_lower.matches(&term_lower).count();
    // Count occurrences in description (each worth 1.5x)
    let desc_lower = skill_desc.to_lowercase();
    let desc_count = desc_lower.matches(&term_lower).count();
    // Count keyword matches as proxy for body occurrences (each worth 1x, capped at 4)
    let body_count = skill_keywords.iter()
        .filter(|k| k.to_lowercase().contains(&term_lower))
        .count()
        .min(4);

    let multiplier = (name_count as f64 * 2.0) + (desc_count as f64 * 1.5) + (body_count as f64);
    // Default to 1.0 if term not found anywhere (shouldn't happen since we matched it)
    if multiplier < 1.0 { 1.0 } else { multiplier }
}

/// Check if a domain synonym matches in text with appropriate boundary rules.
/// Multi-word synonyms (contain space) use substring matching (low false-positive risk).
/// Single-word synonyms use word-boundary matching to prevent "art" matching "start".
pub(crate) fn synonym_matches_in_text(synonym: &str, text: &str, words: &[&str]) -> bool {
    if synonym.contains(' ') {
        text.contains(synonym)
    } else {
        words.iter().any(|w| *w == synonym)
    }
}

/// Infer domain tags from text using the shared taxonomy.
/// Scans the text against each domain's synonym group.
/// Returns the list of detected domain names (e.g., ["security", "backend"]).
pub(crate) fn infer_domains_from_text(text: &str) -> Vec<String> {
    let text_lower = text.to_lowercase();
    let words: Vec<&str> = text_lower.split_whitespace().collect();
    let mut domains = Vec::new();
    for &(domain_name, synonyms) in DOMAIN_TAXONOMY {
        if synonyms.iter().any(|syn| synonym_matches_in_text(syn, &text_lower, &words)) {
            domains.push(domain_name.to_string());
        }
    }
    domains
}

/// Locate the pss-nlp binary for NLP-based negation detection.
/// Search order (TRDD-YC51I1C0 phase 4 — the fetched store IS the production
/// path; the plugin install root is gone): 1. same dir as the current pss
/// binary, 2. $PSS_BINARY_DIR, 3. the fetched store
/// ~/.claude/cache/pss-bin/current (a CONSTANT — sh and Python pin the same
/// path; mirroring get_data_dir()'s conditional here would be a 4th copy of
/// a rule that already drifted once), 4. PATH.
pub(crate) fn find_pss_nlp_binary() -> Option<std::path::PathBuf> {
    // 1. Same directory as the current pss binary
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("pss-nlp");
            if candidate.exists() {
                return Some(candidate);
            }
            // Also check platform-specific names (bin/ directory layout)
            #[cfg(target_os = "macos")]
            {
                let candidate = dir.join("pss-nlp-darwin-arm64");
                if candidate.exists() { return Some(candidate); }
                let candidate = dir.join("pss-nlp-darwin-x86_64");
                if candidate.exists() { return Some(candidate); }
            }
            #[cfg(target_os = "linux")]
            {
                let candidate = dir.join("pss-nlp-linux-x86_64");
                if candidate.exists() { return Some(candidate); }
                let candidate = dir.join("pss-nlp-linux-arm64");
                if candidate.exists() { return Some(candidate); }
            }
        }
    }
    let platform_name = pss_platform_binary_name("pss-nlp");
    // 2. $PSS_BINARY_DIR (operator escape hatch)
    if let Ok(dir) = std::env::var("PSS_BINARY_DIR") {
        if let Some(name) = &platform_name {
            let candidate = std::path::Path::new(&dir).join(name);
            if candidate.exists() { return Some(candidate); }
        }
    }
    // 3. The fetched store — phase 3 flip: it now beats the plugin/repo copy.
    // Same constant the sh shim and the fetcher use.
    if let Some(home) = dirs::home_dir() {
        let current = home.join(".claude").join("cache").join("pss-bin").join("current");
        let candidate = current.join("pss-nlp");
        if candidate.exists() { return Some(candidate); }
        if let Some(name) = &platform_name {
            let candidate = current.join(name);
            if candidate.exists() { return Some(candidate); }
        }
    }
    // 4. Check PATH via which
    if let Ok(output) = std::process::Command::new("which").arg("pss-nlp").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(std::path::PathBuf::from(path));
            }
        }
    }
    None
}

/// The platform-specific binary filename ("pss-darwin-arm64" style) with
/// `prefix` ("pss" or "pss-nlp"), mirroring scripts/pss_paths.py::
/// detect_platform(). Returns None on an unsupported platform, where the
/// caller falls back to the bare "pss-nlp" name it already probes.
pub(crate) fn pss_platform_binary_name(prefix: &str) -> Option<String> {
    let (system, machine) = {
        #[cfg(target_os = "macos")]
        { ("darwin", std::env::consts::ARCH) }
        #[cfg(target_os = "linux")]
        { ("linux", std::env::consts::ARCH) }
        #[cfg(target_os = "windows")]
        { ("windows", "x86_64") }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        { return None }
    };
    let arch = match machine {
        "aarch64" => "arm64",
        "x86_64" | "amd64" => "x86_64",
        _ => return None,
    };
    if system == "windows" {
        Some(format!("{prefix}-windows-{arch}.exe"))
    } else {
        Some(format!("{prefix}-{system}-{arch}"))
    }
}

/// Deadline on the pss-nlp subprocess call (TRDD-AXZAXMDQ). The hook fires on
/// every UserPromptSubmit, so a wedged child must never block the prompt.
pub(crate) const PSS_NLP_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);

/// Cap on the text sent to pss-nlp over its stdin pipe. The write at the call
/// site is synchronous and happens BEFORE wait_child_with_deadline, so it is
/// outside the deadline: a request larger than the OS pipe buffer (~64 KiB)
/// written to a child that is wedged before reading stdin would block the
/// parent in the write itself and hang the prompt hook. Keeping the request
/// far below the pipe buffer guarantees the write always fits in the buffer
/// and returns immediately regardless of child state. Negation phrases live
/// in normal-length prompts; a 100 KB paste gains nothing from full coverage.
///
/// 2 KiB, not more, because BOTH bounds below are worst-case:
/// - JSON escaping inflates up to 6x (control chars become \uXXXX), so 2 KiB
///   of text can serialize to ~12 KiB;
/// - a macOS pipe starts at 16 KiB and only grows opportunistically — under
///   pipe-memory pressure it stays 16 KiB.
/// 12 KiB < 16 KiB keeps the write non-blocking even when both worst cases
/// stack. The test bounds the SERIALIZED request, not the raw text.
pub(crate) const PSS_NLP_MAX_TEXT_BYTES: usize = 2 * 1024;

/// Wait for `child` up to `timeout`; on expiry (or a wait error) kill and reap
/// it, returning None. Poll-based (`try_wait`) so no extra crate is needed.
/// A child that fills the stdout pipe and blocks also hits the deadline and is
/// killed — fail-safe, matching the caller's silent-skip contract.
pub(crate) fn wait_child_with_deadline(
    mut child: std::process::Child,
    timeout: std::time::Duration,
) -> Option<std::process::Output> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return child.wait_with_output().ok(),
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait(); // reap — no zombie left behind
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

/// Truncate the text destined for pss-nlp's stdin at a char boundary, capped
/// at PSS_NLP_MAX_TEXT_BYTES — see that constant for why this must never be
/// removed (an uncapped write can block outside the deadline).
pub(crate) fn cap_nlp_text(prompt: &str) -> &str {
    let mut end = PSS_NLP_MAX_TEXT_BYTES.min(prompt.len());
    while !prompt.is_char_boundary(end) {
        end -= 1;
    }
    &prompt[..end]
}

/// Call the pss-nlp binary to detect negated terms in a prompt.
/// Returns a set of lowercase negated terms, or empty set if pss-nlp is unavailable.
/// This is the key integration point between PSS scoring and NLP-based negation detection.
pub(crate) fn detect_prompt_negations(prompt: &str) -> std::collections::HashSet<String> {
    let mut result = std::collections::HashSet::new();

    let binary = match find_pss_nlp_binary() {
        Some(b) => b,
        None => {
            debug!("pss-nlp binary not found, skipping NLP negation detection");
            return result;
        }
    };

    let capped = cap_nlp_text(prompt);

    // Build JSON request for prompt-mode negation detection
    let request = serde_json::json!({
        "mode": "prompt",
        "text": capped
    });

    // Call pss-nlp as subprocess, bounded by PSS_NLP_TIMEOUT (500ms) below
    let child = std::process::Command::new(&binary)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn();

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            debug!("Failed to spawn pss-nlp: {}", e);
            return result;
        }
    };

    // Write request to stdin
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        let _ = writeln!(stdin, "{}", request);
        // Drop stdin to signal EOF, which triggers pss-nlp to process and respond
    }

    // Wait for the child under a real deadline: a wedged pss-nlp is killed and
    // negation detection silently skipped, so the prompt hook can never hang.
    let output = match wait_child_with_deadline(child, PSS_NLP_TIMEOUT) {
        Some(o) => o,
        None => {
            debug!("pss-nlp timed out or failed within {:?}, killed; skipping NLP negation detection", PSS_NLP_TIMEOUT);
            return result;
        }
    };

    if !output.status.success() {
        debug!("pss-nlp exited with non-zero status");
        return result;
    }

    // Parse JSON response: {"negated_terms": ["react", "angular"], "patterns": [...]}
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(terms) = json.get("negated_terms").and_then(|v| v.as_array()) {
                for term in terms {
                    if let Some(s) = term.as_str() {
                        result.insert(s.to_lowercase());
                    }
                }
            }
        }
    }

    if !result.is_empty() {
        debug!("NLP negation detected terms: {:?}", result);
    }

    result
}

/// * `expanded_prompt` - The prompt after synonym expansion
/// * `index` - The skill index to search
/// * `cwd` - Current working directory for directory context matching
/// * `context` - Project context for platform/framework/language filtering
/// * `incomplete_mode` - If true, skip co_usage boosts (for Pass 2 candidate finding)
#[allow(clippy::too_many_arguments)]
pub(crate) fn find_matches(
    original_prompt: &str,
    expanded_prompt: &str,
    index: &SkillIndex,
    cwd: &str,
    context: &ProjectContext,
    incomplete_mode: bool,
    detected_domains: &DetectedDomains,
    registry: Option<&DomainRegistry>,
) -> Vec<MatchedSkill> {
    let weights = MatchWeights::default();
    let thresholds = ConfidenceThresholds::default();

    // Normalize prompt words: strip trailing punctuation, deduplicate, cap at 100 words.
    // Punctuation stripping: "bun." → "bun" so keyword matching works at sentence ends.
    // Dedup + cap: prevents O(words × skills × keywords) explosion when users paste
    // large code blocks. 1000 repeated "function" = 1000× keyword comparisons per skill;
    // dedup reduces to 1×. 100 unique words is sufficient for intent detection.
    // Without this, 4000-char prompts with repetitive code take 2-4s (linear scaling).
    let original_lower: String = {
        let mut seen = std::collections::HashSet::new();
        original_prompt.to_lowercase()
            .split_whitespace()
            .map(|w| w.trim_end_matches(|c: char| c.is_ascii_punctuation()))
            .filter(|w| !w.is_empty() && seen.insert(w.to_string()))
            .take(100)
            .collect::<Vec<_>>()
            .join(" ")
    };
    let expanded_lower: String = {
        let mut seen = std::collections::HashSet::new();
        expanded_prompt.to_lowercase()
            .split_whitespace()
            .map(|w| w.trim_end_matches(|c: char| c.is_ascii_punctuation()))
            .filter(|w| !w.is_empty() && seen.insert(w.to_string()))
            .take(150) // expanded is larger due to synonym expansion
            .collect::<Vec<_>>()
            .join(" ")
    };

    // LANGUAGE/FRAMEWORK CONFLICT GATE — prompt-level detection
    // Detect languages mentioned in the prompt so we can reject entries with conflicting languages.
    // e.g., "Write Go unit tests" → detect "go" → reject entries with languages: ["python"]
    let lang_signal_patterns: &[(&str, &str)] = &[
        ("swift", "swift"), ("swiftui", "swift"), ("uikit", "swift"),
        ("xcode", "swift"), ("xctest", "swift"), ("storekit", "swift"),
        ("spritekit", "swift"), ("realitykit", "swift"), ("arkit", "swift"),
        ("mapkit", "swift"), ("cloudkit", "swift"), ("widgetkit", "swift"),
        ("appkit", "swift"), ("axiom-", "swift"), ("core-data", "swift"),
        ("ios-", "swift"), ("swiftdata", "swift"), ("app-store", "swift"),
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
        ("gopls", "go"), ("goroutine", "go"),
        ("spring-", "java"), ("quarkus", "java"), ("micronaut", "java"),
        ("flutter", "dart"), ("dart", "dart"),
        ("blazor", "c#"), ("dotnet", "c#"), ("aspnet", "c#"),
        ("laravel", "php"), ("drupal", "php"), ("wordpress", "php"),
        ("phoenix", "elixir"),
    ];

    // Compatible language groups (don't conflict with each other)
    let compatible_lang_groups: &[&[&str]] = &[
        &["javascript", "typescript"],
        &["java", "kotlin"],
        &["c", "cpp"],
        &["shell", "bash"],
    ];

    // Detect languages from prompt words (exact word match for language names)
    let prompt_words_for_lang: Vec<&str> = original_lower.split_whitespace().collect();
    let mut prompt_langs: std::collections::HashSet<String> = std::collections::HashSet::new();
    let direct_lang_names: &[&str] = &[
        "python", "javascript", "typescript", "rust", "go", "swift", "kotlin",
        "java", "ruby", "php", "dart", "elixir", "c#", "csharp", "cpp", "c++",
        "shell", "bash", "lua", "scala", "haskell", "perl",
    ];
    for &lang in direct_lang_names {
        if prompt_words_for_lang.iter().any(|w| *w == lang) {
            prompt_langs.insert(lang.to_string());
        }
    }
    // Language abbreviations and file extensions -> canonical language.
    //
    // These CANNOT go in lang_signal_patterns below: that table uses `contains`,
    // which would fire "ts" on "tests"/"artifacts" and "py" on "numpy" — silently
    // marking a Rust or Python prompt as TypeScript. Matched here as a whole word
    // (after trimming surrounding punctuation, so "ts." and "ts," still count) or
    // as a trailing file extension, so "write ts tests" and "fix main.ts" both
    // resolve while "tests" never does.
    //
    // Without this, the LANGUAGE CONFLICT GATE saw `prompt langs: empty` for
    // "write ts tests" and excluded every typescript-tagged entry, so PSS returned
    // an empty suggestion block (measured 2026-08-02).
    let lang_abbreviations: &[(&str, &str)] = &[
        ("ts", "typescript"),
        ("js", "javascript"),
        ("py", "python"),
        ("rs", "rust"),
    ];
    for &(abbr, lang) in lang_abbreviations {
        let dot_ext = format!(".{}", abbr);
        if prompt_words_for_lang.iter().any(|w| {
            let w = w.trim_matches(|c: char| !c.is_alphanumeric());
            w == abbr || w.ends_with(&dot_ext)
        }) {
            prompt_langs.insert(lang.to_string());
        }
    }
    // Tool/framework-implied languages from prompt (e.g., "django" → python)
    for &(signal, lang) in lang_signal_patterns {
        if original_lower.contains(signal) {
            prompt_langs.insert(lang.to_string());
        }
    }
    // Include project-context languages (from file scan)
    for lang in &context.languages {
        prompt_langs.insert(lang.to_lowercase());
    }
    // Expand with compatible languages (e.g., javascript ↔ typescript)
    let mut expanded_prompt_langs: std::collections::HashSet<String> = prompt_langs.clone();
    for &group in compatible_lang_groups {
        if group.iter().any(|g| prompt_langs.contains(*g)) {
            for &member in group {
                expanded_prompt_langs.insert(member.to_string());
            }
        }
    }

    // PLATFORM SIGNAL DETECTION — detect platform mentions from prompt text
    // Maps tool/framework/keyword signals to platform identifiers.
    // Used for binary exclusion of platform-locked skills when prompt has no matching signal.
    let platform_signal_patterns: &[(&str, &str)] = &[
        // iOS / Apple
        ("swiftui", "ios"), ("uikit", "ios"), ("xcode", "ios"),
        ("xctest", "ios"), ("storekit", "ios"), ("spritekit", "ios"),
        ("realitykit", "ios"), ("arkit", "ios"), ("mapkit", "ios"),
        ("cloudkit", "ios"), ("widgetkit", "ios"), ("appkit", "macos"),
        ("core-data", "ios"), ("swiftdata", "ios"), ("app-store", "ios"),
        ("cocoapods", "ios"), ("testflight", "ios"), ("app store", "ios"),
        ("ios-", "ios"), ("ios ", "ios"), ("iphone", "ios"), ("ipad", "ios"),
        ("watchos", "ios"), ("tvos", "ios"), ("visionos", "ios"),
        ("apple", "ios"), ("macos", "macos"), ("mac os", "macos"),
        ("swift package", "ios"), ("spm", "ios"),
        // Android
        ("android", "android"), ("jetpack", "android"), ("kotlin-", "android"),
        ("gradle", "android"), ("android-studio", "android"),
        ("google-play", "android"), ("play store", "android"),
        ("compose-multiplatform", "android"), ("material-design", "android"),
        // Web (universal — not a filter, just detection)
        ("web-app", "web"), ("webapp", "web"), ("browser", "web"),
        ("pwa", "web"), ("service-worker", "web"),
        // SPA frameworks imply the web platform. Without these the
        // PLATFORM CONFLICT GATE silently excludes web-tagged skills
        // when the user prompts something like "build a react component"
        // (no explicit "web" / "browser" word). See TRDD-014bcc92.
        ("react", "web"), ("vue", "web"), ("angular", "web"),
        ("svelte", "web"), ("preact", "web"), ("solid", "web"),
        ("ember", "web"), ("astro", "web"), ("remix", "web"),
        ("nextjs", "web"), ("next.js", "web"),
        ("nuxt", "web"), ("nuxt.js", "web"),
        ("gatsby", "web"), ("vite", "web"), ("webpack", "web"),
        ("jsx", "web"), ("tsx", "web"),
        ("html", "web"), ("css", "web"), ("dom", "web"),
        // Desktop
        ("electron", "desktop"), ("tauri", "desktop"),
        ("qt", "desktop"), ("gtk", "desktop"), ("wxwidgets", "desktop"),
        // Linux
        ("systemd", "linux"), ("systemctl", "linux"),
        ("apt-get", "linux"), ("dpkg", "linux"), ("yum", "linux"),
        ("pacman", "linux"), ("snap", "linux"), ("flatpak", "linux"),
        // Windows
        ("powershell", "windows"), ("winforms", "windows"),
        ("wpf", "windows"), ("uwp", "windows"), ("winui", "windows"),
        ("msvc", "windows"), (".net-framework", "windows"),
        ("windows-service", "windows"), ("registry", "windows"),
    ];

    // Compatible platform groups (don't conflict with each other)
    let compatible_platform_groups: &[&[&str]] = &[
        &["ios", "macos"],           // Apple ecosystem
        &["web", "desktop"],         // Web apps run on desktop too
        &["linux", "macos"],         // Unix-like
    ];

    // Detect platforms from prompt words (exact word match for platform names)
    let mut prompt_platforms: std::collections::HashSet<String> = std::collections::HashSet::new();
    let direct_platform_names: &[&str] = &[
        "ios", "android", "macos", "linux", "windows", "web",
        "desktop", "mobile", "watchos", "tvos", "visionos",
    ];
    for &plat in direct_platform_names {
        if prompt_words_for_lang.iter().any(|w| *w == plat) {
            prompt_platforms.insert(plat.to_string());
        }
    }
    // Tool/framework-implied platforms from prompt
    for &(signal, plat) in platform_signal_patterns {
        if original_lower.contains(signal) {
            prompt_platforms.insert(plat.to_string());
        }
    }
    // Include project-context platforms
    for plat in &context.platforms {
        prompt_platforms.insert(plat.to_lowercase());
    }
    // Expand with compatible platforms (e.g., ios ↔ macos)
    let mut expanded_prompt_platforms: std::collections::HashSet<String> = prompt_platforms.clone();
    for &group in compatible_platform_groups {
        if group.iter().any(|g| prompt_platforms.contains(*g)) {
            for &member in group {
                expanded_prompt_platforms.insert(member.to_string());
            }
        }
    }
    // "mobile" expands to ios + android
    if prompt_platforms.contains("mobile") {
        expanded_prompt_platforms.insert("ios".to_string());
        expanded_prompt_platforms.insert("android".to_string());
    }

    // Detect competing frameworks from prompt for framework conflict gating
    let competing_fw_groups: &[&[&str]] = &[
        &["react", "vue", "angular", "svelte"],
        &["django", "flask", "fastapi"],
        &["express", "fastify", "koa", "hono", "nestjs"],
        &["nextjs", "nuxt", "sveltekit", "remix"],
        &["jest", "vitest", "mocha"],
        &["pytest", "unittest"],
        &["docker", "podman"],
        &["terraform", "pulumi", "cloudformation"],
        &["flutter", "react-native", "ionic"],
        &["unity", "unreal", "godot"],
        &["pytorch", "tensorflow", "jax"],
        &["playwright", "cypress", "selenium"],
        &["spring", "quarkus", "micronaut"],
        &["rails", "sinatra"],
        &["laravel", "symfony"],
        &["webpack", "vite", "esbuild", "rollup", "parcel", "turbopack"],
    ];
    let mut conflicting_frameworks: std::collections::HashSet<String> = std::collections::HashSet::new();
    for &group in competing_fw_groups {
        let prompt_has: Vec<&&str> = group.iter()
            .filter(|fw| original_lower.contains(**fw))
            .collect();
        if !prompt_has.is_empty() {
            for &fw in group {
                if !original_lower.contains(fw) {
                    conflicting_frameworks.insert(fw.to_string());
                }
            }
        }
    }

    if incomplete_mode {
        debug!("INCOMPLETE MODE: Skipping tier boost and explicit boost fields");
    }

    // NOTE: The global domain gate early-exit (when ALL skills are gated and no
    // keyword matches) is handled in run() BEFORE this function is called. That
    // check uses a flat HashSet scan which is O(K) where K = total unique gate
    // keywords. By the time we reach this loop, at least one keyword matched or
    // some skills are ungated. Per-skill gate checks below handle individual filtering.

    // ========================================================================
    // DOMAIN INFERENCE — pre-compute prompt domains ONCE before the per-skill loop.
    // ========================================================================
    //
    // Domain taxonomy: each domain has strict synonyms (same meaning only).
    // Multi-word synonyms use substring matching.
    // Single-word synonyms use word-boundary matching to avoid false positives
    // (e.g., "art" matching "start", "poetry" matching "poetrydb-api").
    // Stemmed variants included (e.g., "translating" for "translation").
    //
    // "programming" is the default fallback. If a skill has no detectable
    // non-programming domain, it passes through. If it has one, the prompt
    // must mention that domain or the skill is excluded.
    // Use the shared DOMAIN_TAXONOMY constant (defined near synonym_matches_in_text)
    let domain_taxonomy = DOMAIN_TAXONOMY;

    // synonym_matches_in_text is defined as a top-level fn (below find_matches)
    // to work with rayon's par_iter which requires Send+Sync closures.

    // Pre-compute: detect which domains the PROMPT mentions (computed once).
    let prompt_words: Vec<&str> = expanded_lower.split_whitespace().collect();
    let prompt_detected_domains: std::collections::HashSet<&str> = domain_taxonomy.iter()
        .filter(|(_, synonyms)| {
            synonyms.iter().any(|syn| synonym_matches_in_text(syn, &expanded_lower, &prompt_words))
        })
        .map(|(domain_name, _)| *domain_name)
        .collect();

    if !prompt_detected_domains.is_empty() {
        debug!("Prompt inferred domains: {:?}", prompt_detected_domains);
    }

    // NLP-based negation detection: call pss-nlp to identify negated terms in the prompt.
    // E.g., "I don't want React" → negated_terms = {"react"}
    // E.g., "avoid frameworks like Vue and Angular" → negated_terms = {"vue", "angular"}
    // Called ONCE before the par_iter loop; result shared across all skill evaluations.
    let prompt_negated_terms = detect_prompt_negations(original_prompt);

    // Pre-compute word vectors ONCE outside par_iter to avoid per-skill allocation.
    // prompt_words is already computed above (line ~7353).
    let original_words: Vec<&str> = original_lower.split_whitespace().collect();
    let extended_prompt_words: Vec<String> = prompt_words.iter()
        .flat_map(|pw| {
            if pw.contains('-') {
                pw.split('-').map(|s| s.to_string()).collect::<Vec<_>>()
            } else {
                vec![pw.to_string()]
            }
        })
        .collect();

    // Parallel scoring: each skill scored independently across all CPU cores via rayon
    // HashMap key is entry ID (not element name); use entry.name for the element name.
    let mut matches: Vec<MatchedSkill> = index.skills.par_iter().filter_map(|(_entry_id, entry)| {
        let name = &entry.name;
        let mut score: i32 = 0;
        let mut evidence: Vec<String> = Vec::new();
        let mut keyword_matches = 0;
        // Track if ANY keyword/intent match was non-low-signal.
        // When ALL matches came from generic words like "test", "skill", "code",
        // the total score should be capped below HIGH threshold.
        let mut has_non_low_signal_match = false;
        // Track use_case and desc match counts at outer scope for cross-signal synergy bonuses
        let mut outer_uc_match_count: usize = 0;
        let mut outer_desc_match_count: usize = 0;

        // Check negative keywords first (PSS feature) - skip if any match
        let has_negative = entry.negative_keywords.iter().any(|nk| {
            let nk_lower = nk.to_lowercase();
            original_lower.contains(&nk_lower) || expanded_lower.contains(&nk_lower)
        });
        if has_negative {
            debug!("Skipping skill '{}' due to negative keyword match", name);
            return None;
        }

        // NLP-based prompt negation gate: if the user explicitly negated a term that
        // matches this skill's name, keywords, or frameworks, exclude the skill.
        // E.g., "I don't want React" → exclude react-related skills.
        // This uses pss-nlp for scope-aware negation detection (not simple string matching).
        if !prompt_negated_terms.is_empty() {
            let entry_name_lower = name.to_lowercase();
            let is_skill_negated = prompt_negated_terms.iter().any(|neg_term| {
                // Stem the negated term for fuzzy matching (e.g., "testing" → "test")
                let neg_stem = stem_word(neg_term);
                // Check skill name parts match negated term (word-boundary only, NOT substring).
                // "bun" must match "bun-development" but NOT "debug-bundle" or "esbuild-bundler".
                // Split name on delimiters and compare each part as a whole word.
                entry_name_lower.split(|c: char| c == '-' || c == '_' || c == ' ')
                    .any(|part| part == neg_term.as_str() || part == neg_stem || stem_word(part) == neg_stem)
                // Check skill keywords match negated term (exact or stem)
                || entry.keywords.iter().any(|kw| {
                    let kl = kw.to_lowercase();
                    kl == *neg_term || stem_word(&kl) == neg_stem
                })
                // Check skill frameworks match negated term (exact or stem)
                || entry.frameworks.iter().any(|fw| {
                    let fl = fw.to_lowercase();
                    fl == *neg_term || stem_word(&fl) == neg_stem
                })
                // Check description starts with or prominently features negated term
                || entry.description.to_lowercase().split_whitespace().take(5)
                    .any(|w| {
                        let clean = w.trim_matches(|c: char| !c.is_alphanumeric());
                        clean == neg_term.as_str() || stem_word(clean) == neg_stem
                    })
            });
            if is_skill_negated {
                debug!("Skipping skill '{}' — matches NLP-negated term in prompt", name);
                return None;
            }
        }

        // Rule path gates — orthogonal to domain gates, must run unconditionally.
        // A rule with frontmatter `paths:` activation globs (e.g. `["**/*.py"]`) should
        // only be suggested when the current project contains files matching at least one
        // glob's extension. This check is independent of PSS domain gates and must NOT be
        // nested under `if let Some(reg) = registry` — registry-less runs still need
        // path_gate enforcement.
        if entry.skill_type == "rule"
            && !check_path_gates(
                &entry.path_gates,
                &context.file_types,
                &context.languages,
            )
        {
            debug!(
                "Rule '{}': EXCLUDED by path_gates (no matching file types or languages in project)",
                name
            );
            return None;
        }

        // Domain gate hard pre-filter: ALL gates must pass or skill is skipped entirely.
        // This runs before scoring because failing a gate is a hard disqualification.
        if let Some(reg) = registry {
            // If a framework or tool name from this skill explicitly appears in the
            // prompt, bypass domain gates — the user is clearly talking about this
            // technology, so language/platform gates should not block it.
            // E.g. "adopt bun" should match building-with-bun even without saying "javascript".
            // Uses word-boundary matching to avoid "com" matching "comprehensive".
            let orig_words: Vec<&str> = original_lower.split_whitespace().collect();
            // Only bypass domain gates for non-low-signal tech names
            let has_explicit_tech_match = entry.frameworks.iter().any(|fw| {
                let fw_l = fw.to_lowercase();
                // Skip low-signal framework names
                if LOW_SIGNAL_WORDS.contains(fw_l.trim())
                    || LOW_SIGNAL_WORDS.contains(stem_word(fw_l.trim()).as_str())
                {
                    return false;
                }
                if fw_l.contains(' ') { original_lower.contains(&fw_l) }
                else { orig_words.iter().any(|w| *w == fw_l.as_str()) }
            }) || entry.tools.iter().any(|t| {
                let t_l = t.to_lowercase();
                // Skip low-signal tool names
                if LOW_SIGNAL_WORDS.contains(t_l.trim())
                    || LOW_SIGNAL_WORDS.contains(stem_word(t_l.trim()).as_str())
                {
                    return false;
                }
                if t_l.contains(' ') { original_lower.contains(&t_l) }
                else { orig_words.iter().any(|w| *w == t_l.as_str()) }
            });

            if !has_explicit_tech_match {
                // Use expanded prompt for gate checking (W5 fix) — synonym
                // expansions like "typescript" from "ts" should satisfy gates
                let (passes, failed_gate) = check_domain_gates(
                    name,
                    &entry.domain_gates,
                    detected_domains,
                    &expanded_lower,
                    reg,
                );
                if !passes {
                    // Domain gates are BINARY EXCLUSION: if a skill has a domain
                    // gate (e.g. geography) and the prompt doesn't match that domain,
                    // the skill is excluded entirely. A soft penalty would let
                    // high-tier matches (T5 services) overshadow domain-correct
                    // lower-tier skills. Example: geography+openai (80K after 20%
                    // penalty) would beat medicine+bun (11K) even though the prompt
                    // mentions medicine, not geography.
                    debug!(
                        "Skill '{}': EXCLUDED by domain gate '{}' (binary filter)",
                        name,
                        failed_gate.unwrap_or_default()
                    );
                    return None;
                }
            } else {
                debug!(
                    "Skill '{}': framework/tool name found in prompt, bypassing domain gates",
                    name
                );
            }
        }

        // DOMAIN INFERENCE FILTER (for skills without explicit domain_gates)
        // Uses pre-computed domain_taxonomy and prompt_detected_domains (above the loop).
        // Infers skill domain from name + description + keywords + use_cases.
        // Excludes skills whose non-programming domain is not in the prompt.
        if entry.domain_gates.is_empty() {
            // Build searchable text from skill metadata (name, description, keywords, use_cases)
            let entry_name_lower = name.to_lowercase();
            let entry_desc_lower = entry.description.to_lowercase();
            let entry_kw_text: String = entry.keywords.iter()
                .chain(entry.use_cases.iter())
                .map(|k| k.to_lowercase())
                .collect::<Vec<_>>()
                .join(" ");
            let entry_text = format!("{} {} {}", entry_name_lower, entry_desc_lower, entry_kw_text);
            let entry_words: Vec<&str> = entry_text.split_whitespace().collect();

            // Infer skill domain(s) using the pre-computed taxonomy
            let mut skill_domains: Vec<&str> = Vec::new();
            for &(domain_name, synonyms) in domain_taxonomy {
                let matches_domain = synonyms.iter().any(|syn| {
                    synonym_matches_in_text(syn, &entry_text, &entry_words)
                });
                if matches_domain {
                    skill_domains.push(domain_name);
                }
            }

            // If skill has non-programming domain(s), check if prompt mentions any of them
            let has_non_programming = skill_domains.iter().any(|d| *d != "programming");
            if has_non_programming {
                let prompt_has_matching_domain = skill_domains.iter()
                    .filter(|d| **d != "programming")
                    .any(|d| prompt_detected_domains.contains(d));

                if !prompt_has_matching_domain {
                    debug!(
                        "Skill '{}': EXCLUDED by domain inference (skill domains: {:?}, prompt domains: {:?})",
                        name, skill_domains, prompt_detected_domains
                    );
                    return None;
                }
            }

            // SOFTWARE SUB-DOMAIN FILTER: when the prompt mentions specific software
            // sub-domains (security, testing, devops, frontend, backend, etc.), also
            // check skills that are purely "programming" (no sub-domain). If the skill's
            // name+description+keywords don't mention ANY of the prompt's sub-domains,
            // it's a generic utility skill that doesn't match the agent's specialization.
            // This is what filters GitHub/utility skills from security agent profiles.
            let prompt_sub_domains: Vec<&&str> = prompt_detected_domains.iter()
                .filter(|d| **d != "programming")
                .collect();
            if !prompt_sub_domains.is_empty() && !has_non_programming {
                // Skill has no non-programming domain — it's generic programming.
                // Check if any prompt sub-domain synonym appears in the skill's text.
                let mut matches_any_sub = false;
                for &sub_domain in &prompt_sub_domains {
                    for &(domain_name, synonyms) in domain_taxonomy {
                        if domain_name == *sub_domain {
                            if synonyms.iter().any(|syn| synonym_matches_in_text(syn, &entry_text, &entry_words)) {
                                matches_any_sub = true;
                                break;
                            }
                        }
                    }
                    if matches_any_sub { break; }
                }
                if !matches_any_sub {
                    debug!(
                        "Skill '{}': EXCLUDED by sub-domain filter (no {:?} signal in skill text)",
                        name, prompt_sub_domains
                    );
                    return None;
                }
            }
        }

        // DOMAIN OVERLAP BOOST: when the skill's domains[] field overlaps with the
        // prompt's detected domains, give a significant score boost. This promotes
        // security-tagged skills for security prompts, testing-tagged for testing
        // prompts, etc. — solving the "GitHub skills dominate everything" problem
        // where generic programming skills outscored domain-specific ones.
        if !prompt_detected_domains.is_empty() && !entry.domains.is_empty() {
            let overlap_count = entry.domains.iter()
                .filter(|d| prompt_detected_domains.contains(d.as_str()))
                .count();
            if overlap_count > 0 {
                // Each domain overlap is worth 2x a keyword match — strong signal
                let domain_boost = (overlap_count as i32) * weights.keyword * 2;
                score += domain_boost;
                has_non_low_signal_match = true;
                evidence.push(format!("domain_overlap:{}", overlap_count));
            }
        }

        // Project context matching (platform/framework/language)
        // This filters out platform-specific skills that don't match the detected context
        let (context_boost, should_filter) = context.match_skill(entry);
        if should_filter {
            debug!(
                "Skipping skill '{}' due to platform mismatch (skill: {:?}, context: {:?})",
                name, entry.platforms, context.platforms
            );
            return None;
        }
        if context_boost > 0 {
            score += context_boost;
            if !context.platforms.is_empty() && !entry.platforms.is_empty() {
                evidence.push(format!("platform:{:?}", entry.platforms));
            }
            if !context.frameworks.is_empty() && !entry.frameworks.is_empty() {
                evidence.push(format!("framework:{:?}", entry.frameworks));
            }
            if !context.languages.is_empty() && !entry.languages.is_empty() {
                evidence.push(format!("lang:{:?}", entry.languages));
            }
        }

        // Directory context matching
        for dir in &entry.directories {
            if cwd.contains(dir) {
                score += weights.directory;
                has_non_low_signal_match = true;
                evidence.push(format!("dir:{}", dir));
            }
        }

        // Path pattern matching
        for path_pattern in &entry.path_patterns {
            if original_lower.contains(path_pattern) {
                score += weights.path;
                has_non_low_signal_match = true;
                evidence.push(format!("path:{}", path_pattern));
            }
        }

        // Intent (verb) matching — low-signal intents get reduced weight
        for intent in &entry.intents {
            if original_lower.contains(intent) || expanded_lower.contains(intent) {
                // Low-signal intents ("test", "run", "check", etc.) drop to common tier
                let is_low_signal_intent = LOW_SIGNAL_WORDS.contains(intent.as_str());
                let intent_score = if is_low_signal_intent {
                    weights.intent / LOW_SIGNAL_DIVISOR
                } else {
                    has_non_low_signal_match = true;
                    weights.intent
                };
                score += intent_score;
                evidence.push(format!("intent:{}", intent));
            }
        }

        // Pattern (regex) matching — patterns are specific enough to always be non-low-signal
        for pattern in &entry.patterns {
            if let Ok(re) = Regex::new(pattern) {
                if re.is_match(&original_lower) || re.is_match(&expanded_lower) {
                    score += weights.pattern;
                    has_non_low_signal_match = true;
                    evidence.push(format!("pattern:{}", pattern));
                }
            }
        }

        // Framework/tool/protocol name matching — 10x keyword weight because specific
        // names are rarely mentioned unless they are the focus of the discussion.
        // These names (e.g. "bun", "react", "ffmpeg", "graphql") are strong signals.
        // Uses word-boundary matching to avoid "com" matching "comprehensive".
        // original_words is pre-computed outside par_iter (see above)
        for fw in &entry.frameworks {
            let fw_lower = fw.to_lowercase();
            // Skip framework names that are low-signal words (shouldn't happen often,
            // but protects against poorly indexed elements)
            if LOW_SIGNAL_WORDS.contains(fw_lower.trim())
                || LOW_SIGNAL_WORDS.contains(stem_word(fw_lower.trim()).as_str())
            {
                continue;
            }
            // For multi-word frameworks, check substring; for single-word, check word boundary
            // Also check hyphen<->space normalization (W5: "github-actions" matches "github actions")
            let fw_matched = if fw_lower.contains(' ') {
                original_lower.contains(&fw_lower)
                    || original_lower.contains(&fw_lower.replace(' ', "-"))
            } else if fw_lower.contains('-') {
                // Hyphenated framework: check both "github-actions" and "github actions"
                original_words.iter().any(|w| *w == fw_lower.as_str())
                    || original_lower.contains(&fw_lower.replace('-', " "))
            } else {
                original_words.iter().any(|w| *w == fw_lower.as_str())
            };
            if fw_matched {
                // W18: Common programming languages as frameworks get reduced score
                // because "Python" matches 200+ skills, drowning out specific matches.
                // Specific frameworks like "SwiftUI", "React" keep full weight.
                static COMMON_LANGS: &[&str] = &[
                    "python", "javascript", "typescript", "java", "ruby", "go",
                    "rust", "c", "c++", "swift", "kotlin", "php", "perl",
                    "node", "node.js",
                ];
                // GitHub is extremely common as a framework (200+ skills list it).
                // It needs even stronger dampening than common languages.
                static ULTRA_COMMON: &[&str] = &["github"];
                let is_ultra_common = ULTRA_COMMON.iter().any(|u| fw_lower == *u);
                let is_common_lang = COMMON_LANGS.iter().any(|lang| fw_lower == *lang);
                let base_fw = if is_ultra_common {
                    weights.framework_match / 40  // 2.5% for ultra-common (github): 500pts
                } else if is_common_lang {
                    // FM-W3: Reduced from /5 (4000pts) to /10 (2000pts). Common
                    // language matches are too broadly distributed — "python" matches
                    // 200+ skills, drowning out domain-specific signals.
                    weights.framework_match / 10  // 10% for common languages: 2000pts
                } else {
                    weights.framework_match
                };
                // Position-based multiplier: name 2x, desc 1.5x, body 1x (cap 4x)
                let multiplier = position_multiplier(&fw_lower, name, &entry.description, &entry.keywords);
                let fw_score = ((base_fw as f64) * multiplier) as i32;
                // Cap at T4 ceiling (90000)
                score += fw_score.min(90000);
                has_non_low_signal_match = true;
                evidence.push(format!("framework:{}", fw));
            }
        }
        for tool in &entry.tools {
            let tool_lower = tool.to_lowercase();
            // Skip tool names that are low-signal words (e.g. "Skill", "Agent")
            if LOW_SIGNAL_WORDS.contains(tool_lower.trim())
                || LOW_SIGNAL_WORDS.contains(stem_word(tool_lower.trim()).as_str())
            {
                continue;
            }
            // For multi-word tools, check substring; for single-word, check word boundary
            // Also check hyphen<->space normalization (W5: "github-actions" matches "github actions")
            let tool_matched = if tool_lower.contains(' ') {
                original_lower.contains(&tool_lower)
                    || original_lower.contains(&tool_lower.replace(' ', "-"))
            } else if tool_lower.contains('-') {
                original_words.iter().any(|w| *w == tool_lower.as_str())
                    || original_lower.contains(&tool_lower.replace('-', " "))
            } else {
                original_words.iter().any(|w| *w == tool_lower.as_str())
            };
            if tool_matched {
                // W18: Common tools like "python", "bash", "git" get reduced score
                static COMMON_TOOLS: &[&str] = &[
                    "python", "python3", "bash", "git", "npm", "pip",
                    "read", "write", "edit", "grep", "node",
                    "pnpm", "yarn", "cargo", "docker", "make", "rust", "go",
                ];
                // Ultra-common tools need even stronger dampening
                static ULTRA_COMMON_TOOLS: &[&str] = &["github"];
                let is_ultra_common_tool = ULTRA_COMMON_TOOLS.iter().any(|u| tool_lower == *u);
                let is_common_tool = COMMON_TOOLS.iter().any(|t| tool_lower == *t);
                let base_tool = if is_ultra_common_tool {
                    weights.tool_match / 40  // 2.5% for ultra-common tools: 50pts
                } else if is_common_tool {
                    // FM-W3: Reduced from /5 (400pts) to /10 (200pts)
                    weights.tool_match / 10  // 10% for common tools: 200pts
                } else {
                    weights.tool_match
                };
                // Position-based multiplier: name 2x, desc 1.5x, body 1x (cap 4x)
                let multiplier = position_multiplier(&tool_lower, name, &entry.description, &entry.keywords);
                let tool_score = ((base_tool as f64) * multiplier) as i32;
                // Cap at T3 ceiling (9000)
                score += tool_score.min(9000);
                has_non_low_signal_match = true;
                evidence.push(format!("tool:{}", tool));
            }
        }

        // Service/API matching (Tier 5: 100K-900K)
        // Services are external platforms and hosted APIs (aws, openai, stripe, etc.)
        for service in &entry.services {
            let svc_lower = service.to_lowercase();
            // Skip low-signal service names
            if LOW_SIGNAL_WORDS.contains(svc_lower.trim())
                || LOW_SIGNAL_WORDS.contains(stem_word(svc_lower.trim()).as_str())
            {
                continue;
            }
            // Word-boundary + substring matching (same as frameworks/tools)
            let svc_matched = if svc_lower.contains(' ') {
                original_lower.contains(&svc_lower)
                    || original_lower.contains(&svc_lower.replace(' ', "-"))
            } else if svc_lower.contains('-') {
                original_words.iter().any(|w| *w == svc_lower.as_str())
                    || original_lower.contains(&svc_lower.replace('-', " "))
            } else {
                original_words.iter().any(|w| *w == svc_lower.as_str())
            };
            if svc_matched {
                // Common services that appear in hundreds of skills get dampened
                static ULTRA_COMMON_SVCS: &[&str] = &["github"];
                static COMMON_SVCS: &[&str] = &["npm", "git", "node"];
                let is_ultra = ULTRA_COMMON_SVCS.iter().any(|u| svc_lower == *u);
                let is_common = COMMON_SVCS.iter().any(|c| svc_lower == *c);
                let base_svc = if is_ultra {
                    weights.service_match / 20  // 5% for ultra-common
                } else if is_common {
                    weights.service_match / 5  // 20% for common services
                } else {
                    weights.service_match
                };
                // Position-based multiplier: name 2x, desc 1.5x, body 1x (cap 4x)
                let multiplier = position_multiplier(&svc_lower, name, &entry.description, &entry.keywords);
                let svc_score = ((base_svc as f64) * multiplier) as i32;
                // Cap at T5 ceiling (900000)
                score += svc_score.min(900000);
                has_non_low_signal_match = true;
                evidence.push(format!("service:{}", service));
            }
        }

        // ================================================================
        // FM-W2: NAME-IMPLIED FRAMEWORK INFERENCE
        // When a skill's name contains a known framework/tool name (e.g., "xcode"
        // in "axiom-xcode-mcp-setup") AND that framework/tool appears in the prompt,
        // but the skill didn't get an explicit framework/tool match for it, give a
        // partial framework bonus. Only fires for recognized framework/tool names
        // to avoid false positives.
        // ================================================================
        {
            static NAME_INFERABLE_FRAMEWORKS: &[&str] = &[
                "xcode", "mcp", "react", "vue", "angular", "docker", "kubernetes",
                "redis", "postgres", "mongodb", "graphql", "lldb", "metal",
                "arkit", "swiftui", "flutter", "terraform", "webpack", "vite",
            ];
            // Split skill name into parts and check for known framework names
            let name_parts_lower: Vec<String> = name.split(|c: char| c == '-' || c == ':')
                .map(|p| p.to_lowercase())
                .collect();
            // Collect which frameworks were already matched explicitly
            let explicit_fw_matched: HashSet<String> = evidence.iter()
                .filter(|e| e.starts_with("framework:") || e.starts_with("tool:"))
                .map(|e| {
                    let colon = e.find(':').unwrap_or(0);
                    e[colon+1..].to_lowercase()
                })
                .collect();
            let mut name_fw_bonus = 0i32;
            // Count how many name parts (excluding common prefixes like "axiom")
            // match words in the original prompt. Only grant the bonus if at least
            // 2 significant name parts match — this prevents a single framework word
            // like "mcp" from boosting unrelated skills (e.g., axiom-asc-mcp when
            // the prompt is only about MCP, not App Store Connect).
            static IGNORED_NAME_PREFIXES: &[&str] = &["axiom", "ecos", "eama", "eaa", "eoa"];
            let significant_name_parts: Vec<&String> = name_parts_lower.iter()
                .filter(|p| p.len() >= 3)
                .filter(|p| !IGNORED_NAME_PREFIXES.contains(&p.as_str()))
                .filter(|p| !LOW_SIGNAL_WORDS.contains(p.as_str()))
                .collect();
            let name_parts_in_prompt: usize = significant_name_parts.iter()
                .filter(|p| original_words.iter().any(|w| *w == p.as_str()))
                .count();
            // Only infer framework bonus when 2+ significant name parts match prompt
            if name_parts_in_prompt >= 2 {
                for part in &name_parts_lower {
                    if !NAME_INFERABLE_FRAMEWORKS.contains(&part.as_str()) {
                        continue;
                    }
                    // Already matched explicitly — skip
                    if explicit_fw_matched.iter().any(|ef| ef.contains(part.as_str())) {
                        continue;
                    }
                    // Check if this framework/tool name appears in the ORIGINAL prompt
                    // (not expanded, to avoid self-referential expansion loops)
                    if original_words.iter().any(|w| *w == part.as_str()) {
                        // Partial framework bonus: 40% of full framework_match (8000 pts)
                        // This bridges the gap for skills whose names imply the framework
                        // but whose index metadata omits it from the frameworks list.
                        // Lower than explicit framework match to avoid displacing
                        // skills that have correct metadata.
                        name_fw_bonus += weights.framework_match * 2 / 5;
                        has_non_low_signal_match = true;
                        evidence.push(format!("name_fw_infer:{}", part));
                    }
                }
            }
            score += name_fw_bonus;
        }

        // Keyword matching with first-match bonus (from LimorAI)
        // Also includes fuzzy matching for typo tolerance
        // Fixed: Now handles multi-word keyword phrases properly
        // Low-signal word detection: tracks if match came only from generic words
        // prompt_words is pre-computed outside par_iter (see above)

        for keyword in &entry.keywords {
            let kw_lower = keyword.to_lowercase();
            let mut matched = false;
            let mut is_fuzzy = false;
            // Track if the match was driven only by low-signal words (e.g. "test")
            let mut low_signal_match = false;

            // Phase 1: Exact substring match (keyword phrase in prompt)
            if expanded_lower.contains(&kw_lower) {
                matched = true;
                // Only flag as low-signal if the ENTIRE keyword is a single low-signal word.
                // Multi-word phrases like "unit test coverage" are specific enough to be genuine.
                // Also check stemmed form: "testing" → "test" which IS low-signal.
                let kw_words: Vec<&str> = kw_lower.split_whitespace().collect();
                let kw_trimmed = kw_lower.trim();
                low_signal_match = kw_words.len() == 1
                    && (LOW_SIGNAL_WORDS.contains(kw_trimmed)
                        || LOW_SIGNAL_WORDS.contains(stem_word(kw_trimmed).as_str()));
            }

            // Phase 2: Reverse word match (prompt words in keyword phrase)
            // This handles multi-word keywords like "bun bundler setup" when prompt is "implement bun"
            if !matched {
                // Split keyword into words for matching
                let keyword_words: Vec<&str> = kw_lower.split_whitespace().collect();

                // Check if any significant prompt word appears in the keyword phrase
                let mut matched_prompt_word: Option<&str> = None;
                for prompt_word in &prompt_words {
                    // Skip very short words (articles, prepositions)
                    if prompt_word.len() < 3 {
                        continue;
                    }
                    // Check if prompt word matches any keyword word
                    for kw_word in &keyword_words {
                        if *prompt_word == *kw_word {
                            matched = true;
                            matched_prompt_word = Some(prompt_word);
                            break;
                        }
                    }
                    if matched {
                        break;
                    }
                }
                if matched {
                    // If the matching prompt word is low-signal (e.g. "skill", "test"),
                    // penalize regardless of keyword length. A single generic word
                    // from the prompt should not drive full-score matches, even for
                    // multi-word keywords. Phase 1 exact-substring already handles
                    // genuine phrase matches at full score.
                    if let Some(pw) = matched_prompt_word {
                        // Also check stemmed form: "testing" → "test" which IS low-signal
                        low_signal_match = LOW_SIGNAL_WORDS.contains(pw)
                            || LOW_SIGNAL_WORDS.contains(stem_word(pw).as_str());
                    }
                }
            }

            // Phase 2.5: Normalized + stemmed + abbreviation matching
            // Handles separator variants (geojson / geo-json / geo_json),
            // morphological forms (deploys/deploying/deployed → deploy,
            // tests/testing → test, libraries → library),
            // and common abbreviations (config ↔ configuration, repo ↔ repository).
            if !matched {
                let kw_norm = normalize_separators(&kw_lower);
                let kw_stem = stem_word(&kw_norm);
                for prompt_word in &prompt_words {
                    if prompt_word.len() < 2 {
                        continue;
                    }
                    let pw_norm = normalize_separators(prompt_word);
                    // Normalized form match (separator variants)
                    if pw_norm == kw_norm {
                        matched = true;
                        // Check if stemmed form is a low-signal word
                        low_signal_match = LOW_SIGNAL_WORDS.contains(pw_norm.as_str());
                        break;
                    }
                    // Stemmed form match (grammatical variants)
                    let pw_stem = stem_word(&pw_norm);
                    if pw_stem == kw_stem && pw_stem.len() >= 3 {
                        matched = true;
                        // "testing" stems to "test" which is low-signal
                        low_signal_match = LOW_SIGNAL_WORDS.contains(pw_stem.as_str());
                        break;
                    }
                    // Abbreviation match (config ↔ configuration, etc.)
                    if is_abbreviation_match(&pw_norm, &kw_norm) {
                        matched = true;
                        break;
                    }
                }
            }

            // Phase 3: Fuzzy matching for typo tolerance (edit-distance, long words only)
            if !matched {
                // Split keyword into words for multi-word fuzzy matching
                let keyword_words: Vec<&str> = kw_lower.split_whitespace().collect();

                if keyword_words.len() == 1 {
                    // Single-word keyword - existing fuzzy logic
                    for word in &prompt_words {
                        if is_fuzzy_match(word, &kw_lower) {
                            matched = true;
                            is_fuzzy = true;
                            // Fuzzy match of a low-signal word (e.g. "tset" → "test")
                            low_signal_match = LOW_SIGNAL_WORDS.contains(stem_word(word).as_str());
                            break;
                        }
                    }
                } else {
                    // Multi-word keyword - match each prompt word against each keyword word
                    for prompt_word in &prompt_words {
                        if prompt_word.len() < 3 {
                            continue;
                        }
                        for kw_word in &keyword_words {
                            if is_fuzzy_match(prompt_word, kw_word) {
                                matched = true;
                                is_fuzzy = true;
                                low_signal_match = LOW_SIGNAL_WORDS.contains(stem_word(prompt_word).as_str());
                                break;
                            }
                        }
                        if matched {
                            break;
                        }
                    }
                }
            }

            if matched {
                // Low-signal matches (e.g. "test" matching "unit test coverage")
                // get drastically reduced scores to prevent false positives.
                // "test" alone should NOT trigger testing skills at HIGH confidence.
                // Low-signal matches drop from phrase tier to common tier
                let ls_divisor = if low_signal_match { LOW_SIGNAL_DIVISOR } else { 1 };
                if !low_signal_match {
                    has_non_low_signal_match = true;
                }

                if keyword_matches == 0 {
                    // First keyword gets big bonus (reduced for low-signal)
                    score += weights.first_match / ls_divisor;
                } else {
                    // Fuzzy matches get slightly less score than exact matches
                    let kw_score = if is_fuzzy {
                        (weights.keyword * 8) / 10  // 80% for fuzzy
                    } else {
                        weights.keyword
                    };
                    score += kw_score / ls_divisor;
                }
                keyword_matches += 1;

                // Original prompt bonus (not just expanded synonym match)
                if original_lower.contains(&kw_lower) {
                    score += weights.original_bonus / ls_divisor;
                    evidence.push(format!("keyword*:{}", keyword)); // * = original
                } else if is_fuzzy {
                    evidence.push(format!("keyword~:{}", keyword)); // ~ = fuzzy match
                } else {
                    evidence.push(format!("keyword:{}", keyword));
                }
            }
        }

        // ================================================================
        // WHOLE-NAME MATCHING (W18 innovation)
        // If the full skill name (hyphens->spaces OR with hyphens) appears
        // in the expanded prompt, it's a near-certain match. E.g.
        // "test failure analyzer" or "test-failure-analyzer" in prompt
        // matches skill "test-failure-analyzer" perfectly.
        // Synonym expansions add hyphenated names, so we check both forms.
        // ================================================================
        // W20: Replace both hyphens and colons with spaces for whole-name matching
        let name_as_spaces = name.replace('-', " ").replace(':', " ");
        let name_lower = name.to_lowercase();
        if name_as_spaces.len() >= 5 && (expanded_lower.contains(&name_as_spaces) || expanded_lower.contains(&name_lower)) {
            // Massive bonus for whole-name match, scaled by name length.
            // Longer names are more specific, so they deserve a bigger bonus.
            // "profiler" (1 part) gets 2000, "test-failure-analyzer" (3 parts) gets 4000.
            // W20: Split on both hyphens and colons for consistent part counting
            let name_part_count = name.split(|c: char| c == '-' || c == ':').filter(|p| p.len() >= 3).count();
            let whole_name_bonus = 2000 + (name_part_count.saturating_sub(1) as i32) * 1000;
            score += whole_name_bonus;
            has_non_low_signal_match = true;
            evidence.push("whole_name_match".to_string());
        }

        // ================================================================
        // SKILL NAME MATCHING (W4/W5 innovation, +20 hits)
        // Split skill name on hyphens, match non-low-signal name parts
        // against prompt words using word-boundary matching.
        // Also split prompt words on hyphens for matching (W18 fix).
        // ================================================================
        // W20: Split on both hyphens and colons to handle names like
        // "claude-plugin:documentation" -> ["claude", "plugin", "documentation"]
        let name_parts: Vec<&str> = name.split(|c| c == '-' || c == ':').collect();
        let significant_name_parts: Vec<&str> = name_parts.iter()
            .filter(|p| p.len() >= 3)
            .filter(|p| !LOW_SIGNAL_WORDS.contains(**p))
            .filter(|p| !LOW_SIGNAL_WORDS.contains(stem_word(p).as_str()))
            .copied()
            .collect();
        // extended_prompt_words is pre-computed outside par_iter (see above)
        let mut name_match_count = 0;
        for np in &significant_name_parts {
            let np_stem = stem_word(np);
            for pw in &extended_prompt_words {
                if pw.len() < 3 { continue; }
                let pw_stem = stem_word(pw);
                if *np == pw.as_str() || np_stem == pw_stem {
                    name_match_count += 1;
                    break;
                }
            }
        }
        let has_name_match = name_match_count > 0;
        if name_match_count > 0 {
            // W18 tuning: 200 for first match, 400+400*(n-1) for subsequent
            // Name matching is the strongest signal — gold skills average 1.8
            // name matches vs 0.4 for non-gold skills.
            let name_bonus = if name_match_count == 1 {
                200
            } else {
                400 + 400 * (name_match_count - 1)
            };
            score += name_bonus;
            has_non_low_signal_match = true;
            evidence.push(format!("name_match:{}/{}", name_match_count, significant_name_parts.len()));
        }


        // ================================================================
        // DESCRIPTION WORD MATCHING (W5/W8 innovation)
        // Extract significant words from description, match against prompt.
        // ================================================================
        if !entry.description.is_empty() {
            let desc_lower = entry.description.to_lowercase();
            let desc_words: Vec<&str> = desc_lower.split_whitespace().collect();
            let mut desc_match_count = 0;
            let desc_cap = 7;  // Max matches to count
            let desc_points = 70; // Points per match
            for dw in &desc_words {
                if dw.len() < 4 { continue; }
                if LOW_SIGNAL_WORDS.contains(*dw) || LOW_SIGNAL_WORDS.contains(stem_word(dw).as_str()) {
                    continue;
                }
                if desc_match_count >= desc_cap { break; }
                let dw_stem = stem_word(dw);
                for pw in &prompt_words {
                    if pw.len() < 4 { continue; }
                    let pw_stem = stem_word(pw);
                    if *dw == *pw || dw_stem == pw_stem {
                        desc_match_count += 1;
                        break;
                    }
                }
            }
            // Require 2+ matches to reduce false positives
            if desc_match_count >= 2 {
                let desc_bonus = desc_match_count.min(desc_cap) * desc_points;
                score += desc_bonus as i32;
                has_non_low_signal_match = true;
                evidence.push(format!("desc_match:{}", desc_match_count));
            }
            // Propagate to outer scope for cross-signal synergy
            outer_desc_match_count = desc_match_count;
        }

        // ================================================================
        // USE_CASES FIELD MATCHING (W8 innovation)
        // Match significant words from use_cases text against prompt.
        // Gold skills average 2.37 use_case matches vs 1.46 for non-gold.
        // ================================================================
        if !entry.use_cases.is_empty() {
            let uc_text: String = entry.use_cases.iter()
                .map(|uc| uc.to_lowercase())
                .collect::<Vec<_>>()
                .join(" ");
            let uc_words: Vec<&str> = uc_text.split_whitespace().collect();
            let mut uc_match_count = 0;
            let uc_cap = 5;     // Max matches to count
            let uc_points = 125; // Points per match (FM-W3: raised from 75; simulation shows +4)
            // Collect unique UC significant words to avoid double-counting
            let mut seen_uc_words: HashSet<String> = HashSet::new();
            for uw in &uc_words {
                if uw.len() < 4 { continue; }
                if LOW_SIGNAL_WORDS.contains(*uw) || LOW_SIGNAL_WORDS.contains(stem_word(uw).as_str()) {
                    continue;
                }
                let uw_stem = stem_word(uw);
                if seen_uc_words.contains(&uw_stem) { continue; }
                if uc_match_count >= uc_cap { break; }
                for pw in &prompt_words {
                    if pw.len() < 4 { continue; }
                    let pw_stem = stem_word(pw);
                    if *uw == *pw || uw_stem == pw_stem {
                        uc_match_count += 1;
                        seen_uc_words.insert(uw_stem.clone());
                        break;
                    }
                }
            }
            if uc_match_count > 0 {
                // Progressive use-case scoring: first 2 matches at base rate,
                // 3rd+ matches at premium rate (gold skills average more uc matches)
                let base_count = uc_match_count.min(2).min(uc_cap);
                let premium_count = (uc_match_count.min(uc_cap) as i32 - 2).max(0) as usize;
                let uc_bonus = base_count * uc_points + premium_count * 110;
                score += uc_bonus as i32;
                if uc_match_count >= 2 {
                    has_non_low_signal_match = true;
                }
                // FM-W3: High use_case overlap bonus. When 3+ use_case words
                // match, it indicates the skill closely matches the prompt's
                // problem domain, not just vocabulary overlap.
                if uc_match_count >= 3 {
                    // FM-W3: +300 for deep use_case overlap (3+ matching words).
                    // Simulation shows this is the most consistently positive signal.
                    score += 300;
                    evidence.push(format!("uc_depth_bonus:{}", uc_match_count));
                }
                evidence.push(format!("use_case:{}", uc_match_count));
            }
            // Propagate to outer scope for cross-signal synergy
            outer_uc_match_count = uc_match_count;
        }

        // ================================================================
        // MULTI-KEYWORD COHERENCE BONUS (W3 innovation)
        // When 2+ keywords match, it's strong evidence of relevance.
        // +50 per keyword beyond the 1st, capped at 200.
        // ================================================================
        if keyword_matches >= 2 {
            // W18: Increased cap from 200 to 400 to better differentiate
            // skills with many keyword matches (gold skills average more
            // keyword matches than non-gold in their domain).
            let coherence_bonus = ((keyword_matches - 1) * 50).min(400);
            score += coherence_bonus;
            evidence.push(format!("coherence:{}", keyword_matches));
        }

        // ================================================================
        // IDF BONUS FOR RARE KEYWORDS (W3 innovation)
        // Keywords that appear in fewer skills are more discriminative.
        // Range [0.85, 1.5]: common keywords lose max 15%, rare get 50% boost.
        // Applied as a multiplicative factor on the keyword portion of the score.
        // ================================================================
        // (Implemented implicitly via the coherence bonus and name matching
        //  rather than explicit IDF to avoid the penalty trap from Cycle 1)

        // ================================================================
        // KEYWORD ACCUMULATION DAMPING (W11 innovation)
        // Penalizes "crowder" skills that accumulate many generic keyword
        // matches without name relevance. Prevents broad skills like
        // "deployment-engineer" (14 keywords) from filling top-10 slots.
        // ================================================================
        if keyword_matches > 3 && !has_name_match {
            // Level 1: no name match, 4+ keyword matches => -60 per excess, capped at 500
            // Targets "crowder" skills like deployment-engineer (14 keywords),
            // codebase-audit-and-fix (10+ keywords) that fill top-10 slots in
            // 18+ prompts without being gold. Aggressive damping pushes them
            // below genuinely matching skills.
            // FM-W3: Reduced from 60 to 30 per excess keyword. The original
            // aggressive damping was hurting gold skills with many legitimate
            // keyword matches that lacked name_match by coincidence.
            let damping = ((keyword_matches - 3) * 30).min(250);
            score -= damping;
            evidence.push(format!("kw_damp_l1:-{}", damping));
        }
        // Level 2 damping removed: name-matched skills with many keywords are
        // legitimate matches. Penalizing them hurts gold skill recall.

        // ================================================================
        // INTENT + USE_CASE SYNERGY BONUS (W11 innovation)
        // When both intent verb and use_case text match, it's a strong
        // structural signal that generalizes well.
        // ================================================================
        {
            let has_intent_match = entry.intents.iter().any(|intent| {
                original_lower.contains(intent) || expanded_lower.contains(intent)
            });
            let has_uc_match = evidence.iter().any(|e| e.starts_with("use_case:"));
            if has_intent_match && has_uc_match {
                // FM-W3: Raised from 35 to 185. Intent+use_case synergy is a
                // strong structural signal that generalizes well across prompts.
                score += 185;
                evidence.push("intent_uc_synergy".to_string());
            }
        }

        // ================================================================
        // KEYWORD + USE_CASE COMPOUND SYNERGY (FM-W2 iteration 1)
        // Gold skills at rank 11-20 have BOTH 3+ keyword AND 3+ use_case
        // matches 45% of the time, vs only 10% for non-gold at 1-10.
        // This synergy bonus rewards skills with deep multi-signal evidence.
        // ================================================================
        // Scaled kw+uc synergy: higher signal counts get progressively bigger bonuses
        // to break through the 0.6 floor zone crowding
        if keyword_matches >= 3 && outer_uc_match_count >= 2 {
            // Base: 100 for 3kw+2uc, +30 per additional kw, +40 per additional uc
            let base_synergy = 100;
            let kw_extra = ((keyword_matches as i32) - 3).max(0) * 30;
            let uc_extra = ((outer_uc_match_count as i32) - 2).max(0) * 40;
            let kw_uc_synergy = (base_synergy + kw_extra + uc_extra).min(400);
            score += kw_uc_synergy;
            evidence.push(format!("kw_uc_synergy:{}kw+{}uc+{}", keyword_matches, outer_uc_match_count, kw_uc_synergy));
        }

        // ================================================================
        // DESCRIPTION + USE_CASE COMPOUND SYNERGY (FM-W2 iteration 3)
        // Gold skills with both desc and uc matches are 1.5x more likely
        // to be gold vs non-gold at the same positions. This rewards skills
        // whose description AND use_cases both match the prompt.
        // ================================================================
        if outer_desc_match_count >= 2 && outer_uc_match_count >= 2 {
            let desc_uc_synergy = 80;
            score += desc_uc_synergy;
            evidence.push(format!("desc_uc_synergy:{}d+{}uc", outer_desc_match_count, outer_uc_match_count));
        }

        // Triple-signal synergy tested (FM-W2): 3kw+2desc+2uc at 150pts — no net gain
        // beyond kw_uc_synergy + desc_uc_synergy. Removed to keep scoring clean.

        // ================================================================
        // EVIDENCE BREADTH BONUS (FM-W3 innovation)
        // Skills with diverse evidence types (desc_match + use_case +
        // name_match + intent_uc_synergy) are more likely to be genuinely
        // relevant. Each evidence dimension adds a small bonus.
        // Simulation shows +10 when combined with use_case boost.
        // ================================================================
        {
            let has_desc = evidence.iter().any(|e| e.starts_with("desc_match:"));
            let has_uc = evidence.iter().any(|e| e.starts_with("use_case:"));
            let has_synergy = evidence.iter().any(|e| e == "intent_uc_synergy");

            let mut breadth = 0i32;
            if has_desc { breadth += 1; }
            if has_uc { breadth += 1; }
            if has_name_match { breadth += 1; }
            if has_synergy { breadth += 1; }
            if breadth >= 2 {
                // Only apply when 2+ evidence dimensions present
                // to avoid boosting single-signal matches
                score += breadth * 60;
                evidence.push(format!("evidence_breadth:{}", breadth));
            }

            // FM-W3: Triple-signal bonus for skills matching on all three
            // structural dimensions: name parts, description words, and
            // use_case text. This combination strongly predicts gold skills.
            if has_name_match && has_desc && has_uc {
                score += 200;
                evidence.push("triple_signal_bonus".to_string());
            }

            // FM-W3: Name + use_case dual-signal bonus. When a skill's name
            // parts match AND use_case text matches with 2+ words, it's a
            // strong indicator of genuine relevance beyond keyword overlap.
            if has_name_match && has_uc {
                // Extract use_case match count for proportional bonus
                let uc_count: i32 = evidence.iter()
                    .filter(|e| e.starts_with("use_case:"))
                    .filter_map(|e| e.split(':').nth(1).and_then(|v| v.parse().ok()))
                    .next()
                    .unwrap_or(0);
                if uc_count >= 2 {
                    score += 200;
                    evidence.push("name_uc_combo".to_string());
                }
            }
        }

        // Apply tier boost from PSS file (skip in incomplete_mode - populated in Pass 2)
        if !incomplete_mode {
            let tier_boost = match entry.tier.as_str() {
                "primary" => 50,     // Primary elements get phrase-tier boost
                "secondary" => 0,    // Default, no change
                "specialized" => -20, // Specialized elements slightly deprioritized
                _ => 0,
            };
            score += tier_boost;

            // Apply explicit boost from PSS file (scaled to phrase tier)
            score += (entry.boost.clamp(-10, 10)) * 10;
        }

        // Cap score to prevent keyword stuffing
        score = score.min(weights.capped_max);

        // All-low-signal cap: if EVERY match came from generic words (test, skill,
        // code, etc.), cap at common tier maximum (90). No amount of common word
        // matches should reach MEDIUM confidence — only phrases, tools, or frameworks
        // should produce meaningful suggestions.
        if !has_non_low_signal_match && score > ALL_LOW_SIGNAL_CAP {
            score = ALL_LOW_SIGNAL_CAP;
        }

        // LANGUAGE CONFLICT GATE: Binary exclusion for language-specific skills.
        // - Skill with explicit languages (non-empty, non-"any", non-"universal"):
        //   - If prompt has language signals → exclude if no overlap
        //   - If prompt has NO language signals → exclude (language-specific skill
        //     in a language-agnostic prompt)
        // - Skill with no languages or "any"/"universal" → pass through always
        // - Skill with empty languages but inferable language from name/keywords:
        //   same gating as explicit languages
        // Note: expanded_prompt_langs includes project-context languages, so a Python
        // project will have "python" even if the prompt doesn't mention it explicitly.
        if !entry.languages.is_empty()
            && !entry.languages.contains(&"any".to_string())
            && !entry.languages.contains(&"universal".to_string())
        {
            if expanded_prompt_langs.is_empty() {
                // No language signal at all — exclude language-specific skills
                debug!(
                    "Skill '{}': EXCLUDED by LANGUAGE CONFLICT GATE (skill langs: {:?}, prompt langs: empty)",
                    name, entry.languages
                );
                return None;
            }
            let has_lang_overlap = entry.languages.iter().any(|el| {
                expanded_prompt_langs.contains(&el.to_lowercase())
            });
            if !has_lang_overlap {
                // Language mismatch — binary exclusion
                debug!(
                    "Skill '{}': EXCLUDED by LANGUAGE CONFLICT GATE (skill langs: {:?}, prompt langs: {:?})",
                    name, entry.languages, expanded_prompt_langs
                );
                return None;
            }
        } else if entry.languages.is_empty() && !expanded_prompt_langs.is_empty() {
            // Entry has no explicit languages — infer from name/keywords/description
            // Only check when prompt HAS language signals (to catch hidden mismatches)
            let entry_name_lower = name.to_lowercase();
            let kw_text: String = entry.keywords.iter()
                .map(|k| k.to_lowercase())
                .collect::<Vec<_>>()
                .join(" ");
            let desc_lower = entry.description.to_lowercase();
            let search_text = format!("{} {} {}", entry_name_lower, kw_text, desc_lower);
            let mut inferred_lang: Option<&str> = None;
            for &(signal, lang) in lang_signal_patterns {
                if search_text.contains(signal) {
                    inferred_lang = Some(lang);
                    break;
                }
            }
            if let Some(lang) = inferred_lang {
                if !expanded_prompt_langs.contains(lang) {
                    // Binary exclusion for inferred language conflict
                    return None;
                }
            }
        }

        // PLATFORM CONFLICT GATE: Platform-specific skills are binary-excluded
        // when the prompt has NO matching platform signal.
        // Unlike languages (where no signal = pass through), platforms use strict gating:
        // - If skill has platforms (non-empty, non-universal) AND prompt has platform signals
        //   that don't overlap → EXCLUDE (wrong platform)
        // - If skill has platforms (non-empty, non-universal) AND prompt has NO platform signals
        //   → EXCLUDE (platform-specific skill in a platform-agnostic prompt)
        // - If skill has no platforms or "universal" → PASS (platform-agnostic skill)
        if !entry.platforms.is_empty()
            && !entry.platforms.contains(&"universal".to_string())
            && !entry.platforms.contains(&"any".to_string())
        {
            if expanded_prompt_platforms.is_empty() {
                // Prompt mentions no platform at all — exclude platform-specific skills
                debug!(
                    "Skill '{}': EXCLUDED by PLATFORM CONFLICT GATE (skill platforms: {:?}, prompt platforms: empty)",
                    name, entry.platforms
                );
                return None;
            }
            let has_platform_overlap = entry.platforms.iter().any(|ep| {
                expanded_prompt_platforms.contains(&ep.to_lowercase())
            });
            if !has_platform_overlap {
                // Platform mismatch — binary exclusion
                debug!(
                    "Skill '{}': EXCLUDED by PLATFORM CONFLICT GATE (skill platforms: {:?}, prompt platforms: {:?})",
                    name, entry.platforms, expanded_prompt_platforms
                );
                return None;
            }
        }

        // FRAMEWORK CONFLICT GATE: When the prompt mentions a specific framework,
        // entries with a COMPETING framework get a 90% penalty.
        if !conflicting_frameworks.is_empty() {
            let has_fw_conflict = entry.frameworks.iter().any(|ef| {
                conflicting_frameworks.contains(&ef.to_lowercase())
            });
            if has_fw_conflict {
                score = (score as f64 * 0.10) as i32;
                evidence.push("fw_conflict".to_string());
            }
        }

        // Determine confidence level (from reliable)
        let confidence = if score >= thresholds.high {
            Confidence::High
        } else if score >= thresholds.medium {
            Confidence::Medium
        } else {
            Confidence::Low
        };

        // Only include if score is meaningful (at least one common-tier match)
        if score >= 10 {
            Some(MatchedSkill {
                name: name.clone(),
                path: entry.path.clone(),
                skill_type: entry.skill_type.clone(),
                description: entry.description.clone(),
                score,
                confidence,
                evidence,
                plugin: entry.plugin.clone(),
                marketplace: marketplace_of(&entry.source).map(str::to_string),
                origin: entry.origin.clone(),
            })
        } else {
            None
        }
    }).collect();

    // Co-usage boosting (skip in incomplete_mode)
    // If a high-scoring skill lists another skill in usually_with, boost that skill
    if !incomplete_mode {
        // Collect names of high-scoring skills
        let high_score_threshold = 1000;
        let high_scoring: Vec<String> = matches
            .iter()
            .filter(|m| m.score >= high_score_threshold)
            .map(|m| m.name.clone())
            .collect();

        // Build map of skill names that should get co-usage boost, with booster scores
        // Map: related_skill -> Vec<(booster_name, booster_score)>
        let mut co_usage_boosts: std::collections::HashMap<String, Vec<(String, i32)>> =
            std::collections::HashMap::new();

        // Create a score lookup from matches
        let score_lookup: std::collections::HashMap<&str, i32> = matches
            .iter()
            .map(|m| (m.name.as_str(), m.score))
            .collect();

        for matched_name in &high_scoring {
            if let Some(entry) = index.get_by_name(matched_name) {
                let booster_score = *score_lookup.get(matched_name.as_str()).unwrap_or(&0);
                for related in &entry.co_usage.usually_with {
                    co_usage_boosts
                        .entry(related.clone())
                        .or_default()
                        .push((matched_name.clone(), booster_score));
                }
            }
        }

        // Apply co-usage evidence to existing matches (evidence-only, no score boost).
        // FM-W3: Disabled co-usage score boost entirely. Analysis showed co_usage
        // disproportionately helps non-gold skills (744 co_usage entries in blockers
        // vs 369 in gold misses). Evidence is still recorded for transparency.
        for m in &mut matches {
            if let Some(boosters) = co_usage_boosts.get(&m.name) {
                for (booster, _) in boosters {
                    m.evidence.push(format!("co_usage:{}", booster));
                }
            }
        }

        // Also add skills from co_usage that weren't matched at all (minimal score).
        // These are tiebreaker-level injections — they rank below any keyword match.
        for (related_name, boosters) in &co_usage_boosts {
            // Skip if already in matches
            if matches.iter().any(|m| &m.name == related_name) {
                continue;
            }
            // Add the related skill with minimal score (tiebreaker level only)
            if let Some(entry) = index.get_by_name(related_name) {
                let evidence: Vec<String> = boosters
                    .iter()
                    .map(|(b, _)| format!("co_usage:{}", b))
                    .collect();
                // Score is 3% of highest booster score, capped at 30 points
                let max_booster_score = boosters.iter().map(|(_, s)| *s).max().unwrap_or(0);
                let score = std::cmp::min(30, std::cmp::max(5, (max_booster_score * 3) / 100));
                matches.push(MatchedSkill {
                    name: related_name.clone(),
                    path: entry.path.clone(),
                    skill_type: entry.skill_type.clone(),
                    description: entry.description.clone(),
                    plugin: entry.plugin.clone(),
                    marketplace: marketplace_of(&entry.source).map(str::to_string),
                    origin: entry.origin.clone(),
                    score,
                    // Derive confidence from score thresholds (not hardcoded)
                    confidence: if score >= thresholds.high {
                        Confidence::High
                    } else if score >= thresholds.medium {
                        Confidence::Medium
                    } else {
                        Confidence::Low
                    },
                    evidence,
                });
            }
        }
    }

    // Sort by score descending, with skills-first ordering (from LimorAI/Scott Spence pattern)
    // Skills before agents before commands, within same score
    matches.sort_by(|a, b| {
        // First compare by score (descending)
        let score_cmp = b.score.cmp(&a.score);
        if score_cmp != std::cmp::Ordering::Equal {
            return score_cmp;
        }

        // If scores equal, order by type: skill > agent > command
        let type_order = |t: &str| match t {
            "skill" => 0,
            "agent" => 1,
            "command" => 2,
            _ => 3,
        };
        let type_cmp = type_order(&a.skill_type).cmp(&type_order(&b.skill_type));
        if type_cmp != std::cmp::Ordering::Equal {
            return type_cmp;
        }

        // W18: Evidence richness tie-breaking — favor skills with more diverse
        // evidence types (name_match + desc_match + use_case > just keywords).
        // This pushes domain-specific skills above generic keyword accumulators
        // when they're tied on normalized score (the 0.5 floor problem).
        let evidence_richness = |ev: &[String]| -> i32 {
            let mut richness = 0;
            // Strong signals worth more
            if ev.iter().any(|e| e.starts_with("name_match")) { richness += 3; }
            if ev.iter().any(|e| e.starts_with("desc_match")) { richness += 2; }
            if ev.iter().any(|e| e.starts_with("use_case")) { richness += 2; }
            if ev.iter().any(|e| e.starts_with("intent_uc_synergy")) { richness += 2; }
            // FM-W2: Cross-signal synergy bonuses indicate deeper evidence alignment
            if ev.iter().any(|e| e.starts_with("kw_uc_synergy")) { richness += 3; }
            if ev.iter().any(|e| e.starts_with("desc_uc_synergy")) { richness += 2; }
            if ev.iter().any(|e| e.starts_with("coherence")) { richness += 1; }
            if ev.iter().any(|e| e.starts_with("framework:") || e.starts_with("tool:")) { richness += 2; }
            if ev.iter().any(|e| e.starts_with("pattern:")) { richness += 1; }
            richness
        };
        let rich_cmp = evidence_richness(&b.evidence).cmp(&evidence_richness(&a.evidence));
        if rich_cmp != std::cmp::Ordering::Equal {
            return rich_cmp;
        }

        // Deterministic tie-breaking by name (W12 insight: HashMap iteration
        // order changes between compilations, causing non-deterministic results)
        a.name.cmp(&b.name)
    });

    // Limit results
    matches.truncate(MAX_SUGGESTIONS);

    matches
}

/// Calculate relative score (0.0 to 1.0) with absolute floor.
/// The absolute floor prevents one high-scoring skill (e.g., framework match)
/// from crushing all other genuinely matched skills below the min_score filter.
/// Any skill scoring at least ABSOLUTE_ANCHOR/2 raw points always gets at least
/// 0.5 relative score, ensuring it passes the default min_score=0.5 filter.
pub(crate) fn calculate_relative_score(score: i32, max_score: i32) -> f64 {
    if max_score <= 0 {
        return 0.0;
    }
    let relative = (score as f64) / (max_score as f64);
    // W18: Blended scoring — combines relative and absolute components.
    // Pure relative: score/max_score (good for differentiation when scores are close)
    // Pure absolute: score/ANCHOR, capped at 1.0 (good when one skill dominates)
    // Blend: max of relative and (absolute floored at 0.5), plus a small absolute
    // gradient within the floor zone to break ties.
    let absolute_component = (score as f64) / (ABSOLUTE_ANCHOR as f64);
    // Floor: any skill scoring >= ANCHOR/2 gets at least 0.5
    let floor = absolute_component.min(0.5);
    // Gradient: within the floor zone, add a tiny gradient based on raw score
    // to break ties. This gives skills with 800 raw slightly more than 500 raw,
    // even though both are floored at 0.5.
    let gradient = if relative < floor {
        // We're in the floor zone. Add a gradient proportional to raw score.
        let gradient_range = 0.10; // 0.5 to 0.6 range for differentiation
        floor + (absolute_component - floor).max(0.0).min(gradient_range)
    } else {
        relative
    };
    gradient
}

