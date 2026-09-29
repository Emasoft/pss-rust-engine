// Query/inspect subcommands extracted from main.rs (XUD7YUZH step 11).

use crate::agent_archetypes;
use crate::cli::{Cli, Commands, OutputFormat, resolve_format};
use crate::consts::CACHE_DIR;
use crate::index_file::get_db_path;
use crate::loading::{get_index_path, get_registry_path, load_domain_registry, load_index};
use crate::matching::{detect_domains_from_prompt_with_context, find_matches, infer_domains_from_text};
use crate::path::{load_domain_registry_from_db, load_index_from_db, open_db, project_slug};
use crate::text::{correct_typos, expand_synonyms};
use crate::temporal;
use crate::{CoUsageData, Confidence, ProjectContext, SkillEntry, SuggesterError};
use chrono::{DateTime, Utc};
use cozo::{DataValue, DbInstance, ScriptMutability};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::warn;

// Query/Inspect Subcommands
// ============================================================================

/// Open CozoDB for query commands (read-only, no full index load needed).
pub(crate) fn open_db_for_query(cli: &Cli) -> Result<DbInstance, SuggesterError> {
    let db_path = get_db_path(cli.index.as_deref())
        .ok_or_else(|| SuggesterError::IndexNotFound(PathBuf::from("pss-skill-index.db")))?;
    let db = open_db(&db_path)?;
    // Ensure the temporal-history tables exist (idempotent — TRDD-152e697f).
    // Failure here is non-fatal for legacy queries, but warned so users
    // know the temporal subcommands won't have data yet.
    if let Err(e) = temporal::ensure_schema(&db) {
        tracing::debug!("temporal::ensure_schema failed (non-fatal): {}", e);
    }
    Ok(db)
}

/// Valid entry types — whitelist for --type filter. Per COR-4 (audit 20260514)
/// this now covers all 12 element types PSS discovers, not just the legacy 6
/// that live in the `skills` table. The 6 new types (hook/plugin/marketplace/
/// monitor/output-style/theme) live only in `elements_state` — `cmd_list` and
/// `cmd_search` detect them via `NEW_ELEMENT_TYPES` below and route to the
/// elements-state-backed helper.
pub(crate) const VALID_TYPES: &[&str] = &[
    // Legacy 6 (skills table — full keyword/domain/aux indexing).
    "skill", "agent", "command", "rule", "mcp", "lsp",
    // New 6 (elements_state only — basic name/scope/path).
    "hook", "plugin", "marketplace", "monitor", "output-style", "theme",
];

/// Subset of `VALID_TYPES` whose rows live in `elements_state` only — no
/// matching row in the legacy `skills` table. Queries against these types
/// must go through `cmd_list_elements_state` (COR-4 — audit 20260514).
pub(crate) const NEW_ELEMENT_TYPES: &[&str] = &[
    "hook", "plugin", "marketplace", "monitor", "output-style", "theme",
];

/// True if `type_filter` (if Some) is in `NEW_ELEMENT_TYPES` — caller should
/// route the query to `cmd_list_elements_state` instead of the legacy
/// skills-table path.
pub(crate) fn is_new_element_type(type_filter: Option<&str>) -> bool {
    matches!(type_filter, Some(t) if NEW_ELEMENT_TYPES.contains(&t))
}

/// Validate --type filter against whitelist.
/// Returns error if the value is not a known type. Prevents Datalog injection
/// by ensuring only whitelisted values are used in queries.
pub(crate) fn validate_type_filter(t: Option<&str>) -> Result<(), SuggesterError> {
    if let Some(t) = t {
        if !VALID_TYPES.contains(&t) {
            return Err(SuggesterError::IndexParse(format!(
                "Invalid type '{}'. Valid types: {}", t, VALID_TYPES.join(", ")
            )));
        }
    }
    Ok(())
}

/// List rows of a given element type from `elements_state` joined against
/// the latest event (for element_name + scope_path). Per COR-4: this is the
/// elements-state-backed path that handles `hook`, `plugin`, `marketplace`,
/// `monitor`, `output-style`, `theme` — types that have no row in the legacy
/// `skills` table.
///
/// Optional `name_substring` filters by case-insensitive substring match on
/// element_name — used by `cmd_search` to approximate keyword search against
/// new types (which have no keyword aux table).
///
/// Only returns rows where `exists = true` (currently-present elements).
fn cmd_list_elements_state(
    db: &DbInstance,
    type_filter: &str,
    name_substring: Option<&str>,
    top: usize,
    format: &str,
) -> Result<(), SuggesterError> {
    let top = top.min(10000);
    let q = r#"
        ?[element_name, element_type, scope, scope_path, current_path, last_changed_at] :=
            *elements_state{element_id, last_event_id, current_path, last_changed_at, exists: true},
            *events{event_id: last_event_id, element_type, element_name, scope, scope_path},
            element_type = $ftype
        :order element_name
        :limit $top
    "#;
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("ftype".into(), DataValue::Str(type_filter.into()));
    params.insert("top".into(), DataValue::Num(cozo::Num::Int(top as i64)));

    let result = db.run_script(q, params, ScriptMutability::Immutable)
        .map_err(|e| SuggesterError::IndexParse(format!(
            "elements_state list query failed: {}", e
        )))?;

    let needle = name_substring.map(|s| s.to_lowercase());
    let entries: Vec<serde_json::Value> = result.rows.iter()
        .filter(|r| {
            // Apply name-substring filter if provided.
            match &needle {
                None => true,
                Some(n) => dv_to_string(&r[0]).to_lowercase().contains(n),
            }
        })
        .map(|r| serde_json::json!({
            "id": dv_to_string(&r[0]),
            "name": dv_to_string(&r[0]),
            "type": dv_to_string(&r[1]),
            "scope": dv_to_string(&r[2]),
            "scope_path": dv_to_string(&r[3]),
            "path": dv_to_string(&r[4]),
            "last_changed_at": dv_to_string(&r[5]),
            // Legacy field names for compatibility with skills-table output:
            "category": "",
            "description": "",
            "source": dv_to_string(&r[2]),  // alias scope→source for new types
        }))
        .collect();

    if format == "json" {
        println!("{}", serde_json::to_string_pretty(&entries).unwrap_or_default());
    } else {
        print_table(&["NAME", "TYPE", "SCOPE", "SCOPE_PATH", "PATH"],
            &entries.iter().map(|e| vec![
                e["name"].as_str().unwrap_or("").to_string(),
                e["type"].as_str().unwrap_or("").to_string(),
                e["scope"].as_str().unwrap_or("").to_string(),
                e["scope_path"].as_str().unwrap_or("").chars().take(30).collect::<String>(),
                e["path"].as_str().unwrap_or("").chars().take(60).collect::<String>(),
            ]).collect::<Vec<_>>());
    }
    Ok(())
}

/// Validate that a string filter value contains only safe characters.
/// Rejects values containing Datalog control characters that could be used
/// for injection: single quotes, backslashes, newlines, colons, braces.
fn validate_filter_value(name: &str, value: &str) -> Result<(), SuggesterError> {
    if value.contains('\'') || value.contains('\\') || value.contains('\n')
        || value.contains('\r') || value.contains('{') || value.contains('}')
        || value.contains(':') || value.contains(';')
    {
        return Err(SuggesterError::IndexParse(format!(
            "Invalid characters in --{} filter: '{}'", name, value
        )));
    }
    Ok(())
}

/// Resolve a name-or-ID reference to an entry name using CozoDB.
/// If the input looks like a 13-char alphanumeric ID, look it up in skill_ids.
/// Otherwise treat it as a name directly.
pub(crate) fn resolve_name_or_id(db: &DbInstance, ref_str: &str) -> Result<String, SuggesterError> {
    // Check if it looks like an ID (13 chars, all alphanumeric lowercase)
    if ref_str.len() == 13 && ref_str.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()) {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("id".into(), DataValue::Str(ref_str.into()));
        let result = db.run_script(
            "?[name, source] := *skill_ids{ id: $id, name, source }",
            params,
            ScriptMutability::Immutable,
        ).map_err(|e| SuggesterError::IndexParse(format!("ID lookup failed: {}", e)))?;
        if let Some(row) = result.rows.first() {
            if let DataValue::Str(s) = &row[0] {
                return Ok(s.to_string());
            }
        }
        // Fall through to treat as name if ID not found
    }
    Ok(ref_str.to_string())
}

/// Resolve skill names to their indexed path + description.
///
/// The path is what makes the preload gate possible: the index stores none of
/// `disable-model-invocation` / `context` / `agent` / `user-invocable`, so the
/// gate re-reads each `SKILL.md` from disk. A name that does not resolve gets an
/// empty path and is rejected there — never silently dropped, which is the exact
/// failure mode Claude Code itself has with a missing preloaded skill.
fn resolve_skill_refs(
    db: &DbInstance,
    names: &[String],
) -> Result<Vec<agent_archetypes::SkillRef>, SuggesterError> {
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("name".into(), DataValue::Str(name.clone().into()));
        let res = db
            .run_script(
                "?[path, description] := *skills{ name, path, description, skill_type }, \
                 name = $name, skill_type = 'skill'",
                params,
                ScriptMutability::Immutable,
            )
            .map_err(|e| {
                SuggesterError::IndexParse(format!("skill lookup failed for '{}': {}", name, e))
            })?;
        let (path, description) = match res.rows.first() {
            Some(row) => (dv_to_string(&row[0]), dv_to_string(&row[1])),
            None => (String::new(), String::new()),
        };
        out.push(agent_archetypes::SkillRef {
            name: name.clone(),
            path,
            description,
        });
    }
    Ok(out)
}

/// Every skill belonging to one plugin, alphabetically — PLUGIN-OMNI's menu.
fn resolve_plugin_skills(
    db: &DbInstance,
    plugin: &str,
) -> Result<Vec<agent_archetypes::SkillRef>, SuggesterError> {
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("plugin".into(), DataValue::Str(plugin.to_string().into()));
    let res = db
        .run_script(
            "?[name, path, description] := *skills{ name, path, description, skill_type, source }, \
             skill_type = 'skill', str_includes(source, $plugin)",
            params,
            ScriptMutability::Immutable,
        )
        .map_err(|e| {
            SuggesterError::IndexParse(format!("plugin skill lookup failed: {}", e))
        })?;
    let mut out: Vec<agent_archetypes::SkillRef> = res
        .rows
        .iter()
        .map(|row| agent_archetypes::SkillRef {
            name: dv_to_string(&row[0]),
            path: dv_to_string(&row[1]),
            description: dv_to_string(&row[2]),
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Resolve the `description` argument to (text, name-from-frontmatter).
///
/// A caller may pass prose or a path to a `.md` file — the same latitude the
/// plugin generator gives. The path branch is taken only when the file actually
/// exists, so a description that merely *looks* path-like ("agent for src/api
/// work") is still treated as prose rather than failing to open.
fn resolve_description(raw: &str) -> Result<(String, Option<String>), SuggesterError> {
    let candidate = Path::new(raw);
    if !candidate.is_file() {
        return Ok((raw.to_string(), None));
    }
    let content = fs::read_to_string(candidate).map_err(|e| {
        SuggesterError::IndexParse(format!("reading description file {}: {}", raw, e))
    })?;
    // A description file may carry frontmatter (the plugin generator's format).
    // Take `name:` from it when present, and score against the prose body —
    // scoring the YAML too would let field names pollute the match.
    let mut name = None;
    let body = match agent_archetypes::frontmatter_block(&content) {
        Some(fm) => {
            for line in fm.lines() {
                if let Some(v) = line.strip_prefix("name:") {
                    let v = v.trim().trim_matches(['"', '\'']);
                    if !v.is_empty() {
                        name = Some(v.to_string());
                    }
                }
            }
            content[content.find(fm).map(|i| i + fm.len()).unwrap_or(0)..]
                .trim_start_matches(['-', '\r', '\n'])
                .to_string()
        }
        None => content,
    };
    Ok((body, name))
}

/// Pick the skills a description calls for, using PSS's own scorer.
///
/// Deliberately reuses `find_matches` — the same function the suggestion hook
/// runs — rather than a bespoke query. "Which skills match this text" must have
/// exactly one implementation, or the agent generator and the hook would
/// disagree about the same index.
fn select_skills_for_description(
    cli: &Cli,
    db: &DbInstance,
    description: &str,
    top: usize,
) -> Result<Vec<agent_archetypes::SkillRef>, SuggesterError> {
    let index = match load_index_from_db(db) {
        Ok(idx) => idx,
        Err(e) => {
            warn!("make-agent: CozoDB index load failed: {}, using JSON", e);
            load_index(&get_index_path(cli.index.as_deref())?)?
        }
    };
    // The domain registry and detected domains are NOT optional here. Without
    // them find_matches runs with gate filtering disabled, and a request about
    // "Rust memory safety" pulls in spark-optimization (memory tuning), the
    // long-term `memory` skill, and Context Optimization — all matching the word
    // "memory" with no domain to reject them. Measured on the real index: 4 of 8
    // selected skills were off-domain until this was passed.
    let registry = match load_domain_registry_from_db(db) {
        Ok(Some(reg)) => Some(reg),
        _ => get_registry_path(cli.registry.as_deref())
            .and_then(|p| load_domain_registry(&p).ok().flatten()),
    };

    let corrected = correct_typos(description);
    // Fold the description's own domain signals into the expanded query, the way
    // the profiler does — that is what lets the taxonomy reject an off-domain
    // skill instead of merely out-scoring it.
    let domain_signals = infer_domains_from_text(description);
    let expanded = format!(
        "{} {}",
        expand_synonyms(&corrected),
        domain_signals.join(" ")
    );
    let detected_domains = match registry.as_ref() {
        Some(reg) => detect_domains_from_prompt_with_context(&expanded, reg, &domain_signals),
        None => HashMap::new(),
    };

    let matches = find_matches(
        &corrected,
        &expanded,
        &index,
        "",
        &ProjectContext::default(),
        false,
        &detected_domains,
        registry.as_ref(),
    );
    // `top` is a CEILING, not a quota. Taking exactly N regardless of score pads
    // a narrow description out with whatever ranked 5th-8th — on a real index that
    // is how a Rust-memory-safety agent ends up preloading a long-term-memory
    // skill. Below Medium the match is a word collision, and an agent is better
    // off with three right skills than eight of which five are noise.
    // Dedup by name BEFORE taking `top`. The index holds the same skill under
    // several sources (user, project, plugin), and find_matches returns each
    // separately — measured, three `memory` rows filled three of eight slots for
    // one prompt. Preloading a name twice is also meaningless: the agent gets one
    // copy either way, so the duplicate slots were pure loss.
    let mut seen: HashSet<String> = HashSet::new();
    let selected: Vec<agent_archetypes::SkillRef> = matches
        .into_iter()
        .filter(|m| m.skill_type == "skill" && m.confidence != Confidence::Low)
        .filter(|m| seen.insert(m.name.clone()))
        .take(top)
        .map(|m| agent_archetypes::SkillRef {
            name: m.name,
            path: m.path,
            description: m.description,
        })
        .collect();

    if selected.is_empty() {
        return Err(SuggesterError::IndexParse(format!(
            "no skill matched '{}' above low confidence — describe the specialization \
             more concretely, or pass --skills explicitly",
            description.chars().take(60).collect::<String>().trim()
        )));
    }
    Ok(selected)
}

/// Everything `make-agent` was invoked with.
///
/// A struct rather than a dozen positional parameters: the flags are mostly
/// `Option<&str>` and `bool`, so a caller that transposed two of them would
/// still compile and silently generate the wrong agent.
pub(crate) struct MakeAgentArgs<'a> {
    description: Option<&'a str>,
    kind: &'a str,
    name: Option<&'a str>,
    skills: Option<&'a str>,
    plugin: Option<&'a str>,
    summary: Option<&'a str>,
    model: Option<&'a str>,
    effort: Option<&'a str>,
    output: &'a str,
    top: usize,
    dry_run: bool,
    explore: bool,
    filters: agent_archetypes::ElementFilters,
    format: &'a str,
}

fn cmd_make_agent(
    cli: &Cli,
    db: &DbInstance,
    a: MakeAgentArgs<'_>,
) -> Result<(), SuggesterError> {
    let archetype = agent_archetypes::Archetype::parse(a.kind).ok_or_else(|| {
        SuggesterError::IndexParse(format!(
            "unknown --kind '{}' (expected normal, all-in-one, one-for-all or plugin-omni)",
            a.kind
        ))
    })?;

    // A description may be prose or a path to a .md file; the file may name the
    // agent in its frontmatter.
    let (desc_text, desc_name) = match a.description {
        Some(raw) => {
            let (t, n) = resolve_description(raw)?;
            (Some(t), n)
        }
        None => (None, None),
    };

    // Selection precedence: explicit --skills, then --plugin (plugin-omni takes
    // the WHOLE plugin by design), then the description. --no-skill short-circuits
    // all of it — there is no point scoring an index whose result is discarded.
    let refs = if a.filters.no_skill {
        Vec::new()
    } else {
        match (a.skills, a.plugin, desc_text.as_deref()) {
            (Some(list), _, _) => {
                let names: Vec<String> = list
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if names.is_empty() {
                    return Err(SuggesterError::IndexParse("--skills was empty".into()));
                }
                resolve_skill_refs(db, &names)?
            }
            (None, Some(p), _) if archetype == agent_archetypes::Archetype::PluginOmni => {
                resolve_plugin_skills(db, p)?
            }
            (None, _, Some(text)) if !text.trim().is_empty() => {
                select_skills_for_description(cli, db, text, a.top)?
            }
            (None, Some(p), _) => resolve_plugin_skills(db, p)?,
            (None, None, _) => {
                return Err(SuggesterError::IndexParse(
                    "give a description, --skills <a,b,c>, or --plugin <name>".into(),
                ))
            }
        }
    };

    // Name precedence: explicit flag, then a description file's frontmatter,
    // then a slug derived from the prose. Failing with "no name" when the user
    // gave a perfectly good description would be pedantry.
    let name = match (a.name, desc_name.as_deref(), desc_text.as_deref()) {
        (Some(n), _, _) => n.to_string(),
        (None, Some(n), _) => n.to_string(),
        (None, None, Some(text)) => agent_archetypes::derive_name(text),
        (None, None, None) => {
            return Err(SuggesterError::IndexParse(
                "give --name, or a description to derive it from".into(),
            ))
        }
    };

    let description = a
        .summary
        .map(|d| d.to_string())
        .or_else(|| {
            desc_text.as_deref().map(|t| {
                let first = t.trim().split(['.', '\n']).next().unwrap_or(t).trim();
                if first.is_empty() { t.trim().to_string() } else { format!("{}.", first) }
            })
        })
        .unwrap_or_else(|| format!("Generated {} agent.", archetype.as_str()));

    let spec = agent_archetypes::GenSpec {
        kind: archetype,
        name,
        description,
        model: a.model.map(|m| m.to_string()),
        effort: a.effort.map(|e| e.to_string()),
        skills: refs,
        plugin: a.plugin.map(|p| p.to_string()),
        allow_explore: a.explore,
        out_dir: PathBuf::from(a.output),
        agents: Vec::new(),
        mcp: Vec::new(),
        filters: a.filters,
    };

    let (dry_run, format) = (a.dry_run, a.format);
    let em = agent_archetypes::emit(&spec);
    if !dry_run {
        em.commit().map_err(SuggesterError::IndexParse)?;
    }

    if format == "json" {
        let payload = serde_json::json!({
            "agent": spec.name,
            "archetype": archetype.as_str(),
            "dry_run": dry_run,
            "files": em.files.iter()
                .map(|f| f.path.to_string_lossy().to_string())
                .collect::<Vec<_>>(),
            "warnings": em.warnings,
        });
        println!("{}", serde_json::to_string_pretty(&payload).unwrap());
    } else {
        for w in &em.warnings {
            eprintln!("warning: {}", w);
        }
        for f in &em.files {
            println!("{}{}", if dry_run { "would write " } else { "" }, f.path.display());
        }
        // The Explore/micro split is the whole point of one-for-all; printing it
        // keeps a plan that quietly sends every step to the expensive environment
        // from looking like a cheap one.
        if !em.plan.is_empty() {
            print!("\n{}", agent_archetypes::cost_table(&em.plan, agent_archetypes::CostModel::default()));
        }
    }
    Ok(())
}

/// Dispatch query subcommands. Health is handled BEFORE this function in
/// `main()` because it needs special exit-code semantics (0/1/2) that don't
/// fit the `Result<(), SuggesterError>` contract.
pub(crate) fn run_query_command(cli: &Cli, cmd: &Commands) -> Result<(), SuggesterError> {
    let db = open_db_for_query(cli)?;
    match cmd {
        Commands::Search { query, r#type, domain, language, framework, tool,
                           category, file_type, keyword, platform, top, format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_search(&db, query, r#type.as_deref(), domain.as_deref(),
                       language.as_deref(), framework.as_deref(), tool.as_deref(),
                       category.as_deref(), file_type.as_deref(), keyword.as_deref(),
                       platform.as_deref(), *top, fmt.as_str())
        }
        Commands::List { r#type, domain, language, framework, tool,
                         category, file_type, keyword, platform, source_prefix, sort, top, format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_list(&db, r#type.as_deref(), domain.as_deref(),
                     language.as_deref(), framework.as_deref(), tool.as_deref(),
                     category.as_deref(), file_type.as_deref(), keyword.as_deref(),
                     platform.as_deref(), source_prefix.as_deref(), sort, *top, fmt.as_str())
        }
        Commands::MakeAgent { description, kind, name, skills, top, no_mcp, no_skill,
                              no_agent, effort, plugin, summary, model,
                              output, dry_run, explore, format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_make_agent(cli, &db, MakeAgentArgs {
                description: description.as_deref(),
                kind,
                name: name.as_deref(),
                skills: skills.as_deref(),
                plugin: plugin.as_deref(),
                summary: summary.as_deref(),
                model: model.as_deref(),
                effort: effort.as_deref(),
                output,
                top: *top,
                dry_run: *dry_run,
                explore: *explore,
                filters: agent_archetypes::ElementFilters {
                    no_skill: *no_skill,
                    no_agent: *no_agent,
                    no_mcp: *no_mcp,
                },
                format: fmt.as_str(),
            })
        }
        Commands::Inspect { name, format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_inspect(&db, name, fmt.as_str())
        }
        Commands::Compare { name1, name2, format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_compare(&db, name1, name2, fmt.as_str())
        }
        Commands::Stats { format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_stats(&db, fmt.as_str())
        }
        Commands::Vocab { field, r#type, top, format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_vocab(&db, field, r#type.as_deref(), *top, fmt.as_str())
        }
        Commands::Coverage { r#type, format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_coverage(&db, r#type.as_deref(), fmt.as_str())
        }
        Commands::Resolve { ids, format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_resolve(&db, ids, fmt.as_str())
        }
        Commands::GetDescription { names, batch, format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_get_description(&db, names, *batch, fmt.as_str())
        }
        Commands::IndexRules { project_root, format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_index_rules(&db, project_root.as_deref(), fmt.as_str())
        }
        Commands::ListRules { scope, format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_list_rules(&db, scope.as_deref(), fmt.as_str())
        }
        Commands::Export { json, path } =>
            cmd_export(&db, *json, path.as_deref()),
        // --- Phase D (v2.11.0) ---
        Commands::Count { json, format } => {
            let fmt = resolve_format(*json, *format);
            // The cmd_count function's existing `json: bool` signature is kept
            // for minimal surface change — table maps to bool=false (bare int),
            // json maps to bool=true ({"count": N}).  Stub formats fall back to
            // the JSON envelope so the integer is still parseable.
            let want_json = matches!(fmt, OutputFormat::Json | OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown);
            cmd_count(&db, want_json)
        }
        Commands::Get { name, source, json, format } => {
            let fmt = resolve_format(*json, *format);
            // Same rationale as cmd_count: keep the existing bool signature
            // until each format gets its own branch.  Json/stub → JSON envelope.
            let want_json = matches!(fmt, OutputFormat::Json | OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown);
            cmd_get(&db, name, source.as_deref(), want_json)
        }
        Commands::Health { .. } => {
            // Unreachable — Health is dispatched BEFORE run_query_command in main().
            // If we get here, that's a routing bug; fail loudly.
            Err(SuggesterError::IndexParse(
                "internal error: Health command reached run_query_command".to_string(),
            ))
        }
        Commands::ListAddedSince { when, limit, json, format } => {
            let fmt = resolve_format(*json, *format);
            let want_json = matches!(fmt, OutputFormat::Json | OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown);
            cmd_list_added_since(&db, when, *limit, want_json)
        }
        Commands::ListAddedBetween { start, end, limit, json, format } => {
            let fmt = resolve_format(*json, *format);
            let want_json = matches!(fmt, OutputFormat::Json | OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown);
            cmd_list_added_between(&db, start, end, *limit, want_json)
        }
        Commands::ListUpdatedSince { when, limit, json, format } => {
            let fmt = resolve_format(*json, *format);
            let want_json = matches!(fmt, OutputFormat::Json | OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown);
            cmd_list_updated_since(&db, when, *limit, want_json)
        }
        Commands::FindByName { substring, limit, json, regex, format } => {
            let fmt = resolve_format(*json, *format);
            let want_json = matches!(fmt, OutputFormat::Json | OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown);
            cmd_find_by_name(&db, substring, *limit, want_json, *regex)
        }
        Commands::FindByKeyword { keyword, limit, json, format } => {
            let fmt = resolve_format(*json, *format);
            let want_json = matches!(fmt, OutputFormat::Json | OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown);
            cmd_find_by_auxiliary(&db, "skill_keywords", keyword, *limit, want_json)
        }
        Commands::FindByDomain { domain, limit, json, format } => {
            let fmt = resolve_format(*json, *format);
            let want_json = matches!(fmt, OutputFormat::Json | OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown);
            cmd_find_by_auxiliary(&db, "skill_domains", domain, *limit, want_json)
        }
        Commands::FindByLanguage { language, limit, json, format } => {
            let fmt = resolve_format(*json, *format);
            let want_json = matches!(fmt, OutputFormat::Json | OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown);
            cmd_find_by_auxiliary(&db, "skill_languages", language, *limit, want_json)
        }
        Commands::FindByFramework { framework, limit, json, format } => {
            let fmt = resolve_format(*json, *format);
            let want_json = matches!(fmt, OutputFormat::Json | OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown);
            cmd_find_by_auxiliary(&db, "skill_frameworks", framework, *limit, want_json)
        }
        Commands::FindByTool { tool, limit, json, format } => {
            let fmt = resolve_format(*json, *format);
            let want_json = matches!(fmt, OutputFormat::Json | OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown);
            cmd_find_by_auxiliary(&db, "skill_tools", tool, *limit, want_json)
        }
        Commands::FindByPlatform { platform, limit, json, format } => {
            let fmt = resolve_format(*json, *format);
            let want_json = matches!(fmt, OutputFormat::Json | OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown);
            cmd_find_by_auxiliary(&db, "skill_platforms", platform, *limit, want_json)
        }

        // ====================================================================
        // Temporal history index (TRDD-152e697f) — see temporal::* dispatchers.
        // Each dispatcher prints JSON to stdout and returns (); we wrap in Ok.
        // ====================================================================
        Commands::AsOf { date, r#type, scope, scope_path, limit } => {
            temporal::cli::cmd_as_of(&db, date, r#type.as_deref(),
                scope.as_deref(), scope_path.as_deref(), *limit);
            Ok(())
        }
        Commands::ActiveIn { abs_path, as_of, limit, format, json } => {
            // active-in always emits the JSON array (same as as-of); the
            // --format/--json flags are accepted for CLI convention parity
            // but do not change the JSON-array output.
            let _ = resolve_format(*json, *format);
            // Compute the scope-path slug from the absolute path with the SAME
            // algorithm `pss project-slug` uses (single source of truth), so it
            // matches the project/local-scope rows' scope_path in the DB.
            let slug = project_slug(Path::new(abs_path));
            temporal::cli::cmd_active_in(&db, &slug, as_of, *limit);
            Ok(())
        }
        Commands::Timeline { element_id, limit, format, json } => {
            let fmt = resolve_format(*json, *format);
            temporal::cli::cmd_timeline(&db, element_id, *limit, fmt);
            Ok(())
        }
        Commands::Lifespan { element_id } => {
            temporal::cli::cmd_lifespan(&db, element_id);
            Ok(())
        }
        Commands::ChangedBetween { start, end, r#type, limit, format, json } => {
            let fmt = resolve_format(*json, *format);
            temporal::cli::cmd_changed_between(&db, start, end, r#type.as_deref(), *limit, fmt);
            Ok(())
        }
        Commands::RemovedSince { date, limit } => {
            temporal::cli::cmd_removed_since(&db, date, *limit);
            Ok(())
        }
        Commands::ScanLog { limit, format, json } => {
            let fmt = resolve_format(*json, *format);
            temporal::cli::cmd_scan_log(&db, *limit, fmt);
            Ok(())
        }
        Commands::DbStats { format, json } => {
            let fmt = resolve_format(*json, *format);
            temporal::cli::cmd_db_stats(&db, fmt);
            Ok(())
        }
        Commands::Reindex { dry_run } => {
            temporal::cli::cmd_reindex(&db, *dry_run);
            Ok(())
        }
        Commands::PruneHistory { dry_run: _ } => {
            // Unreachable — F10 (TRDD-1Z8SGQ7N) intercepts PruneHistory in
            // main() BEFORE run_query_command so it can hold the two F3 flocks
            // across its event deletions (same pattern as MergeEvents and
            // MigrateElementIds). Reaching here means that intercept was
            // removed.
            Err(SuggesterError::IndexParse(
                "internal error: PruneHistory reached run_query_command".to_string(),
            ))
        }
        Commands::MigrateElementIds {} => {
            // Unreachable — F4 (TRDD-1Z8SGQ7N) intercepts MigrateElementIds in
            // main() BEFORE run_query_command so it can hold the two F3 flocks
            // across its full-table rewrite (same pattern as MergeEvents).
            // Reaching here means that intercept was removed.
            Err(SuggesterError::IndexParse(
                "internal error: MigrateElementIds reached run_query_command".to_string(),
            ))
        }
        Commands::Show { element_id, as_of } => {
            temporal::cli::cmd_show_at(&db, element_id, as_of);
            Ok(())
        }
        Commands::SizeAt { element_id, as_of } => {
            temporal::cli::cmd_size_at(&db, element_id, as_of);
            Ok(())
        }
        Commands::TokensAt { element_id, as_of } => {
            temporal::cli::cmd_tokens_at(&db, element_id, as_of);
            Ok(())
        }
        Commands::Diff { element_id, date1, date2 } => {
            temporal::cli::cmd_diff(&db, element_id, date1, date2);
            Ok(())
        }
        Commands::InstalledBetween { start, end, r#type, limit } => {
            temporal::cli::cmd_installed_between(
                &db, start, end, r#type.as_deref(), *limit,
            );
            Ok(())
        }
        Commands::RemovedBetween { start, end, r#type, limit } => {
            temporal::cli::cmd_removed_between(
                &db, start, end, r#type.as_deref(), *limit,
            );
            Ok(())
        }
        Commands::CurrentlyMissing { r#type, limit }
        | Commands::NeverCurrent { r#type, limit } => {
            temporal::cli::cmd_currently_missing(&db, r#type.as_deref(), *limit);
            Ok(())
        }
        Commands::MultiScope { name, r#type } => {
            temporal::cli::cmd_multi_scope(&db, name, r#type.as_deref());
            Ok(())
        }
        Commands::OverrideHistory { element_id, limit } => {
            temporal::cli::cmd_override_history(&db, element_id, *limit);
            Ok(())
        }
        Commands::EnableHistory { element_id, limit } => {
            temporal::cli::cmd_enable_history(&db, element_id, *limit);
            Ok(())
        }
        Commands::ScopeMoves { name, r#type, limit } => {
            temporal::cli::cmd_scope_moves(&db, name, r#type.as_deref(), *limit);
            Ok(())
        }
        Commands::MarketplaceHistory { limit, format, json } => {
            let fmt = resolve_format(*json, *format);
            temporal::cli::cmd_marketplace_history(&db, *limit, fmt);
            Ok(())
        }
        Commands::PluginHistory { plugin_name, limit, format, json } => {
            let fmt = resolve_format(*json, *format);
            temporal::cli::cmd_plugin_history(&db, plugin_name, *limit, fmt);
            Ok(())
        }
        Commands::MergeEvents { .. } => {
            // Unreachable — F3 (TRDD-1Z8SGQ7N) intercepts MergeEvents in main()
            // BEFORE run_query_command so it can hold the two flocks across the
            // write. Mirrors the Health/DbPath/ProjectSlug intercept pattern
            // above; reaching here means that intercept was removed.
            Err(SuggesterError::IndexParse(
                "internal error: MergeEvents reached run_query_command".to_string(),
            ))
        }
        Commands::SuggestMode { .. } => {
            // Unreachable — intercepted in main() BEFORE the DB is opened, so
            // that switching modes works on a never-indexed install. Mirrors
            // the Health/DbPath/ProjectSlug intercept pattern; reaching here
            // means that intercept was removed.
            Err(SuggesterError::IndexParse(
                "internal error: SuggestMode reached run_query_command".to_string(),
            ))
        }
        Commands::Retention { set } => {
            temporal::cli::cmd_retention(&db, set.as_deref());
            Ok(())
        }
        // ─── Phase 3 Tier A new query subcommands (audit 20260514) ─────
        Commands::ByPlugin { name, r#type, limit } => {
            temporal::cli::cmd_by_plugin(&db, name, r#type.as_deref(), *limit);
            Ok(())
        }
        Commands::ByMarketplace { name, r#type, limit } => {
            temporal::cli::cmd_by_marketplace(&db, name, r#type.as_deref(), *limit);
            Ok(())
        }
        Commands::ScopeDiff { scope1, scope2, r#type, limit } => {
            temporal::cli::cmd_scope_diff(&db, scope1, scope2, r#type.as_deref(), *limit);
            Ok(())
        }
        Commands::ChangesInBatch { scan_id, limit, format, json } => {
            let fmt = resolve_format(*json, *format);
            temporal::cli::cmd_changes_in_batch(&db, scan_id, *limit, fmt);
            Ok(())
        }
        Commands::LastChanges { limit, format, json } => {
            let fmt = resolve_format(*json, *format);
            temporal::cli::cmd_last_changes(&db, *limit, fmt);
            Ok(())
        }
        Commands::StatsByScope { r#type, format, json } => {
            let fmt = resolve_format(*json, *format);
            temporal::cli::cmd_stats_by_scope(&db, r#type.as_deref(), fmt);
            Ok(())
        }
        Commands::VersionHistory { element_id, limit, format, json } => {
            let fmt = resolve_format(*json, *format);
            temporal::cli::cmd_version_history(&db, element_id, *limit, fmt);
            Ok(())
        }
        Commands::ChangesSummary { window, r#type, format, json } => {
            let fmt = resolve_format(*json, *format);
            temporal::cli::cmd_changes_summary(&db, window, r#type.as_deref(), fmt);
            Ok(())
        }
        Commands::EnabledWhere { name, r#type } => {
            temporal::cli::cmd_enabled_where(&db, name, r#type.as_deref());
            Ok(())
        }
        Commands::DedupCandidates { min_count, r#type, limit, format, json } => {
            let fmt = resolve_format(*json, *format);
            temporal::cli::cmd_dedup_candidates(
                &db, *min_count, r#type.as_deref(), *limit, fmt,
            );
            Ok(())
        }
        Commands::CompareSnapshots { date1, date2, r#type, limit, format, json } => {
            let fmt = resolve_format(*json, *format);
            temporal::cli::cmd_compare_snapshots(
                &db, date1, date2, r#type.as_deref(), *limit, fmt,
            );
            Ok(())
        }
        // ─── Phase 3 / v3.7 new top-level overview subcommands ────────────
        Commands::Summary { format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_summary(&db, fmt)
        }
        Commands::Tree { format, json } => {
            let fmt = resolve_format(*json, *format);
            cmd_tree(&db, fmt)
        }
        // Issue #10 P-2 / P-6: db-path and project-slug are intercepted in
        // main() BEFORE run_query_command (they need no DB). Reaching here is a
        // routing bug; fail loudly rather than silently mis-handle. These arms
        // exist only to keep the match exhaustive.
        Commands::DbPath { .. } => Err(SuggesterError::IndexParse(
            "internal error: DbPath command reached run_query_command".to_string(),
        )),
        Commands::ProjectSlug { .. } => Err(SuggesterError::IndexParse(
            "internal error: ProjectSlug command reached run_query_command".to_string(),
        )),
        Commands::NlpBinaryPath => Err(SuggesterError::IndexParse(
            "internal error: NlpBinaryPath command reached run_query_command".to_string(),
        )),
    }
}

/// Export the CozoDB to a JSON snapshot for debugging / git diff workflows.
///
/// Phase B (v2.11.0) deliverable. Reads the CozoDB via load_index_from_db,
/// serialises the SkillIndex to JSON, and atomic-writes it to the requested
/// path. The default path deliberately differs from skill-index.json — the
/// legacy file stays where it is until Phase C, and this export is a
/// separate file so power users can diff the two.
fn cmd_export(db: &DbInstance, json: bool, path: Option<&str>) -> Result<(), SuggesterError> {
    if !json {
        return Err(SuggesterError::IndexParse(
            "Only --json export format is supported today".to_string(),
        ));
    }

    let dest = match path {
        Some(p) => PathBuf::from(p),
        None => {
            // Default: $CLAUDE_PLUGIN_DATA/skill-index.export.json if the
            // env var is set and PSS-scoped, else ~/.claude/cache/.
            let data_dir = std::env::var("CLAUDE_PLUGIN_DATA")
                .ok()
                .map(PathBuf::from)
                .filter(|p| {
                    p.is_absolute()
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .map(|s| s.to_ascii_lowercase().contains("perfect-skill-suggester"))
                            .unwrap_or(false)
                })
                .or_else(|| {
                    dirs::home_dir().map(|h| h.join(".claude").join(CACHE_DIR))
                })
                .ok_or_else(|| SuggesterError::IndexParse(
                    "Could not resolve default export path".to_string()
                ))?;
            data_dir.join("skill-index.export.json")
        }
    };

    eprintln!("Exporting CozoDB → JSON at {:?}", dest);
    let index = load_index_from_db(db)?;

    // Serialise. Entries are keyed by entry ID in the in-memory SkillIndex,
    // but humans expect the composite `source::name` key they saw in the
    // legacy JSON. Re-key for output so git diff against the legacy file
    // stays meaningful.
    let mut rekeyed: std::collections::BTreeMap<String, &SkillEntry> =
        std::collections::BTreeMap::new();
    for entry in index.skills.values() {
        let k = format!("{}::{}", entry.source, entry.name);
        rekeyed.insert(k, entry);
    }

    // Wrap into the top-level envelope that matches skill-index.json format.
    #[derive(Serialize)]
    struct ExportEnvelope<'a> {
        version: &'a str,
        generated: String,
        generator: &'a str,
        skill_count: usize,
        skills: &'a std::collections::BTreeMap<String, &'a SkillEntry>,
    }

    let envelope = ExportEnvelope {
        version: &index.version,
        generated: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        generator: "pss-export-json",
        skill_count: rekeyed.len(),
        skills: &rekeyed,
    };

    // Ensure parent dir exists before writing.
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| SuggesterError::IndexRead {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }

    // Atomic write via temp file + rename (same pattern as pss_merge_queue).
    let tmp_path = {
        let mut t = dest.clone();
        let fn_str = dest.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("skill-index.export.json");
        t.set_file_name(format!(".{}.tmp", fn_str));
        t
    };
    let tmp_str = tmp_path.to_str().unwrap_or_default().to_string();
    {
        let file = fs::File::create(&tmp_path).map_err(|e| SuggesterError::IndexRead {
            path: tmp_path.clone(),
            source: e,
        })?;
        serde_json::to_writer_pretty(file, &envelope).map_err(|e| {
            SuggesterError::IndexParse(format!("JSON serialisation failed: {}", e))
        })?;
    }
    fs::rename(&tmp_path, &dest).map_err(|e| SuggesterError::IndexRead {
        path: dest.clone(),
        source: e,
    })?;

    eprintln!(
        "Exported {} entries to {} (atomic write via {})",
        rekeyed.len(),
        dest.display(),
        tmp_str
    );
    Ok(())
}

/// Helper: extract string from DataValue.
pub(crate) fn dv_to_string(v: &DataValue) -> String {
    match v {
        DataValue::Str(s) => s.to_string(),
        DataValue::Num(cozo::Num::Int(n)) => n.to_string(),
        DataValue::Num(cozo::Num::Float(f)) => f.to_string(),
        _ => String::new(),
    }
}

/// Helper: extract i64 from DataValue.
pub(crate) fn dv_to_i64(v: &DataValue) -> i64 {
    match v {
        DataValue::Num(cozo::Num::Int(n)) => *n,
        DataValue::Num(cozo::Num::Float(f)) => *f as i64,
        _ => 0,
    }
}

/// Print a table with Unicode borders.
pub(crate) fn print_table(headers: &[&str], rows: &[Vec<String>]) {
    if rows.is_empty() {
        println!("(no results)");
        return;
    }
    // Calculate column widths
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(cell.len().min(80));
            }
        }
    }
    // Print header
    let sep: String = widths.iter().map(|w| "─".repeat(*w + 2)).collect::<Vec<_>>().join("┬");
    println!("┌{}┐", sep);
    let hdr: String = headers.iter().enumerate()
        .map(|(i, h)| format!(" {:<width$} ", h, width = widths[i]))
        .collect::<Vec<_>>().join("│");
    println!("│{}│", hdr);
    let sep2: String = widths.iter().map(|w| "═".repeat(*w + 2)).collect::<Vec<_>>().join("╪");
    println!("╞{}╡", sep2);
    // Print rows
    for row in rows {
        let cells: String = row.iter().enumerate()
            .map(|(i, c)| {
                let w = if i < widths.len() { widths[i] } else { 20 };
                let truncated: String = c.chars().take(w).collect();
                format!(" {:<width$} ", truncated, width = w)
            })
            .collect::<Vec<_>>().join("│");
        println!("│{}│", cells);
    }
    let sep3: String = widths.iter().map(|w| "─".repeat(*w + 2)).collect::<Vec<_>>().join("┴");
    println!("└{}┘", sep3);
    println!("({} results)", rows.len());
}

// --- cmd_stats ---

fn cmd_stats(db: &DbInstance, format: &str) -> Result<(), SuggesterError> {
    // Count by type
    let type_result = db.run_script(
        "?[skill_type, count(name)] := *skills{name, skill_type} :order -count(name)",
        Default::default(), ScriptMutability::Immutable,
    ).map_err(|e| SuggesterError::IndexParse(format!("stats query failed: {}", e)))?;

    let mut total: i64 = 0;
    let mut by_type: Vec<(String, i64)> = Vec::new();
    for row in &type_result.rows {
        let t = dv_to_string(&row[0]);
        let c = dv_to_i64(&row[1]);
        total += c;
        by_type.push((t, c));
    }

    // Count by source
    let source_result = db.run_script(
        "?[source, count(name)] := *skills{name, source} :order -count(name)",
        Default::default(), ScriptMutability::Immutable,
    ).map_err(|e| SuggesterError::IndexParse(format!("stats query failed: {}", e)))?;
    let by_source: Vec<(String, i64)> = source_result.rows.iter()
        .map(|r| (dv_to_string(&r[0]), dv_to_i64(&r[1]))).collect();

    // Phase D banner: oldest/newest first_indexed_at + newest last_updated_at.
    // Use min/max aggregates on the timestamp columns. Returns empty string
    // if the DB has no rows (aggregates over empty relation return nothing).
    let fetch_extreme = |order: &str, col: &str| -> (String, String) {
        // First row by sort order. Bound to a separate variable because
        // CozoDB can't return both min(col) and the row's other columns
        // in a single aggregate — we use :order + :limit 1 instead.
        let q = format!(
            "?[name, ts] := *skills{{ name, {col} }}, ts = {col}, ts != '' {order} :limit 1",
            col = col,
            order = order,
        );
        let res = db.run_script(&q, Default::default(), ScriptMutability::Immutable);
        match res {
            Ok(r) => {
                if let Some(row) = r.rows.first() {
                    return (dv_to_string(&row[0]), dv_to_string(&row[1]));
                }
                (String::new(), String::new())
            }
            Err(_) => (String::new(), String::new()),
        }
    };
    let (oldest_name, oldest_ts) = fetch_extreme(":order first_indexed_at", "first_indexed_at");
    let (newest_name, newest_ts) = fetch_extreme(":order -first_indexed_at", "first_indexed_at");
    let (reindex_name, reindex_ts) = fetch_extreme(":order -last_updated_at", "last_updated_at");

    // Top domains, categories, languages, frameworks, platforms, tools (top 20 each)
    let agg_query = |table: &str| -> Vec<(String, i64)> {
        db.run_script(
            &format!("?[value, count(skill_name)] := *{}{{skill_name, value}} :order -count(skill_name) :limit 20", table),
            Default::default(), ScriptMutability::Immutable,
        ).map(|r| r.rows.iter().map(|row| (dv_to_string(&row[0]), dv_to_i64(&row[1]))).collect())
         .unwrap_or_default()
    };

    let by_domain = agg_query("skill_domains");
    let by_category = db.run_script(
        "?[category, count(name)] := *skills{name, category}, category != '' :order -count(name) :limit 20",
        Default::default(), ScriptMutability::Immutable,
    ).map(|r| r.rows.iter().map(|row| (dv_to_string(&row[0]), dv_to_i64(&row[1]))).collect::<Vec<_>>())
     .unwrap_or_default();
    let by_language = agg_query("skill_languages");
    let by_framework = agg_query("skill_frameworks");
    let by_platform = agg_query("skill_platforms");
    let by_tool = agg_query("skill_tools");
    let by_service = agg_query("skill_services");

    if format == "json" {
        let mut stats = serde_json::Map::new();
        stats.insert("total".into(), serde_json::Value::Number(total.into()));
        // Phase D banner: timestamp extremes. Always present in the JSON
        // (null when the DB has no rows or no timestamp data) so scripting
        // callers don't need to branch on key existence.
        let ts_value = |name: &str, ts: &str| -> serde_json::Value {
            if ts.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::json!({"entry": name, "timestamp": ts})
            }
        };
        stats.insert("oldest_first_indexed".into(), ts_value(&oldest_name, &oldest_ts));
        stats.insert("newest_first_indexed".into(), ts_value(&newest_name, &newest_ts));
        stats.insert("last_reindex".into(), ts_value(&reindex_name, &reindex_ts));

        let to_obj = |pairs: &[(String, i64)]| -> serde_json::Value {
            let map: serde_json::Map<String, serde_json::Value> = pairs.iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::Number((*v).into())))
                .collect();
            serde_json::Value::Object(map)
        };
        stats.insert("by_type".into(), to_obj(&by_type));
        stats.insert("by_source".into(), to_obj(&by_source));
        stats.insert("by_domain".into(), to_obj(&by_domain));
        stats.insert("by_category".into(), to_obj(&by_category));
        stats.insert("by_language".into(), to_obj(&by_language));
        stats.insert("by_framework".into(), to_obj(&by_framework));
        stats.insert("by_platform".into(), to_obj(&by_platform));
        stats.insert("by_tool".into(), to_obj(&by_tool));
        stats.insert("by_service".into(), to_obj(&by_service));
        println!("{}", serde_json::to_string_pretty(&serde_json::Value::Object(stats))
            .unwrap_or_default());
    } else {
        // Phase D: "Total: N entries" is the first human-readable line, no
        // leading whitespace, no trailing colon — makes the output parseable
        // by quick `head -1 | awk ...` scripts.
        println!("Total: {} entries", total);
        if !oldest_ts.is_empty() {
            println!("Oldest first_indexed_at: {} (entry: {})", oldest_ts, oldest_name);
        }
        if !newest_ts.is_empty() {
            println!("Newest first_indexed_at: {} (entry: {})", newest_ts, newest_name);
        }
        if !reindex_ts.is_empty() {
            println!("Last reindex (newest last_updated_at): {}", reindex_ts);
        }
        let print_section = |title: &str, pairs: &[(String, i64)]| {
            println!("\n  {} ({})", title, pairs.len());
            for (k, v) in pairs {
                println!("    {:30} {:>6}", k, v);
            }
        };
        print_section("BY TYPE", &by_type);
        print_section("BY SOURCE", &by_source);
        print_section("BY DOMAIN", &by_domain);
        print_section("BY CATEGORY", &by_category);
        print_section("BY LANGUAGE", &by_language);
        print_section("BY FRAMEWORK", &by_framework);
        print_section("BY PLATFORM", &by_platform);
        print_section("BY TOOL", &by_tool);
        print_section("BY SERVICE", &by_service);
    }
    Ok(())
}

// --- cmd_vocab ---

fn cmd_vocab(db: &DbInstance, field: &str, type_filter: Option<&str>, top: usize, format: &str) -> Result<(), SuggesterError> {
    // Validate type_filter before building any query
    validate_type_filter(type_filter)?;
    let top = top.min(10000);

    // COR-4 (audit 20260514): new element types have no aux-table coverage.
    // Most vocab fields (languages/frameworks/domains/tools/...) are simply
    // empty for hook/plugin/etc. Only "types" and "scopes" are meaningful
    // — both queryable from `events`/`elements_state`.
    if is_new_element_type(type_filter) {
        let t = type_filter.unwrap();
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("ftype".into(), DataValue::Str(t.into()));
        params.insert("top".into(), DataValue::Num(cozo::Num::Int(top as i64)));
        let q = match field {
            "scopes" => Some(
                "?[value, count(element_id)] := \
                 *elements_state{element_id, last_event_id, exists: true}, \
                 *events{event_id: last_event_id, element_type, scope: value}, \
                 element_type = $ftype \
                 :order -count(element_id) :limit $top"
            ),
            "types" => Some(
                "?[value, count(element_id)] := \
                 *elements_state{element_id, last_event_id, exists: true}, \
                 *events{event_id: last_event_id, element_type: value}, \
                 element_type = $ftype \
                 :order -count(element_id) :limit $top"
            ),
            _ => None,
        };
        let entries: Vec<(String, i64)> = match q {
            Some(qs) => {
                let result = db.run_script(qs, params, ScriptMutability::Immutable)
                    .map_err(|e| SuggesterError::IndexParse(format!(
                        "vocab query failed: {}", e
                    )))?;
                result.rows.iter()
                    .map(|r| (dv_to_string(&r[0]), dv_to_i64(&r[1])))
                    .collect()
            }
            None => Vec::new(),  // aux fields aren't tracked for new types
        };
        if format == "json" {
            let arr: Vec<serde_json::Value> = entries.iter()
                .map(|(v, c)| serde_json::json!({"value": v, "count": c}))
                .collect();
            println!("{}", serde_json::to_string_pretty(&arr).unwrap_or_default());
        } else {
            println!("  {} ({} distinct values)", field.to_uppercase(), entries.len());
            print_table(&["VALUE", "COUNT"], &entries.iter()
                .map(|(v, c)| vec![v.clone(), c.to_string()])
                .collect::<Vec<_>>());
        }
        return Ok(());
    }

    // Map field name to the appropriate CozoDB table or parametrized query
    let (query, params) = match field {
        "languages" => build_vocab_query("skill_languages", type_filter, top),
        "frameworks" => build_vocab_query("skill_frameworks", type_filter, top),
        "tools" => build_vocab_query("skill_tools", type_filter, top),
        "services" => build_vocab_query("skill_services", type_filter, top),
        "domains" => build_vocab_query("skill_domains", type_filter, top),
        "keywords" => build_vocab_query("skill_keywords", type_filter, top),
        "intents" => build_vocab_query("skill_intents", type_filter, top),
        "platforms" => build_vocab_query("skill_platforms", type_filter, top),
        "file-types" | "file_types" => build_vocab_query("skill_file_types", type_filter, top),
        "categories" => {
            let mut p: BTreeMap<String, DataValue> = BTreeMap::new();
            if let Some(t) = type_filter {
                p.insert("f_type".into(), DataValue::Str(t.into()));
                (format!("?[value, count(name)] := *skills{{name, category: value, skill_type}}, skill_type = $f_type, value != '' :order -count(name) :limit {}", top), p)
            } else {
                (format!("?[value, count(name)] := *skills{{name, category: value}}, value != '' :order -count(name) :limit {}", top), p)
            }
        }
        "types" => {
            (
                "?[value, count(name)] := *skills{name, skill_type: value} :order -count(name)".to_string(),
                BTreeMap::new(),
            )
        }
        _ => return Err(SuggesterError::IndexParse(
            format!("Unknown vocab field '{}'. Valid: languages, frameworks, tools, services, domains, keywords, intents, platforms, file-types, categories, types", field)
        )),
    };

    let result = db.run_script(&query, params, ScriptMutability::Immutable)
        .map_err(|e| SuggesterError::IndexParse(format!("vocab query failed: {}", e)))?;

    let entries: Vec<(String, i64)> = result.rows.iter()
        .map(|r| (dv_to_string(&r[0]), dv_to_i64(&r[1])))
        .collect();

    if format == "json" {
        let arr: Vec<serde_json::Value> = entries.iter()
            .map(|(v, c)| serde_json::json!({"value": v, "count": c}))
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr).unwrap_or_default());
    } else {
        println!("  {} ({} distinct values)", field.to_uppercase(), entries.len());
        print_table(&["VALUE", "COUNT"], &entries.iter()
            .map(|(v, c)| vec![v.clone(), c.to_string()])
            .collect::<Vec<_>>());
    }
    Ok(())
}

fn build_vocab_query(table: &str, type_filter: Option<&str>, top: usize) -> (String, BTreeMap<String, DataValue>) {
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    if let Some(t) = type_filter {
        params.insert("f_type".into(), DataValue::Str(t.into()));
        (format!(
            "?[value, count(skill_name)] := *{}{{skill_name, value}}, *skills{{name: skill_name, skill_type}}, skill_type = $f_type :order -count(skill_name) :limit {}",
            table, top
        ), params)
    } else {
        (format!(
            "?[value, count(skill_name)] := *{}{{skill_name, value}} :order -count(skill_name) :limit {}",
            table, top
        ), params)
    }
}

// --- cmd_resolve ---

fn cmd_resolve(db: &DbInstance, ids: &[String], format: &str) -> Result<(), SuggesterError> {
    let mut results: Vec<serde_json::Value> = Vec::new();
    for ref_str in ids {
        let name = resolve_name_or_id(db, ref_str)?;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("name".into(), DataValue::Str(name.clone().into()));
        let result = db.run_script(
            "?[name, id, path, skill_type, source, description] := *skills{name, id, path, skill_type, source, description}, name = $name",
            params,
            ScriptMutability::Immutable,
        ).map_err(|e| SuggesterError::IndexParse(format!("resolve query failed: {}", e)))?;

        if result.rows.is_empty() {
            results.push(serde_json::json!({
                "ref": ref_str,
                "error": format!("not found: {}", ref_str),
            }));
        } else {
            // Return all matching rows — with composite key (name, source),
            // same name from different sources produces multiple entries
            for row in &result.rows {
                results.push(serde_json::json!({
                    "id": dv_to_string(&row[1]),
                    "name": dv_to_string(&row[0]),
                    "path": dv_to_string(&row[2]),
                    "type": dv_to_string(&row[3]),
                    "source": dv_to_string(&row[4]),
                    "description": dv_to_string(&row[5]),
                }));
            }
        }
    }

    if format == "json" {
        println!("{}", serde_json::to_string_pretty(&results).unwrap_or_default());
    } else {
        print_table(&["ID", "NAME", "TYPE", "SOURCE", "PATH"],
            &results.iter().map(|r| vec![
                r["id"].as_str().unwrap_or("").to_string(),
                r["name"].as_str().unwrap_or("").to_string(),
                r["type"].as_str().unwrap_or("").to_string(),
                r["source"].as_str().unwrap_or("").to_string(),
                r["path"].as_str().unwrap_or("").to_string(),
            ]).collect::<Vec<_>>());
    }
    Ok(())
}

// --- cmd_list ---

#[allow(clippy::too_many_arguments)]
fn cmd_list(db: &DbInstance, type_filter: Option<&str>, domain: Option<&str>,
            language: Option<&str>, framework: Option<&str>, tool: Option<&str>,
            category: Option<&str>, file_type: Option<&str>, keyword: Option<&str>,
            platform: Option<&str>, source_prefix: Option<&str>, sort: &str, top: usize, format: &str,
) -> Result<(), SuggesterError> {
    // Validate inputs — reject injection attempts before building any query
    validate_type_filter(type_filter)?;
    if let Some(c) = category { validate_filter_value("category", c)?; }
    if let Some(sp) = source_prefix {
        validate_filter_value("source_prefix", sp)?;
    }

    // COR-4 (audit 20260514): if the user asked for a new element type that
    // lives only in elements_state (hook/plugin/marketplace/...), route to
    // the elements-state-backed helper. The aux-table filters (domain,
    // language, framework, etc.) don't apply to new types — skills aux
    // tables are populated only for the legacy 6.
    if is_new_element_type(type_filter) {
        let _ = (sort, domain, language, framework, tool, category,
                 file_type, keyword, platform, source_prefix); // intentionally unused
        return cmd_list_elements_state(db, type_filter.unwrap(), None, top, format);
    }

    // Build Datalog query with AND-combined filters via parametrized queries
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    // Scalar filters use parametrized binding ($param), never format!() interpolation
    let mut conditions = String::new();
    if let Some(t) = type_filter {
        conditions.push_str(", skill_type = $f_type");
        params.insert("f_type".into(), DataValue::Str(t.into()));
    }
    if let Some(c) = category {
        conditions.push_str(", category = $f_category");
        params.insert("f_category".into(), DataValue::Str(c.into()));
    }
    // UX-9 (audit 20260514): source_prefix filter via starts_with().
    // Examples: --source-prefix "plugin:" lists all plugin-sourced
    // entries; --source-prefix "marketplace:" lists marketplace ones.
    if let Some(sp) = source_prefix {
        conditions.push_str(", starts_with(source, $f_source_prefix)");
        params.insert("f_source_prefix".into(), DataValue::Str(sp.into()));
    }

    // Normalized table joins for multi-value filters (already parametrized)
    let mut joins = String::new();
    let add_join = |joins: &mut String, params: &mut BTreeMap<String, DataValue>, table: &str, param_name: &str, value: Option<&str>| {
        if let Some(v) = value {
            joins.push_str(&format!(", *{}{{skill_name: name, value: ${}}}", table, param_name));
            params.insert(param_name.into(), DataValue::Str(v.into()));
        }
    };
    add_join(&mut joins, &mut params, "skill_domains", "f_domain", domain);
    add_join(&mut joins, &mut params, "skill_languages", "f_language", language);
    add_join(&mut joins, &mut params, "skill_frameworks", "f_framework", framework);
    add_join(&mut joins, &mut params, "skill_tools", "f_tool", tool);
    add_join(&mut joins, &mut params, "skill_file_types", "f_file_type", file_type);
    add_join(&mut joins, &mut params, "skill_keywords", "f_keyword", keyword);
    add_join(&mut joins, &mut params, "skill_platforms", "f_platform", platform);

    let order = if sort == "category" { ":order category, name" } else { ":order name" };
    let top = top.min(10000); // Cap to prevent excessive memory usage

    let query = format!(
        "?[id, name, skill_type, category, description, path, source] := \
         *skills{{name, id, skill_type, category, description, path, source}}{}{} {} :limit {}",
        conditions, joins, order, top
    );

    let result = db.run_script(&query, params, ScriptMutability::Immutable)
        .map_err(|e| SuggesterError::IndexParse(format!("list query failed: {}", e)))?;

    let entries: Vec<serde_json::Value> = result.rows.iter().map(|r| {
        serde_json::json!({
            "id": dv_to_string(&r[0]),
            "name": dv_to_string(&r[1]),
            "type": dv_to_string(&r[2]),
            "category": dv_to_string(&r[3]),
            "description": dv_to_string(&r[4]),
            "path": dv_to_string(&r[5]),
            "source": dv_to_string(&r[6]),
        })
    }).collect();

    if format == "json" {
        println!("{}", serde_json::to_string_pretty(&entries).unwrap_or_default());
    } else {
        print_table(&["ID", "NAME", "TYPE", "CATEGORY", "DESCRIPTION"],
            &entries.iter().map(|e| vec![
                e["id"].as_str().unwrap_or("").to_string(),
                e["name"].as_str().unwrap_or("").to_string(),
                e["type"].as_str().unwrap_or("").to_string(),
                e["category"].as_str().unwrap_or("").to_string(),
                e["description"].as_str().unwrap_or("").chars().take(60).collect::<String>(),
            ]).collect::<Vec<_>>());
    }
    Ok(())
}

// --- cmd_search ---

fn cmd_search(db: &DbInstance, query: &str, type_filter: Option<&str>, domain: Option<&str>,
              language: Option<&str>, framework: Option<&str>, tool: Option<&str>,
              category: Option<&str>, file_type: Option<&str>, keyword: Option<&str>,
              platform: Option<&str>, top: usize, format: &str,
) -> Result<(), SuggesterError> {
    // Validate inputs — reject injection attempts before building any query
    validate_type_filter(type_filter)?;
    if let Some(c) = category { validate_filter_value("category", c)?; }

    // COR-4 (audit 20260514): new element types live in elements_state and
    // have no aux-table keyword indexing. Approximate `search <q> --type X`
    // for new types as "list elements where name contains <q>".
    if is_new_element_type(type_filter) {
        let _ = (domain, language, framework, tool, category, file_type,
                 keyword, platform); // intentionally unused for new types
        return cmd_list_elements_state(
            db, type_filter.unwrap(), Some(query), top, format
        );
    }

    let query_lower = query.to_lowercase();
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("q".into(), DataValue::Str(query_lower.clone().into()));

    // Build union match: name contains query OR keywords contain query OR description contains query
    // All filters use parametrized binding ($param), never format!() interpolation
    let mut filter_conditions = String::new();
    let mut filter_joins = String::new();

    if let Some(t) = type_filter {
        filter_conditions.push_str(", skill_type = $f_type");
        params.insert("f_type".into(), DataValue::Str(t.into()));
    }
    if let Some(c) = category {
        filter_conditions.push_str(", category = $f_category");
        params.insert("f_category".into(), DataValue::Str(c.into()));
    }

    let add_join = |joins: &mut String, params: &mut BTreeMap<String, DataValue>, table: &str, param_name: &str, value: Option<&str>| {
        if let Some(v) = value {
            joins.push_str(&format!(", *{}{{skill_name: name, value: ${}}}", table, param_name));
            params.insert(param_name.into(), DataValue::Str(v.into()));
        }
    };
    add_join(&mut filter_joins, &mut params, "skill_domains", "f_domain", domain);
    add_join(&mut filter_joins, &mut params, "skill_languages", "f_language", language);
    add_join(&mut filter_joins, &mut params, "skill_frameworks", "f_framework", framework);
    add_join(&mut filter_joins, &mut params, "skill_tools", "f_tool", tool);
    add_join(&mut filter_joins, &mut params, "skill_file_types", "f_file_type", file_type);
    add_join(&mut filter_joins, &mut params, "skill_keywords", "f_keyword", keyword);
    add_join(&mut filter_joins, &mut params, "skill_platforms", "f_platform", platform);

    // Use Datalog: match on name (str_includes) OR keyword (str_includes) OR description (str_includes)
    // CozoDB uses str_includes(haystack, needle) for substring matching
    let top = top.min(10000); // Cap to prevent excessive memory usage
    let script = format!(
        "matches[name] := *skills{{name, skill_type, category}}, str_includes(lowercase(name), $q){fc}{fj}\n\
         matches[name] := *skill_keywords{{skill_name: name, value: kw}}, str_includes(kw, $q), *skills{{name: name, skill_type, category}}{fc}{fj}\n\
         matches[name] := *skills{{name, skill_type, category, description}}, str_includes(lowercase(description), $q){fc}{fj}\n\
         ?[id, name, skill_type, category, description, path] := matches[name], *skills{{name: name, id, skill_type, category, description, path}}\n\
         :order name\n\
         :limit {top}",
        fc = filter_conditions, fj = filter_joins, top = top
    );

    let result = db.run_script(&script, params, ScriptMutability::Immutable)
        .map_err(|e| SuggesterError::IndexParse(format!("search query failed: {}", e)))?;

    let entries: Vec<serde_json::Value> = result.rows.iter().map(|r| {
        serde_json::json!({
            "id": dv_to_string(&r[0]),
            "name": dv_to_string(&r[1]),
            "type": dv_to_string(&r[2]),
            "category": dv_to_string(&r[3]),
            "description": dv_to_string(&r[4]),
            "path": dv_to_string(&r[5]),
        })
    }).collect();

    if format == "json" {
        println!("{}", serde_json::to_string_pretty(&entries).unwrap_or_default());
    } else {
        print_table(&["ID", "NAME", "TYPE", "CATEGORY", "DESCRIPTION"],
            &entries.iter().map(|e| vec![
                e["id"].as_str().unwrap_or("").to_string(),
                e["name"].as_str().unwrap_or("").to_string(),
                e["type"].as_str().unwrap_or("").to_string(),
                e["category"].as_str().unwrap_or("").to_string(),
                e["description"].as_str().unwrap_or("").chars().take(60).collect::<String>(),
            ]).collect::<Vec<_>>());
    }
    Ok(())
}

// --- cmd_inspect ---

fn cmd_inspect(db: &DbInstance, name_or_id: &str, format: &str) -> Result<(), SuggesterError> {
    let name = resolve_name_or_id(db, name_or_id)?;
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("name".into(), DataValue::Str(name.clone().into()));

    // Fetch main entry
    let result = db.run_script(
        "?[name, id, path, skill_type, source, description, tier, boost, category, \
         server_type, server_command, server_args_json, language_ids_json, \
         negative_kw_json, patterns_json, directories_json, path_patterns_json, \
         use_cases_json, co_usage_json, alternatives_json, domain_gates_json, file_types_json, \
         keywords_json, intents_json, tools_json, services_json, frameworks_json, languages_json, platforms_json, domains_json, path_gates_json, \
         first_indexed_at, last_updated_at] := \
         *skills{ name, id, path, skill_type, source, description, tier, boost, category, \
                  server_type, server_command, server_args_json, language_ids_json, \
                  negative_kw_json, patterns_json, directories_json, path_patterns_json, \
                  use_cases_json, co_usage_json, alternatives_json, domain_gates_json, file_types_json, \
                  keywords_json, intents_json, tools_json, services_json, frameworks_json, languages_json, platforms_json, domains_json, path_gates_json, \
                  first_indexed_at, last_updated_at }, name = $name",
        params,
        ScriptMutability::Immutable,
    ).map_err(|e| SuggesterError::IndexParse(format!("inspect query failed: {}", e)))?;

    if result.rows.is_empty() {
        return Err(SuggesterError::IndexParse(format!("Entry not found: '{}'", name_or_id)));
    }

    // With composite key (name, source), multiple rows can match the same name
    // from different sources. Show all of them.
    let multiple = result.rows.len() > 1;
    if multiple && format != "json" {
        eprintln!("Note: {} entries found for '{}' (from different sources):\n", result.rows.len(), name_or_id);
    }

    // Helper: build a full JSON entry from a single row
    let build_json_entry = |row: &[DataValue]| -> serde_json::Value {
        let parse_vec = |idx: usize| -> Vec<String> {
            serde_json::from_str(&dv_to_string(&row[idx])).unwrap_or_default()
        };
        let parse_map = |idx: usize| -> HashMap<String, Vec<String>> {
            serde_json::from_str(&dv_to_string(&row[idx])).unwrap_or_default()
        };
        let mut entry = serde_json::Map::new();
        entry.insert("id".into(), serde_json::json!(dv_to_string(&row[1])));
        entry.insert("name".into(), serde_json::json!(dv_to_string(&row[0])));
        entry.insert("path".into(), serde_json::json!(dv_to_string(&row[2])));
        entry.insert("type".into(), serde_json::json!(dv_to_string(&row[3])));
        entry.insert("source".into(), serde_json::json!(dv_to_string(&row[4])));
        entry.insert("description".into(), serde_json::json!(dv_to_string(&row[5])));
        entry.insert("tier".into(), serde_json::json!(dv_to_string(&row[6])));
        entry.insert("boost".into(), serde_json::json!(dv_to_i64(&row[7])));
        entry.insert("category".into(), serde_json::json!(dv_to_string(&row[8])));
        let st = dv_to_string(&row[9]);
        if !st.is_empty() { entry.insert("server_type".into(), serde_json::json!(st)); }
        let sc = dv_to_string(&row[10]);
        if !sc.is_empty() { entry.insert("server_command".into(), serde_json::json!(sc)); }
        let sa: Vec<String> = parse_vec(11);
        if !sa.is_empty() { entry.insert("server_args".into(), serde_json::json!(sa)); }
        let li: Vec<String> = parse_vec(12);
        if !li.is_empty() { entry.insert("language_ids".into(), serde_json::json!(li)); }
        entry.insert("negative_keywords".into(), serde_json::json!(parse_vec(13)));
        entry.insert("patterns".into(), serde_json::json!(parse_vec(14)));
        entry.insert("directories".into(), serde_json::json!(parse_vec(15)));
        entry.insert("path_patterns".into(), serde_json::json!(parse_vec(16)));
        entry.insert("use_cases".into(), serde_json::json!(parse_vec(17)));
        let co_usage: CoUsageData = serde_json::from_str(&dv_to_string(&row[18])).unwrap_or_default();
        entry.insert("co_usage".into(), serde_json::json!(co_usage));
        entry.insert("alternatives".into(), serde_json::json!(parse_vec(19)));
        entry.insert("domain_gates".into(), serde_json::json!(parse_map(20)));
        entry.insert("file_types".into(), serde_json::json!(parse_vec(21)));
        entry.insert("keywords".into(), serde_json::json!(parse_vec(22)));
        entry.insert("intents".into(), serde_json::json!(parse_vec(23)));
        entry.insert("tools".into(), serde_json::json!(parse_vec(24)));
        entry.insert("services".into(), serde_json::json!(parse_vec(25)));
        entry.insert("frameworks".into(), serde_json::json!(parse_vec(26)));
        entry.insert("languages".into(), serde_json::json!(parse_vec(27)));
        entry.insert("platforms".into(), serde_json::json!(parse_vec(28)));
        entry.insert("domains".into(), serde_json::json!(parse_vec(29)));
        serde_json::Value::Object(entry)
    };

    // Helper: print a single row in table format
    let print_table_entry = |row: &[DataValue]| {
        let parse_vec = |idx: usize| -> Vec<String> {
            serde_json::from_str(&dv_to_string(&row[idx])).unwrap_or_default()
        };
        let parse_map = |idx: usize| -> HashMap<String, Vec<String>> {
            serde_json::from_str(&dv_to_string(&row[idx])).unwrap_or_default()
        };
        println!("━━━ ENTRY: {} ━━━", dv_to_string(&row[0]));
        println!("  ID          : {}", dv_to_string(&row[1]));
        println!("  Type        : {}", dv_to_string(&row[3]));
        println!("  Source      : {}", dv_to_string(&row[4]));
        println!("  Category    : {}", dv_to_string(&row[8]));
        println!("  Tier        : {}", dv_to_string(&row[6]));
        println!("  Boost       : {}", dv_to_i64(&row[7]));
        println!("  Description : {}", dv_to_string(&row[5]));
        println!("  Path        : {}", dv_to_string(&row[2]));
        let print_v = |label: &str, idx: usize| {
            let v: Vec<String> = parse_vec(idx);
            if !v.is_empty() { println!("  {:12}: {}", label, v.join(", ")); }
        };
        print_v("Keywords", 22);
        print_v("Intents", 23);
        print_v("Languages", 27);
        print_v("Frameworks", 26);
        print_v("Platforms", 28);
        print_v("Domains", 29);
        print_v("Tools", 24);
        print_v("Services", 25);
        print_v("File types", 21);
        print_v("Use cases", 17);
        print_v("Alternatives", 19);
        print_v("Neg keywords", 13);
        let gates = parse_map(20);
        if !gates.is_empty() {
            println!("  Domain gates:");
            for (k, v) in &gates {
                println!("    {} → {}", k, v.join(", "));
            }
        }
    };

    if format == "json" {
        if multiple {
            // Multiple entries with same name — return array
            let entries: Vec<serde_json::Value> = result.rows.iter()
                .map(|row| build_json_entry(row))
                .collect();
            println!("{}", serde_json::to_string_pretty(&entries).unwrap_or_default());
        } else {
            // Single entry — return object
            println!("{}", serde_json::to_string_pretty(&build_json_entry(&result.rows[0])).unwrap_or_default());
        }
    } else {
        for (i, row) in result.rows.iter().enumerate() {
            if i > 0 { println!(); }
            print_table_entry(row);
            let co: CoUsageData = serde_json::from_str(&dv_to_string(&row[18])).unwrap_or_default();
            if !co.usually_with.is_empty() { println!("  Usually with: {}", co.usually_with.join(", ")); }
            if !co.precedes.is_empty() { println!("  Precedes    : {}", co.precedes.join(", ")); }
            if !co.follows.is_empty() { println!("  Follows     : {}", co.follows.join(", ")); }
        }
    }
    Ok(())
}

// --- cmd_get_description ---

/// Extract plugin name from the `source` field.
/// Formats: "user" → None, "plugin:owner/name" → "owner/name", "marketplace:name" → "name"
fn extract_plugin_from_source(source: &str) -> Option<String> {
    if let Some(rest) = source.strip_prefix("plugin:") {
        Some(rest.to_string())
    } else if let Some(rest) = source.strip_prefix("marketplace:") {
        Some(rest.to_string())
    } else {
        None
    }
}

/// Derive a human-readable scope label from the `source` field.
/// "user" → "user", "project" → "project",
/// "plugin:owner/name" → "installed", "marketplace:name" → "marketplace"
fn derive_scope(source: &str) -> &'static str {
    if source.starts_with("plugin:") {
        "installed"
    } else if source.starts_with("marketplace:") {
        "marketplace"
    } else if source == "project" {
        "project"
    } else {
        "user"
    }
}

/// Build a lightweight JSON object from a CozoDB row (6 columns:
/// name, skill_type, source, description, path, keywords_json).
fn row_to_description_json(row: &[DataValue]) -> serde_json::Value {
    let source_str = dv_to_string(&row[2]);
    let keywords: Vec<String> = serde_json::from_str(&dv_to_string(&row[5])).unwrap_or_default();

    let mut obj = serde_json::Map::new();
    obj.insert("name".into(), serde_json::json!(dv_to_string(&row[0])));
    obj.insert("type".into(), serde_json::json!(dv_to_string(&row[1])));
    obj.insert("description".into(), serde_json::json!(dv_to_string(&row[3])));
    obj.insert("source_path".into(), serde_json::json!(dv_to_string(&row[4])));
    obj.insert("source".into(), serde_json::json!(&source_str));
    obj.insert("scope".into(), serde_json::json!(derive_scope(&source_str)));
    // Plugin field: extracted from source, null if user-owned
    obj.insert("plugin".into(), match extract_plugin_from_source(&source_str) {
        Some(p) => serde_json::json!(p),
        None => serde_json::Value::Null,
    });
    // Trigger/keywords — first 20 keywords to keep response lightweight
    let trigger: Vec<&String> = keywords.iter().take(20).collect();
    obj.insert("trigger".into(), serde_json::json!(trigger));

    serde_json::Value::Object(obj)
}

/// Parse a possibly-namespaced element reference.
///
/// Claude Code convention (from docs + installed_plugins.json v2):
///   `:` separates namespace from element name
///   `@` separates plugin-name from marketplace-name within a namespace
///
/// Supported formats:
///   "element-name"                                → (None, "element-name")
///   "plugin-name:element-name"                    → (Some("plugin-name"), "element-name")
///   "owner/plugin:element-name"                   → (Some("owner/plugin"), "element-name")
///   "plugin-name@marketplace:element-name"        → (Some("plugin-name@marketplace"), "element-name")
///
/// The namespace is matched against the `source` field.  If the namespace
/// contains `@` (e.g. "cpv@emasoft-plugins"), BOTH the plugin part and the
/// marketplace part must appear in the source string.  This handles:
///   source = "plugin:emasoft-plugins/claude-plugins-validation"
///   namespace = "claude-plugins-validation@emasoft-plugins"
///   → matches because source contains both "claude-plugins-validation" and "emasoft-plugins"
fn parse_namespaced_ref(input: &str) -> (Option<String>, String) {
    let trimmed = input.trim();

    // Try `:` separator — everything after the LAST `:` is the element name
    if let Some(colon_pos) = trimmed.rfind(':') {
        let ns = trimmed[..colon_pos].trim();
        let elem = trimmed[colon_pos + 1..].trim();
        if !ns.is_empty() && !elem.is_empty() {
            return (Some(ns.to_string()), elem.to_string());
        }
    }

    (None, trimmed.to_string())
}

/// Smart lookup: resolves an element reference to one or more matching rows.
///
/// Resolution strategy (in order):
/// 1. If input is a 13-char ID → resolve via skill_ids table
/// 2. If input contains `:` or `@` → parse namespace, query by name + source LIKE
/// 3. Exact name match (case-sensitive)
/// 4. Case-insensitive name match (fallback)
///
/// Returns a Vec of JSON objects.  Empty vec = not found.
/// Multiple results = ambiguity (caller decides how to handle).
fn lookup_descriptions(db: &DbInstance, input: &str) -> Vec<serde_json::Value> {
    let query_cols = "?[name, skill_type, source, description, path, keywords_json]";
    let from_skills = "*skills{ name, skill_type, source, description, path, keywords_json }";

    // Step 1: ID resolution (13-char alphanumeric)
    let resolved_name = match resolve_name_or_id(db, input) {
        Ok(n) => n,
        Err(_) => return vec![],
    };
    // If resolve changed the input (was an ID), use the resolved name directly
    if resolved_name != input {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("name".into(), DataValue::Str(resolved_name.clone().into()));
        if let Ok(result) = db.run_script(
            &format!("{} := {}, name = $name", query_cols, from_skills),
            params, ScriptMutability::Immutable,
        ) {
            return result.rows.iter().map(|r| row_to_description_json(r)).collect();
        }
        return vec![];
    }

    // Step 2: Parse namespace
    let (namespace, element_name) = parse_namespaced_ref(input);

    if let Some(ns) = &namespace {
        // Namespaced lookup: query by element name + source matches namespace.
        //
        // If namespace contains `@` (e.g. "cpv@emasoft-plugins"), split on `@`
        // and require BOTH parts to appear in the source field.  This handles
        // the Claude Code "plugin-name@marketplace" convention matching against
        // source values like "plugin:emasoft-plugins/claude-plugins-validation".
        let ns_lower = ns.to_lowercase();
        let ns_parts: Vec<&str> = ns_lower.split('@').collect();

        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("name".into(), DataValue::Str(element_name.clone().into()));

        let query = if ns_parts.len() >= 2 && !ns_parts[0].is_empty() && !ns_parts[1].is_empty() {
            // "plugin@marketplace" → both parts must match source
            params.insert("ns1".into(), DataValue::Str(ns_parts[0].into()));
            params.insert("ns2".into(), DataValue::Str(ns_parts[1].into()));
            format!("{} := {}, name = $name, str_includes(lowercase(source), $ns1), str_includes(lowercase(source), $ns2)",
                query_cols, from_skills)
        } else {
            // Simple namespace (no @) — single substring match
            params.insert("ns".into(), DataValue::Str(ns_lower.into()));
            format!("{} := {}, name = $name, str_includes(lowercase(source), $ns)",
                query_cols, from_skills)
        };

        if let Ok(result) = db.run_script(&query, params, ScriptMutability::Immutable) {
            if !result.rows.is_empty() {
                return result.rows.iter().map(|r| row_to_description_json(r)).collect();
            }
        }
        // Namespace didn't match — fall through to try element_name alone
    }

    // Step 3: Exact name match (use the element name, which is the full input if no namespace)
    let lookup_name = if namespace.is_some() { &element_name } else { &resolved_name };
    {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("name".into(), DataValue::Str(lookup_name.clone().into()));
        if let Ok(result) = db.run_script(
            &format!("{} := {}, name = $name", query_cols, from_skills),
            params, ScriptMutability::Immutable,
        ) {
            if !result.rows.is_empty() {
                return result.rows.iter().map(|r| row_to_description_json(r)).collect();
            }
        }
    }

    // Step 4: Case-insensitive fallback — bind lowercase(name) then compare
    {
        let name_lower = lookup_name.to_lowercase();
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("q".into(), DataValue::Str(name_lower.into()));
        if let Ok(result) = db.run_script(
            &format!("{} := {}, lc = lowercase(name), lc = $q", query_cols, from_skills),
            params, ScriptMutability::Immutable,
        ) {
            if !result.rows.is_empty() {
                return result.rows.iter().map(|r| row_to_description_json(r)).collect();
            }
        }
    }

    // Step 5: Fall back to the `rules` table — rules are stored separately
    // because they're auto-injected (not suggestable) but still need description lookups
    let rule_results = lookup_rule(db, lookup_name);
    if !rule_results.is_empty() {
        return rule_results;
    }

    vec![]
}

fn cmd_get_description(db: &DbInstance, names: &str, batch: bool, format: &str) -> Result<(), SuggesterError> {
    if batch {
        // Batch mode: comma-separated names → JSON array
        let entries: Vec<serde_json::Value> = names
            .split(',')
            .map(|n| n.trim())
            .filter(|n| !n.is_empty())
            .map(|n| {
                let results = lookup_descriptions(db, n);
                match results.len() {
                    0 => serde_json::Value::Null,
                    1 => results.into_iter().next().unwrap(),
                    // Multiple matches: return array with disambiguation
                    _ => serde_json::json!({
                        "ambiguous": true,
                        "query": n,
                        "matches": results,
                    }),
                }
            })
            .collect();

        if format == "json" {
            println!("{}", serde_json::to_string_pretty(&entries).unwrap_or_default());
        } else {
            // Table format for batch
            println!("{:<30} {:<10} {:<12} {:<40} {:<30}", "NAME", "TYPE", "SCOPE", "DESCRIPTION", "PLUGIN");
            println!("{}", "─".repeat(122));
            for e in &entries {
                if e.is_null() {
                    println!("{:<30} (not found)", "?");
                    continue;
                }
                if e.get("ambiguous").is_some() {
                    // Print each match from the ambiguous result
                    let query = e["query"].as_str().unwrap_or("?");
                    if let Some(matches) = e["matches"].as_array() {
                        for m in matches {
                            let name = m["name"].as_str().unwrap_or("?");
                            let etype = m["type"].as_str().unwrap_or("?");
                            let scope = m["scope"].as_str().unwrap_or("?");
                            let desc = m["description"].as_str().unwrap_or("");
                            let desc_short = if desc.len() > 37 { format!("{}...", &desc[..37]) } else { desc.to_string() };
                            let plugin = m["plugin"].as_str().unwrap_or("-");
                            println!("{:<30} {:<10} {:<12} {:<40} {:<30}  ← ambiguous '{}'", name, etype, scope, desc_short, plugin, query);
                        }
                    }
                    continue;
                }
                let name = e["name"].as_str().unwrap_or("?");
                let etype = e["type"].as_str().unwrap_or("?");
                let scope = e["scope"].as_str().unwrap_or("?");
                let desc = e["description"].as_str().unwrap_or("");
                let desc_short = if desc.len() > 37 { format!("{}...", &desc[..37]) } else { desc.to_string() };
                let plugin = e["plugin"].as_str().unwrap_or("-");
                println!("{:<30} {:<10} {:<12} {:<40} {:<30}", name, etype, scope, desc_short, plugin);
            }
        }
    } else {
        // Single mode
        let results = lookup_descriptions(db, names);
        match results.len() {
            0 => return Err(SuggesterError::IndexParse(format!("Entry not found: '{}'", names))),
            1 => {
                let entry = &results[0];
                if format == "json" {
                    println!("{}", serde_json::to_string_pretty(entry).unwrap_or_default());
                } else {
                    println!("  Name        : {}", entry["name"].as_str().unwrap_or("?"));
                    println!("  Type        : {}", entry["type"].as_str().unwrap_or("?"));
                    println!("  Scope       : {}", entry["scope"].as_str().unwrap_or("?"));
                    println!("  Plugin      : {}", entry["plugin"].as_str().unwrap_or("-"));
                    println!("  Source      : {}", entry["source"].as_str().unwrap_or("?"));
                    println!("  Description : {}", entry["description"].as_str().unwrap_or(""));
                    println!("  Source path : {}", entry["source_path"].as_str().unwrap_or("?"));
                    if let Some(triggers) = entry["trigger"].as_array() {
                        let kws: Vec<&str> = triggers.iter().filter_map(|v| v.as_str()).collect();
                        if !kws.is_empty() {
                            println!("  Trigger     : {}", kws.join(", "));
                        }
                    }
                }
            }
            _ => {
                // Multiple matches — report ambiguity
                if format == "json" {
                    let obj = serde_json::json!({
                        "ambiguous": true,
                        "query": names,
                        "count": results.len(),
                        "hint": "Use namespace:name to disambiguate (e.g. 'trailofbits:element' or 'emasoft-plugins/cpv:element')",
                        "matches": results,
                    });
                    println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
                } else {
                    println!("  AMBIGUOUS: '{}' matched {} entries. Use namespace:name to disambiguate.\n", names, results.len());
                    for entry in &results {
                        println!("  • {} [{}] — scope: {}, plugin: {}",
                            entry["name"].as_str().unwrap_or("?"),
                            entry["type"].as_str().unwrap_or("?"),
                            entry["scope"].as_str().unwrap_or("?"),
                            entry["plugin"].as_str().unwrap_or("-"),
                        );
                        println!("    {}", entry["description"].as_str().unwrap_or(""));
                    }
                }
            }
        }
    }
    Ok(())
}

// --- cmd_index_rules ---

/// Extract a human-readable description from a rule file's content.
/// Takes the first non-empty, non-heading line (skipping `#`-prefixed lines,
/// blank lines, and `**bold**`-only lines that serve as sub-headings).
fn extract_rule_description(content: &str) -> String {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        // Skip markdown headings
        if trimmed.starts_with('#') { continue; }
        // Skip bold-only lines like "**LESSON LEARNED:**" (sub-heading style)
        if trimmed.starts_with("**") && trimmed.ends_with("**") { continue; }
        // Use this line as description, truncated to 200 chars
        let desc = if trimmed.len() > 200 { &trimmed[..200] } else { trimmed };
        return desc.to_string();
    }
    String::new()
}

/// Ensure the `rules` table exists in an existing DB (migration-safe).
/// Silently succeeds if the table already exists.
fn ensure_rules_table(db: &DbInstance) {
    let _ = db.run_script(
        r#"{:create rules {
            name: String, scope: String =>
            description: String,
            source_path: String,
            summary: String,
            keywords_json: String
        }}"#,
        Default::default(),
        ScriptMutability::Mutable,
    );
    // Ignore error — if table already exists, CozoDB returns an error which is fine
}

/// Scan rule file directories and upsert metadata into the `rules` CozoDB table.
fn cmd_index_rules(db: &DbInstance, project_root: Option<&str>, format: &str) -> Result<(), SuggesterError> {
    // Ensure the rules table exists (handles DBs built before this feature)
    ensure_rules_table(db);

    let mut indexed: Vec<serde_json::Value> = Vec::new();

    // Collect rule dirs: user-level (~/.claude/rules/) + project-level (.claude/rules/)
    let mut rule_dirs: Vec<(PathBuf, &str)> = Vec::new();

    // User-level rules
    if let Some(home) = dirs::home_dir() {
        let user_rules = home.join(".claude").join("rules");
        if user_rules.is_dir() {
            rule_dirs.push((user_rules, "user"));
        }
    }

    // Project-level rules
    let proj = project_root
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    let project_rules = proj.join(".claude").join("rules");
    if project_rules.is_dir() {
        rule_dirs.push((project_rules, "project"));
    }

    for (dir, scope) in &rule_dirs {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            // Only .md files
            if path.extension().and_then(|e| e.to_str()) != Some("md") { continue; }
            // Rule name = filename without extension
            let name = match path.file_stem().and_then(|s| s.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            // Read file content and extract description
            let content = match fs::read_to_string(&path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let description = extract_rule_description(&content);
            let source_path = path.to_string_lossy().to_string();

            // Upsert into rules table (name + scope is the composite primary key)
            let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
            params.insert("name".into(), DataValue::Str(name.clone().into()));
            params.insert("scope".into(), DataValue::Str((*scope).into()));
            params.insert("desc".into(), DataValue::Str(description.clone().into()));
            params.insert("path".into(), DataValue::Str(source_path.clone().into()));
            // summary and keywords_json start empty — enriched later by profiler
            params.insert("summary".into(), DataValue::Str("".into()));
            params.insert("kw".into(), DataValue::Str("[]".into()));

            if let Err(e) = db.run_script(
                r#"?[name, scope, description, source_path, summary, keywords_json] <-
                    [[$name, $scope, $desc, $path, $summary, $kw]]
                :put rules { name, scope => description, source_path, summary, keywords_json }"#,
                params,
                ScriptMutability::Mutable,
            ) {
                eprintln!("Warning: failed to index rule '{}': {}", name, e);
                continue;
            }

            indexed.push(serde_json::json!({
                "name": name,
                "scope": scope,
                "description": description,
                "source_path": source_path,
            }));
        }
    }

    // Output results
    if format == "json" {
        let result = serde_json::json!({
            "indexed": indexed.len(),
            "rules": indexed,
        });
        println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
    } else {
        println!("{:<30} {:<10} {:<60}", "NAME", "SCOPE", "DESCRIPTION");
        println!("{}", "─".repeat(100));
        for r in &indexed {
            let name = r["name"].as_str().unwrap_or("?");
            let scope = r["scope"].as_str().unwrap_or("?");
            let desc = r["description"].as_str().unwrap_or("");
            let desc_short = if desc.len() > 57 { format!("{}...", &desc[..57]) } else { desc.to_string() };
            println!("{:<30} {:<10} {:<60}", name, scope, desc_short);
        }
        println!("\nIndexed {} rule(s).", indexed.len());
    }

    Ok(())
}

// --- cmd_list_rules ---

/// List all indexed rules from the `rules` CozoDB table.
fn cmd_list_rules(db: &DbInstance, scope_filter: Option<&str>, format: &str) -> Result<(), SuggesterError> {
    let query = if let Some(scope) = scope_filter {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("scope".into(), DataValue::Str(scope.into()));
        db.run_script(
            "?[name, scope, description, source_path, summary, keywords_json] := *rules{ name, scope, description, source_path, summary, keywords_json }, scope = $scope",
            params,
            ScriptMutability::Immutable,
        )
    } else {
        db.run_script(
            "?[name, scope, description, source_path, summary, keywords_json] := *rules{ name, scope, description, source_path, summary, keywords_json }",
            Default::default(),
            ScriptMutability::Immutable,
        )
    };

    let result = query.map_err(|e| SuggesterError::IndexParse(format!("list-rules query failed: {}", e)))?;

    if format == "json" {
        let rules: Vec<serde_json::Value> = result.rows.iter().map(|row| {
            let keywords: Vec<String> = serde_json::from_str(&dv_to_string(&row[5])).unwrap_or_default();
            serde_json::json!({
                "name": dv_to_string(&row[0]),
                "scope": dv_to_string(&row[1]),
                "description": dv_to_string(&row[2]),
                "source_path": dv_to_string(&row[3]),
                "summary": dv_to_string(&row[4]),
                "keywords": keywords,
                "type": "rule",
            })
        }).collect();
        println!("{}", serde_json::to_string_pretty(&rules).unwrap_or_default());
    } else {
        println!("{:<30} {:<10} {:<60}", "NAME", "SCOPE", "DESCRIPTION");
        println!("{}", "─".repeat(100));
        for row in &result.rows {
            let name = dv_to_string(&row[0]);
            let scope = dv_to_string(&row[1]);
            let desc = dv_to_string(&row[2]);
            let desc_short = if desc.len() > 57 { format!("{}...", &desc[..57]) } else { desc.to_string() };
            println!("{:<30} {:<10} {:<60}", name, scope, desc_short);
        }
        println!("\n{} rule(s) found.", result.rows.len());
    }

    Ok(())
}

/// Look up a rule by name from the `rules` table. Returns a description JSON
/// in the same shape as `row_to_description_json()` for consistency with
/// `get-description` output.
fn lookup_rule(db: &DbInstance, name: &str) -> Vec<serde_json::Value> {
    // Ensure rules table exists (no-op if already present)
    ensure_rules_table(db);

    // Exact match first
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("name".into(), DataValue::Str(name.into()));
    if let Ok(result) = db.run_script(
        "?[name, scope, description, source_path, summary, keywords_json] := *rules{ name, scope, description, source_path, summary, keywords_json }, name = $name",
        params,
        ScriptMutability::Immutable,
    ) {
        if !result.rows.is_empty() {
            return result.rows.iter().map(|row| {
                let keywords: Vec<String> = serde_json::from_str(&dv_to_string(&row[5])).unwrap_or_default();
                let trigger: Vec<&String> = keywords.iter().take(20).collect();
                serde_json::json!({
                    "name": dv_to_string(&row[0]),
                    "type": "rule",
                    "scope": dv_to_string(&row[1]),
                    "description": dv_to_string(&row[2]),
                    "source_path": dv_to_string(&row[3]),
                    "source": format!("rule:{}", dv_to_string(&row[1])),
                    "plugin": serde_json::Value::Null,
                    "trigger": trigger,
                })
            }).collect();
        }
    }

    // Case-insensitive fallback
    let name_lower = name.to_lowercase();
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("q".into(), DataValue::Str(name_lower.into()));
    if let Ok(result) = db.run_script(
        "?[name, scope, description, source_path, summary, keywords_json] := *rules{ name, scope, description, source_path, summary, keywords_json }, lc = lowercase(name), lc = $q",
        params,
        ScriptMutability::Immutable,
    ) {
        if !result.rows.is_empty() {
            return result.rows.iter().map(|row| {
                let keywords: Vec<String> = serde_json::from_str(&dv_to_string(&row[5])).unwrap_or_default();
                let trigger: Vec<&String> = keywords.iter().take(20).collect();
                serde_json::json!({
                    "name": dv_to_string(&row[0]),
                    "type": "rule",
                    "scope": dv_to_string(&row[1]),
                    "description": dv_to_string(&row[2]),
                    "source_path": dv_to_string(&row[3]),
                    "source": format!("rule:{}", dv_to_string(&row[1])),
                    "plugin": serde_json::Value::Null,
                    "trigger": trigger,
                })
            }).collect();
        }
    }

    vec![]
}

// --- cmd_compare ---

fn cmd_compare(db: &DbInstance, ref1: &str, ref2: &str, format: &str) -> Result<(), SuggesterError> {
    // Load both entries via normalized tables for set operations
    let load_sets = |name: &str| -> Result<HashMap<String, HashSet<String>>, SuggesterError> {
        let mut sets: HashMap<String, HashSet<String>> = HashMap::new();
        let tables = [
            ("keywords", "skill_keywords"), ("intents", "skill_intents"),
            ("tools", "skill_tools"), ("services", "skill_services"),
            ("frameworks", "skill_frameworks"),
            ("languages", "skill_languages"), ("platforms", "skill_platforms"),
            ("domains", "skill_domains"), ("file_types", "skill_file_types"),
        ];
        for (field, table) in &tables {
            let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
            params.insert("name".into(), DataValue::Str(name.into()));
            let result = db.run_script(
                &format!("?[value] := *{}{{skill_name: $name, value}}", table),
                params, ScriptMutability::Immutable,
            ).map_err(|e| SuggesterError::IndexParse(format!("compare query failed: {}", e)))?;
            let values: HashSet<String> = result.rows.iter().map(|r| dv_to_string(&r[0])).collect();
            sets.insert(field.to_string(), values);
        }
        Ok(sets)
    };

    let name1 = resolve_name_or_id(db, ref1)?;
    let name2 = resolve_name_or_id(db, ref2)?;

    // Load scalar fields for both. With composite key (name, source), a name
    // query can return multiple rows — use the first and warn on stderr.
    let load_scalars = |name: &str| -> Result<serde_json::Value, SuggesterError> {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("name".into(), DataValue::Str(name.into()));
        let result = db.run_script(
            "?[id, skill_type, source, category, tier, boost, description] := \
             *skills{name: $name, id, skill_type, source, category, tier, boost, description}",
            params, ScriptMutability::Immutable,
        ).map_err(|e| SuggesterError::IndexParse(format!("compare query failed: {}", e)))?;
        let row = result.rows.first()
            .ok_or_else(|| SuggesterError::IndexParse(format!("Entry not found: '{}'", name)))?;
        if result.rows.len() > 1 {
            eprintln!("Warning: '{}' matches {} entries from different sources; comparing first (source: {}). Use IDs for precision.",
                name, result.rows.len(), dv_to_string(&row[2]));
        }
        Ok(serde_json::json!({
            "id": dv_to_string(&row[0]), "type": dv_to_string(&row[1]),
            "source": dv_to_string(&row[2]), "category": dv_to_string(&row[3]),
            "tier": dv_to_string(&row[4]), "boost": dv_to_i64(&row[5]),
            "description": dv_to_string(&row[6]),
        }))
    };

    let scalars_a = load_scalars(&name1)?;
    let scalars_b = load_scalars(&name2)?;
    let sets_a = load_sets(&name1)?;
    let sets_b = load_sets(&name2)?;

    // Compute shared/unique per field
    let fields = ["keywords", "intents", "tools", "frameworks", "languages", "platforms", "domains", "file_types"];
    let mut shared: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut unique_a: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut unique_b: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();

    for field in &fields {
        let empty = HashSet::new();
        let a = sets_a.get(*field).unwrap_or(&empty);
        let b = sets_b.get(*field).unwrap_or(&empty);
        let s: Vec<&String> = a.intersection(b).collect();
        let ua: Vec<&String> = a.difference(b).collect();
        let ub: Vec<&String> = b.difference(a).collect();
        if !s.is_empty() { shared.insert(field.to_string(), serde_json::json!(s)); }
        if !ua.is_empty() { unique_a.insert(field.to_string(), serde_json::json!(ua)); }
        if !ub.is_empty() { unique_b.insert(field.to_string(), serde_json::json!(ub)); }
    }

    // Scalar diffs
    let mut scalar_diffs: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    for key in &["type", "source", "category", "tier"] {
        let va = scalars_a[key].as_str().unwrap_or("");
        let vb = scalars_b[key].as_str().unwrap_or("");
        if va != vb { scalar_diffs.insert(key.to_string(), serde_json::json!([va, vb])); }
    }
    let ba = scalars_a["boost"].as_i64().unwrap_or(0);
    let bb = scalars_b["boost"].as_i64().unwrap_or(0);
    if ba != bb { scalar_diffs.insert("boost".into(), serde_json::json!([ba, bb])); }

    // UX-6 (audit 20260514): Jaccard similarity across all aux-table
    // fields combined. `|A ∩ B| / |A ∪ B|` summed across every field,
    // weighted equally per field. Ranges 0.0 (no overlap) → 1.0
    // (identical aux signature). Empty fields contribute 0 to both
    // numerator and denominator → don't drag the score down to 0.
    let similarity = {
        let mut total_inter: usize = 0;
        let mut total_union: usize = 0;
        for field in &fields {
            let empty = HashSet::new();
            let a = sets_a.get(*field).unwrap_or(&empty);
            let b = sets_b.get(*field).unwrap_or(&empty);
            total_inter += a.intersection(b).count();
            total_union += a.union(b).count();
        }
        if total_union == 0 { 0.0 } else { total_inter as f64 / total_union as f64 }
    };

    if format == "json" {
        let output = serde_json::json!({
            "entry_a": {"name": name1, "scalars": scalars_a},
            "entry_b": {"name": name2, "scalars": scalars_b},
            "shared": serde_json::Value::Object(shared),
            "unique_a": serde_json::Value::Object(unique_a),
            "unique_b": serde_json::Value::Object(unique_b),
            "scalar_diffs": serde_json::Value::Object(scalar_diffs),
            "similarity": (similarity * 10000.0).round() / 10000.0,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
    } else {
        println!("━━━ COMPARISON: {} vs {}  (similarity: {:.2}%) ━━━",
                 name1, name2, similarity * 100.0);
        println!("\n  {} ({})", name1, scalars_a["type"].as_str().unwrap_or(""));
        println!("    {}", scalars_a["description"].as_str().unwrap_or(""));
        println!("\n  {} ({})", name2, scalars_b["type"].as_str().unwrap_or(""));
        println!("    {}", scalars_b["description"].as_str().unwrap_or(""));

        if !scalar_diffs.is_empty() {
            println!("\n  SCALAR DIFFERENCES:");
            for (k, v) in &scalar_diffs { println!("    {:12}: {} vs {}", k, v[0], v[1]); }
        }
        for field in &fields {
            let print_set = |label: &str, map: &serde_json::Map<String, serde_json::Value>| {
                if let Some(v) = map.get(*field) {
                    if let Some(arr) = v.as_array() {
                        let items: Vec<String> = arr.iter().filter_map(|x| x.as_str().map(String::from)).collect();
                        if !items.is_empty() { println!("    {}: {}", label, items.join(", ")); }
                    }
                }
            };
            let has_data = shared.contains_key(*field) || unique_a.contains_key(*field) || unique_b.contains_key(*field);
            if has_data {
                println!("\n  {}:", field.to_uppercase());
                print_set("Shared", &shared);
                print_set(&format!("Only {}", name1), &unique_a);
                print_set(&format!("Only {}", name2), &unique_b);
            }
        }
    }
    Ok(())
}

// --- cmd_coverage ---

fn cmd_coverage(db: &DbInstance, type_filter: Option<&str>, format: &str) -> Result<(), SuggesterError> {
    // Validate type filter against whitelist to prevent injection
    validate_type_filter(type_filter)?;

    // COR-4 (audit 20260514): new element types live in elements_state with
    // no aux-table coverage. Emit a basic-shape coverage report (just the
    // total count) — the language/framework/domain breakdowns are empty
    // because those aux tables aren't populated for new types.
    if is_new_element_type(type_filter) {
        let t = type_filter.unwrap();
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("ftype".into(), DataValue::Str(t.into()));
        let total_result = db.run_script(
            "?[count(element_id)] := \
             *elements_state{element_id, last_event_id, exists: true}, \
             *events{event_id: last_event_id, element_type}, \
             element_type = $ftype",
            params,
            ScriptMutability::Immutable,
        ).map_err(|e| SuggesterError::IndexParse(format!(
            "coverage query (elements_state) failed: {}", e
        )))?;
        let total = total_result.rows.first().map(|r| dv_to_i64(&r[0])).unwrap_or(0);
        if format == "json" {
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                "type": t,
                "total": total,
                "note": "new element types (hook/plugin/marketplace/monitor/output-style/theme) live in elements_state only — no aux-table breakdowns",
                "languages": {}, "frameworks": {}, "domains": {},
                "tools": {}, "services": {}, "platforms": {},
                "language_count": 0, "framework_count": 0, "domain_count": 0,
            })).unwrap_or_default());
        } else {
            println!("━━━ COVERAGE: {} ━━━", t);
            println!("  Total entries: {}", total);
            println!("  (new element types have no aux-table breakdowns)");
        }
        return Ok(());
    }

    // Build parametrized queries — never interpolate user input into Datalog
    let mut total_params: BTreeMap<String, DataValue> = BTreeMap::new();
    let total_query = match type_filter {
        Some(t) => {
            total_params.insert("f_type".into(), DataValue::Str(t.into()));
            "?[count(name)] := *skills{name, skill_type}, skill_type = $f_type".to_string()
        }
        None => "?[count(name)] := *skills{name}".to_string(),
    };

    let total_result = db.run_script(&total_query, total_params, ScriptMutability::Immutable)
        .map_err(|e| SuggesterError::IndexParse(format!("coverage query failed: {}", e)))?;
    let total = total_result.rows.first().map(|r| dv_to_i64(&r[0])).unwrap_or(0);

    // Type filter clause and params for coverage sub-queries
    let (type_clause, type_param) = match type_filter {
        Some(t) => (
            ", *skills{name: skill_name, skill_type}, skill_type = $f_type".to_string(),
            Some(("f_type".to_string(), DataValue::Str(t.into()))),
        ),
        None => (String::new(), None),
    };

    let coverage_query = |table: &str| -> Vec<(String, i64)> {
        let q = format!(
            "?[value, count(skill_name)] := *{}{{skill_name, value}}{} :order -count(skill_name) :limit 50",
            table, type_clause
        );
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        if let Some((ref k, ref v)) = type_param {
            params.insert(k.clone(), v.clone());
        }
        db.run_script(&q, params, ScriptMutability::Immutable)
            .map(|r| r.rows.iter().map(|row| (dv_to_string(&row[0]), dv_to_i64(&row[1]))).collect())
            .unwrap_or_default()
    };

    let languages = coverage_query("skill_languages");
    let frameworks = coverage_query("skill_frameworks");
    let domains = coverage_query("skill_domains");
    let tools = coverage_query("skill_tools");
    let services = coverage_query("skill_services");
    let platforms = coverage_query("skill_platforms");

    if format == "json" {
        let to_obj = |pairs: &[(String, i64)]| -> serde_json::Value {
            let map: serde_json::Map<String, serde_json::Value> = pairs.iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::Number((*v).into())))
                .collect();
            serde_json::Value::Object(map)
        };
        let output = serde_json::json!({
            "type": type_filter.unwrap_or("all"),
            "total": total,
            "languages": to_obj(&languages),
            "frameworks": to_obj(&frameworks),
            "domains": to_obj(&domains),
            "tools": to_obj(&tools),
            "services": to_obj(&services),
            "platforms": to_obj(&platforms),
            "language_count": languages.len(),
            "framework_count": frameworks.len(),
            "domain_count": domains.len(),
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
    } else {
        println!("━━━ COVERAGE: {} ━━━", type_filter.unwrap_or("all types"));
        println!("  Total entries: {}", total);
        let print_coverage = |title: &str, pairs: &[(String, i64)]| {
            println!("\n  {} ({} distinct):", title, pairs.len());
            for (k, v) in pairs {
                let pct = if total > 0 { *v as f64 / total as f64 * 100.0 } else { 0.0 };
                println!("    {:30} {:>6} ({:>5.1}%)", k, v, pct);
            }
        };
        print_coverage("LANGUAGES", &languages);
        print_coverage("FRAMEWORKS", &frameworks);
        print_coverage("DOMAINS", &domains);
        print_coverage("TOOLS", &tools);
        print_coverage("SERVICES", &services);
        print_coverage("PLATFORMS", &platforms);
    }
    Ok(())
}

// ============================================================================
// Phase D (v2.11.0): datetime parsing + query/management subcommands
// ============================================================================
//
// These subcommands expose CozoDB timestamp and search capabilities to the
// CLI without requiring Python. They mirror the helpers in
// `scripts/pss_cozodb.py` byte-for-byte semantically, but run faster.
//
// Datetime formats accepted by parse_date_bound (unified per COR-7 —
// audit 20260514; made direction-aware for TRDD-1Z8SGQ7N / F18):
//   - RFC 3339 strings:  "2026-04-16T22:12:27Z" or "2026-04-16T22:12:27+00:00"
//   - Date only:         "2026-04-16" — names a whole DAY. Which INSTANT the
//                        cutoff resolves to depends on the bound's DIRECTION:
//                        Start -> 00:00:00.000000000, End -> 23:59:59.999999999.
//   - Relative to now:   "1d", "2w", "24h", "30m", "120s"
//   - Keywords:          "now", "yesterday"

/// A cutoff's DIRECTION. A bare date names an INTERVAL (a day); a cutoff needs
/// an INSTANT, and which end of the day you want depends on whether the date
/// is a lower (`Start`) or upper (`End`) bound. Every non-date form already
/// names an instant, so `Bound` is irrelevant to it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Bound {
    Start,
    End,
}

/// Parse a user-supplied datetime into the exact INSTANT a cutoff denotes.
///
/// Returns a `DateTime<Utc>` — NOT a pre-formatted string. The wire format
/// belongs to each STORAGE FAMILY, and a single literal cannot be correct for
/// both (that is F9). The legacy `skills` table stores whole-second Z-form
/// (`…T23:59:59Z`, via `to_rfc3339_opts(Secs, true)`); the temporal
/// `events`/`scan_runs`/`elements_state` tables store offset-form with
/// fractional seconds (`…T23:59:59.999999999+00:00`, via `to_rfc3339()`).
/// Each caller formats the returned instant for ITS OWN table.
///
/// Fail-fast on invalid input — callers are scripts/power-users who want
/// clear errors, not silent fallbacks. Per COR-2 (audit 20260514): garbage
/// inputs like "tomorrow" or "2026/05/14" produce a clear error instead of
/// being silently passed through to CozoDB and matching every row.
///
/// Per COR-7: this is the SINGLE date parser shared by every PSS subcommand —
/// both the legacy `list-added-since` family and the temporal `as-of` /
/// `timeline` / `diff` family. `temporal::cli::resolve_date` is a thin wrapper
/// that formats offset-form and converts the SuggesterError to a String.
pub(crate) fn parse_date_bound(arg: &str, bound: Bound) -> Result<DateTime<Utc>, SuggesterError> {
    let arg = arg.trim();
    if arg.is_empty() {
        return Err(SuggesterError::IndexParse(
            "empty datetime argument".to_string(),
        ));
    }

    // Keyword shortcuts. "now" is the current instant; "yesterday" is 24h ago.
    // Both already name an instant, so `bound` is irrelevant to them.
    if arg == "now" {
        return Ok(Utc::now());
    }
    if arg == "yesterday" {
        return Ok(Utc::now() - chrono::Duration::days(1));
    }

    // Relative shorthand N<unit> (unit s/m/h/d/w): also an instant, `bound` N/A.
    // Cheapest to check: last char in [smhdw], prefix a positive integer.
    let last = arg.chars().last().unwrap();
    if matches!(last, 's' | 'm' | 'h' | 'd' | 'w') {
        let num_part = &arg[..arg.len() - last.len_utf8()];
        if !num_part.is_empty() && num_part.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(n) = num_part.parse::<i64>() {
                let seconds = match last {
                    's' => n,
                    'm' => n * 60,
                    'h' => n * 3600,
                    'd' => n * 86400,
                    'w' => n * 86400 * 7,
                    _ => unreachable!(),
                };
                return Ok(Utc::now() - chrono::Duration::seconds(seconds));
            }
        }
        // Fall through — looked like shorthand but didn't parse cleanly; let
        // the RFC 3339 parser produce the final error.
    }

    // Full RFC 3339 / ISO 8601 (e.g. "2026-04-16T22:12:27Z"): already an instant.
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(arg) {
        return Ok(dt.with_timezone(&Utc));
    }

    // Date-only YYYY-MM-DD names a DAY. TRDD-1Z8SGQ7N / F18: the cutoff instant
    // is DIRECTION-aware — Start is the day's first instant, End its last —
    // because a date-only LOWER bound resolved to end-of-day skipped the whole
    // named day (`changed-between D D` collapsed to a 1-second window; the P1
    // shipped in v3.10.7 returned "(no results)" for "what changed today?").
    if let Ok(date) = chrono::NaiveDate::parse_from_str(arg, "%Y-%m-%d") {
        // Start -> the day's FIRST instant; End -> its LAST (max nanosecond, so
        // no real event within the day can sort after an End cutoff). The nanos
        // survive into the temporal offset-form cutoff (F9); the legacy Z-form
        // path truncates End to `…T23:59:59Z`, which still matches a stored
        // whole-second `…T23:59:59Z`.
        let ndt = match bound {
            Bound::Start => date.and_hms_opt(0, 0, 0),
            Bound::End => date.and_hms_nano_opt(23, 59, 59, 999_999_999),
        };
        if let Some(ndt) = ndt {
            return Ok(ndt.and_utc());
        }
    }

    // Naive datetime without timezone (e.g. "2026-04-16T22:12:27"): an instant.
    if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(arg, "%Y-%m-%dT%H:%M:%S") {
        return Ok(ndt.and_utc());
    }

    Err(SuggesterError::IndexParse(format!(
        "Invalid datetime {:?}. Expected RFC 3339 (2026-04-16T22:12:27Z), \
         date-only (2026-04-16), relative shorthand (1d, 2w, 24h, 30m, 120s), \
         or keyword ('now', 'yesterday').",
        arg
    )))
}

/// Print a compact, ps-style aligned table without borders — one row per line.
/// Truncates cells to their column's max width (last column allowed to overflow).
fn print_ps_table(headers: &[&str], rows: &[Vec<String>]) {
    if rows.is_empty() {
        println!("(no results)");
        return;
    }
    // Compute each column's width as max(header, data) capped at 80 chars.
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(cell.chars().count().min(80));
            }
        }
    }
    // Print header (space-separated, left-aligned)
    let mut header_parts: Vec<String> = Vec::new();
    for (i, h) in headers.iter().enumerate() {
        let w = widths[i];
        header_parts.push(format!("{:<width$}", h, width = w));
    }
    println!("{}", header_parts.join("  "));

    // Print rows — truncate each cell to its column width, except the last
    // column which gets to overflow so descriptions remain readable.
    let last_idx = headers.len().saturating_sub(1);
    for row in rows {
        let mut parts: Vec<String> = Vec::new();
        for (i, cell) in row.iter().enumerate() {
            if i >= widths.len() {
                break;
            }
            let w = widths[i];
            if i == last_idx {
                // Last column: no truncation, no padding on the right
                parts.push(cell.clone());
            } else {
                let truncated: String = cell.chars().take(w).collect();
                parts.push(format!("{:<width$}", truncated, width = w));
            }
        }
        println!("{}", parts.join("  "));
    }
}

/// Shared projection: produce a Vec<(row_fields, json_object)> pair from the
/// CozoDB timestamp queries. Columns are (name, skill_type, source,
/// description, timestamp_iso).
fn rows_to_timestamp_entries(
    rows: &[Vec<DataValue>],
    ts_label: &str,
) -> Vec<serde_json::Value> {
    rows.iter()
        .map(|r| {
            // columns: 0=name, 1=skill_type, 2=source, 3=path, 4=description, 5=timestamp
            serde_json::json!({
                "name": dv_to_string(&r[0]),
                "type": dv_to_string(&r[1]),
                "source": dv_to_string(&r[2]),
                "path": dv_to_string(&r[3]),
                "description": dv_to_string(&r[4]),
                ts_label: dv_to_string(&r[5]),
            })
        })
        .collect()
}

/// Human-readable output for timestamp-based list commands.
fn print_timestamp_list(entries: &[serde_json::Value], ts_label: &str) {
    // Bind temporarily because the `headers` array borrows &str slices that
    // must outlive the `print_ps_table` call.
    let ts_hdr = ts_label_upper(ts_label);
    let headers = ["TYPE", "NAME", "SOURCE", ts_hdr.as_str(), "DESCRIPTION"];
    let rows: Vec<Vec<String>> = entries
        .iter()
        .map(|e| {
            vec![
                e["type"].as_str().unwrap_or("").to_string(),
                e["name"].as_str().unwrap_or("").to_string(),
                e["source"].as_str().unwrap_or("").to_string(),
                e[ts_label].as_str().unwrap_or("").to_string(),
                e["description"]
                    .as_str()
                    .unwrap_or("")
                    .chars()
                    .take(60)
                    .collect(),
            ]
        })
        .collect();
    print_ps_table(&headers, &rows);
}

fn ts_label_upper(label: &str) -> String {
    label.replace('_', " ").to_uppercase()
}

// --- cmd_summary (v3.7, task 3.2) ---

/// Build a `(by_type, by_scope_kind, sources_set)` triple from elements_state +
/// events.  Shared by `cmd_summary` and `cmd_tree`.
///
/// `by_type`: count of currently-active (`exists=true`) elements per
///            element_type ("skill", "agent", ...).
/// `by_scope_kind`: count per scope ("user", "project", "plugin", ...).
/// `sources_set`: full set of distinct `source` strings (user / project /
///                plugin:<name> / marketplace:<name>) currently observed.
fn collect_index_overview(
    db: &DbInstance,
) -> Result<
    (
        BTreeMap<String, i64>,
        BTreeMap<String, i64>,
        std::collections::BTreeSet<String>,
    ),
    SuggesterError,
> {
    // Single query: for every active element, pull (element_type, scope, source)
    // from the `events` row referenced by `elements_state.last_event_id`.
    // We aggregate in Rust because Cozo's multi-dimensional aggregation gets
    // verbose for this case.
    let q = r#"?[element_type, scope, source] :=
        *elements_state{element_id, last_event_id, exists: true},
        *events{event_id: last_event_id, element_type, scope, source}"#;
    let res = db
        .run_script(q, Default::default(), ScriptMutability::Immutable)
        .map_err(|e| SuggesterError::IndexParse(format!("summary query failed: {}", e)))?;
    let mut by_type: BTreeMap<String, i64> = BTreeMap::new();
    let mut by_scope: BTreeMap<String, i64> = BTreeMap::new();
    let mut sources: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for row in &res.rows {
        let et = dv_to_string(&row[0]);
        let sc = dv_to_string(&row[1]);
        let so = dv_to_string(&row[2]);
        if !et.is_empty() {
            *by_type.entry(et).or_insert(0) += 1;
        }
        if !sc.is_empty() {
            *by_scope.entry(sc).or_insert(0) += 1;
        }
        if !so.is_empty() {
            sources.insert(so);
        }
    }
    Ok((by_type, by_scope, sources))
}

/// `pss summary` — one-line overview of the index.
///
/// Default (table): a single dense human-readable line.
/// JSON: structured `{total, by_type, by_scope, by_source, sources}` object.
fn cmd_summary(db: &DbInstance, format: OutputFormat) -> Result<(), SuggesterError> {
    let (by_type, by_scope, sources) = collect_index_overview(db)?;
    let total: i64 = by_type.values().sum();

    match format {
        OutputFormat::Json => {
            // Build by_source breakdown: every distinct `source` and its
            // active-element count.  Bucket "user" / "project" / "local" as
            // their own scope keys; "plugin:..." entries land under "plugin"
            // for the headline, but we also expose the raw `sources` list so
            // callers see plugin names.
            let mut plugin_count: i64 = 0;
            let mut marketplace_count: i64 = 0;
            let mut user_proj_local: i64 = 0;
            for src in &sources {
                if src.starts_with("plugin:") {
                    plugin_count += 1;
                } else if src.starts_with("marketplace:") {
                    marketplace_count += 1;
                } else {
                    user_proj_local += 1;
                }
            }
            let by_source = serde_json::json!({
                "plugin": plugin_count,
                "marketplace": marketplace_count,
                "user_project_local": user_proj_local,
            });
            let out = serde_json::json!({
                "total": total,
                "by_type": by_type,
                "by_scope": by_scope,
                "by_source": by_source,
                "sources": sources.iter().collect::<Vec<_>>(),
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&out).unwrap_or_default()
            );
        }
        OutputFormat::Table => {
            // Build the canonical one-liner.  Format:
            //   PSS index: <total> elements (<n> skills, <n> agents, ...) across
            //   <m> sources (<n> plugins, <n> marketplaces, <n> user/project/local)
            let type_parts: Vec<String> = by_type
                .iter()
                .map(|(k, v)| format!("{} {}{}", v, k, if *v == 1 { "" } else { "s" }))
                .collect();
            let mut plugin_n = 0;
            let mut market_n = 0;
            let mut upl_n = 0;
            for s in &sources {
                if s.starts_with("plugin:") {
                    plugin_n += 1;
                } else if s.starts_with("marketplace:") {
                    market_n += 1;
                } else {
                    upl_n += 1;
                }
            }
            let total_sources = sources.len();
            println!(
                "PSS index: {} elements ({}) across {} sources ({} plugins, {} marketplaces, {} user/project/local)",
                total,
                type_parts.join(", "),
                total_sources,
                plugin_n,
                market_n,
                upl_n,
            );
        }
        OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown => {
            format.print_stub("summary");
        }
    }
    Ok(())
}

// --- cmd_tree (v3.7, task 3.3) ---

/// `pss tree` — directory-tree view of the PSS index, grouped by source
/// then by element type.
///
/// Default (table): Unicode box-drawing characters (├ │ └ ─).
/// JSON: nested object `{source: {type: count, ...}, ...}`.
fn cmd_tree(db: &DbInstance, format: OutputFormat) -> Result<(), SuggesterError> {
    // Per-(source, element_type) count.  We sort sources into the canonical
    // bucket order: user → project → local → plugin:* (alphabetical) →
    // marketplace:* (alphabetical) → anything else.
    let q = r#"?[source, element_type] :=
        *elements_state{element_id, last_event_id, exists: true},
        *events{event_id: last_event_id, source, element_type}"#;
    let res = db
        .run_script(q, Default::default(), ScriptMutability::Immutable)
        .map_err(|e| SuggesterError::IndexParse(format!("tree query failed: {}", e)))?;

    // Map: source → element_type → count
    let mut grouped: BTreeMap<String, BTreeMap<String, i64>> = BTreeMap::new();
    for row in &res.rows {
        let src = dv_to_string(&row[0]);
        let etype = dv_to_string(&row[1]);
        if src.is_empty() || etype.is_empty() {
            continue;
        }
        *grouped
            .entry(src)
            .or_default()
            .entry(etype)
            .or_insert(0) += 1;
    }

    /// Sort priority for a source bucket.  Lower number sorts first.
    fn bucket_priority(src: &str) -> u8 {
        match src {
            "user" => 0,
            "project" => 1,
            "local" => 2,
            _ => {
                if src.starts_with("plugin:") {
                    3
                } else if src.starts_with("marketplace:") {
                    4
                } else {
                    5
                }
            }
        }
    }

    match format {
        OutputFormat::Json => {
            let out = serde_json::to_value(&grouped).unwrap_or(serde_json::Value::Null);
            println!(
                "{}",
                serde_json::to_string_pretty(&out).unwrap_or_default()
            );
        }
        OutputFormat::Table => {
            // Sort sources by (priority, lexical).  Render as a Unicode tree.
            let mut sources: Vec<&String> = grouped.keys().collect();
            sources.sort_by(|a, b| {
                let pa = bucket_priority(a);
                let pb = bucket_priority(b);
                pa.cmp(&pb).then(a.cmp(b))
            });

            // Friendly display label for a source bucket.
            fn pretty_source(s: &str) -> String {
                match s {
                    "user" => "user (~/.claude/)".to_string(),
                    "project" => "project (.claude/)".to_string(),
                    "local" => "local".to_string(),
                    _ => {
                        if let Some(rest) = s.strip_prefix("plugin:") {
                            format!("plugin/{}", rest)
                        } else if let Some(rest) = s.strip_prefix("marketplace:") {
                            format!("marketplace/{}", rest)
                        } else {
                            s.to_string()
                        }
                    }
                }
            }

            println!("PSS Index Tree");
            let total_sources = sources.len();
            for (i, src) in sources.iter().enumerate() {
                let is_last_src = i + 1 == total_sources;
                let src_branch = if is_last_src { "└──" } else { "├──" };
                println!("{} {}", src_branch, pretty_source(src));
                let types = &grouped[*src];
                let mut type_keys: Vec<&String> = types.keys().collect();
                type_keys.sort();
                let total_types = type_keys.len();
                for (j, t) in type_keys.iter().enumerate() {
                    let is_last_t = j + 1 == total_types;
                    let prefix = if is_last_src { "   " } else { "│  " };
                    let t_branch = if is_last_t { "└──" } else { "├──" };
                    let count = types[*t];
                    let label = if count == 1 {
                        format!("{}", t)
                    } else {
                        format!("{}s", t)
                    };
                    println!("{}{} {}/ ({})", prefix, t_branch, label, count);
                }
            }
        }
        OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown => {
            format.print_stub("tree");
        }
    }
    Ok(())
}

// --- cmd_count ---

/// Print the total skill count. Integer on stdout by default; {"count": N}
/// when `--json` is passed. Never exits non-zero when the DB has zero rows —
/// that's a legitimate state (reindex in progress). Use `pss health` for
/// exit-code-based health gating.
fn cmd_count(db: &DbInstance, json: bool) -> Result<(), SuggesterError> {
    let result = db
        .run_script(
            "?[count(name)] := *skills{ name }",
            Default::default(),
            ScriptMutability::Immutable,
        )
        .map_err(|e| SuggesterError::IndexParse(format!("count query failed: {}", e)))?;
    let total = result
        .rows
        .first()
        .and_then(|r| r.first())
        .map(dv_to_i64)
        .unwrap_or(0);
    if json {
        println!("{{\"count\": {}}}", total);
    } else {
        println!("{}", total);
    }
    Ok(())
}

// --- cmd_get ---

/// Fetch a single entry by (name[, source]). Prints the entry as JSON or
/// human-readable text. Exits with error if the entry is not found.
///
/// When `source` is None and multiple entries share the same name, prints
/// all of them (as a JSON array or one block per entry in text mode).
fn cmd_get(
    db: &DbInstance,
    name: &str,
    source: Option<&str>,
    json: bool,
) -> Result<(), SuggesterError> {
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("name".into(), DataValue::Str(name.into()));

    let query = if let Some(src) = source {
        // Reject obviously malformed source strings before building the query
        validate_filter_value("source", src)?;
        params.insert("source".into(), DataValue::Str(src.into()));
        "?[name, id, path, skill_type, source, description, tier, boost, category, \
         first_indexed_at, last_updated_at, keywords_json, domains_json, \
         languages_json, frameworks_json, platforms_json, tools_json] := \
         *skills{ name, id, path, skill_type, source, description, tier, boost, category, \
                  first_indexed_at, last_updated_at, keywords_json, domains_json, \
                  languages_json, frameworks_json, platforms_json, tools_json }, \
         name = $name, source = $source"
            .to_string()
    } else {
        "?[name, id, path, skill_type, source, description, tier, boost, category, \
         first_indexed_at, last_updated_at, keywords_json, domains_json, \
         languages_json, frameworks_json, platforms_json, tools_json] := \
         *skills{ name, id, path, skill_type, source, description, tier, boost, category, \
                  first_indexed_at, last_updated_at, keywords_json, domains_json, \
                  languages_json, frameworks_json, platforms_json, tools_json }, \
         name = $name"
            .to_string()
    };

    let result = db
        .run_script(&query, params, ScriptMutability::Immutable)
        .map_err(|e| SuggesterError::IndexParse(format!("get query failed: {}", e)))?;

    if result.rows.is_empty() {
        return Err(SuggesterError::IndexParse(format!(
            "Entry not found: {:?}{}",
            name,
            source.map(|s| format!(" (source: {})", s)).unwrap_or_default()
        )));
    }

    let parse_vec = |s: &str| -> Vec<String> { serde_json::from_str(s).unwrap_or_default() };
    let row_to_json = |row: &[DataValue]| -> serde_json::Value {
        serde_json::json!({
            "name": dv_to_string(&row[0]),
            "id": dv_to_string(&row[1]),
            "path": dv_to_string(&row[2]),
            "type": dv_to_string(&row[3]),
            "source": dv_to_string(&row[4]),
            "description": dv_to_string(&row[5]),
            "tier": dv_to_string(&row[6]),
            "boost": dv_to_i64(&row[7]),
            "category": dv_to_string(&row[8]),
            "first_indexed_at": dv_to_string(&row[9]),
            "last_updated_at": dv_to_string(&row[10]),
            "keywords": parse_vec(&dv_to_string(&row[11])),
            "domains": parse_vec(&dv_to_string(&row[12])),
            "languages": parse_vec(&dv_to_string(&row[13])),
            "frameworks": parse_vec(&dv_to_string(&row[14])),
            "platforms": parse_vec(&dv_to_string(&row[15])),
            "tools": parse_vec(&dv_to_string(&row[16])),
        })
    };

    if json {
        if result.rows.len() == 1 {
            println!(
                "{}",
                serde_json::to_string_pretty(&row_to_json(&result.rows[0])).unwrap_or_default()
            );
        } else {
            let arr: Vec<serde_json::Value> =
                result.rows.iter().map(|r| row_to_json(r)).collect();
            println!("{}", serde_json::to_string_pretty(&arr).unwrap_or_default());
        }
    } else {
        // Human-readable: one block per row, separated by a blank line.
        for (i, row) in result.rows.iter().enumerate() {
            if i > 0 {
                println!();
            }
            println!("Name        : {}", dv_to_string(&row[0]));
            println!("ID          : {}", dv_to_string(&row[1]));
            println!("Type        : {}", dv_to_string(&row[3]));
            println!("Source      : {}", dv_to_string(&row[4]));
            println!("Category    : {}", dv_to_string(&row[8]));
            println!("Path        : {}", dv_to_string(&row[2]));
            println!("Description : {}", dv_to_string(&row[5]));
            println!("Added       : {}", dv_to_string(&row[9]));
            println!("Updated     : {}", dv_to_string(&row[10]));
            let kws: Vec<String> = parse_vec(&dv_to_string(&row[11]));
            if !kws.is_empty() {
                println!("Keywords    : {}", kws.join(", "));
            }
            let langs: Vec<String> = parse_vec(&dv_to_string(&row[13]));
            if !langs.is_empty() {
                println!("Languages   : {}", langs.join(", "));
            }
            let doms: Vec<String> = parse_vec(&dv_to_string(&row[12]));
            if !doms.is_empty() {
                println!("Domains     : {}", doms.join(", "));
            }
        }
    }
    Ok(())
}

// --- cmd_health ---

/// Exit 0/1/2 health probe. The function itself returns the exit code via
/// the SuggesterError path; callers in main() translate it.
///
/// Semantics:
///   0 = DB reachable AND has >= 1 row
///   1 = DB reachable but empty or query failed (corrupt / schema drift)
///   2 = DB path does not exist (handled in caller before we get here)
///
/// On `--verbose`, a one-line diagnostic is printed to stdout. Silent otherwise.
///
/// This function is unusual: it uses process::exit directly so that exit
/// codes 1 and 2 map to "soft" DB-state signals, not to Rust error paths.
/// Callers should NOT propagate the result through `run_query_command` —
/// main() dispatches Health before the generic query handler.
pub(crate) fn cmd_health(db: Option<&DbInstance>, verbose: bool) -> ! {
    match db {
        None => {
            if verbose {
                println!("DB missing");
            }
            std::process::exit(2);
        }
        Some(db) => {
            // Count rows. Any error → exit 1 (treat as corrupt).
            let result = db.run_script(
                "?[count(name)] := *skills{ name }",
                Default::default(),
                ScriptMutability::Immutable,
            );
            match result {
                Ok(r) => {
                    let total = r
                        .rows
                        .first()
                        .and_then(|row| row.first())
                        .map(dv_to_i64)
                        .unwrap_or(0);
                    if total > 0 {
                        if verbose {
                            println!("OK ({} entries)", total);
                        }
                        std::process::exit(0);
                    } else {
                        if verbose {
                            println!("empty (0 entries)");
                        }
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    if verbose {
                        println!("corrupt ({})", e);
                    }
                    std::process::exit(1);
                }
            }
        }
    }
}

// --- cmd_list_added_since ---

fn cmd_list_added_since(
    db: &DbInstance,
    when: &str,
    limit: usize,
    json: bool,
) -> Result<(), SuggesterError> {
    // F18/F9: `since` is a LOWER bound (query `>= $since`) -> Bound::Start.
    // Legacy `skills` table stores whole-second Z-form, so format Secs+Z.
    let iso = parse_date_bound(when, Bound::Start)?
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let limit = limit.min(10000);
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("since".into(), DataValue::Str(iso.clone().into()));
    let query = format!(
        "?[name, skill_type, source, path, description, first_indexed_at] := \
         *skills{{ name, skill_type, source, path, description, first_indexed_at }}, \
         first_indexed_at >= $since \
         :order first_indexed_at \
         :limit {}",
        limit
    );
    let result = db
        .run_script(&query, params, ScriptMutability::Immutable)
        .map_err(|e| {
            SuggesterError::IndexParse(format!("list-added-since query failed: {}", e))
        })?;
    let entries = rows_to_timestamp_entries(&result.rows, "first_indexed_at");
    if json {
        println!("{}", serde_json::to_string_pretty(&entries).unwrap_or_default());
    } else {
        print_timestamp_list(&entries, "first_indexed_at");
    }
    Ok(())
}

// --- cmd_list_added_between ---

fn cmd_list_added_between(
    db: &DbInstance,
    start: &str,
    end: &str,
    limit: usize,
    json: bool,
) -> Result<(), SuggesterError> {
    // F18/F9: `start` is a LOWER bound (>= $start) -> Start; `end` an UPPER
    // bound (<= $end) -> End. Legacy `skills` table stores whole-second Z-form.
    let start_iso = parse_date_bound(start, Bound::Start)?
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let end_iso = parse_date_bound(end, Bound::End)?
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let limit = limit.min(10000);
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("start".into(), DataValue::Str(start_iso.clone().into()));
    params.insert("end".into(), DataValue::Str(end_iso.clone().into()));
    let query = format!(
        "?[name, skill_type, source, path, description, first_indexed_at] := \
         *skills{{ name, skill_type, source, path, description, first_indexed_at }}, \
         first_indexed_at >= $start, first_indexed_at <= $end \
         :order first_indexed_at \
         :limit {}",
        limit
    );
    let result = db
        .run_script(&query, params, ScriptMutability::Immutable)
        .map_err(|e| {
            SuggesterError::IndexParse(format!("list-added-between query failed: {}", e))
        })?;
    let entries = rows_to_timestamp_entries(&result.rows, "first_indexed_at");
    if json {
        println!("{}", serde_json::to_string_pretty(&entries).unwrap_or_default());
    } else {
        print_timestamp_list(&entries, "first_indexed_at");
    }
    Ok(())
}

// --- cmd_list_updated_since ---

fn cmd_list_updated_since(
    db: &DbInstance,
    when: &str,
    limit: usize,
    json: bool,
) -> Result<(), SuggesterError> {
    // F18/F9: `since` is a LOWER bound (query `>= $since`) -> Bound::Start.
    // Legacy `skills` table stores whole-second Z-form, so format Secs+Z.
    let iso = parse_date_bound(when, Bound::Start)?
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let limit = limit.min(10000);
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("since".into(), DataValue::Str(iso.clone().into()));
    // Descending order — most-recently-updated first is more useful.
    let query = format!(
        "?[name, skill_type, source, path, description, last_updated_at] := \
         *skills{{ name, skill_type, source, path, description, last_updated_at }}, \
         last_updated_at >= $since \
         :order -last_updated_at \
         :limit {}",
        limit
    );
    let result = db
        .run_script(&query, params, ScriptMutability::Immutable)
        .map_err(|e| {
            SuggesterError::IndexParse(format!("list-updated-since query failed: {}", e))
        })?;
    let entries = rows_to_timestamp_entries(&result.rows, "last_updated_at");
    if json {
        println!("{}", serde_json::to_string_pretty(&entries).unwrap_or_default());
    } else {
        print_timestamp_list(&entries, "last_updated_at");
    }
    Ok(())
}

// --- cmd_find_by_name ---

/// Case-insensitive substring match over the `name` column. With
/// `regex=true` (UX-5, audit 20260514) the substring is compiled as a
/// Rust regex and applied to lowercased names — Cozo has no native
/// regex support, so we fetch all rows and filter in Rust. The query
/// is bounded by `limit` post-filter.
fn cmd_find_by_name(
    db: &DbInstance,
    substring: &str,
    limit: usize,
    json: bool,
    use_regex: bool,
) -> Result<(), SuggesterError> {
    let limit = limit.min(10000);

    let rows = if use_regex {
        // UX-5: regex path. Compile pattern (case-insensitive) and
        // post-filter all skills. We fetch all names with their
        // metadata, then run the regex in Rust. Invalid patterns
        // fail loud (exit non-zero).
        let pattern = match regex::RegexBuilder::new(substring)
            .case_insensitive(true)
            .size_limit(1 << 20) // 1 MB compiled-DFA cap
            .build()
        {
            Ok(p) => p,
            Err(e) => {
                return Err(SuggesterError::IndexParse(format!(
                    "find-by-name --regex: invalid pattern {:?}: {}",
                    substring, e
                )));
            }
        };
        let result = db
            .run_script(
                "?[name, skill_type, source, path, description] := \
                 *skills{ name, skill_type, source, path, description } \
                 :order name",
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .map_err(|e| {
                SuggesterError::IndexParse(format!("find-by-name query failed: {}", e))
            })?;
        result
            .rows
            .into_iter()
            .filter(|r| {
                let name = dv_to_string(&r[0]);
                pattern.is_match(&name)
            })
            .take(limit)
            .collect::<Vec<_>>()
    } else {
        // Original case-insensitive substring path.
        let needle = substring.to_lowercase();
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("q".into(), DataValue::Str(needle.into()));
        let query = format!(
            "?[name, skill_type, source, path, description] := \
             *skills{{ name, skill_type, source, path, description }}, \
             str_includes(lowercase(name), $q) \
             :order name \
             :limit {}",
            limit
        );
        let result = db
            .run_script(&query, params, ScriptMutability::Immutable)
            .map_err(|e| {
                SuggesterError::IndexParse(format!("find-by-name query failed: {}", e))
            })?;
        result.rows
    };

    let entries: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": dv_to_string(&r[0]),
                "type": dv_to_string(&r[1]),
                "source": dv_to_string(&r[2]),
                "path": dv_to_string(&r[3]),
                "description": dv_to_string(&r[4]),
            })
        })
        .collect();
    if json {
        println!("{}", serde_json::to_string_pretty(&entries).unwrap_or_default());
    } else {
        let rows: Vec<Vec<String>> = entries
            .iter()
            .map(|e| {
                vec![
                    e["type"].as_str().unwrap_or("").to_string(),
                    e["name"].as_str().unwrap_or("").to_string(),
                    e["source"].as_str().unwrap_or("").to_string(),
                    e["description"]
                        .as_str()
                        .unwrap_or("")
                        .chars()
                        .take(60)
                        .collect(),
                ]
            })
            .collect();
        print_ps_table(&["TYPE", "NAME", "SOURCE", "DESCRIPTION"], &rows);
    }
    Ok(())
}

// --- cmd_find_by_auxiliary (shared implementation) ---

/// Helper: run a find-by-<aux-table> query against one of the 9 auxiliary
/// normalised relations (skill_keywords / skill_domains / skill_languages /
/// etc.), with JOIN back to skills for the output columns.
fn cmd_find_by_auxiliary(
    db: &DbInstance,
    aux_table: &str,
    value: &str,
    limit: usize,
    json: bool,
) -> Result<(), SuggesterError> {
    // Whitelist the aux-table name (no user-controlled value reaches Datalog here,
    // but keep the check tight so an accidental caller bug can't inject).
    const VALID_AUX: &[&str] = &[
        "skill_keywords",
        "skill_intents",
        "skill_tools",
        "skill_services",
        "skill_frameworks",
        "skill_languages",
        "skill_platforms",
        "skill_domains",
        "skill_file_types",
    ];
    if !VALID_AUX.contains(&aux_table) {
        return Err(SuggesterError::IndexParse(format!(
            "invalid auxiliary table: {}",
            aux_table
        )));
    }

    let needle = value.to_lowercase();
    let limit = limit.min(10000);
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("v".into(), DataValue::Str(needle.into()));
    let query = format!(
        "?[name, skill_type, source, path, description] := \
         *{}{{ skill_name: name, value: $v }}, \
         *skills{{ name, skill_type, source, path, description }} \
         :order name \
         :limit {}",
        aux_table, limit
    );
    let result = db
        .run_script(&query, params, ScriptMutability::Immutable)
        .map_err(|e| {
            SuggesterError::IndexParse(format!("find-by query failed ({}): {}", aux_table, e))
        })?;
    let entries: Vec<serde_json::Value> = result
        .rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": dv_to_string(&r[0]),
                "type": dv_to_string(&r[1]),
                "source": dv_to_string(&r[2]),
                "path": dv_to_string(&r[3]),
                "description": dv_to_string(&r[4]),
            })
        })
        .collect();
    if json {
        println!("{}", serde_json::to_string_pretty(&entries).unwrap_or_default());
    } else {
        let rows: Vec<Vec<String>> = entries
            .iter()
            .map(|e| {
                vec![
                    e["type"].as_str().unwrap_or("").to_string(),
                    e["name"].as_str().unwrap_or("").to_string(),
                    e["source"].as_str().unwrap_or("").to_string(),
                    e["description"]
                        .as_str()
                        .unwrap_or("")
                        .chars()
                        .take(60)
                        .collect(),
                ]
            })
            .collect();
        print_ps_table(&["TYPE", "NAME", "SOURCE", "DESCRIPTION"], &rows);
    }
    Ok(())
}

