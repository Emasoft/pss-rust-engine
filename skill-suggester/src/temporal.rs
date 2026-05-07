// Phase-1-of-4 module: many items are wired in by Phase 2 / 3 / 4.
// Silence dead_code warnings until then; the API contract is the public
// surface of this file.
#![allow(dead_code)]

//! Temporal history index — event-sourced model.
//!
//! See `design/tasks/TRDD-152e697f-*.md` for the full design.
//!
//! This module owns:
//! - Schema definitions for `events`, `element_blobs`, `elements_state`, `scan_runs`
//! - `ElementId` computation (stable identity across revisions)
//! - Content hashing (sha256)
//! - File size + tiktoken token-count computation
//! - Migration from legacy `skills`/`rules` tables to event-sourced schema
//!
//! Design rules:
//! - Event log is APPEND-ONLY. No event row is ever updated or deleted
//!   except by `pss prune-history` after retention expiry.
//! - `elements_state` is a materialized view derived from events. It is
//!   the only mutable table (rebuilt per scan).
//! - `observed_at` of every event = `scan_run.finished_at`. Filesystem
//!   mtimes are deliberately NOT consulted (per user's decision, the
//!   indexer's scan time is the canonical event time).
//! - Override resolution applies only to file-based elements
//!   (skill / agent / command / rule / output-style / theme) per the
//!   "local > project > user > plugin" precedence documented at
//!   <https://code.claude.com/docs/en/settings.md>. Hooks merge (array
//!   settings concatenate + dedupe across scopes), so each hook entry is
//!   tracked as its own element_id with no override events.

use cozo::{DataValue, DbInstance, ScriptMutability};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

/// Schema version for the temporal index. Bump when DDL changes.
pub const TEMPORAL_SCHEMA_VERSION: &str = "2";

/// Every event_type the temporal index can emit. Documented in TRDD §6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    Installed,
    Removed,
    ContentChanged,
    SizeChanged,
    FrontmatterChanged,
    DescriptionChanged,
    PathChanged,
    Enabled,
    Disabled,
    ScopeMoved,
    OverrideStarted,
    OverrideEnded,
    MarketplaceAdded,
    MarketplaceRemoved,
    PluginInstalledInScope,
    PluginUninstalledFromScope,
    PluginVersionChanged,
    MetadataChanged,
}

impl EventType {
    /// Stable string form persisted in the events table. Lowercase
    /// snake_case so Datalog filters can pattern-match without alias
    /// tables.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Installed => "installed",
            Self::Removed => "removed",
            Self::ContentChanged => "content_changed",
            Self::SizeChanged => "size_changed",
            Self::FrontmatterChanged => "frontmatter_changed",
            Self::DescriptionChanged => "description_changed",
            Self::PathChanged => "path_changed",
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
            Self::ScopeMoved => "scope_moved",
            Self::OverrideStarted => "override_started",
            Self::OverrideEnded => "override_ended",
            Self::MarketplaceAdded => "marketplace_added",
            Self::MarketplaceRemoved => "marketplace_removed",
            Self::PluginInstalledInScope => "plugin_installed_in_scope",
            Self::PluginUninstalledFromScope => "plugin_uninstalled_from_scope",
            Self::PluginVersionChanged => "plugin_version_changed",
            Self::MetadataChanged => "metadata_changed",
        }
    }
}

/// Element types the temporal index tracks. The list is documented in
/// TRDD §4 and verified against
/// <https://code.claude.com/docs/en/claude-directory.md>.
///
/// Beware: `theme` is global-only (no project scope) — see
/// claude-directory.md "themes/" section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    Skill,
    Agent,
    Command,
    Rule,
    Mcp,
    Lsp,
    Hook,
    Plugin,
    Channel,
    Monitor,
    OutputStyle,
    Theme,
    Marketplace,
}

impl ElementType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Skill => "skill",
            Self::Agent => "agent",
            Self::Command => "command",
            Self::Rule => "rule",
            Self::Mcp => "mcp",
            Self::Lsp => "lsp",
            Self::Hook => "hook",
            Self::Plugin => "plugin",
            Self::Channel => "channel",
            Self::Monitor => "monitor",
            Self::OutputStyle => "output-style",
            Self::Theme => "theme",
            Self::Marketplace => "marketplace",
        }
    }

    /// True iff override-precedence resolution applies to this element
    /// type. Hooks merge across scopes per CC settings precedence; their
    /// rows are independent and never emit override events.
    pub fn has_override_precedence(&self) -> bool {
        matches!(
            self,
            Self::Skill
                | Self::Agent
                | Self::Command
                | Self::Rule
                | Self::OutputStyle
                | Self::Theme
        )
    }
}

/// Stable identity for an element across all of its revisions. Two
/// elements with the same `element_id` are the same conceptual thing
/// observed at different times.
///
/// Format: `<element_type>:<name>@<scope>:<scope_path_slug>`, lowercased.
/// `scope_path_slug` is the absolute path with `/` replaced by `_` so the
/// id stays a single Datalog string token.
pub fn compute_element_id(
    element_type: ElementType,
    name: &str,
    scope: &str,
    scope_path: &str,
) -> String {
    let scope_path_slug = scope_path.replace('/', "_");
    format!(
        "{}:{}@{}:{}",
        element_type.as_str(),
        name.to_lowercase(),
        scope.to_lowercase(),
        scope_path_slug.to_lowercase()
    )
}

/// SHA-256 hex of the canonical bytes for an element. For file
/// elements pass the raw file bytes; for non-file elements (mcp / lsp /
/// hook / plugin / channel / monitor / marketplace) pass the canonical
/// JSON of the config dict (sorted keys).
pub fn content_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    hex_lower(&result)
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = std::fmt::Write::write_fmt(&mut s, format_args!("{:02x}", b));
    }
    s
}

/// Token count of a UTF-8 string using the cl100k_base encoding. This
/// is OpenAI's tokenizer, used here as an approximation of Claude's
/// (Anthropic doesn't publish theirs). Counts are typically within ±10%.
///
/// Returns `-1` if the encoder failed to load (treat as "not applicable").
pub fn token_count_cl100k(text: &str) -> i64 {
    use std::sync::OnceLock;
    use tiktoken_rs::CoreBPE;

    static ENCODER: OnceLock<Option<CoreBPE>> = OnceLock::new();
    let encoder = ENCODER.get_or_init(|| tiktoken_rs::cl100k_base().ok());

    match encoder {
        Some(enc) => enc.encode_with_special_tokens(text).len() as i64,
        None => -1,
    }
}

/// File size in bytes; -1 if file is missing or unreadable.
pub fn file_size_bytes(path: &Path) -> i64 {
    fs_metadata_size(path).unwrap_or(-1)
}

fn fs_metadata_size(path: &Path) -> Option<i64> {
    std::fs::metadata(path).ok().map(|m| m.len() as i64)
}

/// Create the temporal-index tables idempotently. Safe to call on every
/// startup — `:create` is a no-op if the relation already exists.
///
/// Returns `Ok(())` on success, propagates Cozo errors on failure.
pub fn ensure_schema(db: &DbInstance) -> Result<(), String> {
    for ddl in TEMPORAL_DDL.iter() {
        // Cozo returns an error if the relation already exists; we
        // silently treat that as success because :create is the
        // documented idempotent gate. Different Cozo versions phrase
        // the conflict differently — match either form.
        if let Err(e) = db.run_script(ddl, BTreeMap::new(), ScriptMutability::Mutable) {
            let msg = e.to_string();
            let already = msg.contains("already exists")
                || msg.contains("conflicts with an existing")
                || msg.contains("Stored relation");
            if !already {
                return Err(format!("ensure_schema failed: {}", msg));
            }
        }
    }
    // Stamp the schema version so migrations know what they're working with.
    // Cozo's `:put` upserts; the `?[...]` columns must match the relation's
    // key + value columns by name, not position.
    let stamp = format!(
        r#"?[key, value] <- [["schema_version", "{}"]]
           :put pss_metadata {{ key => value }}"#,
        TEMPORAL_SCHEMA_VERSION
    );
    db.run_script(&stamp, BTreeMap::new(), ScriptMutability::Mutable)
        .map_err(|e| format!("schema_version stamp failed: {}", e))?;
    Ok(())
}

/// Read the schema_version from `pss_metadata`. Returns `"1"` (legacy
/// inferred default) if the key is missing or the table doesn't exist.
pub fn read_schema_version(db: &DbInstance) -> String {
    let q = r#"?[v] := *pss_metadata{key: "schema_version", value: v}"#;
    match db.run_script(q, BTreeMap::new(), ScriptMutability::Immutable) {
        Ok(rows) => rows
            .rows
            .first()
            .and_then(|r| r.first())
            .and_then(|d| match d {
                DataValue::Str(s) => Some(s.to_string()),
                _ => None,
            })
            .unwrap_or_else(|| "1".to_string()),
        Err(_) => "1".to_string(),
    }
}

/// All DDL statements for the temporal-index tables. Each is run
/// idempotently by `ensure_schema`. The `pss_metadata` row is included
/// because legacy DBs may not have it yet, and `ensure_schema` writes
/// schema_version there as its final step.
const TEMPORAL_DDL: &[&str] = &[
    // Used by ensure_schema's schema_version stamp; legacy DBs may
    // already have this — :create errors out and we treat it as no-op.
    r#":create pss_metadata { key: String => value: String }"#,
    // Append-only event log. Key (event_id) is a ULID — sortable and
    // monotone. Cozo automatically btree-indexes the key; range queries
    // on observed_at scan the table linearly today (acceptable for ≤100k
    // events; we revisit if it gets slow).
    r#":create events {
        event_id: String =>
        observed_at: String,
        scan_id: String,
        event_type: String,
        element_type: String,
        element_name: String,
        element_id: String,
        scope: String,
        scope_path: String,
        source: String,
        path: String,
        content_hash: String,
        file_size: Int default -1,
        token_count: Int default -1,
        enabled: Bool default true,
        override_status: String default "none",
        diff_json: String default "{}",
        snapshot_ref: String default "",
    }"#,
    // Content-addressed blob store. A blob is stored once even if many
    // events reference it. `ref_count` is bumped/decremented as events
    // are added/pruned.
    r#":create element_blobs {
        hash: String =>
        bytes_b64: String,
        size: Int,
        first_seen_at: String,
        ref_count: Int default 0,
    }"#,
    // Materialized current state per element. Updated transactionally
    // with each batch of events. Suggestion hot path reads this table.
    r#":create elements_state {
        element_id: String =>
        last_event_id: String,
        current_path: String,
        current_hash: String,
        current_size: Int default -1,
        current_token_count: Int default -1,
        enabled: Bool default true,
        override_status: String default "none",
        installed_at: String,
        last_changed_at: String,
        exists: Bool default true,
    }"#,
    // Scan ledger. A subsequent scan that didn't visit a scope_path can
    // not fabricate removal events for elements within it — emission
    // logic checks this table.
    r#":create scan_runs {
        scan_id: String =>
        started_at: String,
        finished_at: String,
        scope_paths_json: String,
        events_emitted: Int default 0,
        rust_binary_version: String,
        pss_version: String,
    }"#,
];

/// Migrate a legacy v1 DB to v2 (event-sourced) schema. Idempotent —
/// reads `pss_metadata.schema_version` and bails out if already at v2.
///
/// What it does:
/// 1. Calls `ensure_schema` to materialize the new tables.
/// 2. Reads every row from the legacy `skills` table.
/// 3. For each, emits a synthetic `installed` event into `events`,
///    populates `elements_state`, and (best-effort) hashes + tokenizes
///    the source file if it still exists on disk.
/// 4. Same for `rules`.
/// 5. Records a synthetic `scan_runs` row for traceability.
/// 6. Stamps `schema_version = "2"` (idempotent).
///
/// On any error, leaves the DB untouched and returns `Err`. Caller may
/// retry safely — partial progress is bounded by Cozo's transactional
/// `:put` semantics.
///
/// NOTE: Phase 1 implementation. The migration emits one synthetic
/// `installed` event per legacy row; computing exact `content_hash` /
/// `file_size` / `token_count` is best-effort: if the source file is
/// missing on disk, we record `-1` / "" placeholders. Phase 2's normal
/// reindex will overwrite these with accurate snapshots on the next run.
pub fn migrate_v1_to_v2(db: &DbInstance) -> Result<MigrationStats, String> {
    if read_schema_version(db) == TEMPORAL_SCHEMA_VERSION {
        return Ok(MigrationStats::already_migrated());
    }
    ensure_schema(db)?;

    let now = chrono::Utc::now().to_rfc3339();
    let scan_id = ulid::Ulid::new().to_string();
    let mut stats = MigrationStats::default();

    // 1. Iterate the legacy `skills` table.
    let skills_q = r#"?[name, source, path, skill_type, description, first_indexed_at, last_updated_at] :=
        *skills{name, source, path, skill_type, description, first_indexed_at, last_updated_at}"#;
    if let Ok(rows) = db.run_script(skills_q, BTreeMap::new(), ScriptMutability::Immutable) {
        for row in rows.rows {
            if row.len() < 7 {
                continue;
            }
            let name = data_str(&row[0]);
            let source = data_str(&row[1]);
            let path = data_str(&row[2]);
            let skill_type = data_str(&row[3]);
            let description = data_str(&row[4]);
            let first_indexed_at = data_str(&row[5]);

            let element_type = match skill_type.as_str() {
                "skill" => ElementType::Skill,
                "agent" => ElementType::Agent,
                "command" => ElementType::Command,
                "rule" => ElementType::Rule,
                "mcp" => ElementType::Mcp,
                "lsp" => ElementType::Lsp,
                _ => ElementType::Skill, // fallback
            };

            let scope = scope_from_source(&source);
            let scope_path = "".to_string(); // legacy schema didn't track project paths
            let element_id = compute_element_id(element_type, &name, &scope, &scope_path);

            let (size, hash, tokens) = read_file_metrics(&path);
            insert_install_event(
                db,
                &scan_id,
                &first_indexed_at,
                element_type,
                &name,
                &element_id,
                &scope,
                &scope_path,
                &source,
                &path,
                &hash,
                size,
                tokens,
                &description,
            )?;
            stats.skills_migrated += 1;
        }
    }

    // 2. Iterate the legacy `rules` table.
    let rules_q = r#"?[name, scope, description, source_path] :=
        *rules{name, scope, description, source_path}"#;
    if let Ok(rows) = db.run_script(rules_q, BTreeMap::new(), ScriptMutability::Immutable) {
        for row in rows.rows {
            if row.len() < 4 {
                continue;
            }
            let name = data_str(&row[0]);
            let scope = data_str(&row[1]);
            let description = data_str(&row[2]);
            let path = data_str(&row[3]);
            let source = scope.clone();
            let scope_path = "".to_string();
            let element_id = compute_element_id(ElementType::Rule, &name, &scope, &scope_path);
            let (size, hash, tokens) = read_file_metrics(&path);
            insert_install_event(
                db,
                &scan_id,
                &now,
                ElementType::Rule,
                &name,
                &element_id,
                &scope,
                &scope_path,
                &source,
                &path,
                &hash,
                size,
                tokens,
                &description,
            )?;
            stats.rules_migrated += 1;
        }
    }

    // 3. Record the synthetic scan run (parameterised — no string interp).
    use cozo::Num;
    let mut scan_params: BTreeMap<String, DataValue> = BTreeMap::new();
    scan_params.insert("scan_id".into(), DataValue::Str(scan_id.clone().into()));
    scan_params.insert("started_at".into(), DataValue::Str(now.clone().into()));
    scan_params.insert("finished_at".into(), DataValue::Str(now.clone().into()));
    scan_params.insert(
        "scope_paths_json".into(),
        DataValue::Str("[\"<migration>\"]".into()),
    );
    scan_params.insert(
        "events_emitted".into(),
        DataValue::Num(Num::Int(
            (stats.skills_migrated + stats.rules_migrated) as i64,
        )),
    );
    scan_params.insert(
        "rust_binary_version".into(),
        DataValue::Str(env!("CARGO_PKG_VERSION").into()),
    );
    scan_params.insert(
        "pss_version".into(),
        DataValue::Str(env!("CARGO_PKG_VERSION").into()),
    );
    let scan_row = r#"?[scan_id, started_at, finished_at, scope_paths_json, events_emitted, rust_binary_version, pss_version] <-
        [[$scan_id, $started_at, $finished_at, $scope_paths_json, $events_emitted, $rust_binary_version, $pss_version]]
       :put scan_runs { scan_id => started_at, finished_at, scope_paths_json, events_emitted, rust_binary_version, pss_version }"#;
    db.run_script(scan_row, scan_params, ScriptMutability::Mutable)
        .map_err(|e| format!("scan_runs insert failed: {}", e))?;

    Ok(stats)
}

/// Statistics returned by `migrate_v1_to_v2`. Useful for logs and tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct MigrationStats {
    pub skills_migrated: u64,
    pub rules_migrated: u64,
    pub already_at_target_version: bool,
}

impl MigrationStats {
    fn already_migrated() -> Self {
        Self {
            already_at_target_version: true,
            ..Self::default()
        }
    }
}

/// Map a legacy `source` value to the canonical scope name. Legacy
/// sources looked like `user`, `project`, `plugin:foo`,
/// `marketplace:bar`. Phase 2 will introduce `local` for
/// settings.local.json-driven elements.
fn scope_from_source(source: &str) -> String {
    if source == "user" || source == "project" || source == "local" {
        source.to_string()
    } else if source.starts_with("plugin:") {
        "plugin".to_string()
    } else if source.starts_with("marketplace:") {
        "marketplace".to_string()
    } else {
        source.to_string() // unknown — preserve verbatim
    }
}

/// Read file size, content hash, and token count. Returns `(-1, "", -1)`
/// if the file is missing or unreadable.
fn read_file_metrics(path: &str) -> (i64, String, i64) {
    if path.is_empty() {
        return (-1, "".to_string(), -1);
    }
    let p = Path::new(path);
    let bytes = match std::fs::read(p) {
        Ok(b) => b,
        Err(_) => return (-1, "".to_string(), -1),
    };
    let size = bytes.len() as i64;
    let hash = content_hash(&bytes);
    let tokens = match std::str::from_utf8(&bytes) {
        Ok(s) => token_count_cl100k(s),
        Err(_) => -1,
    };
    (size, hash, tokens)
}

/// Get a string out of a Cozo `DataValue`, defaulting to empty on
/// non-string variants. Migration tolerates messy legacy data.
fn data_str(v: &DataValue) -> String {
    match v {
        DataValue::Str(s) => s.to_string(),
        _ => "".to_string(),
    }
}

#[allow(clippy::too_many_arguments)]
fn insert_install_event(
    db: &DbInstance,
    scan_id: &str,
    observed_at: &str,
    element_type: ElementType,
    name: &str,
    element_id: &str,
    scope: &str,
    scope_path: &str,
    source: &str,
    path: &str,
    content_hash: &str,
    file_size: i64,
    token_count: i64,
    description: &str,
) -> Result<(), String> {
    use cozo::Num;

    let event_id = ulid::Ulid::new().to_string();
    // diff_json is a JSON object describing what this event records.
    // We build it as a serde_json string so escaping is correct.
    let diff = serde_json::json!({
        "description": description,
        "migrated": true,
    })
    .to_string();

    // Parameterized query — no string interpolation of user data.
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("event_id".into(), DataValue::Str(event_id.clone().into()));
    params.insert("observed_at".into(), DataValue::Str(observed_at.into()));
    params.insert("scan_id".into(), DataValue::Str(scan_id.into()));
    params.insert(
        "event_type".into(),
        DataValue::Str(EventType::Installed.as_str().into()),
    );
    params.insert(
        "element_type".into(),
        DataValue::Str(element_type.as_str().into()),
    );
    params.insert("element_name".into(), DataValue::Str(name.into()));
    params.insert("element_id".into(), DataValue::Str(element_id.into()));
    params.insert("scope".into(), DataValue::Str(scope.into()));
    params.insert("scope_path".into(), DataValue::Str(scope_path.into()));
    params.insert("source".into(), DataValue::Str(source.into()));
    params.insert("path".into(), DataValue::Str(path.into()));
    params.insert("content_hash".into(), DataValue::Str(content_hash.into()));
    params.insert("file_size".into(), DataValue::Num(Num::Int(file_size)));
    params.insert("token_count".into(), DataValue::Num(Num::Int(token_count)));
    params.insert("enabled".into(), DataValue::Bool(true));
    params.insert("override_status".into(), DataValue::Str("active".into()));
    params.insert("diff_json".into(), DataValue::Str(diff.into()));
    params.insert("snapshot_ref".into(), DataValue::Str("".into()));

    let event_q = r#"?[event_id, observed_at, scan_id, event_type, element_type, element_name, element_id, scope, scope_path, source, path, content_hash, file_size, token_count, enabled, override_status, diff_json, snapshot_ref] <-
        [[$event_id, $observed_at, $scan_id, $event_type, $element_type, $element_name, $element_id, $scope, $scope_path, $source, $path, $content_hash, $file_size, $token_count, $enabled, $override_status, $diff_json, $snapshot_ref]]
       :put events { event_id => observed_at, scan_id, event_type, element_type, element_name, element_id, scope, scope_path, source, path, content_hash, file_size, token_count, enabled, override_status, diff_json, snapshot_ref }"#;
    db.run_script(event_q, params.clone(), ScriptMutability::Mutable)
        .map_err(|e| format!("event insert failed: {}", e))?;

    // Now the materialized state row. Reuse params we already have; add
    // last_event_id and rename a few keys for elements_state's schema.
    let mut state_params: BTreeMap<String, DataValue> = BTreeMap::new();
    state_params.insert("element_id".into(), DataValue::Str(element_id.into()));
    state_params.insert("last_event_id".into(), DataValue::Str(event_id.into()));
    state_params.insert("current_path".into(), DataValue::Str(path.into()));
    state_params.insert("current_hash".into(), DataValue::Str(content_hash.into()));
    state_params.insert(
        "current_size".into(),
        DataValue::Num(Num::Int(file_size)),
    );
    state_params.insert(
        "current_token_count".into(),
        DataValue::Num(Num::Int(token_count)),
    );
    state_params.insert("enabled".into(), DataValue::Bool(true));
    state_params.insert("override_status".into(), DataValue::Str("active".into()));
    state_params.insert("installed_at".into(), DataValue::Str(observed_at.into()));
    state_params.insert(
        "last_changed_at".into(),
        DataValue::Str(observed_at.into()),
    );
    state_params.insert("exists".into(), DataValue::Bool(true));

    let state_q = r#"?[element_id, last_event_id, current_path, current_hash, current_size, current_token_count, enabled, override_status, installed_at, last_changed_at, exists] <-
        [[$element_id, $last_event_id, $current_path, $current_hash, $current_size, $current_token_count, $enabled, $override_status, $installed_at, $last_changed_at, $exists]]
       :put elements_state { element_id => last_event_id, current_path, current_hash, current_size, current_token_count, enabled, override_status, installed_at, last_changed_at, exists }"#;
    db.run_script(state_q, state_params, ScriptMutability::Mutable)
        .map_err(|e| format!("state insert failed: {}", e))?;
    Ok(())
}

/// A single observation made during a scan: one element as it was seen
/// at scan time. Compared against the prior `elements_state` row to
/// decide which events to emit.
#[derive(Debug, Clone)]
pub struct Observation {
    pub element_type: ElementType,
    pub name: String,
    pub scope: String,
    pub scope_path: String,
    pub source: String,
    pub path: String,
    pub content_hash: String,
    pub file_size: i64,
    pub token_count: i64,
    pub description: String,
    pub enabled: bool,
}

impl Observation {
    pub fn element_id(&self) -> String {
        compute_element_id(self.element_type, &self.name, &self.scope, &self.scope_path)
    }
}

/// Prior state of an element from `elements_state`. Subset of fields
/// the emission engine needs to decide deltas.
#[derive(Debug, Clone)]
pub struct PriorState {
    pub element_id: String,
    pub current_path: String,
    pub current_hash: String,
    pub current_size: i64,
    pub enabled: bool,
    pub override_status: String,
    pub exists: bool,
}

/// Compare a current `Observation` to its `PriorState` and emit the
/// minimum set of `EventType`s explaining the delta. Pure function — no
/// DB access, no IO. The caller persists these events.
///
/// Rules (TRDD §6):
/// - prior absent or `exists=false` → `Installed`
/// - prior present, hash differs → `ContentChanged` (and `SizeChanged`
///   if size differs too — emit BOTH so per-event diffs are crisp)
/// - prior present, same hash, size differs → `SizeChanged` only
/// - prior present, same hash+size, path differs → `PathChanged`
/// - enabled flag flipped → `Enabled` or `Disabled`
/// - override_status changed → `OverrideStarted` or `OverrideEnded`
///
/// Multiple events can be emitted for one observation (e.g. a content
/// change that also flipped enabled).
pub fn compare_and_emit(prior: Option<&PriorState>, current: &Observation) -> Vec<EventType> {
    let mut out = Vec::with_capacity(2);
    match prior {
        None => {
            out.push(EventType::Installed);
        }
        Some(p) if !p.exists => {
            // Prior tombstone (was removed, now back) — re-install event.
            out.push(EventType::Installed);
        }
        Some(p) => {
            let hash_diff = p.current_hash != current.content_hash;
            let size_diff = p.current_size != current.file_size;
            let path_diff = p.current_path != current.path;
            let enabled_diff = p.enabled != current.enabled;

            if hash_diff {
                out.push(EventType::ContentChanged);
                if size_diff {
                    out.push(EventType::SizeChanged);
                }
            } else if size_diff {
                out.push(EventType::SizeChanged);
            } else if path_diff {
                // Same content+size at a new path within the same scope.
                out.push(EventType::PathChanged);
            }
            if enabled_diff {
                out.push(if current.enabled {
                    EventType::Enabled
                } else {
                    EventType::Disabled
                });
            }
        }
    }
    out
}

/// Build the set of removal events when scanning a scope_path that
/// previously had elements no longer present. Caller passes the list
/// of element_ids that were `exists=true` in `elements_state` for the
/// scope_path AND the set of element_ids actually observed this scan.
/// Anything in the former but not the latter is removed.
///
/// Pure function. The caller wraps each in `Removed` events.
pub fn detect_removals(
    prior_active_in_scope: &std::collections::HashSet<String>,
    observed_this_scan: &std::collections::HashSet<String>,
) -> Vec<String> {
    prior_active_in_scope
        .difference(observed_this_scan)
        .cloned()
        .collect()
}

/// Resolve override priority for a group of elements that share
/// (`element_type`, `name`) but differ in scope. Returns each element's
/// new override_status string.
///
/// Priority order (per <https://code.claude.com/docs/en/settings.md>):
///   `local > project > user > plugin > marketplace`
///
/// Rules:
/// - The highest-priority element gets `"active"` if it's the only one,
///   else `"overrides:<list-of-lower-eids>"` (semicolon-joined).
/// - Lower-priority elements get `"overridden_by:<top-eid>"`.
/// - Element types that don't have override precedence (Hook, Mcp, …)
///   pass through unchanged with status `"active"`.
pub fn resolve_overrides(
    element_type: ElementType,
    candidates: &[(String, String)], // (element_id, scope)
) -> Vec<(String, String)> {
    if !element_type.has_override_precedence() || candidates.len() <= 1 {
        return candidates
            .iter()
            .map(|(eid, _)| (eid.clone(), "active".to_string()))
            .collect();
    }
    let priority = |scope: &str| -> u8 {
        match scope {
            "local" => 5,
            "project" => 4,
            "user" => 3,
            "plugin" => 2,
            "marketplace" => 1,
            _ => 0,
        }
    };
    // Sort candidates by descending priority — top is at index 0.
    let mut sorted: Vec<(String, String)> = candidates.to_vec();
    sorted.sort_by(|a, b| priority(&b.1).cmp(&priority(&a.1)));
    let top_eid = sorted[0].0.clone();
    let mut out = Vec::with_capacity(candidates.len());
    let lower_eids: Vec<String> = sorted.iter().skip(1).map(|(e, _)| e.clone()).collect();
    let top_status = if lower_eids.is_empty() {
        "active".to_string()
    } else {
        format!("overrides:{}", lower_eids.join(";"))
    };
    out.push((top_eid.clone(), top_status));
    for (eid, _) in sorted.into_iter().skip(1) {
        out.push((eid, format!("overridden_by:{}", top_eid)));
    }
    out
}

// ============================================================================
// CLI sub-module — Phase 3 dispatchers. These read CozoDB and print JSON.
// ============================================================================
pub mod cli {
    use super::*;
    use cozo::Num;
    use serde_json::{json, Value as JsonValue};

    /// Convert a Cozo `DataValue` row into a `serde_json::Value`. Strings,
    /// ints, floats, bools, and null preserved; everything else stringified.
    fn data_to_json(d: &DataValue) -> JsonValue {
        match d {
            DataValue::Str(s) => JsonValue::String(s.to_string()),
            DataValue::Bool(b) => JsonValue::Bool(*b),
            DataValue::Num(Num::Int(n)) => json!(*n),
            DataValue::Num(Num::Float(f)) => json!(*f),
            DataValue::Null => JsonValue::Null,
            other => JsonValue::String(format!("{:?}", other)),
        }
    }

    /// Resolve a date shorthand to RFC3339. Accepts:
    /// - "now" → current UTC RFC3339
    /// - "yesterday" → 24h ago UTC RFC3339
    /// - "YYYY-MM-DD" → that date at 23:59:59 UTC
    /// - any RFC3339 string → returned as-is
    fn resolve_date(input: &str) -> String {
        match input {
            "now" => chrono::Utc::now().to_rfc3339(),
            "yesterday" => (chrono::Utc::now() - chrono::Duration::days(1)).to_rfc3339(),
            s if s.len() == 10 && s.chars().nth(4) == Some('-') => {
                format!("{}T23:59:59Z", s)
            }
            s => s.to_string(),
        }
    }

    /// `pss as-of <DATE> [filters]`
    pub fn cmd_as_of(
        db: &DbInstance,
        date: &str,
        type_filter: Option<&str>,
        scope_filter: Option<&str>,
        scope_path_filter: Option<&str>,
        limit: usize,
    ) {
        let cutoff = resolve_date(date);
        // Strategy: walk events table, find for each element_id the event
        // with the highest observed_at <= cutoff. If that event was a
        // `removed`, the element wasn't present at cutoff. Otherwise it
        // was. Filter by element_type / scope / scope_path if requested.
        let q = r#"
            ?[element_id, max_observed_at] :=
                *events{element_id, observed_at},
                observed_at <= $cutoff,
                max_observed_at = max(observed_at)
            :order -max_observed_at
            :limit $limit
        "#;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("cutoff".into(), DataValue::Str(cutoff.into()));
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        let rows = match db.run_script(q, params, ScriptMutability::Immutable) {
            Ok(r) => r.rows,
            Err(e) => {
                eprintln!("as-of query failed: {}", e);
                println!("[]");
                return;
            }
        };
        // For each (element_id, max_observed_at), fetch the event row to
        // see if it was a `removed` and to filter by type/scope.
        let mut out: Vec<JsonValue> = Vec::new();
        for row in rows {
            if row.len() < 2 {
                continue;
            }
            let eid = match &row[0] {
                DataValue::Str(s) => s.to_string(),
                _ => continue,
            };
            let obs_at = match &row[1] {
                DataValue::Str(s) => s.to_string(),
                _ => continue,
            };
            let mut p: BTreeMap<String, DataValue> = BTreeMap::new();
            p.insert("eid".into(), DataValue::Str(eid.clone().into()));
            p.insert("obs_at".into(), DataValue::Str(obs_at.into()));
            let detail_q = r#"
                ?[event_type, element_type, element_name, scope, scope_path, path, content_hash, file_size, token_count, enabled] :=
                    *events{element_id: $eid, observed_at: $obs_at,
                            event_type, element_type, element_name, scope, scope_path,
                            path, content_hash, file_size, token_count, enabled}
            "#;
            if let Ok(d) = db.run_script(detail_q, p, ScriptMutability::Immutable) {
                if let Some(r) = d.rows.first() {
                    let event_type = match &r[0] {
                        DataValue::Str(s) => s.to_string(),
                        _ => "".into(),
                    };
                    if event_type == "removed" {
                        continue; // not present at cutoff
                    }
                    let etype = match &r[1] {
                        DataValue::Str(s) => s.to_string(),
                        _ => "".into(),
                    };
                    if let Some(t) = type_filter {
                        if etype != t {
                            continue;
                        }
                    }
                    let scope = match &r[3] {
                        DataValue::Str(s) => s.to_string(),
                        _ => "".into(),
                    };
                    if let Some(sf) = scope_filter {
                        if scope != sf {
                            continue;
                        }
                    }
                    let scope_path = match &r[4] {
                        DataValue::Str(s) => s.to_string(),
                        _ => "".into(),
                    };
                    if let Some(sp) = scope_path_filter {
                        if scope_path != sp {
                            continue;
                        }
                    }
                    out.push(json!({
                        "element_id": eid,
                        "element_type": etype,
                        "element_name": data_to_json(&r[2]),
                        "scope": scope,
                        "scope_path": scope_path,
                        "path": data_to_json(&r[5]),
                        "content_hash": data_to_json(&r[6]),
                        "file_size": data_to_json(&r[7]),
                        "token_count": data_to_json(&r[8]),
                        "enabled": data_to_json(&r[9]),
                    }));
                }
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&JsonValue::Array(out)).unwrap_or_default()
        );
    }

    /// `pss timeline <ELEMENT_ID>`
    pub fn cmd_timeline(db: &DbInstance, element_id: &str, limit: usize) {
        let q = r#"
            ?[event_id, observed_at, event_type, content_hash, file_size, token_count, diff_json] :=
                *events{element_id: $eid, event_id, observed_at, event_type,
                        content_hash, file_size, token_count, diff_json}
            :order observed_at
            :limit $limit
        "#;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(element_id.into()));
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        match db.run_script(q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let rows: Vec<JsonValue> = r
                    .rows
                    .into_iter()
                    .map(|row| {
                        json!({
                            "event_id": data_to_json(&row[0]),
                            "observed_at": data_to_json(&row[1]),
                            "event_type": data_to_json(&row[2]),
                            "content_hash": data_to_json(&row[3]),
                            "file_size": data_to_json(&row[4]),
                            "token_count": data_to_json(&row[5]),
                            "diff_json": data_to_json(&row[6]),
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                );
            }
            Err(e) => eprintln!("timeline query failed: {}", e),
        }
    }

    /// `pss lifespan <ELEMENT_ID>`
    pub fn cmd_lifespan(db: &DbInstance, element_id: &str) {
        let first_q = r#"
            ?[min(observed_at)] := *events{element_id: $eid, observed_at}
        "#;
        let last_install_q = r#"
            ?[max(observed_at)] := *events{element_id: $eid, observed_at, event_type: "installed"}
        "#;
        let last_removal_q = r#"
            ?[max(observed_at)] := *events{element_id: $eid, observed_at, event_type: "removed"}
        "#;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(element_id.into()));
        let first = db
            .run_script(first_q, params.clone(), ScriptMutability::Immutable)
            .ok()
            .and_then(|r| r.rows.first().cloned())
            .and_then(|row| match row.first() {
                Some(DataValue::Str(s)) => Some(s.to_string()),
                _ => None,
            });
        let last_installed = db
            .run_script(last_install_q, params.clone(), ScriptMutability::Immutable)
            .ok()
            .and_then(|r| r.rows.first().cloned())
            .and_then(|row| match row.first() {
                Some(DataValue::Str(s)) => Some(s.to_string()),
                _ => None,
            });
        let last_removed = db
            .run_script(last_removal_q, params, ScriptMutability::Immutable)
            .ok()
            .and_then(|r| r.rows.first().cloned())
            .and_then(|row| match row.first() {
                Some(DataValue::Str(s)) => Some(s.to_string()),
                _ => None,
            });
        // currently present iff last installed > last removed (or no removal)
        let currently_present = match (&last_installed, &last_removed) {
            (Some(_), None) => true,
            (Some(i), Some(r)) => i > r,
            _ => false,
        };
        let out = json!({
            "element_id": element_id,
            "first_seen_at": first,
            "last_installed_at": last_installed,
            "last_removed_at": last_removed,
            "currently_present": currently_present,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    }

    /// `pss changed-between <START> <END>`
    pub fn cmd_changed_between(
        db: &DbInstance,
        start: &str,
        end: &str,
        type_filter: Option<&str>,
        limit: usize,
    ) {
        let start = resolve_date(start);
        let end = resolve_date(end);
        let q = r#"
            ?[observed_at, event_type, element_type, element_name, element_id, content_hash, file_size, diff_json] :=
                *events{observed_at, event_type, element_type, element_name, element_id, content_hash, file_size, diff_json},
                observed_at >= $start, observed_at <= $end,
                or(
                    event_type == "content_changed",
                    event_type == "size_changed",
                    event_type == "frontmatter_changed",
                    event_type == "description_changed",
                    event_type == "path_changed"
                )
            :order observed_at
            :limit $limit
        "#;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("start".into(), DataValue::Str(start.into()));
        params.insert("end".into(), DataValue::Str(end.into()));
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        match db.run_script(q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let rows: Vec<JsonValue> = r
                    .rows
                    .into_iter()
                    .filter(|row| {
                        if let Some(t) = type_filter {
                            matches!(&row[2], DataValue::Str(s) if s.as_str() == t)
                        } else {
                            true
                        }
                    })
                    .map(|row| {
                        json!({
                            "observed_at": data_to_json(&row[0]),
                            "event_type": data_to_json(&row[1]),
                            "element_type": data_to_json(&row[2]),
                            "element_name": data_to_json(&row[3]),
                            "element_id": data_to_json(&row[4]),
                            "content_hash": data_to_json(&row[5]),
                            "file_size": data_to_json(&row[6]),
                            "diff_json": data_to_json(&row[7]),
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                );
            }
            Err(e) => eprintln!("changed-between query failed: {}", e),
        }
    }

    /// `pss removed-since <DATE>`
    pub fn cmd_removed_since(db: &DbInstance, date: &str, limit: usize) {
        let cutoff = resolve_date(date);
        let q = r#"
            ?[observed_at, element_type, element_name, element_id, scope, scope_path] :=
                *events{observed_at, event_type: "removed",
                        element_type, element_name, element_id, scope, scope_path},
                observed_at >= $cutoff
            :order observed_at
            :limit $limit
        "#;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("cutoff".into(), DataValue::Str(cutoff.into()));
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        match db.run_script(q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let rows: Vec<JsonValue> = r
                    .rows
                    .into_iter()
                    .map(|row| {
                        json!({
                            "observed_at": data_to_json(&row[0]),
                            "element_type": data_to_json(&row[1]),
                            "element_name": data_to_json(&row[2]),
                            "element_id": data_to_json(&row[3]),
                            "scope": data_to_json(&row[4]),
                            "scope_path": data_to_json(&row[5]),
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                );
            }
            Err(e) => eprintln!("removed-since query failed: {}", e),
        }
    }

    /// `pss scan-log [--limit N]`
    pub fn cmd_scan_log(db: &DbInstance, limit: usize) {
        let q = r#"
            ?[scan_id, started_at, finished_at, scope_paths_json, events_emitted, rust_binary_version, pss_version] :=
                *scan_runs{scan_id, started_at, finished_at, scope_paths_json,
                           events_emitted, rust_binary_version, pss_version}
            :order -finished_at
            :limit $limit
        "#;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        match db.run_script(q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let rows: Vec<JsonValue> = r
                    .rows
                    .into_iter()
                    .map(|row| {
                        json!({
                            "scan_id": data_to_json(&row[0]),
                            "started_at": data_to_json(&row[1]),
                            "finished_at": data_to_json(&row[2]),
                            "scope_paths_json": data_to_json(&row[3]),
                            "events_emitted": data_to_json(&row[4]),
                            "rust_binary_version": data_to_json(&row[5]),
                            "pss_version": data_to_json(&row[6]),
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                );
            }
            Err(e) => eprintln!("scan-log query failed: {}", e),
        }
    }

    /// `pss db-stats`
    pub fn cmd_db_stats(db: &DbInstance) {
        let event_count = db
            .run_script(
                r#"?[count(event_id)] := *events{event_id}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .ok()
            .and_then(|r| r.rows.first().and_then(|row| row.first().cloned()))
            .and_then(|d| d.get_int())
            .unwrap_or(0);
        let blob_count = db
            .run_script(
                r#"?[count(hash)] := *element_blobs{hash}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .ok()
            .and_then(|r| r.rows.first().and_then(|row| row.first().cloned()))
            .and_then(|d| d.get_int())
            .unwrap_or(0);
        let state_count = db
            .run_script(
                r#"?[count(element_id)] := *elements_state{element_id}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .ok()
            .and_then(|r| r.rows.first().and_then(|row| row.first().cloned()))
            .and_then(|d| d.get_int())
            .unwrap_or(0);
        let oldest = db
            .run_script(
                r#"?[min(observed_at)] := *events{observed_at}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .ok()
            .and_then(|r| r.rows.first().and_then(|row| row.first().cloned()))
            .and_then(|d| match d {
                DataValue::Str(s) => Some(s.to_string()),
                _ => None,
            });
        let retention = db
            .run_script(
                r#"?[v] := *pss_metadata{key: "retention_window", value: v}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .ok()
            .and_then(|r| r.rows.first().and_then(|row| row.first().cloned()))
            .and_then(|d| match d {
                DataValue::Str(s) => Some(s.to_string()),
                _ => None,
            })
            .unwrap_or_else(|| "9m".to_string());
        let out = json!({
            "schema_version": read_schema_version(db),
            "event_count": event_count,
            "blob_count": blob_count,
            "state_count": state_count,
            "oldest_event_at": oldest,
            "retention_window": retention,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    }

    /// `pss reindex` — placeholder; full implementation in Phase 4.
    /// For now, prints a message instructing the user to use the legacy
    /// `pss-reindex-skills` slash command until Phase 4 wires the new
    /// event-emission orchestration.
    pub fn cmd_reindex(_db: &DbInstance, dry_run: bool) {
        let msg = if dry_run {
            "pss reindex --dry-run is a Phase 4 deliverable. \
             Use `/pss-reindex-skills` for now (no event emission yet)."
        } else {
            "pss reindex is a Phase 4 deliverable. \
             Use `/pss-reindex-skills` for now (no event emission yet)."
        };
        eprintln!("{}", msg);
    }

    /// Parse a retention duration string. Accepts ISO 8601 ("P9M",
    /// "P30D", "P1Y") and shorthand ("9m" = 9 months, "30d", "1y", "180d").
    /// Returns the cutoff RFC3339 = `now - duration`. None on parse failure.
    fn cutoff_for_retention(raw: &str) -> Option<String> {
        let now = chrono::Utc::now();
        let t = raw.trim().to_lowercase();
        // ISO 8601 form: PnY / PnM / PnD
        let (num_str, unit_char) = if let Some(rest) = t.strip_prefix('p') {
            let last = rest.chars().last()?;
            let n = &rest[..rest.len().saturating_sub(1)];
            (n.to_string(), last)
        } else {
            // shorthand: "9m" / "30d" / "1y"
            let last = t.chars().last()?;
            let n = &t[..t.len().saturating_sub(1)];
            (n.to_string(), last)
        };
        let n: i64 = num_str.parse().ok()?;
        let cutoff = match unit_char {
            'd' => now - chrono::Duration::days(n),
            'w' => now - chrono::Duration::weeks(n),
            'm' => now - chrono::Duration::days(n * 30),
            'y' => now - chrono::Duration::days(n * 365),
            _ => return None,
        };
        Some(cutoff.to_rfc3339())
    }

    /// `pss prune-history [--dry-run]` — drop events older than retention,
    /// preserving install events of currently-existing elements (so their
    /// timelines remain anchored). Idempotent — safe to run frequently.
    pub fn cmd_prune_history(db: &DbInstance, dry_run: bool) {
        // Read retention window from pss_metadata (default 9m).
        let retention = db
            .run_script(
                r#"?[v] := *pss_metadata{key: "retention_window", value: v}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .ok()
            .and_then(|r| r.rows.first().and_then(|row| row.first().cloned()))
            .and_then(|d| match d {
                DataValue::Str(s) => Some(s.to_string()),
                _ => None,
            })
            .unwrap_or_else(|| "9m".to_string());
        let cutoff = match cutoff_for_retention(&retention) {
            Some(c) => c,
            None => {
                eprintln!(
                    "prune-history: invalid retention_window '{}' (must be like 9m, 30d, 1y, P9M)",
                    retention
                );
                return;
            }
        };

        // Find candidate events: those older than cutoff that are NOT the
        // most recent install event for an element whose elements_state
        // exists=true. We delete them.
        // For correctness, we collect targeted event_ids first, then issue
        // a single :rm.
        let candidates_q = r#"
            ?[event_id] :=
                *events{event_id, observed_at, event_type, element_id},
                observed_at < $cutoff,
                not (
                    *elements_state{element_id, exists: true},
                    event_type == "installed"
                )
        "#;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("cutoff".into(), DataValue::Str(cutoff.clone().into()));
        let candidates = match db.run_script(candidates_q, params, ScriptMutability::Immutable) {
            Ok(r) => r.rows,
            Err(e) => {
                eprintln!("prune-history candidate query failed: {}", e);
                return;
            }
        };
        let count = candidates.len();
        if dry_run {
            println!(
                "{}",
                serde_json::json!({
                    "dry_run": true,
                    "retention_window": retention,
                    "cutoff": cutoff,
                    "candidates": count,
                })
            );
            return;
        }
        // Issue one delete per candidate (Cozo doesn't have a native batch
        // delete, but the scale is bounded by retention).
        let mut deleted = 0u64;
        for row in candidates {
            if let Some(DataValue::Str(eid)) = row.first() {
                let mut p: BTreeMap<String, DataValue> = BTreeMap::new();
                p.insert("eid".into(), DataValue::Str(eid.to_string().into()));
                let q = r#"?[event_id] <- [[$eid]] :rm events { event_id }"#;
                if db.run_script(q, p, ScriptMutability::Mutable).is_ok() {
                    deleted += 1;
                }
            }
        }
        println!(
            "{}",
            serde_json::json!({
                "deleted_events": deleted,
                "retention_window": retention,
                "cutoff": cutoff,
            })
        );
    }

    /// `pss retention [--set <DURATION>]` — read or write retention
    /// window in `pss_metadata`.
    pub fn cmd_retention(db: &DbInstance, set: Option<&str>) {
        if let Some(value) = set {
            // Validate the duration string. Accept ISO 8601 ("P9M",
            // "P30D") and shorthand ("9m", "30d", "1y"). Phase 4 will
            // formalise this; for now we accept any non-empty string.
            if value.is_empty() {
                eprintln!("retention --set requires a non-empty duration");
                return;
            }
            let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
            params.insert("key".into(), DataValue::Str("retention_window".into()));
            params.insert("value".into(), DataValue::Str(value.into()));
            let q = r#"?[key, value] <- [[$key, $value]]
                       :put pss_metadata { key => value }"#;
            match db.run_script(q, params, ScriptMutability::Mutable) {
                Ok(_) => println!("{{\"retention_window\":\"{}\"}}", value),
                Err(e) => eprintln!("retention set failed: {}", e),
            }
        } else {
            let v = db
                .run_script(
                    r#"?[v] := *pss_metadata{key: "retention_window", value: v}"#,
                    BTreeMap::new(),
                    ScriptMutability::Immutable,
                )
                .ok()
                .and_then(|r| r.rows.first().and_then(|row| row.first().cloned()))
                .and_then(|d| match d {
                    DataValue::Str(s) => Some(s.to_string()),
                    _ => None,
                })
                .unwrap_or_else(|| "9m".to_string());
            println!("{{\"retention_window\":\"{}\"}}", v);
        }
    }
}

/// Drop all temporal tables — used by tests only. Production code
/// should never call this.
#[cfg(test)]
pub fn drop_temporal_tables(db: &DbInstance) -> Result<(), String> {
    let names = [
        "events",
        "element_blobs",
        "elements_state",
        "scan_runs",
    ];
    for n in names.iter() {
        let q = format!("::remove {}", n);
        let _ = db.run_script(&q, BTreeMap::new(), ScriptMutability::Mutable);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_id_is_stable_and_lowercased() {
        let id = compute_element_id(
            ElementType::Skill,
            "MyCoolSkill",
            "Project",
            "/Users/foo/Bar",
        );
        assert_eq!(id, "skill:mycoolskill@project:_users_foo_bar");
    }

    #[test]
    fn element_id_distinguishes_scopes() {
        let user = compute_element_id(ElementType::Agent, "x", "user", "");
        let proj = compute_element_id(ElementType::Agent, "x", "project", "/a/b");
        let local = compute_element_id(ElementType::Agent, "x", "local", "/a/b");
        assert_ne!(user, proj);
        assert_ne!(proj, local);
    }

    #[test]
    fn content_hash_is_sha256_hex() {
        assert_eq!(
            content_hash(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            content_hash(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn token_count_cl100k_known_strings() {
        // Known: cl100k_base tokenizes "hello world" as 2 tokens.
        let n = token_count_cl100k("hello world");
        assert!(n > 0, "expected positive token count, got {}", n);
        // Empty string is 0 tokens.
        assert_eq!(token_count_cl100k(""), 0);
    }

    #[test]
    fn override_precedence_only_for_file_elements() {
        assert!(ElementType::Skill.has_override_precedence());
        assert!(ElementType::Agent.has_override_precedence());
        assert!(ElementType::Command.has_override_precedence());
        assert!(ElementType::Rule.has_override_precedence());
        assert!(ElementType::OutputStyle.has_override_precedence());
        assert!(ElementType::Theme.has_override_precedence());
        // Hooks merge per docs/en/settings.md "Array settings merge".
        assert!(!ElementType::Hook.has_override_precedence());
        // Plugins have their own scope semantics (installed_plugins.json
        // tracks per-scope membership rather than precedence).
        assert!(!ElementType::Plugin.has_override_precedence());
        // MCP servers are allowlisted/denylisted, not overridden.
        assert!(!ElementType::Mcp.has_override_precedence());
    }

    #[test]
    fn event_type_strings_are_snake_case() {
        assert_eq!(EventType::Installed.as_str(), "installed");
        assert_eq!(EventType::OverrideStarted.as_str(), "override_started");
        assert_eq!(
            EventType::PluginVersionChanged.as_str(),
            "plugin_version_changed"
        );
    }

    fn legacy_skills_v1_db() -> DbInstance {
        let db = DbInstance::new("mem", "", "").expect("mem db");
        // Legacy schema (subset — only the columns migration reads).
        db.run_script(
            r#":create skills {
                name: String, source: String =>
                id: String default "",
                path: String,
                skill_type: String,
                description: String,
                first_indexed_at: String,
                last_updated_at: String,
            }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        )
        .expect("create skills");
        db.run_script(
            r#":create rules {
                name: String, scope: String =>
                description: String,
                source_path: String,
                summary: String default "",
                keywords_json: String default "[]",
            }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        )
        .expect("create rules");
        db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        )
        .expect("create pss_metadata");
        // Seed a row in each.
        db.run_script(
            r#"?[name, source, id, path, skill_type, description, first_indexed_at, last_updated_at] <-
                [["my-skill", "user", "abc", "/tmp/nonexistent.md", "skill", "test desc", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"]]
               :put skills { name, source => id, path, skill_type, description, first_indexed_at, last_updated_at }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        )
        .expect("seed skill");
        db.run_script(
            r#"?[name, scope, description, source_path] <-
                [["worktree-merge", "user", "remind to merge worktree", "/tmp/wt.md"]]
               :put rules { name, scope => description, source_path }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        )
        .expect("seed rule");
        db
    }

    fn obs(name: &str, scope: &str, hash: &str, size: i64, path: &str) -> Observation {
        Observation {
            element_type: ElementType::Skill,
            name: name.to_string(),
            scope: scope.to_string(),
            scope_path: "".to_string(),
            source: scope.to_string(),
            path: path.to_string(),
            content_hash: hash.to_string(),
            file_size: size,
            token_count: 100,
            description: "".to_string(),
            enabled: true,
        }
    }

    fn prior_of(o: &Observation) -> PriorState {
        PriorState {
            element_id: o.element_id(),
            current_path: o.path.clone(),
            current_hash: o.content_hash.clone(),
            current_size: o.file_size,
            enabled: o.enabled,
            override_status: "active".to_string(),
            exists: true,
        }
    }

    #[test]
    fn emit_install_when_no_prior() {
        let o = obs("x", "user", "h1", 100, "/p");
        assert_eq!(compare_and_emit(None, &o), vec![EventType::Installed]);
    }

    #[test]
    fn emit_install_when_prior_was_tombstoned() {
        let o = obs("x", "user", "h1", 100, "/p");
        let mut p = prior_of(&o);
        p.exists = false;
        assert_eq!(
            compare_and_emit(Some(&p), &o),
            vec![EventType::Installed]
        );
    }

    #[test]
    fn emit_nothing_when_unchanged() {
        let o = obs("x", "user", "h1", 100, "/p");
        let p = prior_of(&o);
        assert!(compare_and_emit(Some(&p), &o).is_empty());
    }

    #[test]
    fn emit_content_and_size_when_both_change() {
        let o = obs("x", "user", "h2", 200, "/p");
        let mut p = prior_of(&o);
        p.current_hash = "h1".to_string();
        p.current_size = 100;
        assert_eq!(
            compare_and_emit(Some(&p), &o),
            vec![EventType::ContentChanged, EventType::SizeChanged]
        );
    }

    #[test]
    fn emit_size_only_when_only_size_changes() {
        let o = obs("x", "user", "h1", 200, "/p");
        let mut p = prior_of(&o);
        p.current_size = 100;
        assert_eq!(compare_and_emit(Some(&p), &o), vec![EventType::SizeChanged]);
    }

    #[test]
    fn emit_path_change_when_only_path_differs() {
        let o = obs("x", "user", "h1", 100, "/new/p");
        let mut p = prior_of(&o);
        p.current_path = "/old/p".to_string();
        assert_eq!(compare_and_emit(Some(&p), &o), vec![EventType::PathChanged]);
    }

    #[test]
    fn emit_disabled_when_enabled_flag_flips() {
        let mut o = obs("x", "user", "h1", 100, "/p");
        o.enabled = false;
        let p = prior_of(&Observation { enabled: true, ..o.clone() });
        assert_eq!(compare_and_emit(Some(&p), &o), vec![EventType::Disabled]);
    }

    #[test]
    fn detect_removals_flags_missing_elements() {
        use std::collections::HashSet;
        let prior: HashSet<String> =
            ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let observed: HashSet<String> = ["a", "c"].iter().map(|s| s.to_string()).collect();
        let removed = detect_removals(&prior, &observed);
        assert_eq!(removed, vec!["b".to_string()]);
    }

    #[test]
    fn resolve_overrides_single_element_is_active() {
        let r = resolve_overrides(
            ElementType::Skill,
            &[("eid1".to_string(), "user".to_string())],
        );
        assert_eq!(r, vec![("eid1".to_string(), "active".to_string())]);
    }

    #[test]
    fn resolve_overrides_local_beats_project_beats_user() {
        let r = resolve_overrides(
            ElementType::Skill,
            &[
                ("user-eid".to_string(), "user".to_string()),
                ("local-eid".to_string(), "local".to_string()),
                ("proj-eid".to_string(), "project".to_string()),
            ],
        );
        // local wins; it overrides project + user
        assert_eq!(r[0].0, "local-eid");
        assert_eq!(r[0].1, "overrides:proj-eid;user-eid");
        // project overridden_by local
        let proj = r.iter().find(|(e, _)| e == "proj-eid").unwrap();
        assert_eq!(proj.1, "overridden_by:local-eid");
        let user = r.iter().find(|(e, _)| e == "user-eid").unwrap();
        assert_eq!(user.1, "overridden_by:local-eid");
    }

    #[test]
    fn resolve_overrides_passes_through_for_hooks() {
        // Hooks merge per docs/en/settings.md — never overridden.
        let r = resolve_overrides(
            ElementType::Hook,
            &[
                ("a".to_string(), "user".to_string()),
                ("b".to_string(), "project".to_string()),
            ],
        );
        for (_, s) in r {
            assert_eq!(s, "active");
        }
    }

    #[test]
    fn migrate_v1_to_v2_emits_install_events() {
        let db = legacy_skills_v1_db();
        // schema_version absent => migration must run.
        assert_eq!(read_schema_version(&db), "1");
        let stats = migrate_v1_to_v2(&db).expect("migration");
        assert_eq!(stats.skills_migrated, 1);
        assert_eq!(stats.rules_migrated, 1);
        assert!(!stats.already_at_target_version);

        // events table populated — Cozo uses count() directly in the head.
        let count = db
            .run_script(
                r#"?[count(event_id)] := *events{event_id}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .expect("count events");
        let n = count.rows[0][0].get_int().unwrap_or(0);
        assert_eq!(n, 2);

        // elements_state populated
        let states = db
            .run_script(
                r#"?[count(element_id)] := *elements_state{element_id}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .expect("count states");
        let m = states.rows[0][0].get_int().unwrap_or(0);
        assert_eq!(m, 2);

        // schema_version stamped to "2"
        assert_eq!(read_schema_version(&db), "2");
    }

    #[test]
    fn migrate_is_idempotent_on_v2_db() {
        let db = legacy_skills_v1_db();
        let s1 = migrate_v1_to_v2(&db).expect("first");
        assert_eq!(s1.skills_migrated, 1);
        let s2 = migrate_v1_to_v2(&db).expect("second");
        assert!(s2.already_at_target_version);
        assert_eq!(s2.skills_migrated, 0);
    }

    #[test]
    fn ensure_schema_idempotent() {
        // Use an in-memory cozo store so we don't pollute the real DB.
        let db = DbInstance::new("mem", "", "").expect("mem db");
        // pss_metadata may not exist yet; create it minimally.
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("first call");
        ensure_schema(&db).expect("second call (idempotent)");
        assert_eq!(read_schema_version(&db), TEMPORAL_SCHEMA_VERSION);
    }
}
