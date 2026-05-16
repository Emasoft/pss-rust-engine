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

/// DI-1 wave 1 (audit 20260514): a stable 32-char hex digest of an
/// element's `description` field, used by the description_changed
/// detector. Empty descriptions hash to the well-known empty-SHA-256
/// prefix `"e3b0c44298fc1c14"` (16 bytes), which still differs from
/// `""` — so a transition `<no description>` ↔ `"foo"` is detectable.
pub fn description_hash(desc: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(desc.as_bytes());
    let result = hasher.finalize();
    // 16 bytes = 32 hex chars: collision-safe for ~10k elements and
    // half the storage cost of a full 64-char SHA-256.
    let truncated = &result[..16];
    hex_lower(truncated)
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
    // ────────────────────────────────────────────────────────────────────
    // DBE-1 (audit 20260514) — secondary indexes on `events`.
    //
    // Without these, every filter on element_id / observed_at / event_type /
    // scope / element_name does an O(N) full scan over the events table.
    // Measured 14–62 ms today (~9k events), projecting to 90–360 ms at 60k
    // events after 1 year. With covering indexes, every lifecycle query
    // drops to sub-millisecond.
    //
    // Cozo's `::index create rel:idx { fields }` creates a btree-backed
    // covering index. ensure_schema's "already exists" suppressor treats
    // re-runs as no-ops, so this is safe on every startup.
    //
    // Trade-off: each index adds ~5k k-v rows (~600 KB total disk today;
    // ~3–5 MB at 60k events). Negligible compared to the read latency win.
    // ────────────────────────────────────────────────────────────────────
    r#"::index create events:by_element_id { element_id }"#,
    r#"::index create events:by_observed_at { observed_at }"#,
    r#"::index create events:by_event_type { event_type }"#,
    r#"::index create events:by_scope { scope }"#,
    r#"::index create events:by_element_name { element_name }"#,
    // DI-1 wave 1 (audit 20260514): per-element description tracking so
    // the writer can detect description_changed events without touching
    // the elements_state schema (which would force a v2→v3 migration).
    // The hash is a hex-encoded 16-byte truncated SHA-256 of the raw
    // description text; storing the text alongside lets us emit a
    // crisp diff into events.diff_json (and enables a future
    // `pss description-history` subcommand).
    //
    // Storage cost: ~9k rows × (~64 hash + ~200 text) ≈ 2.4 MB at
    // current scale. Acceptable — events table is already ~12 MB.
    r#":create element_descriptions {
        element_id: String =>
        description_hash: String,
        description_text: String default "",
        last_updated_at: String,
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

    /// Resolve a date string to RFC3339. Thin wrapper around the unified
    /// `crate::parse_date` (per COR-7 — audit 20260514). Returns String error
    /// on garbage input so the legacy `temporal::cli` module doesn't need to
    /// know about `SuggesterError`.
    ///
    /// Per COR-2 (audit 20260514): garbage like "tomorrow" or "2026/05/14"
    /// now produces a clear error instead of being silently passed through to
    /// CozoDB (where it would string-compare-true against every row).
    ///
    /// Accepts: "now", "yesterday", "1d"/"2w"/"24h"/"30m"/"120s" relative
    /// shorthand, "YYYY-MM-DD" (validated, end-of-day UTC), full RFC3339,
    /// or naive datetime ("YYYY-MM-DDTHH:MM:SS").
    fn resolve_date(input: &str) -> Result<String, String> {
        crate::parse_date(input).map_err(|e| e.to_string())
    }

    /// Format an `Invalid date` error consistently and emit `[]` to stdout
    /// so callers can `return;` after a single line.
    fn print_date_err(label: &str, input: &str, err: &str) {
        eprintln!("Invalid {} '{}': {}", label, input, err);
        println!("[]");
    }

    /// `pss as-of <DATE> [filters]`
    ///
    /// DBE-2 + COR-1 (audit 20260514): single-query rewrite. The old
    /// implementation walked every element_id in `elements_state`, then
    /// issued a separate Cozo query per element to fetch the latest
    /// at-or-before event — 1 + 9131 round-trips, ~2.2 min on the live DB.
    /// The COR-1 bug was a `take(limit)` applied to the element_id slice
    /// BEFORE the --type / --scope filters, so `--type skill --limit 100`
    /// could return 0 rows (the first 100 element_ids weren't skills).
    ///
    /// New strategy: one sorted Datalog query fetches every event
    /// at-or-before cutoff (with `--type` / `--scope` / `--scope-path`
    /// pushed into the WHERE clause), sorted by (element_id desc,
    /// observed_at desc). Rust then dedupes by element_id (first
    /// occurrence per id = latest observed_at — Cozo's numeric `max()`
    /// can't aggregate the RFC3339 strings, hence the sort+dedup pattern),
    /// excludes events whose latest type is `removed`, and applies the
    /// `:limit` LAST. With DBE-1 indexes also landing, the sort is
    /// effectively over a presorted observed_at column, dropping latency
    /// from ~2 min to <50 ms on today's DB.
    pub fn cmd_as_of(
        db: &DbInstance,
        date: &str,
        type_filter: Option<&str>,
        scope_filter: Option<&str>,
        scope_path_filter: Option<&str>,
        limit: usize,
    ) {
        use std::collections::HashSet;

        let cutoff = match resolve_date(date) {
            Ok(s) => s,
            Err(e) => {
                print_date_err("date", date, &e);
                return;
            }
        };

        // Parametrized filters — never f-string interpolated. Type filter
        // is whitelisted by validate_type_filter() in main.rs; scope and
        // scope_path arrive from CLI flags and must flow through $params.
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("cutoff".into(), DataValue::Str(cutoff.into()));

        let mut filter_clauses = String::new();
        if let Some(t) = type_filter {
            filter_clauses.push_str(", element_type = $f_type");
            params.insert("f_type".into(), DataValue::Str(t.into()));
        }
        if let Some(s) = scope_filter {
            filter_clauses.push_str(", scope = $f_scope");
            params.insert("f_scope".into(), DataValue::Str(s.into()));
        }
        if let Some(sp) = scope_path_filter {
            filter_clauses.push_str(", scope_path = $f_scope_path");
            params.insert("f_scope_path".into(), DataValue::Str(sp.into()));
        }

        // Single sorted query: fetch every event at-or-before cutoff that
        // passes the --type / --scope / --scope-path filters, sorted so
        // that the FIRST row per element_id is its latest event.
        let query = format!(
            r#"
            ?[element_id, observed_at, event_type, element_type, element_name,
              scope, scope_path, path, content_hash, file_size, token_count, enabled] :=
                *events{{element_id, observed_at, event_type, element_type, element_name,
                         scope, scope_path, path, content_hash, file_size, token_count, enabled}},
                observed_at <= $cutoff{filters}
            :order element_id, -observed_at
            "#,
            filters = filter_clauses
        );

        let result = match db.run_script(&query, params, ScriptMutability::Immutable) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("as-of query failed: {}", e);
                println!("[]");
                return;
            }
        };

        // Dedupe by element_id (first occurrence per id = latest by
        // observed_at). Skip rows where the latest event is `removed`
        // (element wasn't present at cutoff). Apply :limit AFTER filtering
        // — this is the COR-1 fix.
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<JsonValue> = Vec::new();
        for r in result.rows.iter() {
            let eid = match &r[0] {
                DataValue::Str(s) => s.to_string(),
                _ => continue,
            };
            if !seen.insert(eid.clone()) {
                continue; // not the latest occurrence
            }
            let event_type = match &r[2] {
                DataValue::Str(s) => s.to_string(),
                _ => "".into(),
            };
            if event_type == "removed" {
                continue; // latest event is a removal — not present at cutoff
            }
            out.push(json!({
                "element_id": eid,
                "event_type": event_type,
                "element_type": data_to_json(&r[3]),
                "element_name": data_to_json(&r[4]),
                "scope": data_to_json(&r[5]),
                "scope_path": data_to_json(&r[6]),
                "path": data_to_json(&r[7]),
                "content_hash": data_to_json(&r[8]),
                "file_size": data_to_json(&r[9]),
                "token_count": data_to_json(&r[10]),
                "enabled": data_to_json(&r[11]),
            }));
            if out.len() >= limit {
                break;
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
    ///
    /// COR-5 (audit 20260514): the prior implementation used Cozo's numeric
    /// `min()` / `max()` aggregates against RFC3339 strings — those silently
    /// returned a `DataValue` variant that didn't unwrap to String, so
    /// `lifespan` always returned `null` for first/last timestamps. Switch
    /// to `:order observed_at :limit 1` which works on string-sortable
    /// timestamps and benefits from the DBE-1 `events:by_observed_at` index.
    pub fn cmd_lifespan(db: &DbInstance, element_id: &str) {
        let first_q = r#"
            ?[observed_at] := *events{element_id: $eid, observed_at}
            :order observed_at
            :limit 1
        "#;
        let last_install_q = r#"
            ?[observed_at] := *events{element_id: $eid, observed_at, event_type: "installed"}
            :order -observed_at
            :limit 1
        "#;
        let last_removal_q = r#"
            ?[observed_at] := *events{element_id: $eid, observed_at, event_type: "removed"}
            :order -observed_at
            :limit 1
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
        let start = match resolve_date(start) {
            Ok(s) => s,
            Err(e) => { print_date_err("start date", start, &e); return; }
        };
        let end = match resolve_date(end) {
            Ok(s) => s,
            Err(e) => { print_date_err("end date", end, &e); return; }
        };
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

    /// `pss by-plugin <NAME>` (Phase 3 Tier A — audit 20260514)
    ///
    /// List every currently-active element whose `source` is exactly
    /// `plugin:<NAME>` — i.e. provided by the given plugin. Reads from
    /// `elements_state` joined against `events.source` via last_event_id.
    pub fn cmd_by_plugin(
        db: &DbInstance,
        plugin_name: &str,
        type_filter: Option<&str>,
        limit: usize,
    ) {
        let needle = format!("plugin:{}", plugin_name);
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("source".into(), DataValue::Str(needle.into()));
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        let mut filter_clauses = String::new();
        if let Some(t) = type_filter {
            filter_clauses.push_str(", element_type = $f_type");
            params.insert("f_type".into(), DataValue::Str(t.into()));
        }
        let q = format!(
            r#"?[element_id, element_type, element_name, scope_path, current_path, last_changed_at] :=
                *elements_state{{element_id, last_event_id, current_path, last_changed_at, exists: true}},
                *events{{event_id: last_event_id, element_type, element_name, scope_path, source}},
                source = $source{filters}
            :order element_type, element_name
            :limit $limit"#,
            filters = filter_clauses
        );
        match db.run_script(&q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let out: Vec<JsonValue> = r.rows.iter().map(|row| {
                    json!({
                        "element_id": data_to_json(&row[0]),
                        "element_type": data_to_json(&row[1]),
                        "element_name": data_to_json(&row[2]),
                        "scope_path": data_to_json(&row[3]),
                        "path": data_to_json(&row[4]),
                        "last_changed_at": data_to_json(&row[5]),
                    })
                }).collect();
                println!("{}", serde_json::to_string_pretty(&JsonValue::Array(out)).unwrap_or_default());
            }
            Err(e) => {
                eprintln!("by-plugin query failed: {}", e);
                println!("[]");
            }
        }
    }

    /// `pss by-marketplace <NAME>` (Phase 3 Tier A — F-2, audit 20260514)
    ///
    /// List every currently-active element whose `source` starts with
    /// `marketplace:<NAME>` — i.e. installed from the given marketplace.
    /// The discoverer encodes marketplace-installed plugins with
    /// `source = "marketplace:<name>"` so we match on a prefix here.
    /// Reads from `elements_state` joined against `events.source` via
    /// last_event_id.
    pub fn cmd_by_marketplace(
        db: &DbInstance,
        marketplace_name: &str,
        type_filter: Option<&str>,
        limit: usize,
    ) {
        let prefix = format!("marketplace:{}", marketplace_name);
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("prefix".into(), DataValue::Str(prefix.into()));
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        let mut filter_clauses = String::new();
        if let Some(t) = type_filter {
            filter_clauses.push_str(", element_type = $f_type");
            params.insert("f_type".into(), DataValue::Str(t.into()));
        }
        let q = format!(
            r#"?[element_id, element_type, element_name, scope_path, current_path, last_changed_at, source] :=
                *elements_state{{element_id, last_event_id, current_path, last_changed_at, exists: true}},
                *events{{event_id: last_event_id, element_type, element_name, scope_path, source}},
                starts_with(source, $prefix){filters}
            :order element_type, element_name
            :limit $limit"#,
            filters = filter_clauses
        );
        match db.run_script(&q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let out: Vec<JsonValue> = r.rows.iter().map(|row| {
                    json!({
                        "element_id": data_to_json(&row[0]),
                        "element_type": data_to_json(&row[1]),
                        "element_name": data_to_json(&row[2]),
                        "scope_path": data_to_json(&row[3]),
                        "path": data_to_json(&row[4]),
                        "last_changed_at": data_to_json(&row[5]),
                        "source": data_to_json(&row[6]),
                    })
                }).collect();
                println!("{}", serde_json::to_string_pretty(&JsonValue::Array(out)).unwrap_or_default());
            }
            Err(e) => {
                eprintln!("by-marketplace query failed: {}", e);
                println!("[]");
            }
        }
    }

    /// F-17 (audit 20260514): list every event from a specific scan_id.
    pub fn cmd_changes_in_batch(db: &DbInstance, scan_id: &str, limit: usize) {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("scan_id".into(), DataValue::Str(scan_id.into()));
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        let q = r#"
            ?[observed_at, event_type, element_type, element_name, element_id, scope, diff_json] :=
                *events{event_id, scan_id, observed_at, event_type,
                        element_type, element_name, element_id, scope, diff_json},
                scan_id = $scan_id
            :order observed_at, element_name
            :limit $limit
        "#;
        match db.run_script(q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let rows: Vec<JsonValue> = r.rows.into_iter().map(|row| {
                    json!({
                        "observed_at": data_to_json(&row[0]),
                        "event_type": data_to_json(&row[1]),
                        "element_type": data_to_json(&row[2]),
                        "element_name": data_to_json(&row[3]),
                        "element_id": data_to_json(&row[4]),
                        "scope": data_to_json(&row[5]),
                        "diff_json": data_to_json(&row[6]),
                    })
                }).collect();
                println!("{}", serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default());
            }
            Err(e) => {
                eprintln!("changes-in-batch query failed: {}", e);
                println!("[]");
            }
        }
    }

    /// F-18 (audit 20260514): emit every event from the most recent scan.
    pub fn cmd_last_changes(db: &DbInstance, limit: usize) {
        // Find the latest scan_id by `started_at` from scan_runs.
        let latest_q = r#"
            ?[scan_id, started_at] :=
                *scan_runs{scan_id, started_at}
            :order -started_at
            :limit 1
        "#;
        let latest = match db.run_script(latest_q, BTreeMap::new(), ScriptMutability::Immutable) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("last-changes: scan_runs query failed: {}", e);
                println!("[]");
                return;
            }
        };
        let scan_id = match latest.rows.first().and_then(|r| r.first()) {
            Some(DataValue::Str(s)) => s.to_string(),
            _ => {
                eprintln!("last-changes: no scan_runs found");
                println!("[]");
                return;
            }
        };
        cmd_changes_in_batch(db, &scan_id, limit);
    }

    /// F-19 (audit 20260514): count elements per scope, optionally
    /// filtered by element_type. Reads `elements_state` joined against
    /// `events.scope` via last_event_id.
    pub fn cmd_stats_by_scope(db: &DbInstance, type_filter: Option<&str>) {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        let mut filters = String::new();
        if let Some(t) = type_filter {
            filters.push_str(", element_type = $f_type");
            params.insert("f_type".into(), DataValue::Str(t.into()));
        }
        let q = format!(
            r#"?[scope, element_type, count(element_id)] :=
                *elements_state{{element_id, last_event_id, exists: true}},
                *events{{event_id: last_event_id, scope, element_type}}{filters}
            :order scope, element_type"#,
            filters = filters
        );
        match db.run_script(&q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let mut by_scope: std::collections::BTreeMap<String, std::collections::BTreeMap<String, i64>> =
                    std::collections::BTreeMap::new();
                let mut total: i64 = 0;
                for row in &r.rows {
                    let scope = if let DataValue::Str(s) = &row[0] { s.to_string() } else { continue };
                    let etype = if let DataValue::Str(s) = &row[1] { s.to_string() } else { continue };
                    let count = match &row[2] {
                        DataValue::Num(Num::Int(n)) => *n,
                        _ => 0,
                    };
                    by_scope.entry(scope).or_default().insert(etype, count);
                    total += count;
                }
                let output = json!({
                    "by_scope": by_scope,
                    "total_elements": total,
                });
                println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
            }
            Err(e) => {
                eprintln!("stats-by-scope query failed: {}", e);
                println!("{{}}");
            }
        }
    }

    /// `pss scope-diff <SCOPE1> <SCOPE2>` (Phase 3 Tier A — F-6, audit 20260514)
    ///
    /// Show elements present in one scope but not the other. Output is
    /// a JSON object with three keys:
    ///   - `only_in_scope1`: list of element_names present in scope1 but not scope2
    ///   - `only_in_scope2`: list of element_names present in scope2 but not scope1
    ///   - `shared`: list of element_names present in both
    /// All comparisons are by (element_type, element_name) tuple so a
    /// rule and a skill with the same name don't collide.
    pub fn cmd_scope_diff(
        db: &DbInstance,
        scope1: &str,
        scope2: &str,
        type_filter: Option<&str>,
        limit: usize,
    ) {
        // Helper: fetch all (element_type, element_name) tuples for a scope.
        let load_scope = |scope: &str| -> Result<std::collections::HashSet<(String, String)>, String> {
            let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
            params.insert("scope".into(), DataValue::Str(scope.into()));
            let mut filters = String::new();
            if let Some(t) = type_filter {
                filters.push_str(", element_type = $f_type");
                params.insert("f_type".into(), DataValue::Str(t.into()));
            }
            let q = format!(
                r#"?[element_type, element_name] :=
                    *elements_state{{element_id, last_event_id, exists: true}},
                    *events{{event_id: last_event_id, element_type, element_name, scope}},
                    scope = $scope{filters}"#,
                filters = filters
            );
            let r = db.run_script(&q, params, ScriptMutability::Immutable)
                .map_err(|e| e.to_string())?;
            Ok(r.rows.iter().filter_map(|row| {
                let etype = if let DataValue::Str(s) = &row[0] { s.to_string() } else { return None };
                let ename = if let DataValue::Str(s) = &row[1] { s.to_string() } else { return None };
                Some((etype, ename))
            }).collect())
        };

        let set1 = match load_scope(scope1) {
            Ok(s) => s,
            Err(e) => { eprintln!("scope-diff query failed (scope1): {}", e); println!("{{}}"); return; }
        };
        let set2 = match load_scope(scope2) {
            Ok(s) => s,
            Err(e) => { eprintln!("scope-diff query failed (scope2): {}", e); println!("{{}}"); return; }
        };

        let to_json = |tuples: &std::collections::HashSet<(String, String)>| -> Vec<JsonValue> {
            let mut sorted: Vec<(String, String)> = tuples.iter().cloned().collect();
            sorted.sort();
            sorted.into_iter().take(limit).map(|(t, n)| {
                json!({"element_type": t, "element_name": n})
            }).collect()
        };

        let only1: std::collections::HashSet<(String, String)> =
            set1.difference(&set2).cloned().collect();
        let only2: std::collections::HashSet<(String, String)> =
            set2.difference(&set1).cloned().collect();
        let shared: std::collections::HashSet<(String, String)> =
            set1.intersection(&set2).cloned().collect();

        let output = json!({
            "scope1": scope1,
            "scope2": scope2,
            "only_in_scope1": to_json(&only1),
            "only_in_scope2": to_json(&only2),
            "shared": to_json(&shared),
            "counts": {
                "only_in_scope1": only1.len(),
                "only_in_scope2": only2.len(),
                "shared": shared.len(),
            },
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
    }

    /// `pss changes-summary --window <DURATION>` (Phase 3 Tier A)
    ///
    /// Count events by `event_type` in the time window before now. Reads
    /// from `events` with an observed_at >= cutoff filter. Output is a JSON
    /// object: `{"event_type": count, ...}` plus a `window` field for
    /// audit. Uses the unified `crate::parse_date` helper so the same date
    /// shorthand (`24h`, `7d`, `2w`, RFC3339, `yesterday`) is accepted as
    /// every other temporal subcommand.
    pub fn cmd_changes_summary(
        db: &DbInstance,
        window: &str,
        type_filter: Option<&str>,
    ) {
        let cutoff = match resolve_date(window) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Invalid --window '{}': {}", window, e);
                println!("{{}}");
                return;
            }
        };
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("cutoff".into(), DataValue::Str(cutoff.clone().into()));
        let mut filter_clauses = String::new();
        if let Some(t) = type_filter {
            filter_clauses.push_str(", element_type = $f_type");
            params.insert("f_type".into(), DataValue::Str(t.into()));
        }
        let q = format!(
            r#"?[event_type, count(event_id)] :=
                *events{{event_id, observed_at, event_type, element_type}},
                observed_at >= $cutoff{filters}"#,
            filters = filter_clauses
        );
        match db.run_script(&q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let mut counts = serde_json::Map::new();
                for row in &r.rows {
                    let etype = data_to_json(&row[0]);
                    let count = data_to_json(&row[1]);
                    if let JsonValue::String(s) = etype {
                        counts.insert(s, count);
                    }
                }
                let out = json!({
                    "window": window,
                    "cutoff": cutoff,
                    "type_filter": type_filter,
                    "counts": JsonValue::Object(counts),
                });
                println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
            }
            Err(e) => {
                eprintln!("changes-summary query failed: {}", e);
                println!("{{}}");
            }
        }
    }

    /// `pss enabled-where <NAME>` (Phase 3 Tier A)
    ///
    /// Return every (scope, scope_path) tuple where the given element_name
    /// is currently present (`exists=true`) AND enabled (`enabled=true`).
    /// Useful for "is plugin X actually on anywhere?" or detecting a
    /// half-disabled-half-enabled scope drift.
    pub fn cmd_enabled_where(
        db: &DbInstance,
        name: &str,
        type_filter: Option<&str>,
    ) {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("name".into(), DataValue::Str(name.into()));
        let mut filter_clauses = String::new();
        if let Some(t) = type_filter {
            filter_clauses.push_str(", element_type = $f_type");
            params.insert("f_type".into(), DataValue::Str(t.into()));
        }
        let q = format!(
            r#"?[element_id, element_type, scope, scope_path] :=
                *elements_state{{element_id, last_event_id, exists: true, enabled: true}},
                *events{{event_id: last_event_id, element_type, element_name, scope, scope_path}},
                element_name = $name{filters}
            :order scope, scope_path"#,
            filters = filter_clauses
        );
        match db.run_script(&q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let out: Vec<JsonValue> = r.rows.iter().map(|row| {
                    json!({
                        "element_id": data_to_json(&row[0]),
                        "element_type": data_to_json(&row[1]),
                        "scope": data_to_json(&row[2]),
                        "scope_path": data_to_json(&row[3]),
                    })
                }).collect();
                println!("{}", serde_json::to_string_pretty(&JsonValue::Array(out)).unwrap_or_default());
            }
            Err(e) => {
                eprintln!("enabled-where query failed: {}", e);
                println!("[]");
            }
        }
    }

    /// `pss compare-snapshots <DATE1> <DATE2>` (Phase 3 Tier A F-5)
    ///
    /// Diff the active-element sets at two points in time. Reuses the same
    /// dedup-by-element_id pattern as `cmd_as_of` (Cozo can't `max()` on
    /// RFC3339 strings, so we sort+dedupe in Rust). Output is a structured
    /// JSON object with `only_at_date1`, `only_at_date2`, and
    /// `common_count` — enabling "what changed?" audits between two
    /// arbitrary historical snapshots.
    pub fn cmd_compare_snapshots(
        db: &DbInstance,
        date1: &str,
        date2: &str,
        type_filter: Option<&str>,
        limit: usize,
    ) {
        let c1 = match resolve_date(date1) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Invalid date1 '{}': {}", date1, e);
                println!("null");
                return;
            }
        };
        let c2 = match resolve_date(date2) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Invalid date2 '{}': {}", date2, e);
                println!("null");
                return;
            }
        };
        // Helper closure: run the same sorted-dedupe query as cmd_as_of
        // and return the set of element_ids that were present at cutoff.
        let present_at = |cutoff: &str| -> std::collections::HashSet<String> {
            use std::collections::HashSet;
            let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
            params.insert("cutoff".into(), DataValue::Str(cutoff.into()));
            let mut filter_clauses = String::new();
            if let Some(t) = type_filter {
                filter_clauses.push_str(", element_type = $f_type");
                params.insert("f_type".into(), DataValue::Str(t.into()));
            }
            let q = format!(
                r#"?[element_id, observed_at, event_type] :=
                    *events{{element_id, observed_at, event_type, element_type}},
                    observed_at <= $cutoff{filters}
                :order element_id, -observed_at"#,
                filters = filter_clauses
            );
            let mut seen: HashSet<String> = HashSet::new();
            let mut present: HashSet<String> = HashSet::new();
            if let Ok(r) = db.run_script(&q, params, ScriptMutability::Immutable) {
                for row in r.rows.iter() {
                    let eid = if let DataValue::Str(s) = &row[0] {
                        s.to_string()
                    } else { continue };
                    if !seen.insert(eid.clone()) { continue; }
                    let etype = if let DataValue::Str(s) = &row[2] {
                        s.to_string()
                    } else { String::new() };
                    if etype != "removed" {
                        present.insert(eid);
                    }
                    if present.len() >= limit { break; }
                }
            }
            present
        };

        let s1 = present_at(&c1);
        let s2 = present_at(&c2);
        let only_1: Vec<String> = s1.difference(&s2).cloned().collect();
        let only_2: Vec<String> = s2.difference(&s1).cloned().collect();
        let common_count = s1.intersection(&s2).count();
        let mut only_1_sorted = only_1;
        only_1_sorted.sort();
        let mut only_2_sorted = only_2;
        only_2_sorted.sort();
        let out = json!({
            "date1": c1,
            "date2": c2,
            "type_filter": type_filter,
            "only_at_date1": only_1_sorted,
            "only_at_date2": only_2_sorted,
            "common_count": common_count,
            "added_between_count": only_2_sorted.len(),
            "removed_between_count": only_1_sorted.len(),
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    }

    /// `pss dedup-candidates [--min-count N] [--type T]` (Phase 3 Tier A F-8)
    ///
    /// Group currently-active elements by `(element_type, element_name)` and
    /// return groups whose distinct-scope count is ≥ `min_count`. Highlights
    /// accidental duplicates between user/project/plugin installations of
    /// the same name — exactly the "is this skill installed twice?" question
    /// the audit identified as missing.
    pub fn cmd_dedup_candidates(
        db: &DbInstance,
        min_count: usize,
        type_filter: Option<&str>,
        limit: usize,
    ) {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("min".into(), DataValue::Num(Num::Int(min_count as i64)));
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        let mut filter_clauses = String::new();
        if let Some(t) = type_filter {
            filter_clauses.push_str(", etype = $f_type");
            params.insert("f_type".into(), DataValue::Str(t.into()));
        }
        // Two-rule Datalog: count distinct element_ids per (type, name),
        // then filter for count >= min_count.
        let q = format!(
            r#"
            counts[etype, ename, count(eid)] :=
                *elements_state{{element_id: eid, last_event_id, exists: true}},
                *events{{event_id: last_event_id, element_type: etype, element_name: ename}}

            ?[etype, ename, c] :=
                counts[etype, ename, c],
                c >= $min{filters}
            :order -c, etype, ename
            :limit $limit
            "#,
            filters = filter_clauses
        );
        match db.run_script(&q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let out: Vec<JsonValue> = r.rows.iter().map(|row| {
                    json!({
                        "element_type": data_to_json(&row[0]),
                        "element_name": data_to_json(&row[1]),
                        "scope_count": data_to_json(&row[2]),
                    })
                }).collect();
                println!("{}", serde_json::to_string_pretty(&JsonValue::Array(out)).unwrap_or_default());
            }
            Err(e) => {
                eprintln!("dedup-candidates query failed: {}", e);
                println!("[]");
            }
        }
    }

    /// `pss removed-since <DATE>`
    pub fn cmd_removed_since(db: &DbInstance, date: &str, limit: usize) {
        let cutoff = match resolve_date(date) {
            Ok(s) => s,
            Err(e) => { print_date_err("date", date, &e); return; }
        };
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
                // COR-5 (audit 20260514): see cmd_lifespan rationale.
                r#"?[observed_at] := *events{observed_at} :order observed_at :limit 1"#,
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

    /// `pss reindex` — COR-8 (audit 20260514): the placeholder that
    /// referenced "Phase 4 deliverable" has been replaced with a real
    /// orchestrator wrapper. We execute `scripts/pss_reindex.py` (the
    /// canonical Python orchestrator that already wires discover →
    /// enrich → merge-events → temporal merge-events) and surface its
    /// exit code verbatim. The placeholder lived since v3.3.0 against
    /// the original TRDD §9.2 promise; surfacing the real path keeps
    /// the CLI surface honest and lets `pss reindex` work from scripts
    /// without needing the `/pss-reindex-skills` slash command.
    ///
    /// Resolution order for the orchestrator script:
    ///   1. $CLAUDE_PLUGIN_ROOT/scripts/pss_reindex.py — production
    ///      (when launched inside Claude Code).
    ///   2. Sibling `scripts/pss_reindex.py` relative to the Rust
    ///      binary's parent's parent — dev path so `cargo run` works
    ///      without env vars.
    pub fn cmd_reindex(_db: &DbInstance, dry_run: bool) {
        use std::path::PathBuf;

        let script = (|| -> Option<PathBuf> {
            if let Ok(plugin_root) = std::env::var("CLAUDE_PLUGIN_ROOT") {
                let p = PathBuf::from(plugin_root).join("scripts/pss_reindex.py");
                if p.is_file() {
                    return Some(p);
                }
            }
            if let Ok(exe) = std::env::current_exe() {
                // bin/pss-darwin-arm64 → up 1 = bin/ → up 1 = repo root.
                let candidate = exe
                    .parent()
                    .and_then(|p| p.parent())
                    .map(|root| root.join("scripts/pss_reindex.py"));
                if let Some(c) = candidate {
                    if c.is_file() {
                        return Some(c);
                    }
                }
                // Fallback: cargo target/release/pss → up 3 = repo root.
                let alt = exe
                    .parent()
                    .and_then(|p| p.parent())
                    .and_then(|p| p.parent())
                    .and_then(|p| p.parent())
                    .map(|root| root.join("scripts/pss_reindex.py"));
                if let Some(c) = alt {
                    if c.is_file() {
                        return Some(c);
                    }
                }
            }
            None
        })();

        let script_path = match script {
            Some(p) => p,
            None => {
                eprintln!(
                    "pss reindex: cannot locate scripts/pss_reindex.py. \
                     Set CLAUDE_PLUGIN_ROOT or run from the plugin's repo."
                );
                std::process::exit(2);
            }
        };

        if dry_run {
            eprintln!(
                "pss reindex --dry-run: would invoke {} (use without --dry-run to execute)",
                script_path.display()
            );
            return;
        }

        let status = std::process::Command::new("uv")
            .arg("run")
            .arg("--script")
            .arg(&script_path)
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!(
                    "pss reindex: orchestrator exited with status {} (script: {})",
                    s.code().unwrap_or(-1),
                    script_path.display()
                );
                std::process::exit(s.code().unwrap_or(1));
            }
            Err(e) => {
                eprintln!(
                    "pss reindex: failed to spawn `uv run --script {}`: {}",
                    script_path.display(),
                    e
                );
                std::process::exit(2);
            }
        }
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

    // ============================================================
    // merge-events: the only event writer in the normal reindex flow.
    // ============================================================

    /// Map a JSONL `type` field to an `ElementType`. Returns None for
    /// unknown values (the caller skips the row rather than guessing).
    fn parse_element_type(s: &str) -> Option<ElementType> {
        match s {
            "skill" => Some(ElementType::Skill),
            "agent" => Some(ElementType::Agent),
            "command" => Some(ElementType::Command),
            "rule" => Some(ElementType::Rule),
            "mcp" => Some(ElementType::Mcp),
            "lsp" => Some(ElementType::Lsp),
            "hook" => Some(ElementType::Hook),
            "plugin" => Some(ElementType::Plugin),
            "channel" => Some(ElementType::Channel),
            "monitor" => Some(ElementType::Monitor),
            "output-style" => Some(ElementType::OutputStyle),
            "theme" => Some(ElementType::Theme),
            "marketplace" => Some(ElementType::Marketplace),
            _ => None,
        }
    }

    /// Map a discovery `source` field to a canonical scope name.
    /// `"user"` → `"user"`, `"project"` (or `"project:<name>"`) → `"project"`,
    /// `"local"` → `"local"`, `"plugin:..."` → `"plugin"`,
    /// `"marketplace:..."` → `"marketplace"`. Returns the source verbatim
    /// for unknown forms so we never silently lose data.
    pub(crate) fn scope_from_discovery_source(source: &str) -> String {
        // DI-10 (audit 20260514): project-installed plugins are encoded
        // as `project:<name>/plugin:<plugin>` by the discoverer. The
        // previous logic matched the `project:` prefix first and
        // classified them as scope "project" — wrong. Plugins installed
        // into a project are still plugin-scope (overridden by the
        // project's own elements but still part of the plugin's
        // namespace). Check the composite form before the bare project
        // prefix.
        if source.starts_with("project:") && source.contains("/plugin:") {
            return "plugin".to_string();
        }
        if source == "user" || source.starts_with("user:") {
            "user".to_string()
        } else if source == "project" || source.starts_with("project:") {
            "project".to_string()
        } else if source == "local" || source.starts_with("local:") {
            "local".to_string()
        } else if source.starts_with("plugin:") {
            "plugin".to_string()
        } else if source.starts_with("marketplace:") {
            "marketplace".to_string()
        } else if source == "built-in" {
            "user".to_string() // LSP servers from the registry
        } else {
            source.to_string()
        }
    }

    /// For a discovery record, derive the scope_path that goes into the
    /// element_id. For project-scoped records that include a project
    /// name (`project:foo`) we use the project name as the slug; for
    /// plugin-scoped records (`plugin:<marketplace>/<name>`) we use that
    /// composite. Empty string for global scopes (user, marketplace).
    ///
    /// DI-10 (audit 20260514): for `project:<name>/plugin:<plugin>`
    /// composite (a plugin installed into a specific project) we keep
    /// the FULL composite (`<name>/plugin:<plugin>`) as the scope_path
    /// so two projects with the same plugin don't collide on
    /// element_id. The composite path is unique per (project, plugin).
    pub(crate) fn scope_path_from_discovery_source(source: &str) -> String {
        if let Some(rest) = source.strip_prefix("project:") {
            rest.to_string()
        } else if let Some(rest) = source.strip_prefix("local:") {
            rest.to_string()
        } else if let Some(rest) = source.strip_prefix("plugin:") {
            rest.to_string()
        } else if let Some(rest) = source.strip_prefix("user:") {
            rest.to_string()
        } else if let Some(rest) = source.strip_prefix("marketplace:") {
            rest.to_string()
        } else {
            "".to_string()
        }
    }

    /// Compute the canonical content blob for hashing an element. For
    /// file-based types we return the file bytes. For synthetic types
    /// (hook / plugin / marketplace / mcp / lsp / channel) we hash a
    /// canonical JSON of the discovery record so reorders don't fire
    /// spurious changes but real edits do.
    fn canonical_content(record: &serde_json::Value, path: &str) -> Vec<u8> {
        // Strip the JSON-pointer fragment for synthetic locators
        // (e.g. `/path/to/settings.json#hooks.X[0]`).
        let real_path = match path.find('#') {
            Some(idx) => &path[..idx],
            None => path,
        };
        // For file-based elements, hash the actual bytes — a description
        // tweak in the JSONL record shouldn't fire ContentChanged when
        // the file is untouched.
        let is_file_type = matches!(
            record.get("type").and_then(|v| v.as_str()),
            Some("skill") | Some("agent") | Some("command") | Some("rule")
                | Some("output-style") | Some("theme") | Some("monitor")
        );
        if is_file_type && !real_path.is_empty() {
            if let Ok(b) = std::fs::read(real_path) {
                return b;
            }
        }
        // Synthetic / unreadable: hash a canonical JSON of the record.
        // We strip volatile fields (description, use_context, preview)
        // first — those are derived metadata, not the element's
        // identity. Bumping them shouldn't fire ContentChanged.
        let mut clone = record.clone();
        if let Some(obj) = clone.as_object_mut() {
            for k in &["description", "use_context", "preview", "first_indexed_at",
                       "last_updated_at", "plugin_installed_at"] {
                obj.remove(*k);
            }
        }
        let canonical = canonical_json(&clone);
        canonical.into_bytes()
    }

    /// Render a JSON value as canonical (sorted-key) string. Recursive,
    /// stable across runs. Used for synthetic element hashing.
    fn canonical_json(v: &serde_json::Value) -> String {
        match v {
            serde_json::Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let parts: Vec<String> = keys
                    .into_iter()
                    .map(|k| format!("{}:{}", k, canonical_json(&map[k])))
                    .collect();
                format!("{{{}}}", parts.join(","))
            }
            serde_json::Value::Array(arr) => {
                let parts: Vec<String> = arr.iter().map(canonical_json).collect();
                format!("[{}]", parts.join(","))
            }
            other => other.to_string(),
        }
    }

    /// Read the full PriorState row for a given element_id from
    /// elements_state, or None if absent.
    fn read_prior(db: &DbInstance, element_id: &str) -> Option<PriorState> {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(element_id.into()));
        let q = r#"
            ?[current_path, current_hash, current_size, enabled, override_status, exists] :=
                *elements_state{element_id: $eid, current_path, current_hash,
                                current_size, enabled, override_status, exists}
        "#;
        let rows = db
            .run_script(q, params, ScriptMutability::Immutable)
            .ok()?
            .rows;
        let r = rows.first()?;
        Some(PriorState {
            element_id: element_id.to_string(),
            current_path: data_str(&r[0]),
            current_hash: data_str(&r[1]),
            current_size: r[2].get_int().unwrap_or(-1),
            enabled: matches!(&r[3], DataValue::Bool(true)),
            override_status: data_str(&r[4]),
            exists: !matches!(&r[5], DataValue::Bool(false)),
        })
    }

    /// DI-1 wave 1 (audit 20260514): read the prior description hash
    /// from `element_descriptions`. Returns None when the element has
    /// never been observed before (first install — no prior to compare
    /// against). The side-table approach avoids an elements_state
    /// schema migration.
    fn read_prior_description_hash(db: &DbInstance, element_id: &str) -> Option<String> {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(element_id.into()));
        let q = r#"
            ?[description_hash] :=
                *element_descriptions{element_id: $eid, description_hash}
        "#;
        let rows = db
            .run_script(q, params, ScriptMutability::Immutable)
            .ok()?
            .rows;
        let r = rows.first()?;
        match &r[0] {
            DataValue::Str(s) => Some(s.to_string()),
            _ => None,
        }
    }

    /// DI-1 wave 1: upsert the (element_id, description_hash,
    /// description_text, last_updated_at) row. Called once per
    /// observation in merge-events. Idempotent — same hash → identical
    /// row → no-op semantically.
    fn upsert_element_description(
        db: &DbInstance,
        element_id: &str,
        description: &str,
        description_hash: &str,
        observed_at: &str,
    ) -> Result<(), String> {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(element_id.into()));
        params.insert("hash".into(), DataValue::Str(description_hash.into()));
        // Truncate to 200 chars to bound storage; the full description
        // lives in skills.description anyway.
        let truncated: String = description.chars().take(200).collect();
        params.insert("text".into(), DataValue::Str(truncated.into()));
        params.insert("ts".into(), DataValue::Str(observed_at.into()));
        let q = r#"
            ?[element_id, description_hash, description_text, last_updated_at] <-
                [[$eid, $hash, $text, $ts]]
            :put element_descriptions {
                element_id => description_hash, description_text, last_updated_at
            }
        "#;
        db.run_script(q, params, ScriptMutability::Mutable)
            .map(|_| ())
            .map_err(|e| format!("upsert_element_description failed: {}", e))
    }

    /// Insert one event row + upsert the corresponding elements_state row.
    /// The two writes happen sequentially under the caller's lock; the
    /// merge-events orchestrator never partially commits because the
    /// transaction is bounded by the binary's lifetime.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn persist_event_and_state(
        db: &DbInstance,
        scan_id: &str,
        observed_at: &str,
        event_type: EventType,
        obs: &Observation,
        diff_json: &str,
        update_state: bool,
        // DI-2 (audit 20260514): the writer previously hard-coded
        // `override_status: "active"` for every event, so multi-scope
        // override resolution (resolve_overrides() — defined since v3.4.0
        // but never invoked) could never see real status. Adding an
        // explicit parameter (with a default of "active" supplied by
        // every existing caller) keeps the prior behavior intact while
        // letting the new override-resolution pass below the main loop
        // emit OverrideStarted/OverrideEnded events with the right
        // values.
        override_status: &str,
    ) -> Result<(), String> {
        use cozo::Num;
        let event_id = ulid::Ulid::new().to_string();
        let element_id = obs.element_id();
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("event_id".into(), DataValue::Str(event_id.clone().into()));
        params.insert("observed_at".into(), DataValue::Str(observed_at.into()));
        params.insert("scan_id".into(), DataValue::Str(scan_id.into()));
        params.insert(
            "event_type".into(),
            DataValue::Str(event_type.as_str().into()),
        );
        params.insert(
            "element_type".into(),
            DataValue::Str(obs.element_type.as_str().into()),
        );
        params.insert("element_name".into(), DataValue::Str(obs.name.clone().into()));
        params.insert("element_id".into(), DataValue::Str(element_id.clone().into()));
        params.insert("scope".into(), DataValue::Str(obs.scope.clone().into()));
        params.insert("scope_path".into(), DataValue::Str(obs.scope_path.clone().into()));
        params.insert("source".into(), DataValue::Str(obs.source.clone().into()));
        params.insert("path".into(), DataValue::Str(obs.path.clone().into()));
        params.insert(
            "content_hash".into(),
            DataValue::Str(obs.content_hash.clone().into()),
        );
        params.insert("file_size".into(), DataValue::Num(Num::Int(obs.file_size)));
        params.insert(
            "token_count".into(),
            DataValue::Num(Num::Int(obs.token_count)),
        );
        params.insert("enabled".into(), DataValue::Bool(obs.enabled));
        params.insert("override_status".into(), DataValue::Str(override_status.into()));
        params.insert("diff_json".into(), DataValue::Str(diff_json.into()));
        params.insert("snapshot_ref".into(), DataValue::Str("".into()));

        let event_q = r#"?[event_id, observed_at, scan_id, event_type, element_type, element_name, element_id, scope, scope_path, source, path, content_hash, file_size, token_count, enabled, override_status, diff_json, snapshot_ref] <-
            [[$event_id, $observed_at, $scan_id, $event_type, $element_type, $element_name, $element_id, $scope, $scope_path, $source, $path, $content_hash, $file_size, $token_count, $enabled, $override_status, $diff_json, $snapshot_ref]]
           :put events { event_id => observed_at, scan_id, event_type, element_type, element_name, element_id, scope, scope_path, source, path, content_hash, file_size, token_count, enabled, override_status, diff_json, snapshot_ref }"#;
        db.run_script(event_q, params, ScriptMutability::Mutable)
            .map_err(|e| format!("event insert failed: {}", e))?;

        if !update_state {
            return Ok(());
        }
        // Upsert elements_state. installed_at: keep prior if any (we read
        // it back — if the event_type is `removed` or the prior row
        // existed we preserve installed_at; if this is a new install, we
        // use observed_at).
        let prior_installed_at = read_installed_at(db, &element_id)
            .unwrap_or_else(|| observed_at.to_string());
        let exists = !matches!(event_type, EventType::Removed);
        let mut state_params: BTreeMap<String, DataValue> = BTreeMap::new();
        state_params.insert("element_id".into(), DataValue::Str(element_id.clone().into()));
        state_params.insert("last_event_id".into(), DataValue::Str(event_id.clone().into()));
        state_params.insert("current_path".into(), DataValue::Str(obs.path.clone().into()));
        state_params.insert(
            "current_hash".into(),
            DataValue::Str(obs.content_hash.clone().into()),
        );
        state_params.insert("current_size".into(), DataValue::Num(Num::Int(obs.file_size)));
        state_params.insert(
            "current_token_count".into(),
            DataValue::Num(Num::Int(obs.token_count)),
        );
        state_params.insert("enabled".into(), DataValue::Bool(obs.enabled));
        state_params.insert("override_status".into(), DataValue::Str(override_status.into()));
        state_params.insert("installed_at".into(), DataValue::Str(prior_installed_at.into()));
        state_params.insert("last_changed_at".into(), DataValue::Str(observed_at.into()));
        state_params.insert("exists".into(), DataValue::Bool(exists));

        let state_q = r#"?[element_id, last_event_id, current_path, current_hash, current_size, current_token_count, enabled, override_status, installed_at, last_changed_at, exists] <-
            [[$element_id, $last_event_id, $current_path, $current_hash, $current_size, $current_token_count, $enabled, $override_status, $installed_at, $last_changed_at, $exists]]
           :put elements_state { element_id => last_event_id, current_path, current_hash, current_size, current_token_count, enabled, override_status, installed_at, last_changed_at, exists }"#;
        db.run_script(state_q, state_params, ScriptMutability::Mutable)
            .map_err(|e| format!("state upsert failed: {}", e))?;
        Ok(())
    }

    /// Read the installed_at timestamp for an element if present.
    /// Used to preserve install anchor across content_changed/size_changed
    /// events without losing it on every upsert.
    fn read_installed_at(db: &DbInstance, element_id: &str) -> Option<String> {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(element_id.into()));
        let q = r#"?[installed_at] := *elements_state{element_id: $eid, installed_at}"#;
        db.run_script(q, params, ScriptMutability::Immutable)
            .ok()
            .and_then(|r| r.rows.first().cloned())
            .and_then(|row| match row.first() {
                Some(DataValue::Str(s)) => Some(s.to_string()),
                _ => None,
            })
    }

    /// `pss merge-events` — Phase-2 wiring. Reads JSONL observations from
    /// stdin and emits events. This is the ONLY writer of `events` and
    /// `elements_state` during normal reindex flow.
    pub fn cmd_merge_events(db: &DbInstance, quiet: bool) -> Result<(), String> {
        use std::io::BufReader;
        let stdin = std::io::stdin();
        let reader = BufReader::new(stdin.lock());
        merge_events_from_reader(db, reader, quiet)
    }

    /// Reader-driven variant of [`cmd_merge_events`] so tests can pipe a
    /// `Cursor<&str>` without touching stdin. Production code always goes
    /// through `cmd_merge_events`, which builds the reader from
    /// `std::io::stdin()` and delegates here.
    pub fn merge_events_from_reader<R: std::io::BufRead>(
        db: &DbInstance,
        reader: R,
        quiet: bool,
    ) -> Result<(), String> {
        use std::collections::{HashMap, HashSet};

        ensure_schema(db)?;

        let scan_id = ulid::Ulid::new().to_string();
        let started_at = chrono::Utc::now().to_rfc3339();
        let mut observed_eids: HashSet<String> = HashSet::new();
        let mut visited_scope_paths: HashSet<String> = HashSet::new();
        let mut events_emitted: u64 = 0;
        let mut lines_read: u64 = 0;
        // Group observations per (element_type, name) so override
        // resolution sees all candidates before deciding the active row.
        let mut by_type_and_name: HashMap<(String, String), Vec<Observation>> = HashMap::new();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => return Err(format!("read line failed: {}", e)),
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            lines_read += 1;
            let value: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue, // tolerate malformed lines
            };
            // DI-4 (audit 20260514): leading manifest line lists every
            // scope_path the discoverer walked, even those with zero
            // observations. Without this, a plugin uninstall that left
            // zero observations for its scope_path was never detected
            // as a removal (visited_scope_paths populated only from
            // successful observations missed the dead plugin's path).
            if value.get("_pss_manifest").and_then(|v| v.as_bool()).unwrap_or(false) {
                if let Some(arr) = value.get("visited_scope_paths").and_then(|v| v.as_array()) {
                    for sp in arr {
                        if let Some(s) = sp.as_str() {
                            visited_scope_paths.insert(s.to_string());
                        }
                    }
                }
                continue; // manifest line consumed — don't try to parse as observation
            }
            let etype_str = value
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let element_type = match parse_element_type(etype_str) {
                Some(t) => t,
                None => continue, // unknown type — skip
            };
            let name = value
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if name.is_empty() {
                continue;
            }
            let source = value
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("user")
                .to_string();
            let path = value
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let description = value
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let scope = scope_from_discovery_source(&source);
            let scope_path = scope_path_from_discovery_source(&source);

            // Compute content metrics. canonical_content() handles
            // file-vs-synthetic logic. token_count is best-effort: only
            // computed when we have UTF-8 text bytes.
            let bytes = canonical_content(&value, &path);
            let file_size = bytes.len() as i64;
            let content_hash_str = content_hash(&bytes);
            let token_count = match std::str::from_utf8(&bytes) {
                Ok(s) => token_count_cl100k(s),
                Err(_) => -1,
            };

            // DI-3 (audit 20260514): the previous writer hard-coded
            // `enabled: true`, so the temporal index could never detect a
            // plugin/mcp/lsp being toggled off in settings.json. The
            // discoverer now emits `enabled` per element (defaults to true
            // for file-based types whose state is implicit). Reading it
            // from the JSONL line is the writer-side half of DI-3 and
            // wires up Enabled/Disabled event emission.
            let enabled = value
                .get("enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            let obs = Observation {
                element_type,
                name,
                scope,
                scope_path: scope_path.clone(),
                source,
                path,
                content_hash: content_hash_str,
                file_size,
                token_count,
                description,
                enabled,
            };
            observed_eids.insert(obs.element_id());
            visited_scope_paths.insert(scope_path);
            let key = (
                element_type.as_str().to_string(),
                obs.name.clone(),
            );
            by_type_and_name.entry(key).or_default().push(obs);
        }

        // Emit events for every observation.
        for (_, observations) in &by_type_and_name {
            for obs in observations {
                let element_id = obs.element_id();
                let prior = read_prior(db, &element_id);
                let evts = compare_and_emit(prior.as_ref(), obs);
                for evt in evts {
                    let diff = serde_json::json!({
                        "description": obs.description,
                        "event": evt.as_str(),
                    })
                    .to_string();
                    // DI-3 (audit 20260514): pass obs.enabled (not hard-true)
                    // so the events row reflects the actual enabled state
                    // observed in this scan. Without this, even after the
                    // Enabled/Disabled events fire, the events row still
                    // says enabled=true.
                    persist_event_and_state(
                        db, &scan_id, &started_at, evt, obs, &diff, obs.enabled,
                        // Per-observation events default to "active" — the
                        // override-resolution pass below this loop emits
                        // separate Override* events with the resolved
                        // status when scopes actually conflict.
                        "active",
                    )?;
                    events_emitted += 1;
                }

                // DI-1 wave 1 (audit 20260514): description change
                // detection. Read the prior description_hash from the
                // side table, compute the new one, and emit a
                // DescriptionChanged event when they differ AND the
                // element wasn't just installed (an Installed event
                // already implies the description is "new"). Skip the
                // comparison when there's no prior row at all (first
                // observation — Installed event fired above).
                let new_desc_hash = description_hash(&obs.description);
                let prior_desc_hash = read_prior_description_hash(db, &element_id);
                let element_existed = prior.as_ref().map(|p| p.exists).unwrap_or(false);
                if let Some(prior_hash) = &prior_desc_hash {
                    if element_existed && prior_hash != &new_desc_hash {
                        let diff = serde_json::json!({
                            "previous_description_hash": prior_hash,
                            "new_description_hash": new_desc_hash,
                            "new_description": obs.description.chars().take(200).collect::<String>(),
                            "event": EventType::DescriptionChanged.as_str(),
                        })
                        .to_string();
                        persist_event_and_state(
                            db,
                            &scan_id,
                            &started_at,
                            EventType::DescriptionChanged,
                            obs,
                            &diff,
                            obs.enabled,
                            "active",
                        )?;
                        events_emitted += 1;
                    }
                }
                // Always upsert so subsequent observations have a prior
                // to compare against. The upsert is idempotent — same
                // hash → no behavioural change.
                upsert_element_description(
                    db,
                    &element_id,
                    &obs.description,
                    &new_desc_hash,
                    &started_at,
                )?;
                // If no events fired (unchanged), still refresh
                // last_changed_at? No — the spec says events table is
                // append-only and elements_state is materialized FROM
                // events. Skipping is correct.
            }
        }

        // DI-2 (audit 20260514): override resolution pass. For each
        // (element_type, name) group with multiple scope candidates, run
        // resolve_overrides() to determine the effective active row, then
        // compare against each candidate's prior override_status. If the
        // status differs, emit an OverrideStarted (becoming overridden) or
        // OverrideEnded (no longer overridden) event. This finally
        // surfaces the resolver that has been defined and unit-tested but
        // never wired into the writer (audit §4.B DI-2).
        for ((etype_str, _name), observations) in by_type_and_name.iter() {
            // Single-scope groups don't have override decisions — skip.
            if observations.len() <= 1 {
                continue;
            }
            let element_type = match parse_element_type(etype_str) {
                Some(t) => t,
                None => continue,
            };
            if !element_type.has_override_precedence() {
                continue; // hooks / mcp / lsp don't override by file precedence
            }
            let candidates: Vec<(String, String)> = observations
                .iter()
                .map(|o| (o.element_id(), o.scope.clone()))
                .collect();
            let resolved = resolve_overrides(element_type, &candidates);
            for (eid, new_status) in &resolved {
                // What did elements_state say BEFORE this scan?
                let prior_status = read_prior(db, eid)
                    .map(|p| p.override_status)
                    .unwrap_or_else(|| "active".to_string());
                if new_status == &prior_status {
                    continue; // unchanged — no event needed
                }
                // Find the matching observation so we can persist with the
                // right scope/path/hash metadata.
                let obs = match observations.iter().find(|o| o.element_id() == *eid) {
                    Some(o) => o,
                    None => continue,
                };
                // Decide event direction: was-active → now-not = Override
                // started; was-not-active → now-active = Override ended.
                let evt = if prior_status == "active" {
                    EventType::OverrideStarted
                } else if new_status == "active" {
                    EventType::OverrideEnded
                } else {
                    // Both non-active but different (e.g. overridden_by
                    // pointer changed). Treat as Started for clarity.
                    EventType::OverrideStarted
                };
                let diff = serde_json::json!({
                    "previous_override_status": prior_status,
                    "new_override_status": new_status,
                    "event": evt.as_str(),
                })
                .to_string();
                persist_event_and_state(
                    db,
                    &scan_id,
                    &started_at,
                    evt,
                    obs,
                    &diff,
                    obs.enabled,
                    new_status,
                )?;
                events_emitted += 1;
            }
        }

        // Detect removals: anything in elements_state with exists=true
        // whose scope_path is in visited_scope_paths but whose
        // element_id was NOT observed this scan.
        let prior_active = read_active_in_scope_paths(db, &visited_scope_paths)?;
        let removed = detect_removals(&prior_active, &observed_eids);
        for eid in &removed {
            // Read the prior obs metadata to attach to the removal event.
            let prior_meta = read_prior_meta_for_removal(db, eid);
            if let Some(meta) = prior_meta {
                let diff = serde_json::json!({
                    "event": "removed",
                    "previous_path": meta.path,
                })
                .to_string();
                persist_event_and_state(
                    db,
                    &scan_id,
                    &started_at,
                    EventType::Removed,
                    &meta,
                    &diff,
                    true,
                    // Removal events: override_status no longer relevant.
                    "active",
                )?;
                events_emitted += 1;
            }
        }

        // Record the scan_runs row.
        let finished_at = chrono::Utc::now().to_rfc3339();
        let scope_paths_json: String = serde_json::to_string(
            &visited_scope_paths.iter().cloned().collect::<Vec<_>>(),
        )
        .unwrap_or_else(|_| "[]".to_string());
        use cozo::Num;
        let mut scan_params: BTreeMap<String, DataValue> = BTreeMap::new();
        scan_params.insert("scan_id".into(), DataValue::Str(scan_id.clone().into()));
        scan_params.insert("started_at".into(), DataValue::Str(started_at.into()));
        scan_params.insert("finished_at".into(), DataValue::Str(finished_at.into()));
        scan_params.insert(
            "scope_paths_json".into(),
            DataValue::Str(scope_paths_json.into()),
        );
        scan_params.insert(
            "events_emitted".into(),
            DataValue::Num(Num::Int(events_emitted as i64)),
        );
        scan_params.insert(
            "rust_binary_version".into(),
            DataValue::Str(env!("CARGO_PKG_VERSION").into()),
        );
        scan_params.insert(
            "pss_version".into(),
            DataValue::Str(env!("CARGO_PKG_VERSION").into()),
        );
        let scan_q = r#"?[scan_id, started_at, finished_at, scope_paths_json, events_emitted, rust_binary_version, pss_version] <-
            [[$scan_id, $started_at, $finished_at, $scope_paths_json, $events_emitted, $rust_binary_version, $pss_version]]
           :put scan_runs { scan_id => started_at, finished_at, scope_paths_json, events_emitted, rust_binary_version, pss_version }"#;
        db.run_script(scan_q, scan_params, ScriptMutability::Mutable)
            .map_err(|e| format!("scan_runs insert failed: {}", e))?;

        if !quiet {
            eprintln!(
                "[merge-events] scan_id={} lines={} events={} removed={} scopes={}",
                scan_id,
                lines_read,
                events_emitted,
                removed.len(),
                visited_scope_paths.len(),
            );
        }
        Ok(())
    }

    /// Read element_ids that are currently `exists=true` in the given
    /// set of scope_paths. Used by removal detection.
    fn read_active_in_scope_paths(
        db: &DbInstance,
        scope_paths: &std::collections::HashSet<String>,
    ) -> Result<std::collections::HashSet<String>, String> {
        // We can't parameterise an IN clause in Cozo as cleanly as SQL;
        // walk the table once and filter in Rust. For ≤100k elements
        // this is fine.
        let q = r#"
            ?[element_id, scope_path] :=
                *elements_state{element_id, scope_path, exists: true}
        "#;
        // Note: scope_path lives in the events table, not elements_state.
        // elements_state has installed_at + last_changed_at + the latest
        // hash/size, but no scope_path. We resolve scope_path from the
        // most recent event for each element_id.
        let _ = q; // silence unused warning if we change strategy below.

        let q = r#"
            ?[element_id] := *elements_state{element_id, exists: true}
        "#;
        let rows = db
            .run_script(q, BTreeMap::new(), ScriptMutability::Immutable)
            .map_err(|e| format!("active query failed: {}", e))?
            .rows;
        let mut out: std::collections::HashSet<String> = std::collections::HashSet::new();
        for row in rows {
            let eid = match row.first() {
                Some(DataValue::Str(s)) => s.to_string(),
                _ => continue,
            };
            // Resolve scope_path from the most recent event for this eid.
            // Use :order -observed_at :limit 1 instead of max() so we
            // handle the empty-result case (no events for this eid) without
            // raising "Evaluation of expression failed".
            let mut p: BTreeMap<String, DataValue> = BTreeMap::new();
            p.insert("eid".into(), DataValue::Str(eid.clone().into()));
            // Cozo requires sort keys to appear in the output projection.
            // observed_at is in the head solely for :order — row[1] is
            // ignored downstream.
            let scope_q = r#"
                ?[scope_path, observed_at] :=
                    *events{element_id: $eid, scope_path, observed_at}
                :order -observed_at
                :limit 1
            "#;
            if let Ok(r) = db.run_script(scope_q, p, ScriptMutability::Immutable) {
                if let Some(row) = r.rows.first() {
                    let sp = data_str(&row[0]);
                    if scope_paths.contains(&sp) {
                        out.insert(eid);
                    }
                }
            }
        }
        Ok(out)
    }

    /// Read enough metadata about a previously-existing element to emit
    /// a `removed` event for it. Returns None if no prior event row.
    /// Uses :order/:limit instead of max() so empty-result cases don't
    /// raise "Evaluation of expression failed".
    fn read_prior_meta_for_removal(db: &DbInstance, element_id: &str) -> Option<Observation> {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(element_id.into()));
        // Cozo requires sort keys to appear in the output projection.
        // observed_at is included in the head only for :order; downstream
        // ignores it.
        let q = r#"
            ?[element_type, element_name, scope, scope_path, source, path,
              content_hash, file_size, token_count, observed_at] :=
                *events{element_id: $eid, observed_at, element_type, element_name,
                        scope, scope_path, source, path, content_hash,
                        file_size, token_count}
            :order -observed_at
            :limit 1
        "#;
        let rows = db.run_script(q, params, ScriptMutability::Immutable).ok()?.rows;
        let r = rows.first()?;
        let element_type_str = data_str(&r[0]);
        let element_type = parse_element_type(&element_type_str)?;
        Some(Observation {
            element_type,
            name: data_str(&r[1]),
            scope: data_str(&r[2]),
            scope_path: data_str(&r[3]),
            source: data_str(&r[4]),
            path: data_str(&r[5]),
            content_hash: data_str(&r[6]),
            file_size: r[7].get_int().unwrap_or(-1),
            token_count: r[8].get_int().unwrap_or(-1),
            description: "".to_string(),
            enabled: true,
        })
    }

    // ============================================================
    // Secondary temporal queries (TRDD §9.1).
    // All return JSON. All read CozoDB directly — zero LLM calls.
    // ============================================================

    /// Helper: read the last event for `element_id` whose
    /// `observed_at <= cutoff`. Returns None if no such event.
    /// Uses `:order -observed_at :limit 1` instead of an aggregate so
    /// it works on element_ids that have no events at all (the aggregate
    /// form raises "Evaluation of expression failed" on empty result).
    fn read_event_at_or_before(
        db: &DbInstance,
        element_id: &str,
        cutoff: &str,
    ) -> Option<BTreeMap<String, JsonValue>> {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(element_id.into()));
        params.insert("cutoff".into(), DataValue::Str(cutoff.into()));
        let q = r#"
            ?[event_type, path, content_hash, file_size, token_count, enabled,
              override_status, observed_at] :=
                *events{element_id: $eid, observed_at, event_type, path,
                        content_hash, file_size, token_count, enabled,
                        override_status},
                observed_at <= $cutoff
            :order -observed_at
            :limit 1
        "#;
        let rows = db.run_script(q, params, ScriptMutability::Immutable).ok()?.rows;
        let r = rows.first()?;
        let mut out = BTreeMap::new();
        out.insert("event_type".to_string(), data_to_json(&r[0]));
        out.insert("path".to_string(), data_to_json(&r[1]));
        out.insert("content_hash".to_string(), data_to_json(&r[2]));
        out.insert("file_size".to_string(), data_to_json(&r[3]));
        out.insert("token_count".to_string(), data_to_json(&r[4]));
        out.insert("enabled".to_string(), data_to_json(&r[5]));
        out.insert("override_status".to_string(), data_to_json(&r[6]));
        out.insert("observed_at".to_string(), data_to_json(&r[7]));
        Some(out)
    }

    /// `pss show <ELEMENT_ID> --as-of <DATE>` — full snapshot at a date.
    pub fn cmd_show_at(db: &DbInstance, element_id: &str, date: &str) {
        let cutoff = match resolve_date(date) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Invalid date '{}': {}", date, e);
                println!("null");
                return;
            }
        };
        match read_event_at_or_before(db, element_id, &cutoff) {
            Some(snap) => {
                // If the most recent event was a removal, mark exists=false.
                let is_removed = matches!(
                    snap.get("event_type"),
                    Some(JsonValue::String(s)) if s == "removed"
                );
                let mut json: serde_json::Map<String, JsonValue> = serde_json::Map::new();
                json.insert("element_id".into(), JsonValue::String(element_id.into()));
                json.insert("as_of".into(), JsonValue::String(cutoff));
                json.insert("exists".into(), JsonValue::Bool(!is_removed));
                for (k, v) in snap {
                    json.insert(k, v);
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonValue::Object(json)).unwrap_or_default()
                );
            }
            None => {
                println!(
                    "{}",
                    json!({
                        "element_id": element_id,
                        "as_of": cutoff,
                        "exists": false,
                        "note": "no event found at or before this date",
                    })
                );
            }
        }
    }

    /// `pss size-at <ELEMENT_ID> --as-of <DATE>` — file_size at a date.
    pub fn cmd_size_at(db: &DbInstance, element_id: &str, date: &str) {
        let cutoff = match resolve_date(date) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Invalid date '{}': {}", date, e);
                println!("null");
                return;
            }
        };
        let snap = read_event_at_or_before(db, element_id, &cutoff);
        let size = snap
            .as_ref()
            .and_then(|m| m.get("file_size"))
            .cloned()
            .unwrap_or(JsonValue::Null);
        let observed_at = snap
            .as_ref()
            .and_then(|m| m.get("observed_at"))
            .cloned()
            .unwrap_or(JsonValue::Null);
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "element_id": element_id,
                "as_of": cutoff,
                "file_size": size,
                "observed_at": observed_at,
            }))
            .unwrap_or_default()
        );
    }

    /// `pss tokens-at <ELEMENT_ID> --as-of <DATE>` — token_count at a date.
    pub fn cmd_tokens_at(db: &DbInstance, element_id: &str, date: &str) {
        let cutoff = match resolve_date(date) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Invalid date '{}': {}", date, e);
                println!("null");
                return;
            }
        };
        let snap = read_event_at_or_before(db, element_id, &cutoff);
        let tokens = snap
            .as_ref()
            .and_then(|m| m.get("token_count"))
            .cloned()
            .unwrap_or(JsonValue::Null);
        let observed_at = snap
            .as_ref()
            .and_then(|m| m.get("observed_at"))
            .cloned()
            .unwrap_or(JsonValue::Null);
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "element_id": element_id,
                "as_of": cutoff,
                "token_count": tokens,
                "observed_at": observed_at,
            }))
            .unwrap_or_default()
        );
    }

    /// `pss diff <ELEMENT_ID> <DATE1> <DATE2>` — show the deltas between
    /// two snapshots of an element.
    pub fn cmd_diff(db: &DbInstance, element_id: &str, date1: &str, date2: &str) {
        let c1 = match resolve_date(date1) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Invalid date1 '{}': {}", date1, e);
                println!("null");
                return;
            }
        };
        let c2 = match resolve_date(date2) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Invalid date2 '{}': {}", date2, e);
                println!("null");
                return;
            }
        };
        let s1 = read_event_at_or_before(db, element_id, &c1);
        let s2 = read_event_at_or_before(db, element_id, &c2);
        let mut deltas = serde_json::Map::new();
        if let (Some(a), Some(b)) = (&s1, &s2) {
            for k in &["path", "content_hash", "file_size", "token_count", "enabled", "override_status"] {
                let av = a.get(*k);
                let bv = b.get(*k);
                if av != bv {
                    deltas.insert(
                        k.to_string(),
                        json!({"before": av, "after": bv}),
                    );
                }
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "element_id": element_id,
                "date1": c1,
                "date2": c2,
                "snapshot1": s1.as_ref().map(|m| {
                    let v: serde_json::Map<String, JsonValue> = m.iter()
                        .map(|(k, v)| (k.clone(), v.clone())).collect();
                    JsonValue::Object(v)
                }).unwrap_or(JsonValue::Null),
                "snapshot2": s2.as_ref().map(|m| {
                    let v: serde_json::Map<String, JsonValue> = m.iter()
                        .map(|(k, v)| (k.clone(), v.clone())).collect();
                    JsonValue::Object(v)
                }).unwrap_or(JsonValue::Null),
                "deltas": deltas,
            }))
            .unwrap_or_default()
        );
    }

    /// `pss installed-between <START> <END> [--type T]` — every install
    /// event in a time window.
    pub fn cmd_installed_between(
        db: &DbInstance,
        start: &str,
        end: &str,
        type_filter: Option<&str>,
        limit: usize,
    ) {
        emit_event_window(db, start, end, "installed", type_filter, limit);
    }

    /// `pss removed-between <START> <END> [--type T]` — every removal
    /// event in a time window.
    pub fn cmd_removed_between(
        db: &DbInstance,
        start: &str,
        end: &str,
        type_filter: Option<&str>,
        limit: usize,
    ) {
        emit_event_window(db, start, end, "removed", type_filter, limit);
    }

    /// Internal: filter events table by event_type + window + element_type.
    fn emit_event_window(
        db: &DbInstance,
        start: &str,
        end: &str,
        event_type: &str,
        type_filter: Option<&str>,
        limit: usize,
    ) {
        let s = match resolve_date(start) {
            Ok(s) => s,
            Err(e) => { print_date_err("start date", start, &e); return; }
        };
        let e = match resolve_date(end) {
            Ok(s) => s,
            Err(err) => { print_date_err("end date", end, &err); return; }
        };
        let q = r#"
            ?[observed_at, element_type, element_name, element_id, scope, scope_path,
              path, content_hash, file_size, token_count] :=
                *events{observed_at, event_type, element_type, element_name,
                        element_id, scope, scope_path, path, content_hash,
                        file_size, token_count},
                event_type == $etype,
                observed_at >= $start, observed_at <= $end
            :order observed_at
            :limit $limit
        "#;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("etype".into(), DataValue::Str(event_type.into()));
        params.insert("start".into(), DataValue::Str(s.into()));
        params.insert("end".into(), DataValue::Str(e.into()));
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        match db.run_script(q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let rows: Vec<JsonValue> = r
                    .rows
                    .into_iter()
                    .filter(|row| {
                        type_filter.is_none_or(|t| {
                            matches!(&row[1], DataValue::Str(s) if s.as_str() == t)
                        })
                    })
                    .map(|row| {
                        json!({
                            "observed_at": data_to_json(&row[0]),
                            "element_type": data_to_json(&row[1]),
                            "element_name": data_to_json(&row[2]),
                            "element_id": data_to_json(&row[3]),
                            "scope": data_to_json(&row[4]),
                            "scope_path": data_to_json(&row[5]),
                            "path": data_to_json(&row[6]),
                            "content_hash": data_to_json(&row[7]),
                            "file_size": data_to_json(&row[8]),
                            "token_count": data_to_json(&row[9]),
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                );
            }
            Err(err) => eprintln!("event-window query failed: {}", err),
        }
    }

    /// `pss currently-missing-but-once-was [--type T]` — element_ids that
    /// have at least one event but whose elements_state row is
    /// `exists=false` (or absent). Synonym: `never-current`.
    ///
    /// Strategy: walk elements_state for `exists=false` rows, then
    /// resolve element_type and element_name from the latest event.
    pub fn cmd_currently_missing(db: &DbInstance, type_filter: Option<&str>, limit: usize) {
        // Pull element_ids whose elements_state.exists=false. These are
        // exactly the "once was, currently missing" set.
        let q = r#"
            ?[element_id, last_changed_at] :=
                *elements_state{element_id, exists: false, last_changed_at}
            :order -last_changed_at
            :limit $limit
        "#;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        let rows = match db.run_script(q, params, ScriptMutability::Immutable) {
            Ok(r) => r.rows,
            Err(e) => {
                eprintln!("currently-missing query failed: {}", e);
                println!("[]");
                return;
            }
        };
        let mut out: Vec<JsonValue> = Vec::new();
        for row in rows {
            let eid = data_str(&row[0]);
            let last_seen = data_to_json(&row[1]);
            // Resolve element_type and element_name from the latest event.
            let mut p: BTreeMap<String, DataValue> = BTreeMap::new();
            p.insert("eid".into(), DataValue::Str(eid.clone().into()));
            // Cozo requires sort keys to appear in the output projection.
            // observed_at appears in the head solely for :order ordering.
            let detail_q = r#"
                ?[element_type, element_name, observed_at] :=
                    *events{element_id: $eid, element_type, element_name, observed_at}
                :order -observed_at
                :limit 1
            "#;
            let (etype, ename) = db
                .run_script(detail_q, p, ScriptMutability::Immutable)
                .ok()
                .and_then(|r| r.rows.first().cloned())
                .map(|row| (data_str(&row[0]), data_to_json(&row[1])))
                .unwrap_or_else(|| ("".into(), JsonValue::Null));
            if let Some(t) = type_filter {
                if etype != t {
                    continue;
                }
            }
            out.push(json!({
                "element_id": eid,
                "element_type": etype,
                "element_name": ename,
                "last_seen_at": last_seen,
            }));
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&JsonValue::Array(out)).unwrap_or_default()
        );
    }

    /// `pss multi-scope <NAME> [--type T]` — find an element name that
    /// exists at multiple scopes simultaneously.
    pub fn cmd_multi_scope(
        db: &DbInstance,
        name: &str,
        type_filter: Option<&str>,
    ) {
        // Pull every element_id whose name matches AND has elements_state
        // exists=true. Group by (element_type, name) and emit groups
        // with size > 1.
        let q = r#"
            ?[element_id, element_type, scope, scope_path] :=
                *events{element_id, element_type, element_name, scope, scope_path,
                        observed_at},
                *elements_state{element_id, exists: true},
                element_name == $name
        "#;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("name".into(), DataValue::Str(name.into()));
        let rows = match db.run_script(q, params, ScriptMutability::Immutable) {
            Ok(r) => r.rows,
            Err(e) => {
                eprintln!("multi-scope query failed: {}", e);
                println!("[]");
                return;
            }
        };
        // Dedupe on element_id and group by element_type.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut by_type: std::collections::HashMap<String, Vec<JsonValue>> =
            std::collections::HashMap::new();
        for row in rows {
            let eid = data_str(&row[0]);
            if seen.contains(&eid) {
                continue;
            }
            seen.insert(eid.clone());
            let etype = data_str(&row[1]);
            if let Some(t) = type_filter {
                if etype != t {
                    continue;
                }
            }
            by_type.entry(etype.clone()).or_default().push(json!({
                "element_id": eid,
                "scope": data_to_json(&row[2]),
                "scope_path": data_to_json(&row[3]),
            }));
        }
        let mut groups: Vec<JsonValue> = Vec::new();
        for (etype, scopes) in by_type {
            if scopes.len() > 1 {
                groups.push(json!({
                    "element_type": etype,
                    "name": name,
                    "scopes": scopes,
                }));
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&JsonValue::Array(groups)).unwrap_or_default()
        );
    }

    /// `pss override-history <ELEMENT_ID>` — every override_started /
    /// override_ended event for an element.
    pub fn cmd_override_history(db: &DbInstance, element_id: &str, limit: usize) {
        emit_filtered_timeline(
            db,
            element_id,
            &["override_started", "override_ended"],
            limit,
        );
    }

    /// `pss enable-history <ELEMENT_ID>` — every enabled / disabled event.
    pub fn cmd_enable_history(db: &DbInstance, element_id: &str, limit: usize) {
        emit_filtered_timeline(db, element_id, &["enabled", "disabled"], limit);
    }

    /// `pss scope-moves <NAME> [--type T]` — every scope_moved event
    /// matching a name (and optional element_type).
    pub fn cmd_scope_moves(
        db: &DbInstance,
        name: &str,
        type_filter: Option<&str>,
        limit: usize,
    ) {
        let q = r#"
            ?[observed_at, element_type, element_id, scope, scope_path, diff_json] :=
                *events{event_type: "scope_moved", element_name, observed_at,
                        element_type, element_id, scope, scope_path, diff_json},
                element_name == $name
            :order observed_at
            :limit $limit
        "#;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("name".into(), DataValue::Str(name.into()));
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        match db.run_script(q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let rows: Vec<JsonValue> = r
                    .rows
                    .into_iter()
                    .filter(|row| {
                        type_filter.is_none_or(|t| {
                            matches!(&row[1], DataValue::Str(s) if s.as_str() == t)
                        })
                    })
                    .map(|row| {
                        json!({
                            "observed_at": data_to_json(&row[0]),
                            "element_type": data_to_json(&row[1]),
                            "element_id": data_to_json(&row[2]),
                            "scope": data_to_json(&row[3]),
                            "scope_path": data_to_json(&row[4]),
                            "diff_json": data_to_json(&row[5]),
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                );
            }
            Err(e) => eprintln!("scope-moves query failed: {}", e),
        }
    }

    /// `pss marketplace-history` — all marketplace_added / marketplace_removed
    /// events. Optionally filter by date window.
    pub fn cmd_marketplace_history(db: &DbInstance, limit: usize) {
        let q = r#"
            ?[observed_at, event_type, element_name, element_id, diff_json] :=
                *events{event_type, element_type: "marketplace", observed_at,
                        element_name, element_id, diff_json},
                or(event_type == "marketplace_added",
                   event_type == "marketplace_removed",
                   event_type == "installed",
                   event_type == "removed")
            :order observed_at
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
                            "observed_at": data_to_json(&row[0]),
                            "event_type": data_to_json(&row[1]),
                            "element_name": data_to_json(&row[2]),
                            "element_id": data_to_json(&row[3]),
                            "diff_json": data_to_json(&row[4]),
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                );
            }
            Err(e) => eprintln!("marketplace-history query failed: {}", e),
        }
    }

    /// `pss plugin-history <PLUGIN_NAME>` — every event whose
    /// element_type=="plugin" AND element_name matches.
    ///
    /// DI-9 (audit 20260514): the discoverer stores plugin element_name
    /// as the composite `<name>@<marketplace>` (e.g.
    /// `perfect-skill-suggester@emasoft-plugins`). Looking up by just
    /// `perfect-skill-suggester` used to return `[]` because the
    /// equality match required the user to know the exact marketplace
    /// they installed from. The fix accepts BOTH forms:
    ///   - `name@marketplace` → exact equality match (original
    ///     behaviour, still works).
    ///   - `name` (no `@`) → prefix match across all marketplaces
    ///     (`name@*`), so a user can find their plugin without
    ///     remembering the marketplace.
    pub fn cmd_plugin_history(db: &DbInstance, plugin_name: &str, limit: usize) {
        // DI-9: dual-mode lookup.
        let has_at = plugin_name.contains('@');
        let q = if has_at {
            // Exact match (original semantics).
            r#"
                ?[observed_at, event_type, element_name, element_id, scope, diff_json] :=
                    *events{event_type, element_type: "plugin", observed_at,
                            element_name, element_id, scope, diff_json},
                    element_name == $name
                :order observed_at
                :limit $limit
            "#
        } else {
            // Prefix match: `<name>@*`. Datalog has no native LIKE, but
            // we can express the prefix bound via the lex order of the
            // matched string: element_name >= "name@" AND
            // element_name < "name@~" (the `~` byte sorts above `}`,
            // which is the highest ASCII byte plugin names typically
            // contain — safe for the marketplace suffix).
            r#"
                ?[observed_at, event_type, element_name, element_id, scope, diff_json] :=
                    *events{event_type, element_type: "plugin", observed_at,
                            element_name, element_id, scope, diff_json},
                    starts_with(element_name, $prefix)
                :order observed_at
                :limit $limit
            "#
        };
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        if has_at {
            params.insert("name".into(), DataValue::Str(plugin_name.into()));
        } else {
            params.insert(
                "prefix".into(),
                DataValue::Str(format!("{}@", plugin_name).into()),
            );
        }
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        match db.run_script(q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let rows: Vec<JsonValue> = r
                    .rows
                    .into_iter()
                    .map(|row| {
                        json!({
                            "observed_at": data_to_json(&row[0]),
                            "event_type": data_to_json(&row[1]),
                            "element_name": data_to_json(&row[2]),
                            "element_id": data_to_json(&row[3]),
                            "scope": data_to_json(&row[4]),
                            "diff_json": data_to_json(&row[5]),
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                );
            }
            Err(e) => eprintln!("plugin-history query failed: {}", e),
        }
    }

    /// Internal: events for one element_id matching one of `event_types`.
    fn emit_filtered_timeline(
        db: &DbInstance,
        element_id: &str,
        event_types: &[&str],
        limit: usize,
    ) {
        // Cozo has no easy "in-list" predicate; we OR them.
        let q = r#"
            ?[event_id, observed_at, event_type, diff_json] :=
                *events{element_id: $eid, event_id, observed_at, event_type, diff_json}
            :order observed_at
            :limit $limit
        "#;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(element_id.into()));
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        match db.run_script(q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let allowed: std::collections::HashSet<&str> =
                    event_types.iter().copied().collect();
                let rows: Vec<JsonValue> = r
                    .rows
                    .into_iter()
                    .filter(|row| {
                        matches!(&row[2], DataValue::Str(s) if allowed.contains(s.as_str()))
                    })
                    .map(|row| {
                        json!({
                            "event_id": data_to_json(&row[0]),
                            "observed_at": data_to_json(&row[1]),
                            "event_type": data_to_json(&row[2]),
                            "diff_json": data_to_json(&row[3]),
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                );
            }
            Err(e) => eprintln!("filtered timeline query failed: {}", e),
        }
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
    use cozo::Num;

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

    // ====================================================================
    // Phase 1.2 — DBE-1 + DBE-2 + COR-1 (audit 20260514)
    // ====================================================================

    #[test]
    fn ensure_schema_creates_events_indexes() {
        // DBE-1 (audit 20260514): all 5 secondary indexes on events must be
        // present after ensure_schema. Without these, every lifecycle query
        // does a full scan.
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("first call");

        // Each index covers a different column, so probe each with its own
        // column binding. The `, false` predicate forces an empty result so
        // the test stays cheap on a freshly-created in-memory DB.
        let probes: &[(&str, &str)] = &[
            ("events:by_element_id", "element_id"),
            ("events:by_observed_at", "observed_at"),
            ("events:by_event_type", "event_type"),
            ("events:by_scope", "scope"),
            ("events:by_element_name", "element_name"),
        ];
        for (idx, col) in probes {
            let q = format!("?[c] := *{}{{{}: c}}, false", idx, col);
            let r = db.run_script(&q, BTreeMap::new(), ScriptMutability::Immutable);
            assert!(r.is_ok(), "index {} must exist: {:?}", idx, r.err());
        }
    }

    #[test]
    fn ensure_schema_indexes_are_idempotent() {
        // DBE-1: re-running ensure_schema (e.g., on every binary startup)
        // must not error out when indexes already exist.
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("first");
        ensure_schema(&db).expect("second (idempotent)");
        ensure_schema(&db).expect("third (still idempotent)");
    }

    /// Helper: build a small in-memory cozo DB populated with the temporal
    /// schema and seed events for DBE-2 / COR-1 tests.
    fn build_test_db_with_events() -> DbInstance {
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("schema");

        // Seed: 3 elements (1 skill, 1 agent, 1 plugin), each with one
        // 'installed' event at 2026-01-01.
        let seed = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["e1", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "sk1", "skill:sk1@user:_home", "user", "/home", "user",
                 "/home/sk1.md", "h1", 100, 50, true, "active", "{}", ""],
                ["e2", "2026-01-01T00:00:00Z", "s1", "installed", "agent",
                 "ag1", "agent:ag1@user:_home", "user", "/home", "user",
                 "/home/ag1.md", "h2", 200, 80, true, "active", "{}", ""],
                ["e3", "2026-01-01T00:00:00Z", "s1", "installed", "plugin",
                 "pl1@mkt", "plugin:pl1@user:_home", "user", "/home", "user",
                 "/home/pl1", "h3", 300, 120, true, "active", "{}", ""]
              ]
            :put events { event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }
        "#;
        db.run_script(seed, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed events");
        db
    }

    /// Helper: emulate cmd_as_of's internal logic in test code so we can
    /// assert on the rows it would print. Returns (element_id, element_type)
    /// pairs after dedup + removed-filter + limit.
    fn as_of_rows_for_test(
        db: &DbInstance,
        cutoff: &str,
        type_filter: Option<&str>,
        limit: usize,
    ) -> Vec<(String, String)> {
        use std::collections::HashSet;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("cutoff".into(), DataValue::Str(cutoff.into()));
        let mut filters = String::new();
        if let Some(t) = type_filter {
            filters.push_str(", element_type = $f_type");
            params.insert("f_type".into(), DataValue::Str(t.into()));
        }
        let q = format!(
            r#"?[element_id, observed_at, event_type, element_type] :=
                *events{{element_id, observed_at, event_type, element_type}},
                observed_at <= $cutoff{filters}
            :order element_id, -observed_at"#,
            filters = filters
        );
        let result = db.run_script(&q, params, ScriptMutability::Immutable)
            .expect("query must run");
        let mut seen: HashSet<String> = HashSet::new();
        let mut out = Vec::new();
        for r in &result.rows {
            // Projection: [element_id, observed_at, event_type, element_type]
            let eid = if let DataValue::Str(s) = &r[0] { s.to_string() } else { continue };
            if !seen.insert(eid.clone()) { continue; }
            let etype_event = if let DataValue::Str(s) = &r[2] { s.to_string() } else { String::new() };
            if etype_event == "removed" { continue; }
            let elem_type = if let DataValue::Str(s) = &r[3] { s.to_string() } else { String::new() };
            out.push((eid, elem_type));
            if out.len() >= limit { break; }
        }
        out
    }

    #[test]
    fn cmd_as_of_query_returns_rows_for_seeded_events() {
        // DBE-2: cmd_as_of's single-query rewrite must return the latest
        // event per element, deduped in Rust (Cozo's numeric max() can't
        // aggregate RFC3339 strings, hence the sort+dedup pattern).
        let db = build_test_db_with_events();
        let rows = as_of_rows_for_test(&db, "2026-12-01T00:00:00Z", None, 100);
        assert_eq!(rows.len(), 3, "expected 3 elements, got {}", rows.len());
    }

    #[test]
    fn cmd_as_of_type_filter_applies_before_limit() {
        // COR-1 (audit 20260514): the bug was that --limit applied BEFORE
        // --type filter, so `--type skill --limit 100` could return 0 rows
        // if the first 100 element_ids weren't skills. The new
        // implementation pushes the type filter into the Datalog WHERE
        // clause and applies :limit LAST.
        let db = build_test_db_with_events();
        let rows = as_of_rows_for_test(&db, "2026-12-01T00:00:00Z", Some("skill"), 100);
        assert_eq!(rows.len(), 1, "expected 1 skill row, got {}", rows.len());
        assert_eq!(rows[0].1, "skill");
    }

    #[test]
    fn cmd_as_of_excludes_removed_events() {
        // DBE-2: an element whose latest event is `removed` must not appear.
        let db = build_test_db_with_events();
        // Add a removal event for sk1 at 2026-02-01.
        let removal = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["e4", "2026-02-01T00:00:00Z", "s2", "removed", "skill",
                 "sk1", "skill:sk1@user:_home", "user", "/home", "user",
                 "/home/sk1.md", "h1", -1, -1, true, "active", "{}", ""]
              ]
            :put events { event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }
        "#;
        db.run_script(removal, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("removal event");

        let rows = as_of_rows_for_test(&db, "2026-12-01T00:00:00Z", None, 100);
        // sk1 was removed → should NOT be in results. ag1 + pl1 remain.
        assert_eq!(rows.len(), 2, "expected 2 elements after sk1 removed");
        for (eid, _) in &rows {
            assert_ne!(eid, "skill:sk1@user:_home",
                       "removed element must not appear");
        }
    }

    #[test]
    fn cmd_as_of_dedup_picks_latest_per_element() {
        // DBE-2: an element with multiple events at-or-before cutoff must
        // appear exactly once, and the row chosen must be the latest
        // observed_at — not an older one. Sort-then-dedup yields the
        // first occurrence per element_id, which is the latest because of
        // `:order element_id, -observed_at`.
        let db = build_test_db_with_events();

        // Add an older 'installed' event for sk1 (older than the seed).
        let older = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["e0", "2025-12-01T00:00:00Z", "s0", "installed", "skill",
                 "sk1", "skill:sk1@user:_home", "user", "/home", "user",
                 "/home/sk1.md", "OLD_HASH", 50, 25, true, "active", "{}", ""]
              ]
            :put events { event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }
        "#;
        db.run_script(older, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("older event");

        // Query with the dedup helper. sk1 has two events at-or-before
        // cutoff (older at 2025-12-01, newer at 2026-01-01). The query
        // must return sk1 exactly once.
        let rows = as_of_rows_for_test(&db, "2026-12-01T00:00:00Z", Some("skill"), 100);
        assert_eq!(rows.len(), 1, "sk1 must appear exactly once");
    }

    #[test]
    fn cmd_as_of_limit_applied_after_filter() {
        // COR-1: limit=2 with 3 events of different types must respect the
        // filter — `--type=skill --limit=2` must still return only 1
        // (the lone skill), not 2 (skill + agent or similar).
        let db = build_test_db_with_events();
        let rows = as_of_rows_for_test(&db, "2026-12-01T00:00:00Z", Some("skill"), 2);
        assert_eq!(rows.len(), 1, "type filter applied first, then limit");
        assert_eq!(rows[0].1, "skill");
    }

    // ====================================================================
    // Phase 3 Tier A — new query subcommands (audit 20260514 — v3.6.1)
    // ====================================================================

    /// Helper: emulate cmd_by_plugin's underlying query for testing.
    fn by_plugin_rows_for_test(
        db: &DbInstance,
        plugin_name: &str,
        type_filter: Option<&str>,
    ) -> Vec<(String, String)> {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        let needle = format!("plugin:{}", plugin_name);
        params.insert("source".into(), DataValue::Str(needle.into()));
        let mut filters = String::new();
        if let Some(t) = type_filter {
            filters.push_str(", element_type = $f_type");
            params.insert("f_type".into(), DataValue::Str(t.into()));
        }
        let q = format!(
            r#"?[element_id, element_type] :=
                *elements_state{{element_id, last_event_id, exists: true}},
                *events{{event_id: last_event_id, element_type, source}},
                source = $source{filters}"#,
            filters = filters
        );
        let r = db.run_script(&q, params, ScriptMutability::Immutable)
            .expect("by-plugin query must run");
        r.rows.iter().filter_map(|row| {
            let eid = if let DataValue::Str(s) = &row[0] { s.to_string() } else { return None };
            let etype = if let DataValue::Str(s) = &row[1] { s.to_string() } else { return None };
            Some((eid, etype))
        }).collect()
    }

    /// Build a test DB with elements from two distinct plugin sources.
    /// Populates BOTH events AND elements_state (the by_plugin/enabled-where
    /// queries join the two via `last_event_id`).
    fn build_test_db_with_plugin_sources() -> DbInstance {
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("schema");

        // Three events: sk1+ag1 from plugin:foo, sk2 from plugin:bar.
        let seed_events = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["e1", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "sk1", "skill:sk1@plugin:foo", "plugin", "foo/skills", "plugin:foo",
                 "/foo/sk1.md", "h1", 100, 50, true, "active", "{}", ""],
                ["e2", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "sk2", "skill:sk2@plugin:bar", "plugin", "bar/skills", "plugin:bar",
                 "/bar/sk2.md", "h2", 200, 80, true, "active", "{}", ""],
                ["e3", "2026-01-01T00:00:00Z", "s1", "installed", "agent",
                 "ag1", "agent:ag1@plugin:foo", "plugin", "foo/agents", "plugin:foo",
                 "/foo/ag1.md", "h3", 50, 25, true, "active", "{}", ""]
              ]
            :put events { event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }
        "#;
        db.run_script(seed_events, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed events");

        // Mirror in elements_state — exists=true so the joins find them.
        let seed_state = r#"
            ?[element_id, last_event_id, current_path, current_hash,
              current_size, current_token_count, enabled, override_status,
              installed_at, last_changed_at, exists] <- [
                ["skill:sk1@plugin:foo", "e1", "/foo/sk1.md", "h1", 100, 50, true, "active", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true],
                ["skill:sk2@plugin:bar", "e2", "/bar/sk2.md", "h2", 200, 80, true, "active", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true],
                ["agent:ag1@plugin:foo", "e3", "/foo/ag1.md", "h3", 50, 25, true, "active", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true]
              ]
            :put elements_state { element_id =>
              last_event_id, current_path, current_hash, current_size,
              current_token_count, enabled, override_status, installed_at,
              last_changed_at, exists }
        "#;
        db.run_script(seed_state, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed elements_state");
        db
    }

    #[test]
    fn cmd_by_plugin_filters_by_source() {
        // F-1 (audit 20260514): listing elements provided by a given plugin
        // must filter on the literal source `plugin:<name>` — not match
        // other plugins.
        let db = build_test_db_with_plugin_sources();
        let foo_rows = by_plugin_rows_for_test(&db, "foo", None);
        assert_eq!(foo_rows.len(), 2, "plugin:foo provides sk1 + ag1");
        let bar_rows = by_plugin_rows_for_test(&db, "bar", None);
        assert_eq!(bar_rows.len(), 1, "plugin:bar provides sk2");
    }

    #[test]
    fn cmd_by_plugin_respects_type_filter() {
        // F-1: `--type skill` must return only skills (not agents)
        // when the plugin provides both.
        let db = build_test_db_with_plugin_sources();
        let foo_skills = by_plugin_rows_for_test(&db, "foo", Some("skill"));
        assert_eq!(foo_skills.len(), 1, "plugin:foo has 1 skill (sk1)");
        assert_eq!(foo_skills[0].1, "skill");
    }

    #[test]
    fn cmd_by_plugin_returns_empty_for_unknown_plugin() {
        // F-1: unknown plugin returns empty, no error.
        let db = build_test_db_with_plugin_sources();
        let rows = by_plugin_rows_for_test(&db, "nonexistent-plugin", None);
        assert!(rows.is_empty());
    }

    // ────────────────────────────────────────────────────────────────────
    // F-2 (audit 20260514): by-marketplace
    // ────────────────────────────────────────────────────────────────────

    /// Helper: emulate cmd_by_marketplace's underlying query for testing.
    fn by_marketplace_rows_for_test(
        db: &DbInstance,
        marketplace_name: &str,
        type_filter: Option<&str>,
    ) -> Vec<(String, String, String)> {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        let prefix = format!("marketplace:{}", marketplace_name);
        params.insert("prefix".into(), DataValue::Str(prefix.into()));
        let mut filters = String::new();
        if let Some(t) = type_filter {
            filters.push_str(", element_type = $f_type");
            params.insert("f_type".into(), DataValue::Str(t.into()));
        }
        let q = format!(
            r#"?[element_id, element_type, source] :=
                *elements_state{{element_id, last_event_id, exists: true}},
                *events{{event_id: last_event_id, element_type, source}},
                starts_with(source, $prefix){filters}"#,
            filters = filters
        );
        let r = db.run_script(&q, params, ScriptMutability::Immutable)
            .expect("by-marketplace query must run");
        r.rows.iter().filter_map(|row| {
            let eid = if let DataValue::Str(s) = &row[0] { s.to_string() } else { return None };
            let etype = if let DataValue::Str(s) = &row[1] { s.to_string() } else { return None };
            let src = if let DataValue::Str(s) = &row[2] { s.to_string() } else { return None };
            Some((eid, etype, src))
        }).collect()
    }

    fn build_test_db_with_marketplace_sources() -> DbInstance {
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("schema");

        // Three events: pluginA from marketplace "emasoft-plugins",
        // pluginB from same marketplace, pluginC from "other-mp".
        let seed = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["e1", "2026-01-01T00:00:00Z", "s1", "installed", "plugin",
                 "pluginA@emasoft-plugins", "plugin:pluginA@marketplace:emasoft-plugins",
                 "marketplace", "emasoft-plugins", "marketplace:emasoft-plugins",
                 "/mp/pluginA.json", "h1", 100, 50, true, "active", "{}", ""],
                ["e2", "2026-01-01T00:00:00Z", "s1", "installed", "plugin",
                 "pluginB@emasoft-plugins", "plugin:pluginB@marketplace:emasoft-plugins",
                 "marketplace", "emasoft-plugins", "marketplace:emasoft-plugins",
                 "/mp/pluginB.json", "h2", 200, 80, true, "active", "{}", ""],
                ["e3", "2026-01-01T00:00:00Z", "s1", "installed", "plugin",
                 "pluginC@other-mp", "plugin:pluginC@marketplace:other-mp",
                 "marketplace", "other-mp", "marketplace:other-mp",
                 "/mp/pluginC.json", "h3", 50, 25, true, "active", "{}", ""]
              ]
            :put events { event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }
        "#;
        db.run_script(seed, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed events");

        let seed_state = r#"
            ?[element_id, last_event_id, current_path, current_hash,
              current_size, current_token_count, enabled, override_status,
              installed_at, last_changed_at, exists] <- [
                ["plugin:pluginA@marketplace:emasoft-plugins", "e1", "/mp/pluginA.json", "h1", 100, 50, true, "active", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true],
                ["plugin:pluginB@marketplace:emasoft-plugins", "e2", "/mp/pluginB.json", "h2", 200, 80, true, "active", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true],
                ["plugin:pluginC@marketplace:other-mp", "e3", "/mp/pluginC.json", "h3", 50, 25, true, "active", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true]
              ]
            :put elements_state { element_id =>
              last_event_id, current_path, current_hash, current_size,
              current_token_count, enabled, override_status, installed_at,
              last_changed_at, exists }
        "#;
        db.run_script(seed_state, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed elements_state");
        db
    }

    #[test]
    fn cmd_by_marketplace_filters_by_source_prefix() {
        // F-2: by-marketplace must find every element from a given marketplace.
        let db = build_test_db_with_marketplace_sources();
        let rows = by_marketplace_rows_for_test(&db, "emasoft-plugins", None);
        assert_eq!(rows.len(), 2, "emasoft-plugins provides pluginA + pluginB");
        for (_, _, src) in &rows {
            assert!(src.starts_with("marketplace:emasoft-plugins"));
        }
        let other = by_marketplace_rows_for_test(&db, "other-mp", None);
        assert_eq!(other.len(), 1, "other-mp provides pluginC");
    }

    #[test]
    fn cmd_by_marketplace_returns_empty_for_unknown_marketplace() {
        // F-2: unknown marketplace returns empty, no error.
        let db = build_test_db_with_marketplace_sources();
        let rows = by_marketplace_rows_for_test(&db, "nonexistent-mp", None);
        assert!(rows.is_empty());
    }

    #[test]
    fn cmd_by_marketplace_respects_type_filter() {
        // F-2: --type plugin must filter to plugin elements only.
        let db = build_test_db_with_marketplace_sources();
        let rows = by_marketplace_rows_for_test(&db, "emasoft-plugins", Some("plugin"));
        assert_eq!(rows.len(), 2);
        for (_, etype, _) in &rows {
            assert_eq!(etype, "plugin");
        }
        let rows_skill = by_marketplace_rows_for_test(&db, "emasoft-plugins", Some("skill"));
        assert!(rows_skill.is_empty(), "no skills in seeded marketplace data");
    }

    // ────────────────────────────────────────────────────────────────────
    // F-6 (audit 20260514): scope-diff
    // ────────────────────────────────────────────────────────────────────

    fn build_test_db_with_two_scopes() -> DbInstance {
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("schema");

        // user scope has skills "a" and "b"; project scope has "b" and "c".
        // Expected diff: only_user=["a"], only_project=["c"], shared=["b"].
        let seed = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["e1", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "a", "skill:a@user:_home", "user", "/home", "user",
                 "/home/a.md", "h1", 100, 50, true, "active", "{}", ""],
                ["e2", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "b", "skill:b@user:_home", "user", "/home", "user",
                 "/home/b.md", "h2", 100, 50, true, "active", "{}", ""],
                ["e3", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "b", "skill:b@project:demo", "project", "demo", "project:demo",
                 "/proj/b.md", "h3", 100, 50, true, "active", "{}", ""],
                ["e4", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "c", "skill:c@project:demo", "project", "demo", "project:demo",
                 "/proj/c.md", "h4", 100, 50, true, "active", "{}", ""]
              ]
            :put events { event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }
        "#;
        db.run_script(seed, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed events");

        let seed_state = r#"
            ?[element_id, last_event_id, current_path, current_hash,
              current_size, current_token_count, enabled, override_status,
              installed_at, last_changed_at, exists] <- [
                ["skill:a@user:_home", "e1", "/home/a.md", "h1", 100, 50, true, "active", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true],
                ["skill:b@user:_home", "e2", "/home/b.md", "h2", 100, 50, true, "active", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true],
                ["skill:b@project:demo", "e3", "/proj/b.md", "h3", 100, 50, true, "active", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true],
                ["skill:c@project:demo", "e4", "/proj/c.md", "h4", 100, 50, true, "active", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true]
              ]
            :put elements_state { element_id =>
              last_event_id, current_path, current_hash, current_size,
              current_token_count, enabled, override_status, installed_at,
              last_changed_at, exists }
        "#;
        db.run_script(seed_state, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed elements_state");
        db
    }

    /// Helper that runs the scope-diff core query (without the println).
    fn scope_diff_for_test(
        db: &DbInstance,
        scope: &str,
    ) -> std::collections::HashSet<(String, String)> {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("scope".into(), DataValue::Str(scope.into()));
        let q = r#"?[element_type, element_name] :=
            *elements_state{element_id, last_event_id, exists: true},
            *events{event_id: last_event_id, element_type, element_name, scope},
            scope = $scope"#;
        let r = db.run_script(q, params, ScriptMutability::Immutable)
            .expect("scope query");
        r.rows.iter().filter_map(|row| {
            let etype = if let DataValue::Str(s) = &row[0] { s.to_string() } else { return None };
            let ename = if let DataValue::Str(s) = &row[1] { s.to_string() } else { return None };
            Some((etype, ename))
        }).collect()
    }

    #[test]
    fn cmd_scope_diff_user_vs_project() {
        let db = build_test_db_with_two_scopes();
        let user_set = scope_diff_for_test(&db, "user");
        let proj_set = scope_diff_for_test(&db, "project");
        let only_user: Vec<(String, String)> = user_set.difference(&proj_set).cloned().collect();
        let only_proj: Vec<(String, String)> = proj_set.difference(&user_set).cloned().collect();
        let shared: Vec<(String, String)> = user_set.intersection(&proj_set).cloned().collect();

        let only_user_names: Vec<&str> = only_user.iter().map(|(_, n)| n.as_str()).collect();
        let only_proj_names: Vec<&str> = only_proj.iter().map(|(_, n)| n.as_str()).collect();
        let shared_names: Vec<&str> = shared.iter().map(|(_, n)| n.as_str()).collect();

        assert_eq!(only_user_names, vec!["a"]);
        assert_eq!(only_proj_names, vec!["c"]);
        assert_eq!(shared_names, vec!["b"]);
    }

    #[test]
    fn cmd_scope_diff_returns_empty_for_unknown_scope() {
        let db = build_test_db_with_two_scopes();
        let bogus = scope_diff_for_test(&db, "nonexistent");
        assert!(bogus.is_empty());
    }

    // ────────────────────────────────────────────────────────────────────
    // F-19 (audit 20260514): stats-by-scope query
    // ────────────────────────────────────────────────────────────────────

    /// F-19: count elements per scope. We can run the Datalog directly
    /// against the same seed data used by scope-diff tests.
    #[test]
    fn cmd_stats_by_scope_counts_per_scope() {
        let db = build_test_db_with_two_scopes();
        // user has 2 (a, b); project has 2 (b, c).
        let q = r#"?[scope, count(element_id)] :=
            *elements_state{element_id, last_event_id, exists: true},
            *events{event_id: last_event_id, scope}
            :order scope"#;
        let r = db.run_script(q, BTreeMap::new(), ScriptMutability::Immutable)
            .expect("query");
        let mut counts: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
        for row in &r.rows {
            let scope = if let DataValue::Str(s) = &row[0] { s.to_string() } else { continue };
            let n = if let DataValue::Num(Num::Int(n)) = &row[1] { *n } else { 0 };
            counts.insert(scope, n);
        }
        assert_eq!(counts.get("user").copied(), Some(2));
        assert_eq!(counts.get("project").copied(), Some(2));
    }

    // ────────────────────────────────────────────────────────────────────
    // F-17 (audit 20260514): changes-in-batch
    // ────────────────────────────────────────────────────────────────────

    /// F-17: lookup events by scan_id should return all 4 seeded events.
    #[test]
    fn cmd_changes_in_batch_finds_events_for_scan_id() {
        let db = build_test_db_with_two_scopes();
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("scan_id".into(), DataValue::Str("s1".into()));
        let q = r#"?[count(event_id)] :=
            *events{event_id, scan_id},
            scan_id = $scan_id"#;
        let r = db.run_script(q, params, ScriptMutability::Immutable)
            .expect("query");
        let count = if let DataValue::Num(Num::Int(n)) = &r.rows[0][0] { *n } else { -1 };
        assert_eq!(count, 4, "expected 4 events for scan_id=s1");
    }

    #[test]
    fn cmd_dedup_candidates_finds_duplicate_names() {
        // F-8 (audit 20260514): dedup-candidates must find element names
        // that appear in 2+ scopes. We seed a duplicate (same name "foo",
        // different scopes) and expect it back; a unique name must NOT
        // appear when --min-count=2.
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("schema");

        // Seed events: "foo" skill in user + plugin scopes (DUPLICATE),
        // "bar" skill in user only (UNIQUE).
        let seed = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["e1", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "foo", "skill:foo@user:_home", "user", "/home", "user",
                 "/home/foo.md", "h1", 100, 50, true, "active", "{}", ""],
                ["e2", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "foo", "skill:foo@plugin:p", "plugin", "p", "plugin:p",
                 "/p/foo.md", "h2", 100, 50, true, "active", "{}", ""],
                ["e3", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "bar", "skill:bar@user:_home", "user", "/home", "user",
                 "/home/bar.md", "h3", 100, 50, true, "active", "{}", ""]
              ]
            :put events { event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }
        "#;
        db.run_script(seed, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed events");
        let seed_state = r#"
            ?[element_id, last_event_id, current_path, current_hash,
              current_size, current_token_count, enabled, override_status,
              installed_at, last_changed_at, exists] <- [
                ["skill:foo@user:_home", "e1", "/home/foo.md", "h1", 100, 50, true, "active", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true],
                ["skill:foo@plugin:p",  "e2", "/p/foo.md",    "h2", 100, 50, true, "active", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true],
                ["skill:bar@user:_home", "e3", "/home/bar.md", "h3", 100, 50, true, "active", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true]
              ]
            :put elements_state { element_id =>
              last_event_id, current_path, current_hash, current_size,
              current_token_count, enabled, override_status, installed_at,
              last_changed_at, exists }
        "#;
        db.run_script(seed_state, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed elements_state");

        // Run the same query the cmd_dedup_candidates impl uses.
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("min".into(), DataValue::Num(Num::Int(2)));
        let q = r#"
            counts[etype, ename, count(eid)] :=
                *elements_state{element_id: eid, last_event_id, exists: true},
                *events{event_id: last_event_id, element_type: etype, element_name: ename}

            ?[etype, ename, c] :=
                counts[etype, ename, c],
                c >= $min
            :order -c, etype, ename
        "#;
        let r = db.run_script(q, params, ScriptMutability::Immutable)
            .expect("dedup query must run");
        // Expected: ("skill", "foo", 2) appears; "bar" does not.
        let mut found_foo = false;
        for row in &r.rows {
            if let DataValue::Str(name) = &row[1] {
                if name.as_str() == "foo" {
                    found_foo = true;
                }
                assert_ne!(name.as_str(), "bar", "unique name must not appear");
            }
        }
        assert!(found_foo, "duplicate skill 'foo' must be detected");
    }

    #[test]
    fn cmd_changes_summary_groups_by_event_type() {
        // F-7 (audit 20260514): changes-summary must aggregate event_type
        // counts in the cutoff window — providing the "what changed?"
        // dashboard the audit specifically called out as missing.
        let db = build_test_db_with_plugin_sources();
        // All 3 seeded events are 'installed' on 2026-01-01.
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert(
            "cutoff".into(),
            DataValue::Str("2025-12-01T00:00:00Z".into()),
        );
        let q = r#"?[event_type, count(event_id)] :=
            *events{event_id, observed_at, event_type},
            observed_at >= $cutoff"#;
        let r = db.run_script(q, params, ScriptMutability::Immutable)
            .expect("changes-summary query");
        let installed_count: i64 = r.rows.iter()
            .find_map(|row| match (&row[0], &row[1]) {
                (DataValue::Str(s), DataValue::Num(Num::Int(n))) if s.as_str() == "installed" => Some(*n),
                _ => None,
            })
            .unwrap_or(0);
        assert_eq!(installed_count, 3, "expected 3 installed events");
    }

    /// DI-2 (audit 20260514): when the merge-events writer sees the same
    /// (element_type, name) coming from BOTH a user scope and a plugin
    /// scope, the override resolver must run and emit an
    /// `override_started` event for the lower-priority candidate. Prior
    /// to v3.6.4 the resolver existed but was never wired, so this case
    /// silently produced `override_status: "active"` on both rows.
    #[test]
    fn cmd_merge_events_wires_override_resolver() {
        use std::io::Cursor;

        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );

        // Two observations of the SAME skill name "shared" — one from
        // user scope (higher priority), one from a plugin (lower).
        // resolve_overrides() should mark the plugin one as
        // `overridden_by:skill:shared@user:_home` and the user one as
        // `overrides:skill:shared@plugin:foo`.
        let jsonl = r#"{"_pss_manifest": true, "visited_scope_paths": ["/home", "foo/skills"]}
{"type": "skill", "name": "shared", "source": "user", "path": "/home/shared.md", "description": "user scope", "enabled": true}
{"type": "skill", "name": "shared", "source": "plugin:foo", "path": "foo/skills/shared.md", "description": "plugin scope", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(jsonl), true)
            .expect("merge-events must succeed");

        // Verify both override-resolution events were emitted with the
        // RIGHT override_status (not the placeholder "active").
        let q = r#"
            ?[element_id, event_type, override_status] :=
                *events{element_id, event_type, override_status},
                event_type = "override_started"
        "#;
        let r = db.run_script(q, BTreeMap::new(), ScriptMutability::Immutable)
            .expect("query events");
        assert!(
            !r.rows.is_empty(),
            "DI-2 wiring must emit override_started events when 2+ scopes share a name; got 0 rows"
        );

        // Inspect every override_started row: status must NOT be the
        // pre-v3.6.4 placeholder "active".
        for row in &r.rows {
            let status = if let DataValue::Str(s) = &row[2] {
                s.as_str().to_string()
            } else {
                String::new()
            };
            assert_ne!(
                status, "active",
                "override_started row must carry a real override_status (overrides:/overridden_by:); got 'active'"
            );
            assert!(
                status.starts_with("overrides:") || status.starts_with("overridden_by:"),
                "expected resolver-marker prefix, got: {}",
                status
            );
        }
    }

    /// DI-2: when only ONE scope reports the element, the resolver must
    /// NOT emit an override_started event (single-scope = no override
    /// decision to make). Negative test guarding against false positives.
    #[test]
    fn cmd_merge_events_skips_override_for_single_scope() {
        use std::io::Cursor;

        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );

        // One observation only — resolver must skip.
        let jsonl = r#"{"_pss_manifest": true, "visited_scope_paths": ["/home"]}
{"type": "skill", "name": "solo", "source": "user", "path": "/home/solo.md", "description": "only here", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(jsonl), true)
            .expect("merge-events must succeed");

        let q = r#"
            ?[count(event_id)] :=
                *events{event_id, event_type},
                event_type = "override_started"
        "#;
        let r = db.run_script(q, BTreeMap::new(), ScriptMutability::Immutable)
            .expect("query events");
        let count = if let DataValue::Num(Num::Int(n)) = &r.rows[0][0] {
            *n
        } else {
            -1
        };
        assert_eq!(count, 0, "single-scope element must not produce override_started");
    }

    // ========================================================================
    // DI-1 wave 1 (audit 20260514): description_changed event tests
    // ========================================================================

    /// DI-1 wave 1: the description-hash helper is deterministic and
    /// collision-resistant for typical inputs. Empty input still hashes
    /// (not the empty string) so empty-vs-non-empty transitions are
    /// detectable.
    #[test]
    fn description_hash_basic_properties() {
        let a = description_hash("Run security audit");
        let b = description_hash("Run security audit");
        assert_eq!(a, b, "same input → same hash");
        let c = description_hash("Run security audit.");
        assert_ne!(a, c, "trailing period must change the hash");
        let empty = description_hash("");
        assert_eq!(empty.len(), 32, "16-byte truncated SHA-256 = 32 hex chars");
        assert_ne!(empty, a, "empty must differ from non-empty");
    }

    /// DI-1 wave 1: changing the description on a subsequent scan emits
    /// a description_changed event. First scan produces Installed +
    /// stores the hash; second scan with a different description must
    /// emit DescriptionChanged.
    #[test]
    fn cmd_merge_events_emits_description_changed() {
        use std::io::Cursor;

        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );

        // First scan: install with description "v1".
        let scan1 = r#"{"_pss_manifest": true, "visited_scope_paths": ["/home"]}
{"type": "skill", "name": "foo", "source": "user", "path": "/home/foo.md", "description": "v1 description", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan1), true)
            .expect("first merge-events");

        // Second scan: same element, new description.
        let scan2 = r#"{"_pss_manifest": true, "visited_scope_paths": ["/home"]}
{"type": "skill", "name": "foo", "source": "user", "path": "/home/foo.md", "description": "v2 description", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan2), true)
            .expect("second merge-events");

        let q = r#"
            ?[event_type, diff_json] :=
                *events{event_type, diff_json},
                event_type = "description_changed"
        "#;
        let r = db.run_script(q, BTreeMap::new(), ScriptMutability::Immutable)
            .expect("query events");
        assert_eq!(r.rows.len(), 1, "expected exactly one description_changed event");
        let diff = match &r.rows[0][1] {
            DataValue::Str(s) => s.to_string(),
            _ => String::new(),
        };
        assert!(diff.contains("previous_description_hash"));
        assert!(diff.contains("new_description_hash"));
        assert!(diff.contains("v2 description"));
    }

    /// DI-1 wave 1: a re-scan with the SAME description must NOT emit
    /// description_changed (negative case — false-positive guard).
    #[test]
    fn cmd_merge_events_skips_description_unchanged() {
        use std::io::Cursor;

        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );

        let scan = r#"{"_pss_manifest": true, "visited_scope_paths": ["/home"]}
{"type": "skill", "name": "foo", "source": "user", "path": "/home/foo.md", "description": "unchanged", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan), true)
            .expect("first merge-events");
        // Same input, same hash → no DescriptionChanged.
        cli::merge_events_from_reader(&db, Cursor::new(scan), true)
            .expect("second merge-events");

        let q = r#"
            ?[count(event_id)] :=
                *events{event_id, event_type},
                event_type = "description_changed"
        "#;
        let r = db.run_script(q, BTreeMap::new(), ScriptMutability::Immutable)
            .expect("query events");
        let count = if let DataValue::Num(Num::Int(n)) = &r.rows[0][0] {
            *n
        } else {
            -1
        };
        assert_eq!(count, 0, "same description must not produce description_changed");
    }

    /// DI-1 wave 1: first observation never emits description_changed
    /// (no prior to compare against). Installed event covers it.
    #[test]
    fn cmd_merge_events_first_install_no_description_changed() {
        use std::io::Cursor;

        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );

        let scan = r#"{"_pss_manifest": true, "visited_scope_paths": ["/home"]}
{"type": "skill", "name": "fresh", "source": "user", "path": "/home/fresh.md", "description": "brand new", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan), true)
            .expect("merge-events");

        let q = r#"
            ?[event_type, count(event_id)] :=
                *events{event_id, event_type}
        "#;
        let r = db.run_script(q, BTreeMap::new(), ScriptMutability::Immutable)
            .expect("query events");
        // Expect exactly one Installed event, no DescriptionChanged.
        let mut found_installed = false;
        for row in &r.rows {
            if let DataValue::Str(s) = &row[0] {
                if s.as_str() == "installed" {
                    found_installed = true;
                }
                assert_ne!(
                    s.as_str(),
                    "description_changed",
                    "first install must NOT emit description_changed"
                );
            }
        }
        assert!(found_installed, "expected installed event for first scan");
    }

    // ========================================================================
    // DI-10 (audit 20260514): project-installed plugins must classify as scope "plugin"
    // ========================================================================

    /// DI-10: `project:foo/plugin:bar` MUST classify as scope "plugin",
    /// not "project". The discoverer encodes project-installed plugins
    /// in this composite form; the previous logic matched `project:`
    /// first and mis-classified them.
    #[test]
    fn scope_from_discovery_source_project_plugin_composite() {
        // The composite form a project-installed plugin produces.
        assert_eq!(
            cli::scope_from_discovery_source("project:myproj/plugin:foo"),
            "plugin",
            "project-installed plugin must classify as plugin scope"
        );
        // Bare project (not a plugin) still classifies as project.
        assert_eq!(
            cli::scope_from_discovery_source("project:myproj"),
            "project"
        );
        // Plain plugin source unchanged.
        assert_eq!(
            cli::scope_from_discovery_source("plugin:foo"),
            "plugin"
        );
    }

    /// DI-10: scope_path for the composite form keeps both parts so
    /// `<project1>/plugin:foo` and `<project2>/plugin:foo` don't collide.
    #[test]
    fn scope_path_from_discovery_source_preserves_composite() {
        assert_eq!(
            cli::scope_path_from_discovery_source("project:projA/plugin:foo"),
            "projA/plugin:foo"
        );
        assert_eq!(
            cli::scope_path_from_discovery_source("project:projB/plugin:foo"),
            "projB/plugin:foo"
        );
        // The two scope_paths above differ, so element_ids will too.
        assert_ne!(
            cli::scope_path_from_discovery_source("project:projA/plugin:foo"),
            cli::scope_path_from_discovery_source("project:projB/plugin:foo")
        );
    }
}
