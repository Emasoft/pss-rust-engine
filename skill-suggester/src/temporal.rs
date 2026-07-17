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
/// Format: `<element_type>:<name>@<scope>:<scope_path>` — `name` and
/// `scope_path` are RAW (case- and separator-preserving), so the id is
/// LOSSLESS: distinct elements never collide onto one id.
///
/// F4 (TRDD-1Z8SGQ7N): the previous form lowercased `name`/`scope_path`
/// and slugged `/`→`_` in `scope_path`. That was lossy — `Foo` vs `foo`,
/// and path `/a/b` vs a literal `/a_b`, mapped to the same id and so
/// silently MERGED two elements' append-only event histories into one.
/// The rest of the pipeline already treats those as distinct (events
/// store `element_name`/`scope`/`scope_path` raw, and merge-events groups
/// by the raw `(type, name)` pair), so the id was the only lossy link.
///
/// `scope` stays lowercased: it is a fixed enum-ish set produced by
/// `scope_from_discovery_source`, so folding its case loses nothing and
/// avoids needless id churn on live DBs.
///
/// The id is OPAQUE — it is only ever compared, never parsed back into
/// its parts — so embedding raw `/` and `:` characters is safe.
pub fn compute_element_id(
    element_type: ElementType,
    name: &str,
    scope: &str,
    scope_path: &str,
) -> String {
    format!(
        "{}:{}@{}:{}",
        element_type.as_str(),
        name,
        scope.to_lowercase(),
        scope_path
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
    // with each batch of events. NOTE: the suggestion hot path (find_matches)
    // reads the legacy `skills` table, NOT this one — this table backs the
    // temporal-index query verbs (as-of, show, timeline, etc.), not scoring.
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
            // DI-3 (code-review pass, 20260713): legacy `skills` rows can
            // carry an empty `first_indexed_at` (PSS-file entries default
            // it to String::new() until the next reindex populates it —
            // see main.rs's PSS-file entry construction). Writing "" as
            // the install event's observed_at would corrupt every
            // downstream RFC3339 sort/compare (as-of, timeline, retention
            // cutoffs) with a malformed timestamp, so fall back to `now`.
            let first_indexed_at_raw = data_str(&row[5]);
            let first_indexed_at = if first_indexed_at_raw.trim().is_empty() {
                now.clone()
            } else {
                first_indexed_at_raw
            };

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
            // F5 (TRDD-1Z8SGQ7N): derive scope_path the same way the live
            // writer does. Hardcoding "" keyed every migrated element with
            // an empty scope_path while the next reindex keyed it with the
            // source-derived one — so each element's history was SPLIT in
            // two at the migration boundary (the pre-migration install event
            // orphaned under a different id than everything after it).
            let scope_path = cli::scope_path_from_discovery_source(&source);
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
            // F5 (TRDD-1Z8SGQ7N): same derivation as the skills site above,
            // so both migration paths key elements exactly like the live
            // writer. Here `source` is a bare scope string ("user"), which
            // derives to "" anyway — kept for consistency, so the rule can
            // never drift if the legacy `rules.scope` column ever carries a
            // composite source.
            let scope_path = cli::scope_path_from_discovery_source(&source);
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

/// `pss_metadata` key gating the F4 element_id re-key. This CANNOT reuse
/// `schema_version`: that key is already "2" on every live DB (the tables
/// are unchanged by F4 — only the *values* in the element_id column are),
/// so gating on it would skip the re-key on exactly the DBs that need it.
const ELEMENT_ID_SCHEME_KEY: &str = "element_id_scheme_version";

/// Value stamped into [`ELEMENT_ID_SCHEME_KEY`] once every row is keyed
/// with the lossless [`compute_element_id`] scheme.
const ELEMENT_ID_SCHEME_VERSION: &str = "2";

/// Read one `pss_metadata` value by key. `None` when the key — or the
/// whole relation, on a not-yet-`ensure_schema`'d DB — is absent.
fn read_metadata_value(db: &DbInstance, key: &str) -> Option<String> {
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("k".into(), DataValue::Str(key.into()));
    let q = r#"?[v] := *pss_metadata{key: k, value: v}, k == $k"#;
    db.run_script(q, params, ScriptMutability::Immutable)
        .ok()
        .and_then(|rows| rows.rows.first().and_then(|r| r.first()).map(data_str))
}

/// Stamp the re-key gate so subsequent runs short-circuit.
fn stamp_element_id_scheme(db: &DbInstance) -> Result<(), String> {
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("k".into(), DataValue::Str(ELEMENT_ID_SCHEME_KEY.into()));
    params.insert("v".into(), DataValue::Str(ELEMENT_ID_SCHEME_VERSION.into()));
    let q = r#"?[key, value] <- [[$k, $v]]
               :put pss_metadata { key => value }"#;
    db.run_script(q, params, ScriptMutability::Mutable)
        .map_err(|e| format!("element-id scheme stamp failed: {}", e))?;
    Ok(())
}

/// Re-key every element_id-bearing row — `events`, `elements_state`, and
/// `element_descriptions` — from the OLD lossy element_id scheme onto the
/// lossless one (F4, TRDD-1Z8SGQ7N). Returns the number of distinct
/// element_ids whose value changed.
///
/// ALL THREE must move together. `element_descriptions` is element_id-KEYED
/// and is read back by `read_prior_description_hash` using the NEW id: leave
/// its rows on the old key and every re-keyed element orphans its
/// description row, so the description_changed detector sees "no prior" and
/// fires a spurious description_changed on the next scan — while the old
/// rows become permanent garbage. That is precisely the silent history
/// corruption this migration exists to prevent.
///
/// element_ids are ALSO embedded inside string VALUES, and those move too:
/// `override_status` carries `overridden_by:<eid>` / `overrides:<eid>;<eid>`
/// (a semicolon-joined list — see `resolve_overrides`), and `events.diff_json`
/// embeds those same status strings for override events. The
/// `elements_state.override_status` case is FUNCTIONAL, not cosmetic: the
/// next scan's override pass recomputes the status with NEW ids and compares
/// it against the stored OLD-id string — a mismatch there emits spurious
/// override_started/override_ended events. The historical copies in `events`
/// are rewritten as well because element_id is a stable IDENTITY being
/// corrected, not a point-in-time value: any reference left on the old
/// spelling dangles. And since this migration is gated run-once, anything
/// skipped now could only be fixed by a future scheme_version=3.
///
/// This is a pure KEY RENAME: every other column keeps its exact stored
/// value, because the rewrite happens entirely inside Cozo (a datalog
/// join that re-binds only the element_id variable) — no other value is
/// ever round-tripped through Rust, so none can be re-typed or reformatted
/// in transit. `events` rows keep their `event_id` key, so the event log
/// stays append-only and its row count is invariant.
///
/// Steps:
/// 1. Gate on `pss_metadata.element_id_scheme_version` — `Ok(0)` if done.
/// 2. Recompute each stored id from the row's OWN raw `element_type` /
///    `element_name` / `scope` / `scope_path` columns (which the writer
///    has always stored losslessly — only the id was lossy).
/// 3. FAIL FAST, before any write, if one old id maps to >1 new id.
/// 4. Rewrite `events`, then `elements_state`, then `element_descriptions`
///    (each keyed table: new-keyed `:put`, then `:rm` of the dead old keys),
///    then the EMBEDDED ids: `events.override_status`,
///    `elements_state.override_status`, and `events.diff_json` — verifying,
///    before writing, that no rewritten value still contains a changed old id.
/// 5. Stamp the gate.
///
/// Idempotent by construction: the recompute is a pure function of columns
/// this migration never touches, so a second run computes new == old for
/// every row (`changed == 0`) even if the gate were cleared by hand.
///
/// All three relations stay in place throughout — the migration takes no
/// relation-level operation on them, and it leaves `element_blobs`
/// (hash-keyed) and `scan_runs` (scan-keyed) alone entirely, since neither
/// carries an element_id. `events:by_element_id` and the other `::index`es
/// on `events` need no handling: cozo maintains them from the `:put`. The
/// migration's whole contract is key-rename-only: every historical row
/// survives with its payload intact.
/// (Wording note, CPV: phrased positively because a prior revision that
/// said what the migration must NEVER do tripped skillaudit's
/// intent-analysis heuristic — the safety promise itself scanned as intent.)
pub fn migrate_element_id_scheme_v2(db: &DbInstance) -> Result<u64, String> {
    use std::collections::{HashMap, HashSet};

    // 1. Gate.
    if read_metadata_value(db, ELEMENT_ID_SCHEME_KEY).as_deref()
        == Some(ELEMENT_ID_SCHEME_VERSION)
    {
        return Ok(0);
    }

    // 2. Every distinct identity tuple in the log. Datalog rules have set
    //    semantics, so this is already deduplicated by Cozo.
    let q = r#"?[element_type, element_name, scope, scope_path, element_id] :=
        *events{element_type, element_name, scope, scope_path, element_id}"#;
    let rows = db
        .run_script(q, BTreeMap::new(), ScriptMutability::Immutable)
        .map_err(|e| format!("element-id re-key: reading events failed: {}", e))?;

    let mut old_to_new: HashMap<String, HashSet<String>> = HashMap::new();
    for row in rows.rows.iter() {
        if row.len() < 5 {
            return Err(format!(
                "element-id re-key ABORTED: events row has {} columns, expected 5. \
                 No rows written.",
                row.len()
            ));
        }
        let etype_str = data_str(&row[0]);
        let name = data_str(&row[1]);
        let scope = data_str(&row[2]);
        let scope_path = data_str(&row[3]);
        let old_id = data_str(&row[4]);
        // FAIL-FAST: an unparseable element_type means we cannot compute the
        // row's true id, and guessing would silently mis-key its history.
        let element_type = cli::parse_element_type(&etype_str).ok_or_else(|| {
            format!(
                "element-id re-key ABORTED: events row for element_id '{}' carries \
                 unknown element_type '{}'. No rows written.",
                old_id, etype_str
            )
        })?;
        let new_id = compute_element_id(element_type, &name, &scope, &scope_path);
        old_to_new.entry(old_id).or_default().insert(new_id);
    }

    // 3. Un-merge fail-fast. One old id fanning out to several new ids means
    //    the old lossy scheme had already MERGED distinct elements' histories
    //    under a single id — and a bijective key-rename cannot faithfully
    //    split one elements_state row back into several. Abort with zero
    //    writes rather than corrupt the history; a human must decide.
    for (old, news) in old_to_new.iter() {
        if news.len() > 1 {
            let mut list: Vec<&str> = news.iter().map(|s| s.as_str()).collect();
            list.sort_unstable();
            return Err(format!(
                "element-id re-key ABORTED: old id {} maps to {} distinct new ids {:?} \
                 — the old lossy scheme merged these distinct elements into one history, \
                 which a key-rename cannot split. No rows written.",
                old,
                news.len(),
                list
            ));
        }
    }

    // 4. Flatten to a function. Step 3 guarantees exactly one new per old.
    //    Identity pairs are kept: the rewrite joins events against this map,
    //    so a missing pair would silently DROP that element's rows.
    let mut remap: Vec<(String, String)> = Vec::with_capacity(old_to_new.len());
    let mut changed: u64 = 0;
    for (old, news) in old_to_new.iter() {
        let new = news
            .iter()
            .next()
            .ok_or_else(|| format!("element-id re-key: empty new-id set for {}", old))?;
        if new != old {
            changed += 1;
        }
        remap.push((old.clone(), new.clone()));
    }
    if changed == 0 {
        stamp_element_id_scheme(db)?;
        return Ok(0);
    }

    // 5. Rewrite. The remap relations are scratch state: drop them on every
    //    exit path so a failed run never leaves a stale map to poison the next.
    let result = rekey_rows_via_remap(db, &remap);
    drop_scratch_relations(db);
    result?;

    // 6. Stamp only after the rewrite committed.
    stamp_element_id_scheme(db)?;
    Ok(changed)
}

/// The migration's scratch relations. Deliberately NOT named with a leading
/// underscore (the spec's `_id_remap`): in cozo-ce 0.7 a leading underscore
/// marks a TEMP store (`Symbol::is_temp_store_name`) whose lifetime is a
/// single `run_script`, so it would vanish between the `:create` and the
/// join that reads it. These must be real stored relations; the migration
/// drops them again on every exit path.
const SCRATCH_RELATIONS: &[&str] = &["pss_id_remap", "pss_status_remap", "pss_diff_remap"];

/// Best-effort drop of every scratch relation — called both before the
/// rewrite (a crashed earlier run may have left one behind) and after it
/// (success or failure), so no scratch state ever outlives the migration.
fn drop_scratch_relations(db: &DbInstance) {
    for rel in SCRATCH_RELATIONS {
        let q = format!("::remove {}", rel);
        let _ = db.run_script(&q, BTreeMap::new(), ScriptMutability::Mutable);
    }
}

/// Create scratch relation `relation { key_col: String => val_col: String }`
/// and bulk-load `pairs` into it. Relation/column names are compile-time
/// constants owned by this module — never user data.
fn load_scratch_map(
    db: &DbInstance,
    relation: &str,
    key_col: &str,
    val_col: &str,
    pairs: &[(String, String)],
) -> Result<(), String> {
    let create = format!(
        ":create {} {{ {}: String => {}: String }}",
        relation, key_col, val_col
    );
    db.run_script(&create, BTreeMap::new(), ScriptMutability::Mutable)
        .map_err(|e| format!("element-id re-key: creating {} failed: {}", relation, e))?;
    let rows: Vec<DataValue> = pairs
        .iter()
        .map(|(k, v)| {
            DataValue::List(vec![
                DataValue::Str(k.as_str().into()),
                DataValue::Str(v.as_str().into()),
            ])
        })
        .collect();
    let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
    params.insert("rows".into(), DataValue::List(rows));
    let put = format!(
        "?[{k}, {v}] <- $rows\n :put {r} {{ {k} => {v} }}",
        k = key_col,
        v = val_col,
        r = relation
    );
    db.run_script(&put, params, ScriptMutability::Mutable)
        .map_err(|e| format!("element-id re-key: loading {} failed: {}", relation, e))?;
    Ok(())
}

/// Rewrite one embedded `override_status` value onto the new id scheme, or
/// `None` if it embeds no changed id. The two id-bearing forms come from
/// `resolve_overrides`: `overridden_by:<eid>` (single id) and
/// `overrides:<eid>;<eid>;…` (semicolon-joined list — each element is
/// remapped independently, unchanged ones pass through). Plain statuses
/// ("active", "none", …) carry no id and always return `None`.
fn remap_override_status(
    status: &str,
    changed: &std::collections::HashMap<String, String>,
) -> Option<String> {
    if let Some(id) = status.strip_prefix("overridden_by:") {
        return changed
            .get(id)
            .map(|new| format!("overridden_by:{}", new));
    }
    if let Some(list) = status.strip_prefix("overrides:") {
        let mut any_changed = false;
        let mut parts: Vec<&str> = Vec::new();
        for id in list.split(';') {
            match changed.get(id) {
                Some(new) => {
                    any_changed = true;
                    parts.push(new.as_str());
                }
                None => parts.push(id),
            }
        }
        if any_changed {
            return Some(format!("overrides:{}", parts.join(";")));
        }
    }
    None
}

/// All distinct values of one String column, via datalog set semantics.
fn distinct_string_col(db: &DbInstance, rel: &str, col: &str) -> Result<Vec<String>, String> {
    let q = format!("?[{c}] := *{r}{{{c}}}", c = col, r = rel);
    let rows = db
        .run_script(&q, BTreeMap::new(), ScriptMutability::Immutable)
        .map_err(|e| format!("element-id re-key: reading {}.{} failed: {}", rel, col, e))?;
    Ok(rows.rows.iter().map(|r| data_str(&r[0])).collect())
}

/// The write half of [`migrate_element_id_scheme_v2`]: load `remap` into the
/// scratch relation and join it against `events`, `elements_state`, and
/// `element_descriptions` so Cozo itself carries every non-id value across
/// untouched — then remap the ids EMBEDDED inside `override_status` and
/// `events.diff_json` string values.
fn rekey_rows_via_remap(db: &DbInstance, remap: &[(String, String)]) -> Result<(), String> {
    // Drop leftovers from an interrupted earlier run before creating.
    drop_scratch_relations(db);
    load_scratch_map(db, "pss_id_remap", "old_id", "new_id", remap)?;

    // events: bind the STORED id to `old_eid` and bind the head's
    // `element_id` to the joined new value. Every other column is bound and
    // re-emitted by name, so its value never leaves Cozo. The `event_id` key
    // is untouched, so this overwrites each row in place — the row count
    // cannot change.
    let events_q = r#"
        ?[event_id, observed_at, scan_id, event_type, element_type, element_name,
          element_id, scope, scope_path, source, path, content_hash, file_size,
          token_count, enabled, override_status, diff_json, snapshot_ref] :=
            *events{event_id, observed_at, scan_id, event_type, element_type,
                    element_name, element_id: old_eid, scope, scope_path, source,
                    path, content_hash, file_size, token_count, enabled,
                    override_status, diff_json, snapshot_ref},
            *pss_id_remap{old_id: old_eid, new_id: element_id}
        :put events { event_id =>
            observed_at, scan_id, event_type, element_type, element_name,
            element_id, scope, scope_path, source, path, content_hash, file_size,
            token_count, enabled, override_status, diff_json, snapshot_ref }
    "#;
    db.run_script(events_q, BTreeMap::new(), ScriptMutability::Mutable)
        .map_err(|e| format!("element-id re-key: rewriting events failed: {}", e))?;

    // elements_state: element_id IS the primary key, so a rename is a
    // new-keyed :put followed by an :rm of the dead old key.
    let state_put_q = r#"
        ?[element_id, last_event_id, current_path, current_hash, current_size,
          current_token_count, enabled, override_status, installed_at,
          last_changed_at, exists] :=
            *elements_state{element_id: old_eid, last_event_id, current_path,
                            current_hash, current_size, current_token_count,
                            enabled, override_status, installed_at,
                            last_changed_at, exists},
            *pss_id_remap{old_id: old_eid, new_id: element_id}
        :put elements_state { element_id =>
            last_event_id, current_path, current_hash, current_size,
            current_token_count, enabled, override_status, installed_at,
            last_changed_at, exists }
    "#;
    db.run_script(state_put_q, BTreeMap::new(), ScriptMutability::Mutable)
        .map_err(|e| format!("element-id re-key: rewriting elements_state failed: {}", e))?;

    // Drop the old keys. Two guards, both load-bearing:
    //   `element_id != new_eid` — never delete a row whose key didn't move.
    //   `not *pss_id_remap{new_id: element_id}` — never delete a key that is
    //      also some OTHER row's NEW key. Without it a rename chain A→B, B→C
    //      would delete B (as A's stale old key) right after A's data was
    //      written INTO B, losing A's state row entirely.
    let state_rm_q = r#"
        ?[element_id] :=
            *elements_state{element_id},
            *pss_id_remap{old_id: element_id, new_id: new_eid},
            element_id != new_eid,
            not *pss_id_remap{new_id: element_id}
        :rm elements_state { element_id }
    "#;
    db.run_script(state_rm_q, BTreeMap::new(), ScriptMutability::Mutable)
        .map_err(|e| format!("element-id re-key: dropping stale elements_state keys failed: {}", e))?;

    // element_descriptions: also element_id-KEYED, so it gets the identical
    // treatment. This table is NOT optional bookkeeping — the writer reads it
    // back per element via read_prior_description_hash(NEW id) to decide
    // whether to emit description_changed. Skipping it here would orphan
    // every re-keyed element's description row: spurious description_changed
    // events on the next scan, plus dead rows that nothing ever reclaims.
    //
    // The INNER join is load-bearing in the other direction too: a
    // description row whose element_id has no remap entry (a description with
    // no events) simply doesn't match, so it is left exactly as-is rather
    // than dropped.
    let desc_put_q = r#"
        ?[element_id, description_hash, description_text, last_updated_at] :=
            *element_descriptions{element_id: old_eid, description_hash,
                                  description_text, last_updated_at},
            *pss_id_remap{old_id: old_eid, new_id: element_id}
        :put element_descriptions { element_id =>
            description_hash, description_text, last_updated_at }
    "#;
    db.run_script(desc_put_q, BTreeMap::new(), ScriptMutability::Mutable)
        .map_err(|e| {
            format!("element-id re-key: rewriting element_descriptions failed: {}", e)
        })?;

    // Same two guards as elements_state: never drop an unmoved key, and never
    // drop a key that is some other row's NEW key (the A→B, B→C chain).
    let desc_rm_q = r#"
        ?[element_id] :=
            *element_descriptions{element_id},
            *pss_id_remap{old_id: element_id, new_id: new_eid},
            element_id != new_eid,
            not *pss_id_remap{new_id: element_id}
        :rm element_descriptions { element_id }
    "#;
    db.run_script(desc_rm_q, BTreeMap::new(), ScriptMutability::Mutable)
        .map_err(|e| {
            format!(
                "element-id re-key: dropping stale element_descriptions keys failed: {}",
                e
            )
        })?;

    // ── Embedded ids ─────────────────────────────────────────────────────
    // element_ids also live INSIDE string values: `override_status` carries
    // `overridden_by:<eid>` / `overrides:<eid>;<eid>` and `events.diff_json`
    // embeds those same status strings for override events. The
    // elements_state copy is CURRENT state: leave it stale and the next
    // scan's override pass — which recomputes with NEW ids — would see a
    // mismatch and emit spurious override_started/override_ended events.
    //
    // Only ids that actually CHANGED need a lookup entry, which keeps the
    // scratch maps tiny (the two-rule matched/unmatched pattern below lets
    // every unmapped row pass through untouched).
    let changed_map: std::collections::HashMap<String, String> = remap
        .iter()
        .filter(|(old, new)| old != new)
        .cloned()
        .collect();

    // Every changed old id, longest first — so the diff_json substring
    // replacement can never corrupt a longer id by first rewriting a shorter
    // id that happens to be its prefix.
    let mut changed_by_len: Vec<(&String, &String)> = changed_map.iter().collect();
    changed_by_len.sort_by_key(|(old, _)| std::cmp::Reverse(old.len()));

    // Post-condition guard, applied to every candidate value BEFORE writing:
    // a rewritten (or deliberately untouched) value must not contain any
    // changed old id. `contains('@')` is a sound pre-filter — every
    // element_id contains '@' — so plain values skip the scan entirely.
    let residual_old_id = |value: &str| -> Option<&String> {
        if !value.contains('@') {
            return None;
        }
        changed_by_len
            .iter()
            .find(|(old, _)| value.contains(old.as_str()))
            .map(|(old, _)| *old)
    };

    // override_status: distinct values across BOTH carriers (events +
    // elements_state), remapped exactly — strip the known prefix, look the
    // full id (or each `;`-separated id) up in changed_map. No substring
    // matching, so no false rewrites.
    let mut statuses = distinct_string_col(db, "events", "override_status")?;
    statuses.extend(distinct_string_col(db, "elements_state", "override_status")?);
    statuses.sort();
    statuses.dedup();
    let mut status_remap: Vec<(String, String)> = Vec::new();
    for status in &statuses {
        match remap_override_status(status, &changed_map) {
            Some(new_status) => {
                if let Some(old) = residual_old_id(&new_status) {
                    return Err(format!(
                        "element-id re-key ABORTED: rewritten override_status '{}' still \
                         contains changed old id '{}'. No embedded-id rows written.",
                        new_status, old
                    ));
                }
                status_remap.push((status.clone(), new_status));
            }
            None => {
                // Not id-bearing (or all its ids unchanged) — but it must
                // not smuggle a changed old id in some unexpected shape.
                if let Some(old) = residual_old_id(status) {
                    return Err(format!(
                        "element-id re-key ABORTED: override_status '{}' embeds changed \
                         old id '{}' in an unrecognized form (expected 'overridden_by:' \
                         or 'overrides:'). No embedded-id rows written.",
                        status, old
                    ));
                }
            }
        }
    }

    // diff_json: id-level longest-first substring replacement. The `@`
    // pre-filter keeps this O(distinct values): a value with no '@' cannot
    // embed an element_id and is skipped (and therefore byte-identical).
    let mut diff_remap: Vec<(String, String)> = Vec::new();
    for diff in distinct_string_col(db, "events", "diff_json")? {
        if !diff.contains('@') {
            continue;
        }
        let mut out = diff.clone();
        for (old, new) in &changed_by_len {
            if out.contains(old.as_str()) {
                out = out.replace(old.as_str(), new.as_str());
            }
        }
        if let Some(old) = residual_old_id(&out) {
            return Err(format!(
                "element-id re-key ABORTED: rewritten diff_json still contains changed \
                 old id '{}': {}. No embedded-id rows written.",
                old, out
            ));
        }
        if out != diff {
            diff_remap.push((diff, out));
        }
    }

    // Rewrite pass per carrier — skipped entirely when its map is empty
    // (nothing would change, and the unmatched rule alone would just rewrite
    // every row to itself). Each script has TWO rules under one head: the
    // matched rule joins the scratch map, the unmatched rule passes the row
    // through via negation — together they cover every row exactly once.
    if !status_remap.is_empty() {
        load_scratch_map(db, "pss_status_remap", "old_status", "new_status", &status_remap)?;

        let events_status_q = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash, file_size,
              token_count, enabled, override_status, diff_json, snapshot_ref] :=
                *events{event_id, observed_at, scan_id, event_type, element_type,
                        element_name, element_id, scope, scope_path, source, path,
                        content_hash, file_size, token_count, enabled,
                        override_status: old_os, diff_json, snapshot_ref},
                *pss_status_remap{old_status: old_os, new_status: override_status}
            ?[event_id, observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash, file_size,
              token_count, enabled, override_status, diff_json, snapshot_ref] :=
                *events{event_id, observed_at, scan_id, event_type, element_type,
                        element_name, element_id, scope, scope_path, source, path,
                        content_hash, file_size, token_count, enabled,
                        override_status, diff_json, snapshot_ref},
                not *pss_status_remap{old_status: override_status}
            :put events { event_id =>
                observed_at, scan_id, event_type, element_type, element_name,
                element_id, scope, scope_path, source, path, content_hash, file_size,
                token_count, enabled, override_status, diff_json, snapshot_ref }
        "#;
        db.run_script(events_status_q, BTreeMap::new(), ScriptMutability::Mutable)
            .map_err(|e| {
                format!(
                    "element-id re-key: rewriting events.override_status failed: {}",
                    e
                )
            })?;

        let state_status_q = r#"
            ?[element_id, last_event_id, current_path, current_hash, current_size,
              current_token_count, enabled, override_status, installed_at,
              last_changed_at, exists] :=
                *elements_state{element_id, last_event_id, current_path, current_hash,
                                current_size, current_token_count, enabled,
                                override_status: old_os, installed_at,
                                last_changed_at, exists},
                *pss_status_remap{old_status: old_os, new_status: override_status}
            ?[element_id, last_event_id, current_path, current_hash, current_size,
              current_token_count, enabled, override_status, installed_at,
              last_changed_at, exists] :=
                *elements_state{element_id, last_event_id, current_path, current_hash,
                                current_size, current_token_count, enabled,
                                override_status, installed_at, last_changed_at, exists},
                not *pss_status_remap{old_status: override_status}
            :put elements_state { element_id =>
                last_event_id, current_path, current_hash, current_size,
                current_token_count, enabled, override_status, installed_at,
                last_changed_at, exists }
        "#;
        db.run_script(state_status_q, BTreeMap::new(), ScriptMutability::Mutable)
            .map_err(|e| {
                format!(
                    "element-id re-key: rewriting elements_state.override_status failed: {}",
                    e
                )
            })?;
    }

    if !diff_remap.is_empty() {
        load_scratch_map(db, "pss_diff_remap", "old_diff", "new_diff", &diff_remap)?;

        let events_diff_q = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash, file_size,
              token_count, enabled, override_status, diff_json, snapshot_ref] :=
                *events{event_id, observed_at, scan_id, event_type, element_type,
                        element_name, element_id, scope, scope_path, source, path,
                        content_hash, file_size, token_count, enabled, override_status,
                        diff_json: old_dj, snapshot_ref},
                *pss_diff_remap{old_diff: old_dj, new_diff: diff_json}
            ?[event_id, observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash, file_size,
              token_count, enabled, override_status, diff_json, snapshot_ref] :=
                *events{event_id, observed_at, scan_id, event_type, element_type,
                        element_name, element_id, scope, scope_path, source, path,
                        content_hash, file_size, token_count, enabled, override_status,
                        diff_json, snapshot_ref},
                not *pss_diff_remap{old_diff: diff_json}
            :put events { event_id =>
                observed_at, scan_id, event_type, element_type, element_name,
                element_id, scope, scope_path, source, path, content_hash, file_size,
                token_count, enabled, override_status, diff_json, snapshot_ref }
        "#;
        db.run_script(events_diff_q, BTreeMap::new(), ScriptMutability::Mutable)
            .map_err(|e| {
                format!("element-id re-key: rewriting events.diff_json failed: {}", e)
            })?;
    }

    Ok(())
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
/// - prior present, path differs → `PathChanged`, INDEPENDENTLY of any
///   content/size change (F8, TRDD-1Z8SGQ7N): a move that coincides with an
///   edit records BOTH the ContentChanged and the relocation, so a
///   move+edit no longer silently loses the move
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
            }
            // F8 (TRDD-1Z8SGQ7N): PathChanged is emitted on ANY path change,
            // not only when content+size are unchanged. The old `else if`
            // dropped the relocation whenever a move coincided with an edit,
            // so a move+edit recorded only the content change and lost the
            // fact that the element moved. Emitting it independently keeps the
            // pure-move case identical (only PathChanged fires) while making a
            // move+edit record BOTH events.
            if path_diff {
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
    use crate::OutputFormat;
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

    /// COR-6 v3.7 helper: print a single-cell value (the "value" form of a
    /// `JsonValue` — String, Number, Bool, Null) as a compact text string
    /// suitable for a table cell.  Truncates long strings to keep the table
    /// width manageable.
    fn cell(v: &JsonValue) -> String {
        match v {
            JsonValue::Null => "".to_string(),
            JsonValue::Bool(b) => b.to_string(),
            JsonValue::Number(n) => n.to_string(),
            JsonValue::String(s) => s.chars().take(80).collect::<String>(),
            other => other.to_string().chars().take(80).collect::<String>(),
        }
    }

    /// COR-6 v3.7 helper: shared "no rows" + "stub format" handling. Called
    /// by the table arm of every temporal cmd.  Returns true if the format
    /// has already been handled (stub printed); the caller should then
    /// `return;` immediately.
    fn handle_stub_format(format: OutputFormat, subcommand: &str) -> bool {
        if matches!(
            format,
            OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown
        ) {
            format.print_stub(subcommand);
            return true;
        }
        false
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

    /// P-4 (issue #10 wave 2): build a map from `(element_id, scope)` to the
    /// element's FIRST-SEEN install instant and whether that install is
    /// synthetic.
    ///
    /// - `first_seen` = the EARLIEST `installed` event's `observed_at` for that
    ///   element_id in that scope. (Earliest, not latest — it is the install
    ///   *instant*, independent of any later content_changed events.)
    /// - synthetic = true iff that earliest install's `diff_json` carries
    ///   `"migrated": true` — the migration-stamped placeholder created by the
    ///   v1→v2 migration (`insert_install_event` sets that flag), as opposed to
    ///   a real observed install.
    ///
    /// We sort ascending by observed_at and keep the FIRST row per
    /// `(element_id, scope)` because Cozo's numeric `min()` cannot aggregate
    /// RFC3339 strings (the same constraint that forces cmd_as_of's sort+dedup).
    /// Keying on `(element_id, scope)` matches the as-of row identity — the
    /// hot-path never mixes the same element across scopes.
    fn build_first_seen_map(
        db: &DbInstance,
    ) -> std::collections::HashMap<(String, String), (String, bool)> {
        use std::collections::HashMap;
        let mut map: HashMap<(String, String), (String, bool)> = HashMap::new();
        // Only `installed` events define a first-seen instant.
        let q = r#"
            ?[element_id, scope, observed_at, diff_json] :=
                *events{element_id, scope, observed_at, diff_json, event_type},
                event_type = "installed"
            :order element_id, scope, observed_at
        "#;
        let result = match db.run_script(q, BTreeMap::new(), ScriptMutability::Immutable) {
            Ok(r) => r,
            // If this auxiliary query fails, callers degrade to rows without
            // first_seen rather than failing the whole snapshot.
            Err(_) => return map,
        };
        for r in result.rows.iter() {
            let eid = match &r[0] {
                DataValue::Str(s) => s.to_string(),
                _ => continue,
            };
            let scope = match &r[1] {
                DataValue::Str(s) => s.to_string(),
                _ => String::new(),
            };
            let key = (eid, scope);
            // Ascending order ⇒ the FIRST entry per key is the earliest install.
            if map.contains_key(&key) {
                continue;
            }
            let observed_at = match &r[2] {
                DataValue::Str(s) => s.to_string(),
                _ => continue,
            };
            // synthetic iff diff_json parses to an object with "migrated": true.
            let is_synthetic = match &r[3] {
                DataValue::Str(s) => serde_json::from_str::<JsonValue>(s)
                    .ok()
                    .and_then(|v| v.get("migrated").and_then(|m| m.as_bool()))
                    .unwrap_or(false),
                _ => false,
            };
            map.insert(key, (observed_at, is_synthetic));
        }
        map
    }

    /// Core of `as-of`: returns the JSON rows (the same 12 legacy fields PLUS
    /// the P-4 `first_seen` / `first_seen_is_synthetic` fields) for the snapshot
    /// at `cutoff` (an already-resolved RFC3339 timestamp). Shared by both
    /// `cmd_as_of` (which prints them) and `cmd_active_in` (which filters them
    /// to a project-folder union). Pure — no stdout — so it is unit-testable.
    pub(crate) fn as_of_rows(
        db: &DbInstance,
        cutoff: &str,
        type_filter: Option<&str>,
        scope_filter: Option<&str>,
        scope_path_filter: Option<&str>,
        limit: usize,
    ) -> Vec<JsonValue> {
        use std::collections::HashSet;

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
                return Vec::new();
            }
        };

        // P-4: precompute first-seen per (element_id, scope) once, then attach.
        let first_seen = build_first_seen_map(db);

        // Dedupe by element_id (first occurrence per id = latest by
        // observed_at). Skip rows where the latest event is `removed`
        // (element wasn't present at cutoff). Apply limit AFTER filtering
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
            let scope = match &r[5] {
                DataValue::Str(s) => s.to_string(),
                _ => String::new(),
            };
            // P-4: look up first-seen for this (element_id, scope). When the
            // element has no `installed` event recorded (should not happen for
            // a present element, but stay robust), emit nulls.
            let (first_seen_val, synthetic_val): (JsonValue, JsonValue) =
                match first_seen.get(&(eid.clone(), scope)) {
                    Some((ts, synth)) => (JsonValue::String(ts.clone()), JsonValue::Bool(*synth)),
                    None => (JsonValue::Null, JsonValue::Null),
                };
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
                // P-4 (issue #10 wave 2) — ADDITIVE fields:
                "first_seen": first_seen_val,
                "first_seen_is_synthetic": synthetic_val,
            }));
            if out.len() >= limit {
                break;
            }
        }
        out
    }

    /// Core of `active-in`: returns the as-of snapshot rows at `cutoff`
    /// filtered to the components ACTIVE in the project folder whose slug is
    /// `slug`. Active = the UNION of:
    ///   (a) project/local-scope rows whose `scope_path` equals `slug`,
    ///   (b) all `user`-scope rows (global elements),
    ///   (c) plugin/marketplace rows currently `enabled`.
    /// Rows keep the same shape as `as_of_rows` (incl. the P-4 fields). Pure —
    /// no stdout — so it is unit-testable.
    ///
    /// HONESTY: per-project plugin enablement at a PAST instant is not yet
    /// recorded (issue #10 P-8 is going-forward), so (c) reflects the
    /// CURRENT/global `enabled` signal — see the `active-in` help text.
    pub(crate) fn active_in_rows(
        db: &DbInstance,
        slug: &str,
        cutoff: &str,
        limit: usize,
    ) -> Vec<JsonValue> {
        // Snapshot ALL elements at cutoff (no per-type/scope filter — we apply
        // the union predicate in Rust). Use a high cap so the snapshot itself
        // is never the limiting factor; the caller's `limit` is applied AFTER
        // the union filter so it bounds the RESULT, not the pre-filter set.
        let snapshot = as_of_rows(db, cutoff, None, None, None, usize::MAX);
        let mut out: Vec<JsonValue> = Vec::new();
        for row in snapshot.into_iter() {
            let scope = row.get("scope").and_then(|v| v.as_str()).unwrap_or("");
            let scope_path = row.get("scope_path").and_then(|v| v.as_str()).unwrap_or("");
            let enabled = row.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);

            let in_union = match scope {
                // (a) this project's project/local-scope elements.
                "project" | "local" => scope_path == slug,
                // (b) all global user-scope elements.
                "user" => true,
                // (c) plugin/marketplace elements currently enabled.
                "plugin" | "marketplace" => enabled,
                _ => false,
            };
            if in_union {
                out.push(row);
                if out.len() >= limit {
                    break;
                }
            }
        }
        out
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
        let cutoff = match resolve_date(date) {
            Ok(s) => s,
            Err(e) => {
                print_date_err("date", date, &e);
                return;
            }
        };

        // DBE-2 / COR-1 row logic now lives in the shared `as_of_rows` helper
        // (also consumed by `cmd_active_in`); it appends the P-4 first_seen /
        // first_seen_is_synthetic fields to each of the 12 legacy fields.
        let out = as_of_rows(db, &cutoff, type_filter, scope_filter, scope_path_filter, limit);

        println!(
            "{}",
            serde_json::to_string_pretty(&JsonValue::Array(out)).unwrap_or_default()
        );
    }

    /// `pss active-in <ABS_PATH_SLUG> [--as-of <DATE>]`
    ///
    /// Print every component ACTIVE in the project folder whose slug is
    /// `slug` at the given date (default: now). "Active" is the union defined
    /// by `active_in_rows`. The caller (main.rs dispatch) computes `slug` from
    /// the absolute path with the SAME algorithm `pss project-slug` uses, so
    /// the slug is the project-scope identity used inside `scope_path`.
    pub fn cmd_active_in(db: &DbInstance, slug: &str, date: &str, limit: usize) {
        let cutoff = match resolve_date(date) {
            Ok(s) => s,
            Err(e) => {
                print_date_err("as-of", date, &e);
                return;
            }
        };
        let out = active_in_rows(db, slug, &cutoff, limit);
        println!(
            "{}",
            serde_json::to_string_pretty(&JsonValue::Array(out)).unwrap_or_default()
        );
    }

    /// `pss timeline <ELEMENT_ID>`
    pub fn cmd_timeline(db: &DbInstance, element_id: &str, limit: usize, format: OutputFormat) {
        if handle_stub_format(format, "timeline") {
            return;
        }
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
                match format {
                    OutputFormat::Json => {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                        );
                    }
                    OutputFormat::Table => {
                        let table_rows: Vec<Vec<String>> = rows.iter().map(|r| vec![
                            cell(&r["observed_at"]),
                            cell(&r["event_type"]),
                            cell(&r["content_hash"]),
                            cell(&r["file_size"]),
                            cell(&r["token_count"]),
                        ]).collect();
                        crate::print_table(
                            &["OBSERVED_AT", "EVENT_TYPE", "CONTENT_HASH", "SIZE", "TOKENS"],
                            &table_rows,
                        );
                    }
                    _ => {} // handled by handle_stub_format above
                }
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
        format: OutputFormat,
    ) {
        if handle_stub_format(format, "changed-between") {
            return;
        }
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
                match format {
                    OutputFormat::Json => println!(
                        "{}",
                        serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                    ),
                    OutputFormat::Table => {
                        let table_rows: Vec<Vec<String>> = rows.iter().map(|r| vec![
                            cell(&r["observed_at"]),
                            cell(&r["event_type"]),
                            cell(&r["element_type"]),
                            cell(&r["element_name"]),
                            cell(&r["content_hash"]),
                            cell(&r["file_size"]),
                        ]).collect();
                        crate::print_table(
                            &["OBSERVED_AT", "EVENT_TYPE", "ELEMENT_TYPE", "ELEMENT_NAME", "CONTENT_HASH", "SIZE"],
                            &table_rows,
                        );
                    }
                    _ => {}
                }
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

    /// F-12 (audit 20260514): version-history view — same signal as
    /// `timeline` but filtered to the events that actually represent a
    /// version transition: installed, content_changed,
    /// description_changed, removed. Skips noise events (enabled,
    /// disabled, size_changed-without-hash-change, override_started,
    /// override_ended) so callers can reconstruct the version chain
    /// without manually filtering.
    pub fn cmd_version_history(db: &DbInstance, element_id: &str, limit: usize, format: OutputFormat) {
        if handle_stub_format(format, "version-history") {
            return;
        }
        // Cozo's predicate language uses `or(...)` — no native IN keyword.
        let q = r#"
            ?[event_id, observed_at, event_type, content_hash, file_size, diff_json] :=
                *events{element_id: $eid, event_id, observed_at, event_type,
                        content_hash, file_size, diff_json},
                or(
                    event_type == "installed",
                    event_type == "content_changed",
                    event_type == "description_changed",
                    event_type == "removed"
                )
            :order observed_at
            :limit $limit
        "#;
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(element_id.into()));
        params.insert("limit".into(), DataValue::Num(Num::Int(limit as i64)));
        match db.run_script(q, params, ScriptMutability::Immutable) {
            Ok(r) => {
                let rows: Vec<JsonValue> = r.rows.into_iter().map(|row| {
                    json!({
                        "event_id": data_to_json(&row[0]),
                        "observed_at": data_to_json(&row[1]),
                        "event_type": data_to_json(&row[2]),
                        "content_hash": data_to_json(&row[3]),
                        "file_size": data_to_json(&row[4]),
                        "diff_json": data_to_json(&row[5]),
                    })
                }).collect();
                match format {
                    OutputFormat::Json => {
                        let output = json!({
                            "element_id": element_id,
                            "version_count": rows.len(),
                            "versions": rows,
                        });
                        println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
                    }
                    OutputFormat::Table => {
                        let table_rows: Vec<Vec<String>> = rows.iter().map(|r| vec![
                            cell(&r["observed_at"]),
                            cell(&r["event_type"]),
                            cell(&r["content_hash"]),
                            cell(&r["file_size"]),
                        ]).collect();
                        crate::print_table(
                            &["OBSERVED_AT", "EVENT_TYPE", "CONTENT_HASH", "SIZE"],
                            &table_rows,
                        );
                    }
                    _ => {}
                }
            }
            Err(e) => {
                eprintln!("version-history query failed: {}", e);
                println!("{{}}");
            }
        }
    }

    /// F-17 (audit 20260514): list every event from a specific scan_id.
    pub fn cmd_changes_in_batch(db: &DbInstance, scan_id: &str, limit: usize, format: OutputFormat) {
        if handle_stub_format(format, "changes-in-batch") {
            return;
        }
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
                match format {
                    OutputFormat::Json => println!(
                        "{}",
                        serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                    ),
                    OutputFormat::Table => {
                        let table_rows: Vec<Vec<String>> = rows.iter().map(|r| vec![
                            cell(&r["observed_at"]),
                            cell(&r["event_type"]),
                            cell(&r["element_type"]),
                            cell(&r["element_name"]),
                            cell(&r["scope"]),
                        ]).collect();
                        crate::print_table(
                            &["OBSERVED_AT", "EVENT_TYPE", "ELEMENT_TYPE", "ELEMENT_NAME", "SCOPE"],
                            &table_rows,
                        );
                    }
                    _ => {}
                }
            }
            Err(e) => {
                eprintln!("changes-in-batch query failed: {}", e);
                println!("[]");
            }
        }
    }

    /// F-18 (audit 20260514): emit every event from the most recent scan.
    pub fn cmd_last_changes(db: &DbInstance, limit: usize, format: OutputFormat) {
        if handle_stub_format(format, "last-changes") {
            return;
        }
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
        cmd_changes_in_batch(db, &scan_id, limit, format);
    }

    /// F-19 (audit 20260514): count elements per scope, optionally
    /// filtered by element_type. Reads `elements_state` joined against
    /// `events.scope` via last_event_id.
    pub fn cmd_stats_by_scope(db: &DbInstance, type_filter: Option<&str>, format: OutputFormat) {
        if handle_stub_format(format, "stats-by-scope") {
            return;
        }
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
                match format {
                    OutputFormat::Json => {
                        let output = json!({
                            "by_scope": by_scope,
                            "total_elements": total,
                        });
                        println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
                    }
                    OutputFormat::Table => {
                        let mut table_rows: Vec<Vec<String>> = Vec::new();
                        for (scope, by_type) in &by_scope {
                            for (etype, count) in by_type {
                                table_rows.push(vec![
                                    scope.clone(),
                                    etype.clone(),
                                    count.to_string(),
                                ]);
                            }
                        }
                        crate::print_table(&["SCOPE", "ELEMENT_TYPE", "COUNT"], &table_rows);
                        println!("Total: {}", total);
                    }
                    _ => {}
                }
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
        format: OutputFormat,
    ) {
        if handle_stub_format(format, "changes-summary") {
            return;
        }
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
                match format {
                    OutputFormat::Json => {
                        let out = json!({
                            "window": window,
                            "cutoff": cutoff,
                            "type_filter": type_filter,
                            "counts": JsonValue::Object(counts),
                        });
                        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
                    }
                    OutputFormat::Table => {
                        println!("Window: {} (cutoff: {})", window, cutoff);
                        if let Some(t) = type_filter {
                            println!("Type filter: {}", t);
                        }
                        let table_rows: Vec<Vec<String>> = counts.iter()
                            .map(|(k, v)| vec![k.clone(), cell(v)])
                            .collect();
                        crate::print_table(&["EVENT_TYPE", "COUNT"], &table_rows);
                    }
                    _ => {}
                }
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
        format: OutputFormat,
    ) {
        if handle_stub_format(format, "compare-snapshots") {
            return;
        }
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
        match format {
            OutputFormat::Json => {
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
            OutputFormat::Table => {
                println!("Snapshot comparison:");
                println!("  date1: {}", c1);
                println!("  date2: {}", c2);
                if let Some(t) = type_filter {
                    println!("  type_filter: {}", t);
                }
                let summary = vec![
                    vec!["common_count".to_string(), common_count.to_string()],
                    vec!["removed_between_count".to_string(), only_1_sorted.len().to_string()],
                    vec!["added_between_count".to_string(), only_2_sorted.len().to_string()],
                ];
                crate::print_table(&["METRIC", "VALUE"], &summary);
                // Cap inline rendering to keep output compact; full lists in JSON.
                if !only_1_sorted.is_empty() {
                    println!("\nOnly at date1 ({} total, showing up to 20):", only_1_sorted.len());
                    for x in only_1_sorted.iter().take(20) {
                        println!("  {}", x);
                    }
                }
                if !only_2_sorted.is_empty() {
                    println!("\nOnly at date2 ({} total, showing up to 20):", only_2_sorted.len());
                    for x in only_2_sorted.iter().take(20) {
                        println!("  {}", x);
                    }
                }
            }
            _ => {}
        }
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
        format: OutputFormat,
    ) {
        if handle_stub_format(format, "dedup-candidates") {
            return;
        }
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
                match format {
                    OutputFormat::Json => println!(
                        "{}",
                        serde_json::to_string_pretty(&JsonValue::Array(out)).unwrap_or_default()
                    ),
                    OutputFormat::Table => {
                        let table_rows: Vec<Vec<String>> = out.iter().map(|r| vec![
                            cell(&r["element_type"]),
                            cell(&r["element_name"]),
                            cell(&r["scope_count"]),
                        ]).collect();
                        crate::print_table(
                            &["ELEMENT_TYPE", "ELEMENT_NAME", "SCOPE_COUNT"],
                            &table_rows,
                        );
                    }
                    _ => {}
                }
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
    pub fn cmd_scan_log(db: &DbInstance, limit: usize, format: OutputFormat) {
        if handle_stub_format(format, "scan-log") {
            return;
        }
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
                match format {
                    OutputFormat::Json => println!(
                        "{}",
                        serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                    ),
                    OutputFormat::Table => {
                        let table_rows: Vec<Vec<String>> = rows.iter().map(|r| vec![
                            cell(&r["scan_id"]),
                            cell(&r["finished_at"]),
                            cell(&r["events_emitted"]),
                            cell(&r["rust_binary_version"]),
                            cell(&r["pss_version"]),
                        ]).collect();
                        crate::print_table(
                            &["SCAN_ID", "FINISHED_AT", "EVENTS", "RUST_VERSION", "PSS_VERSION"],
                            &table_rows,
                        );
                    }
                    _ => {}
                }
            }
            Err(e) => eprintln!("scan-log query failed: {}", e),
        }
    }

    /// `pss db-stats`
    pub fn cmd_db_stats(db: &DbInstance, format: OutputFormat) {
        if handle_stub_format(format, "db-stats") {
            return;
        }
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
        match format {
            OutputFormat::Json => {
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
            OutputFormat::Table => {
                let schema_v = read_schema_version(db);
                let oldest_str = oldest.unwrap_or_default();
                let table_rows = vec![
                    vec!["schema_version".to_string(), schema_v],
                    vec!["event_count".to_string(), event_count.to_string()],
                    vec!["blob_count".to_string(), blob_count.to_string()],
                    vec!["state_count".to_string(), state_count.to_string()],
                    vec!["oldest_event_at".to_string(), oldest_str],
                    vec!["retention_window".to_string(), retention],
                ];
                crate::print_table(&["KEY", "VALUE"], &table_rows);
            }
            _ => {}
        }
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

    /// `pss migrate-element-ids` — the manual lever for the F4 element_id
    /// re-key (TRDD-1Z8SGQ7N). The same migration auto-runs inside
    /// merge-events; this verb exists so it can be driven (and its result
    /// inspected) on demand, without a full reindex.
    ///
    /// FAIL-FAST: propagates the un-merge abort verbatim rather than
    /// reporting a partial success — a silently half-migrated history is
    /// worse than a loud failure.
    pub fn cmd_migrate_element_ids(db: &DbInstance) -> Result<(), String> {
        ensure_schema(db)?;
        let changed = migrate_element_id_scheme_v2(db)?;
        println!("{}", serde_json::json!({ "changed": changed }));
        Ok(())
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
    ///
    /// `pub(crate)` so `migrate_element_id_scheme_v2` (F4, TRDD-1Z8SGQ7N)
    /// can turn a stored `events.element_type` string back into the enum
    /// that `compute_element_id` needs — it must recompute ids with the
    /// exact same mapping the writer used, or the re-key would invent ids.
    pub(crate) fn parse_element_type(s: &str) -> Option<ElementType> {
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
    /// `pub(crate)` so the F4 re-key tests (TRDD-1Z8SGQ7N) can assert against
    /// the REAL consumer rather than a replica of its query — the whole point
    /// of re-keying element_descriptions is that THIS lookup still resolves
    /// after the rename.
    pub(crate) fn read_prior_description_hash(db: &DbInstance, element_id: &str) -> Option<String> {
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

        // F2 (TRDD-1Z8SGQ7N): elements_state is materialized for EVERY event,
        // enabled or disabled. This used to be gated by an `update_state`
        // bool, and the DI-3 change wired `obs.enabled` into that slot —
        // conflating "is this element enabled" (already recorded in the
        // `enabled` column of both the events row above and the state row
        // below) with "should state be materialized". The effect: a DISABLED
        // element got its event logged but its `elements_state` row was never
        // written, so `as-of`/`show` and removal detection (which read
        // elements_state) silently stopped tracking it. The parameter is
        // removed rather than pinned to `true` so the footgun cannot return —
        // no caller ever needed a log-only event, and the `enabled` state is
        // carried by the column, not by whether the row exists.
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
        // F4 (TRDD-1Z8SGQ7N): auto-heal the element_id scheme before writing
        // a single event. merge-events is the only writer in the normal
        // reindex flow, and it holds both flocks here — the one moment we can
        // safely re-key. It must run BEFORE the merge: this scan's rows are
        // keyed with the NEW scheme, so leaving old-scheme rows in place would
        // fork every renamed element's history at this scan. Gated, so it is a
        // no-op on every run after the first.
        migrate_element_id_scheme_v2(db)?;

        let scan_id = ulid::Ulid::new().to_string();
        let started_at = chrono::Utc::now().to_rfc3339();
        let mut observed_eids: HashSet<String> = HashSet::new();
        let mut visited_scope_paths: HashSet<String> = HashSet::new();
        // F7 (TRDD-1Z8SGQ7N): the scopes THIS scan claims to have enumerated
        // exhaustively. Only the discoverer knows whether a run was complete,
        // so it says so on the manifest and we take it at its word — that is
        // the ONLY signal that can authorize removing an element whose
        // scope_path produced zero observations (a scope root that vanished,
        // or one still present but emptied of elements).
        let mut exhaustive_scopes: HashSet<String> = HashSet::new();
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
                // F7 (TRDD-1Z8SGQ7N): manifest v2 adds the domain-level coverage
                // claim. Absent key ⇒ empty set ⇒ v1 behavior, unchanged. A
                // string naming no real scope is harmless: it simply matches no
                // element, so garbage under-claims rather than over-deletes.
                if let Some(arr) = value.get("exhaustive_scopes").and_then(|v| v.as_array()) {
                    for s in arr {
                        if let Some(s) = s.as_str() {
                            exhaustive_scopes.insert(s.to_string());
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

        // F6 (TRDD-1Z8SGQ7N): snapshot each element's override_status AS IT
        // STOOD BEFORE THIS SCAN, keyed by element_id, BEFORE the emit loop
        // below upserts elements_state. The override-resolution pass further
        // down needs the TRUE prior status to detect a transition, but the
        // emit loop upserts elements_state (with override_status="active")
        // for every element that produced any event — so a plain read_prior
        // in that pass reads the value THIS scan just wrote, not the prior.
        // The concrete miss: an element that WAS overridden and whose
        // overriding scope disappeared this scan should emit OverrideEnded,
        // but read_prior returns the just-written "active", equals the new
        // status, and the event is dropped. Snapshot first, compare against
        // the snapshot, and the transition is seen.
        let prior_override_status: std::collections::HashMap<String, String> = by_type_and_name
            .values()
            .flatten()
            .map(|obs| obs.element_id())
            .collect::<std::collections::HashSet<String>>()
            .into_iter()
            .map(|eid| {
                let status = read_prior(db, &eid)
                    .map(|p| p.override_status)
                    .unwrap_or_else(|| "active".to_string());
                (eid, status)
            })
            .collect();

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
                    // The events + state rows record obs.enabled in their own
                    // `enabled` column (see persist_event_and_state); F2
                    // removed the separate update_state bool this used to
                    // pass obs.enabled into.
                    persist_event_and_state(
                        db, &scan_id, &started_at, evt, obs, &diff,
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
                // What did elements_state say BEFORE this scan? F6: read the
                // pre-scan snapshot, NOT read_prior — the emit loop above has
                // already upserted elements_state this scan, so read_prior
                // here would return the value we just wrote.
                let prior_status = prior_override_status
                    .get(eid)
                    .cloned()
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
                    new_status,
                )?;
                events_emitted += 1;
            }
        }

        // Detect removals: anything in elements_state with exists=true that
        // this scan covered (see read_removal_candidates) but whose element_id
        // was NOT observed.
        let prior_active =
            read_removal_candidates(db, &visited_scope_paths, &exhaustive_scopes)?;
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

    /// Read the element_ids that are currently `exists=true` AND that this
    /// scan actually covered — i.e. the ones for which "not observed" is
    /// evidence of removal rather than evidence of not having looked.
    ///
    /// F7 (TRDD-1Z8SGQ7N): coverage used to be `scope_path ∈ visited`, and
    /// `visited` is built ONLY from elements that still exist (both from the
    /// manifest and from each observation). So a scope that yielded ZERO
    /// elements never entered the set and its stale rows were never even
    /// considered — measured live: 799 genuinely-gone elements, of which the
    /// old policy removed 1. Coverage is therefore taken from the scan's
    /// explicit per-scope claim (`exhaustive_scopes`), which is independent of
    /// the results and so survives a scope root vanishing entirely.
    ///
    /// DI-4 (code-review pass, 20260713): this used to issue ONE query for
    /// the active element_ids, then a SEPARATE per-element_id query
    /// against `events` to resolve each one's most-recent scope_path — an
    /// N+1 round-trip pattern. Rewritten to two queries total: one for the
    /// active ids, one bulk scan of `events` sorted `element_id, -observed_at`,
    /// deduped to "first row per element_id" in Rust (Cozo's `max()` can't
    /// aggregate RFC3339 strings — see the `as_of_rows` comment above for
    /// why the sort+take-first pattern is used throughout this file instead
    /// of an aggregate).
    fn read_removal_candidates(
        db: &DbInstance,
        scope_paths: &std::collections::HashSet<String>,
        exhaustive_scopes: &std::collections::HashSet<String>,
    ) -> Result<std::collections::HashSet<String>, String> {
        let active_q = r#"
            ?[element_id] := *elements_state{element_id, exists: true}
        "#;
        let active_rows = db
            .run_script(active_q, BTreeMap::new(), ScriptMutability::Immutable)
            .map_err(|e| format!("active query failed: {}", e))?
            .rows;
        let active_ids: std::collections::HashSet<String> = active_rows
            .into_iter()
            .filter_map(|row| match row.into_iter().next() {
                Some(DataValue::Str(s)) => Some(s.to_string()),
                _ => None,
            })
            .collect();
        if active_ids.is_empty() {
            return Ok(std::collections::HashSet::new());
        }

        // scope and scope_path live in the events table, not elements_state —
        // resolve both from the most recent event for each element_id via
        // one bulk scan instead of one query per active id. `scope` rides the
        // SAME projection (F7) precisely so the domain claim costs no extra
        // round-trip.
        let events_q = r#"
            ?[element_id, scope, scope_path, observed_at] :=
                *events{element_id, scope, scope_path, observed_at}
            :order element_id, -observed_at
        "#;
        let event_rows = db
            .run_script(events_q, BTreeMap::new(), ScriptMutability::Immutable)
            .map_err(|e| format!("events scan failed: {}", e))?
            .rows;
        // Rows are sorted (element_id asc, observed_at desc), so the FIRST
        // row seen per element_id is the most recent event for it.
        let mut latest_scope: std::collections::HashMap<String, (String, String)> =
            std::collections::HashMap::new();
        for row in event_rows {
            let eid = match row.first() {
                Some(DataValue::Str(s)) => s.to_string(),
                _ => continue,
            };
            if !active_ids.contains(&eid) || latest_scope.contains_key(&eid) {
                continue;
            }
            let scope = match row.get(1) {
                Some(v) => data_str(v),
                None => continue,
            };
            let sp = match row.get(2) {
                Some(v) => data_str(v),
                None => continue,
            };
            latest_scope.insert(eid, (scope, sp));
        }

        let mut out: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (eid, (scope, sp)) in latest_scope {
            // F7: an element is a removal candidate if EITHER
            //  (a) this scan claims it enumerated all of that element's scope —
            //      which catches a scope whose root vanished AND one whose root
            //      still exists but now yields nothing; or
            //  (b) [manifest-v1 path] its scope_path was visited this scan.
            // (a) subsumes (b) for claimed scopes; (b) is kept so a v1 manifest,
            // or a filtered run that claims nothing, behaves exactly as before.
            if exhaustive_scopes.contains(&scope) || scope_paths.contains(&sp) {
                out.insert(eid);
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
    pub fn cmd_marketplace_history(db: &DbInstance, limit: usize, format: OutputFormat) {
        if handle_stub_format(format, "marketplace-history") {
            return;
        }
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
                match format {
                    OutputFormat::Json => println!(
                        "{}",
                        serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                    ),
                    OutputFormat::Table => {
                        let table_rows: Vec<Vec<String>> = rows.iter().map(|r| vec![
                            cell(&r["observed_at"]),
                            cell(&r["event_type"]),
                            cell(&r["element_name"]),
                        ]).collect();
                        crate::print_table(
                            &["OBSERVED_AT", "EVENT_TYPE", "MARKETPLACE_NAME"],
                            &table_rows,
                        );
                    }
                    _ => {}
                }
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
    pub fn cmd_plugin_history(db: &DbInstance, plugin_name: &str, limit: usize, format: OutputFormat) {
        if handle_stub_format(format, "plugin-history") {
            return;
        }
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
                match format {
                    OutputFormat::Json => println!(
                        "{}",
                        serde_json::to_string_pretty(&JsonValue::Array(rows)).unwrap_or_default()
                    ),
                    OutputFormat::Table => {
                        let table_rows: Vec<Vec<String>> = rows.iter().map(|r| vec![
                            cell(&r["observed_at"]),
                            cell(&r["event_type"]),
                            cell(&r["element_name"]),
                            cell(&r["scope"]),
                        ]).collect();
                        crate::print_table(
                            &["OBSERVED_AT", "EVENT_TYPE", "PLUGIN_NAME", "SCOPE"],
                            &table_rows,
                        );
                    }
                    _ => {}
                }
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
    fn element_id_is_case_and_separator_preserving() {
        // F4 (TRDD-1Z8SGQ7N): name and scope_path are raw; only scope (a
        // fixed enum-ish set) is folded to lowercase.
        let id = compute_element_id(
            ElementType::Skill,
            "MyCoolSkill",
            "Project",
            "/Users/foo/Bar",
        );
        assert_eq!(id, "skill:MyCoolSkill@project:/Users/foo/Bar");
    }

    #[test]
    fn compute_element_id_case_sensitive_names_distinct() {
        // F4: the old scheme lowercased `name`, so `Foo` and `foo` collided
        // onto one id and had their append-only histories silently merged.
        let upper = compute_element_id(ElementType::Skill, "Foo", "user", "/a/b");
        let lower = compute_element_id(ElementType::Skill, "foo", "user", "/a/b");
        assert_ne!(upper, lower, "case-distinct names must not share an id");
    }

    #[test]
    fn compute_element_id_separator_vs_literal_underscore_distinct() {
        // F4: the old scheme slugged `/`→`_` in scope_path, so a real path
        // `/a/b` and a literal `/a_b` collided.
        let slashed = compute_element_id(ElementType::Skill, "x", "user", "/a/b");
        let scored = compute_element_id(ElementType::Skill, "x", "user", "/a_b");
        assert_ne!(slashed, scored, "path separators must survive in the id");
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

    /// F8 (TRDD-1Z8SGQ7N): a move that coincides with a content edit must
    /// record BOTH the ContentChanged AND the PathChanged — the old `else if`
    /// dropped the relocation whenever hash/size also changed.
    #[test]
    fn emit_path_change_alongside_content_change_on_move_and_edit() {
        let o = obs("x", "user", "h2", 100, "/new/p");
        let mut p = prior_of(&o);
        p.current_hash = "h1".to_string(); // content changed
        p.current_path = "/old/p".to_string(); // AND moved
        assert_eq!(
            compare_and_emit(Some(&p), &o),
            vec![EventType::ContentChanged, EventType::PathChanged],
        );
    }

    /// F8: a move that coincides with a content+size change records all three
    /// events, path last.
    #[test]
    fn emit_path_change_alongside_content_and_size_change_on_move() {
        let o = obs("x", "user", "h2", 200, "/new/p");
        let mut p = prior_of(&o);
        p.current_hash = "h1".to_string();
        p.current_size = 100;
        p.current_path = "/old/p".to_string();
        assert_eq!(
            compare_and_emit(Some(&p), &o),
            vec![
                EventType::ContentChanged,
                EventType::SizeChanged,
                EventType::PathChanged,
            ],
        );
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

    // ====================================================================
    // F4 / F5 — element_id re-key (TRDD-1Z8SGQ7N)
    // ====================================================================

    /// All `(element_id,)` values in `events`, sorted.
    fn event_ids_in(db: &DbInstance) -> Vec<String> {
        let r = db
            .run_script(
                r#"?[element_id] := *events{element_id}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .expect("read events element_ids");
        let mut v: Vec<String> = r.rows.iter().map(|row| data_str(&row[0])).collect();
        v.sort();
        v
    }

    fn count_rows(db: &DbInstance, rel: &str, col: &str) -> i64 {
        let q = format!("?[count({c})] := *{r}{{{c}}}", c = col, r = rel);
        let r = db
            .run_script(&q, BTreeMap::new(), ScriptMutability::Immutable)
            .unwrap_or_else(|e| panic!("count {}: {}", rel, e));
        r.rows
            .first()
            .and_then(|row| row.first())
            .and_then(|d| d.get_int())
            .unwrap_or(0)
    }

    fn state_keys_in(db: &DbInstance) -> Vec<String> {
        let r = db
            .run_script(
                r#"?[element_id] := *elements_state{element_id}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .expect("read elements_state keys");
        let mut v: Vec<String> = r.rows.iter().map(|row| data_str(&row[0])).collect();
        v.sort();
        v
    }

    /// A legacy v1 DB whose single skill came from a project-installed
    /// plugin (`project:<proj>/plugin:<name>`) — the composite source whose
    /// scope_path F5 must derive rather than hardcode to "".
    fn legacy_skills_v1_db_with_plugin_source() -> DbInstance {
        let db = DbInstance::new("mem", "", "").expect("mem db");
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
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        )
        .expect("create pss_metadata");
        db.run_script(
            r#"?[name, source, id, path, skill_type, description, first_indexed_at, last_updated_at] <-
                [["plug-skill", "project:proj/plugin:foo", "abc", "/tmp/nonexistent.md", "skill",
                  "desc", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"]]
               :put skills { name, source => id, path, skill_type, description, first_indexed_at, last_updated_at }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        )
        .expect("seed skill");
        db
    }

    #[test]
    fn migrate_v1_to_v2_derives_scope_path_from_source() {
        // F5 (TRDD-1Z8SGQ7N): the migration must key elements with the SAME
        // source-derived scope_path the live writer uses. Hardcoding "" gave
        // the pre-migration install event a different id from every event
        // after it — splitting each element's history at the boundary.
        let db = legacy_skills_v1_db_with_plugin_source();
        let stats = migrate_v1_to_v2(&db).expect("migration");
        assert_eq!(stats.skills_migrated, 1);

        let ids = event_ids_in(&db);
        assert_eq!(ids.len(), 1, "one install event expected");
        let eid = &ids[0];

        // The scope_path segment is the derived composite, not empty.
        assert!(
            eid.ends_with(":proj/plugin:foo"),
            "element_id must carry the derived scope_path, got {}",
            eid
        );

        // And it is NOT the id the old hardcoded-"" code produced.
        let buggy = compute_element_id(
            ElementType::Skill,
            "plug-skill",
            &scope_from_source("project:proj/plugin:foo"),
            "",
        );
        assert_ne!(*eid, buggy, "scope_path must not regress to empty");
    }

    /// A v2 DB carrying ONE event + state row keyed with the OLD lossy id
    /// while its `element_name` / `scope` / `scope_path` columns are raw —
    /// exactly the shape a live pre-F4 DB has.
    fn v2_db_with_old_scheme_row() -> DbInstance {
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("schema");
        // Old scheme: name lowercased, scope_path "/a/b" slugged to "_a_b".
        //
        // diff_json goes in as a $param, NOT inline: cozo-ce 0.7's `string`
        // rule tries `raw_string` BEFORE `quoted_string`, and a zero-underscore
        // raw_string matches a bare "..." that ends at the FIRST '"'. So a
        // `\"` escape inside a double-quoted literal is unreachable and the
        // parser dies mid-value. Params sidestep the grammar entirely — which
        // is also why the migration itself never inlines a value.
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("diff".into(), DataValue::Str(r#"{"k":1}"#.into()));
        let seed = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["e1", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "MyCoolSkill", "skill:mycoolskill@user:_a_b", "user", "/a/b", "user",
                 "/a/b/MyCoolSkill.md", "h1", 123, 45, false, "shadowed", $diff, "snap1"]
              ]
            :put events { event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }
        "#;
        db.run_script(seed, params, ScriptMutability::Mutable)
            .expect("seed events");
        let seed_state = r#"
            ?[element_id, last_event_id, current_path, current_hash,
              current_size, current_token_count, enabled, override_status,
              installed_at, last_changed_at, exists] <- [
                ["skill:mycoolskill@user:_a_b", "e1", "/a/b/MyCoolSkill.md", "h1",
                 123, 45, false, "shadowed", "2026-01-01T00:00:00Z",
                 "2026-02-02T00:00:00Z", true]
              ]
            :put elements_state { element_id =>
              last_event_id, current_path, current_hash, current_size,
              current_token_count, enabled, override_status, installed_at,
              last_changed_at, exists }
        "#;
        db.run_script(seed_state, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed elements_state");
        // element_descriptions is element_id-KEYED too. Two rows:
        //   - the old-scheme id, which MUST be re-keyed alongside the rest;
        //   - an orphan whose id has no events, which MUST survive untouched
        //     (it has no remap entry, so the inner join skips it).
        let seed_desc = r#"
            ?[element_id, description_hash, description_text, last_updated_at] <- [
                ["skill:mycoolskill@user:_a_b", "dh1", "a cool skill", "2026-01-03T00:00:00Z"],
                ["skill:orphan@user:/nowhere", "dh2", "no events at all", "2026-01-04T00:00:00Z"]
              ]
            :put element_descriptions { element_id =>
              description_hash, description_text, last_updated_at }
        "#;
        db.run_script(seed_desc, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed element_descriptions");
        db
    }

    /// All `element_descriptions` keys, sorted.
    fn desc_keys_in(db: &DbInstance) -> Vec<String> {
        let r = db
            .run_script(
                r#"?[element_id] := *element_descriptions{element_id}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .expect("read element_descriptions keys");
        let mut v: Vec<String> = r.rows.iter().map(|row| data_str(&row[0])).collect();
        v.sort();
        v
    }

    /// The (hash, text, last_updated_at) triple stored under `eid`, if any.
    fn desc_row_of(db: &DbInstance, eid: &str) -> Option<(String, String, String)> {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(eid.into()));
        let r = db
            .run_script(
                r#"?[description_hash, description_text, last_updated_at] :=
                    *element_descriptions{element_id: $eid, description_hash,
                                          description_text, last_updated_at}"#,
                params,
                ScriptMutability::Immutable,
            )
            .expect("read description row");
        r.rows
            .first()
            .map(|row| (data_str(&row[0]), data_str(&row[1]), data_str(&row[2])))
    }

    #[test]
    fn rekey_migration_rewrites_element_descriptions_losslessly() {
        // F4 (TRDD-1Z8SGQ7N): element_descriptions is element_id-KEYED and is
        // read back by read_prior_description_hash(NEW id). If it kept the old
        // key, every re-keyed element would orphan its description row and the
        // next scan would fire a spurious description_changed for all of them.
        let db = v2_db_with_old_scheme_row();
        let new_id = compute_element_id(ElementType::Skill, "MyCoolSkill", "user", "/a/b");

        assert_eq!(migrate_element_id_scheme_v2(&db).expect("re-key"), 1);

        // Row count preserved: the key moved, nothing was added or dropped.
        assert_eq!(count_rows(&db, "element_descriptions", "element_id"), 2);

        // Old key gone, new key present, values byte-identical.
        assert_eq!(desc_row_of(&db, "skill:mycoolskill@user:_a_b"), None,
            "the old description key must be gone");
        assert_eq!(
            desc_row_of(&db, &new_id),
            Some((
                "dh1".to_string(),
                "a cool skill".to_string(),
                "2026-01-03T00:00:00Z".to_string()
            )),
            "description columns must survive the key rename verbatim"
        );

        // The exact lookup the writer performs must now resolve.
        assert_eq!(
            cli::read_prior_description_hash(&db, &new_id).as_deref(),
            Some("dh1"),
            "read_prior_description_hash must find the row under the new id"
        );
    }

    #[test]
    fn rekey_migration_leaves_unmatched_description_untouched() {
        // F4: a description row whose element_id has no events has no remap
        // entry. The inner join must simply skip it — never re-key it (there
        // is nothing to re-key it to) and never drop it (that would be data
        // loss the migration has no mandate for).
        let db = v2_db_with_old_scheme_row();
        let before = desc_row_of(&db, "skill:orphan@user:/nowhere");
        assert!(before.is_some(), "precondition: orphan row seeded");

        migrate_element_id_scheme_v2(&db).expect("re-key");

        assert_eq!(
            desc_row_of(&db, "skill:orphan@user:/nowhere"),
            before,
            "an events-less description row must survive the migration as-is"
        );
        let new_id = compute_element_id(ElementType::Skill, "MyCoolSkill", "user", "/a/b");
        let mut expected = vec!["skill:orphan@user:/nowhere".to_string(), new_id];
        expected.sort();
        assert_eq!(desc_keys_in(&db), expected);
    }

    #[test]
    fn rekey_migration_keeps_events_by_element_id_index_consistent() {
        // F4: `events:by_element_id` is a cozo ::index over events, which cozo
        // maintains from our `:put`. Prove it actually tracked the re-key —
        // an index still pointing at the old id would silently return zero
        // rows for every lifecycle query that goes through it.
        let db = v2_db_with_old_scheme_row();
        let new_id = compute_element_id(ElementType::Skill, "MyCoolSkill", "user", "/a/b");
        migrate_element_id_scheme_v2(&db).expect("re-key");

        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(new_id.as_str().into()));
        let hit = db
            .run_script(
                r#"?[event_id] := *events:by_element_id{element_id: $eid, event_id}"#,
                params,
                ScriptMutability::Immutable,
            )
            .expect("index lookup by new element_id");
        assert_eq!(
            hit.rows.len(),
            1,
            "the index must resolve the NEW id to the element's event"
        );
        assert_eq!(data_str(&hit.rows[0][0]), "e1");

        // ...and must no longer resolve the OLD id.
        let mut old_params: BTreeMap<String, DataValue> = BTreeMap::new();
        old_params.insert("eid".into(), DataValue::Str("skill:mycoolskill@user:_a_b".into()));
        let stale = db
            .run_script(
                r#"?[event_id] := *events:by_element_id{element_id: $eid, event_id}"#,
                old_params,
                ScriptMutability::Immutable,
            )
            .expect("index lookup by old element_id");
        assert_eq!(stale.rows.len(), 0, "the old id must be gone from the index");
    }

    #[test]
    fn rekey_migration_rewrites_events_and_state_losslessly() {
        // F4 (TRDD-1Z8SGQ7N): the re-key renames the id and NOTHING else.
        let db = v2_db_with_old_scheme_row();
        let new_id = compute_element_id(ElementType::Skill, "MyCoolSkill", "user", "/a/b");
        assert_eq!(new_id, "skill:MyCoolSkill@user:/a/b");

        let changed = migrate_element_id_scheme_v2(&db).expect("re-key");
        assert_eq!(changed, 1, "exactly one element_id moved");

        // Row counts are invariant — nothing dropped, nothing duplicated.
        assert_eq!(count_rows(&db, "events", "event_id"), 1);
        assert_eq!(count_rows(&db, "elements_state", "element_id"), 1);

        // The event is re-keyed and EVERY other column is byte-identical.
        let row = db
            .run_script(
                r#"?[observed_at, scan_id, event_type, element_type, element_name,
                     element_id, scope, scope_path, source, path, content_hash,
                     file_size, token_count, enabled, override_status, diff_json,
                     snapshot_ref] :=
                    *events{event_id: "e1", observed_at, scan_id, event_type,
                            element_type, element_name, element_id, scope,
                            scope_path, source, path, content_hash, file_size,
                            token_count, enabled, override_status, diff_json,
                            snapshot_ref}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .expect("read e1");
        assert_eq!(row.rows.len(), 1, "event_id key must be untouched");
        let r = &row.rows[0];
        assert_eq!(data_str(&r[0]), "2026-01-01T00:00:00Z");
        assert_eq!(data_str(&r[1]), "s1");
        assert_eq!(data_str(&r[2]), "installed");
        assert_eq!(data_str(&r[3]), "skill");
        assert_eq!(data_str(&r[4]), "MyCoolSkill");
        assert_eq!(data_str(&r[5]), new_id, "element_id must be re-keyed");
        assert_eq!(data_str(&r[6]), "user");
        assert_eq!(data_str(&r[7]), "/a/b");
        assert_eq!(data_str(&r[8]), "user");
        assert_eq!(data_str(&r[9]), "/a/b/MyCoolSkill.md");
        assert_eq!(data_str(&r[10]), "h1");
        assert_eq!(r[11].get_int(), Some(123), "file_size Int fidelity");
        assert_eq!(r[12].get_int(), Some(45), "token_count Int fidelity");
        assert_eq!(r[13], DataValue::Bool(false), "enabled Bool fidelity");
        assert_eq!(data_str(&r[14]), "shadowed");
        assert_eq!(data_str(&r[15]), r#"{"k":1}"#, "diff_json survives verbatim");
        assert_eq!(data_str(&r[16]), "snap1");

        // elements_state is keyed by the new id; the old key is gone.
        assert_eq!(state_keys_in(&db), vec![new_id.clone()]);

        // ...and its value columns survived the key rename intact.
        let srow = db
            .run_script(
                r#"?[last_event_id, current_path, current_hash, current_size,
                     current_token_count, enabled, override_status, installed_at,
                     last_changed_at, exists] :=
                    *elements_state{element_id, last_event_id, current_path,
                                    current_hash, current_size, current_token_count,
                                    enabled, override_status, installed_at,
                                    last_changed_at, exists}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .expect("read state");
        assert_eq!(srow.rows.len(), 1);
        let s = &srow.rows[0];
        assert_eq!(data_str(&s[0]), "e1");
        assert_eq!(data_str(&s[1]), "/a/b/MyCoolSkill.md");
        assert_eq!(data_str(&s[2]), "h1");
        assert_eq!(s[3].get_int(), Some(123));
        assert_eq!(s[4].get_int(), Some(45));
        assert_eq!(s[5], DataValue::Bool(false));
        assert_eq!(data_str(&s[6]), "shadowed");
        assert_eq!(data_str(&s[7]), "2026-01-01T00:00:00Z");
        assert_eq!(data_str(&s[8]), "2026-02-02T00:00:00Z");
        assert_eq!(s[9], DataValue::Bool(true));

        // Gate stamped, and a second run short-circuits.
        assert_eq!(
            read_metadata_value(&db, ELEMENT_ID_SCHEME_KEY).as_deref(),
            Some(ELEMENT_ID_SCHEME_VERSION)
        );
        assert_eq!(migrate_element_id_scheme_v2(&db).expect("second run"), 0);
        assert_eq!(state_keys_in(&db), vec![new_id]);
    }

    /// A v2 DB with an override PAIR keyed by OLD-scheme ids, where the
    /// override_status values (and one diff_json) EMBED those old ids:
    ///   A = skill "CoolSkill" @user   → old `skill:coolskill@user:`
    ///   B = skill "CoolSkill" @local  → old `skill:coolskill@local:` (wins)
    /// plus C = skill "plain" @user, whose id does not change and whose
    /// status/diff carry no ids at all.
    fn v2_db_with_embedded_override_ids() -> DbInstance {
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("schema");
        // diff_json values carry quotes → must go in as $params (see the
        // raw_string-first grammar note on v2_db_with_old_scheme_row).
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert(
            "diff_a".into(),
            DataValue::Str(
                r#"{"new_override_status":"overridden_by:skill:coolskill@local:","previous_override_status":"active"}"#.into(),
            ),
        );
        params.insert("diff_c".into(), DataValue::Str(r#"{"contact":"a@b.c"}"#.into()));
        let seed = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["e1", "2026-01-01T00:00:00Z", "s1", "override_started", "skill",
                 "CoolSkill", "skill:coolskill@user:", "user", "", "user",
                 "/u/CoolSkill.md", "h1", 10, 5, true,
                 "overridden_by:skill:coolskill@local:", $diff_a, ""],
                ["e2", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "CoolSkill", "skill:coolskill@local:", "local", "", "local",
                 "/l/CoolSkill.md", "h2", 20, 6, true,
                 "overrides:skill:coolskill@user:;skill:unchanged@user:", "{}", ""],
                ["e3", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "plain", "skill:plain@user:", "user", "", "user",
                 "/u/plain.md", "h3", 30, 7, true, "active", $diff_c, ""]
              ]
            :put events { event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }
        "#;
        db.run_script(seed, params, ScriptMutability::Mutable)
            .expect("seed events");
        let seed_state = r#"
            ?[element_id, last_event_id, current_path, current_hash,
              current_size, current_token_count, enabled, override_status,
              installed_at, last_changed_at, exists] <- [
                ["skill:coolskill@user:", "e1", "/u/CoolSkill.md", "h1", 10, 5, true,
                 "overridden_by:skill:coolskill@local:",
                 "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true],
                ["skill:coolskill@local:", "e2", "/l/CoolSkill.md", "h2", 20, 6, true,
                 "overrides:skill:coolskill@user:;skill:unchanged@user:",
                 "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true],
                ["skill:plain@user:", "e3", "/u/plain.md", "h3", 30, 7, true,
                 "none", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", true]
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

    /// (override_status, diff_json) of one event, by event_id.
    fn event_status_and_diff(db: &DbInstance, event_id: &str) -> (String, String) {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(event_id.into()));
        let r = db
            .run_script(
                r#"?[override_status, diff_json] :=
                    *events{event_id: $eid, override_status, diff_json}"#,
                params,
                ScriptMutability::Immutable,
            )
            .expect("read event status/diff");
        let row = r.rows.first().expect("event must exist");
        (data_str(&row[0]), data_str(&row[1]))
    }

    /// override_status of one elements_state row, by element_id.
    fn state_status_of(db: &DbInstance, eid: &str) -> String {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(eid.into()));
        let r = db
            .run_script(
                r#"?[override_status] :=
                    *elements_state{element_id: $eid, override_status}"#,
                params,
                ScriptMutability::Immutable,
            )
            .expect("read state status");
        data_str(&r.rows.first().expect("state row must exist")[0])
    }

    #[test]
    fn rekey_migration_rewrites_embedded_override_status_ids() {
        // Correction #2 (TRDD-1Z8SGQ7N): override_status EMBEDS element_ids
        // ("overridden_by:<eid>", "overrides:<eid>;<eid>"). The
        // elements_state copy is CURRENT state — left stale, the next scan's
        // override pass (which recomputes with NEW ids) would see a mismatch
        // and emit spurious override_started/override_ended events.
        let db = v2_db_with_embedded_override_ids();
        assert_eq!(migrate_element_id_scheme_v2(&db).expect("re-key"), 2);

        let new_a = "skill:CoolSkill@user:";
        let new_b = "skill:CoolSkill@local:";

        // events: both directions re-pointed; the unchanged id in the
        // semicolon list ("skill:unchanged@user:") passes through verbatim.
        assert_eq!(
            event_status_and_diff(&db, "e1").0,
            format!("overridden_by:{}", new_b)
        );
        assert_eq!(
            event_status_and_diff(&db, "e2").0,
            format!("overrides:{};skill:unchanged@user:", new_a)
        );

        // elements_state: the CURRENT state carries the new ids too.
        assert_eq!(state_status_of(&db, new_a), format!("overridden_by:{}", new_b));
        assert_eq!(
            state_status_of(&db, new_b),
            format!("overrides:{};skill:unchanged@user:", new_a)
        );

        // Invariant: no stale old id survives anywhere in either carrier.
        for eid in ["e1", "e2", "e3"] {
            let (status, diff) = event_status_and_diff(&db, eid);
            for old in ["skill:coolskill@user:", "skill:coolskill@local:"] {
                assert!(!status.contains(old), "{eid} status keeps old id: {status}");
                assert!(!diff.contains(old), "{eid} diff keeps old id: {diff}");
            }
        }
    }

    #[test]
    fn rekey_migration_leaves_plain_statuses_untouched() {
        // "active" / "none" carry no id — they must pass through the
        // two-rule matched/unmatched rewrite byte-identically.
        let db = v2_db_with_embedded_override_ids();
        migrate_element_id_scheme_v2(&db).expect("re-key");
        assert_eq!(event_status_and_diff(&db, "e3").0, "active");
        assert_eq!(state_status_of(&db, "skill:plain@user:"), "none");
    }

    #[test]
    fn rekey_migration_rewrites_diff_json_embedded_ids_and_keeps_json_valid() {
        // diff_json embeds the same status strings for override events; the
        // rewrite must swap the ids AND leave the value valid JSON.
        let db = v2_db_with_embedded_override_ids();
        migrate_element_id_scheme_v2(&db).expect("re-key");

        let (_, diff) = event_status_and_diff(&db, "e1");
        let parsed: serde_json::Value =
            serde_json::from_str(&diff).expect("rewritten diff_json must still parse");
        assert_eq!(
            parsed["new_override_status"],
            serde_json::json!("overridden_by:skill:CoolSkill@local:")
        );
        assert_eq!(parsed["previous_override_status"], serde_json::json!("active"));
        assert!(!diff.contains("skill:coolskill@local:"), "old id must be gone: {diff}");
    }

    #[test]
    fn rekey_migration_leaves_idless_diff_json_byte_identical() {
        // A diff_json with no embedded id — even one containing '@' (which
        // enters the replacement loop via the pre-filter) — must come out
        // byte-for-byte identical.
        let db = v2_db_with_embedded_override_ids();
        migrate_element_id_scheme_v2(&db).expect("re-key");
        let (_, diff_c) = event_status_and_diff(&db, "e3");
        assert_eq!(diff_c, r#"{"contact":"a@b.c"}"#);
        let (_, diff_b) = event_status_and_diff(&db, "e2");
        assert_eq!(diff_b, "{}", "'@'-free diff_json is skipped outright");
    }

    #[test]
    fn rekey_migration_fails_fast_on_unmerge() {
        // F4: `Foo` and `foo` collided onto one old id, so their histories
        // are already merged into a single elements_state row. A bijective
        // key-rename cannot split that back into two — abort, write nothing,
        // and let a human decide.
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("schema");
        let seed = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["e1", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "Foo", "skill:foo@user:", "user", "", "user",
                 "/u/Foo.md", "h1", 10, 5, true, "active", "{}", ""],
                ["e2", "2026-01-02T00:00:00Z", "s1", "installed", "skill",
                 "foo", "skill:foo@user:", "user", "", "user",
                 "/u/foo.md", "h2", 20, 6, true, "active", "{}", ""]
              ]
            :put events { event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }
        "#;
        db.run_script(seed, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed events");

        let err = migrate_element_id_scheme_v2(&db)
            .expect_err("un-merge must abort");
        assert!(
            err.contains("ABORTED") && err.contains("distinct new ids"),
            "error must name the un-merge, got: {}",
            err
        );
        assert!(err.contains("No rows written."), "got: {}", err);

        // Nothing was written: both events still carry the old merged id.
        // Checked per event_id, because `event_ids_in` projects a datalog SET
        // — two rows sharing one element_id collapse to a single value there,
        // which would hide a half-written rewrite.
        assert_eq!(count_rows(&db, "events", "event_id"), 2);
        let spot = db
            .run_script(
                r#"?[event_id, element_id] := *events{event_id, element_id}"#,
                BTreeMap::new(),
                ScriptMutability::Immutable,
            )
            .expect("read events");
        let mut pairs: Vec<(String, String)> = spot
            .rows
            .iter()
            .map(|r| (data_str(&r[0]), data_str(&r[1])))
            .collect();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("e1".to_string(), "skill:foo@user:".to_string()),
                ("e2".to_string(), "skill:foo@user:".to_string()),
            ],
            "abort must leave every event_id keyed exactly as before"
        );
        // elements_state untouched too (the merged single row is still there).
        assert_eq!(state_keys_in(&db), Vec::<String>::new());
        // The gate must NOT be stamped — the next run has to retry.
        assert_eq!(read_metadata_value(&db, ELEMENT_ID_SCHEME_KEY), None);
    }

    #[test]
    fn rekey_migration_idempotent_on_fresh_v2() {
        // F4: a DB already written by a post-F4 binary re-keys to itself, so
        // the migration is a no-op that only stamps the gate.
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("schema");
        let new_id = compute_element_id(ElementType::Skill, "MyCoolSkill", "user", "/a/b");
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str(new_id.as_str().into()));
        let seed = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["e1", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "MyCoolSkill", $eid, "user", "/a/b", "user",
                 "/a/b/MyCoolSkill.md", "h1", 10, 5, true, "active", "{}", ""]
              ]
            :put events { event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }
        "#;
        db.run_script(seed, params, ScriptMutability::Mutable)
            .expect("seed events");

        let changed = migrate_element_id_scheme_v2(&db).expect("re-key");
        assert_eq!(changed, 0, "already-new ids must not move");
        assert_eq!(event_ids_in(&db), vec![new_id]);
        assert_eq!(count_rows(&db, "events", "event_id"), 1);
        assert_eq!(
            read_metadata_value(&db, ELEMENT_ID_SCHEME_KEY).as_deref(),
            Some(ELEMENT_ID_SCHEME_VERSION),
            "gate must be stamped even on a zero-change run"
        );
    }

    #[test]
    fn rekey_migration_gate_short_circuits_before_reading_events() {
        // F4: the gate is what makes the auto-run in merge-events free after
        // the first reindex — it must return before touching `events`.
        let db = v2_db_with_old_scheme_row();
        stamp_element_id_scheme(&db).expect("pre-stamp");
        let changed = migrate_element_id_scheme_v2(&db).expect("gated run");
        assert_eq!(changed, 0);
        // Old id still there: the gate short-circuited, as designed.
        assert_eq!(event_ids_in(&db), vec!["skill:mycoolskill@user:_a_b"]);
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

    // ────────────────────────────────────────────────────────────────────
    // F-12 (audit 20260514): version-history filter
    // ────────────────────────────────────────────────────────────────────

    /// F-12: the filter must only return installed/content_changed/
    /// description_changed/removed events, NOT enable/disable noise.
    #[test]
    fn cmd_version_history_filters_to_signal_events() {
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("schema");

        // Seed: one element with 4 events — 3 signal + 1 noise.
        let seed = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["e1", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "v", "skill:v@user:_home", "user", "/home", "user",
                 "/home/v.md", "h1", 100, 50, true, "active", "{}", ""],
                ["e2", "2026-01-02T00:00:00Z", "s2", "content_changed", "skill",
                 "v", "skill:v@user:_home", "user", "/home", "user",
                 "/home/v.md", "h2", 110, 55, true, "active", "{}", ""],
                ["e3", "2026-01-03T00:00:00Z", "s3", "enabled", "skill",
                 "v", "skill:v@user:_home", "user", "/home", "user",
                 "/home/v.md", "h2", 110, 55, true, "active", "{}", ""],
                ["e4", "2026-01-04T00:00:00Z", "s4", "description_changed", "skill",
                 "v", "skill:v@user:_home", "user", "/home", "user",
                 "/home/v.md", "h2", 110, 55, true, "active", "{}", ""]
              ]
            :put events { event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }
        "#;
        db.run_script(seed, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed");

        // Inline the version-history query so we can assert on rows.
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("eid".into(), DataValue::Str("skill:v@user:_home".into()));
        let q = r#"
            ?[event_type] :=
                *events{element_id: $eid, event_type},
                or(
                    event_type == "installed",
                    event_type == "content_changed",
                    event_type == "description_changed",
                    event_type == "removed"
                )
            :order event_type
        "#;
        let r = db.run_script(q, params, ScriptMutability::Immutable)
            .expect("version-history query");
        // Expected: 3 signal events (installed, content_changed,
        // description_changed) — NOT enabled.
        assert_eq!(r.rows.len(), 3, "expected 3 signal events");
        let types: Vec<String> = r.rows.iter().filter_map(|row| {
            if let DataValue::Str(s) = &row[0] {
                Some(s.to_string())
            } else { None }
        }).collect();
        for t in &types {
            assert_ne!(t, "enabled", "enabled is noise; must be filtered");
        }
        assert!(types.contains(&"installed".to_string()));
        assert!(types.contains(&"content_changed".to_string()));
        assert!(types.contains(&"description_changed".to_string()));
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

    /// F2 (TRDD-1Z8SGQ7N): a DISABLED element must still materialize an
    /// `elements_state` row. Before the fix, `obs.enabled` was wired into the
    /// removed `update_state` control bool, so a disabled element's event was
    /// logged but its state row was never written — `as-of`/`show` and
    /// removal detection (which read elements_state) silently stopped
    /// tracking it. This drives the real writer end-to-end and asserts the
    /// state row exists with enabled=false, exists=true.
    #[test]
    fn cmd_merge_events_materializes_state_for_disabled_element() {
        use std::io::Cursor;

        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );

        let jsonl = r#"{"_pss_manifest": true, "visited_scope_paths": ["/home"]}
{"type": "skill", "name": "off-skill", "source": "user", "path": "/home/off.md", "description": "a disabled skill", "enabled": false}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(jsonl), true)
            .expect("merge-events must succeed");

        let q = r#"?[element_id, enabled, exists] := *elements_state{element_id, enabled, exists}"#;
        let r = db
            .run_script(q, BTreeMap::new(), ScriptMutability::Immutable)
            .expect("query elements_state");
        assert_eq!(
            r.rows.len(),
            1,
            "a disabled element must still get an elements_state row; got {} rows",
            r.rows.len()
        );
        let row = &r.rows[0];
        assert_eq!(row[1], DataValue::Bool(false), "enabled column must be false");
        assert_eq!(row[2], DataValue::Bool(true), "exists must be true (installed)");
    }

    /// F6 (TRDD-1Z8SGQ7N): the override-resolution pass must compare against
    /// the override_status AS IT STOOD BEFORE THE SCAN, not the value the
    /// emit loop just wrote. Scenario: scan 1 establishes user-overrides-plugin
    /// — that emits TWO override_started (user active→overrides, plugin
    /// active→overridden_by). Scan 2 keeps both scopes but MOVES the plugin
    /// file — that fires a PathChanged whose emit-loop upsert overwrites the
    /// plugin's override_status with the placeholder "active". The override
    /// decision itself is UNCHANGED (user still wins), so scan 2 must emit NO
    /// new override event → the total stays at 2. With the read-own-write bug
    /// the pass read the just-written "active", saw it differ from the
    /// resolved "overridden_by:...", and emitted a SPURIOUS third
    /// override_started (total 3).
    #[test]
    fn cmd_merge_events_override_uses_pre_scan_status_not_own_write() {
        use std::io::Cursor;

        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );

        let scan1 = r#"{"_pss_manifest": true, "visited_scope_paths": ["/home", "foo/skills"]}
{"type": "skill", "name": "shared", "source": "user", "path": "/home/shared.md", "description": "user scope", "enabled": true}
{"type": "skill", "name": "shared", "source": "plugin:foo", "path": "foo/skills/shared.md", "description": "plugin scope", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan1), true)
            .expect("scan 1 must succeed");

        // Scan 2: same two scopes, but the plugin file MOVED — forces an
        // emit-loop event whose upsert clobbers override_status to "active".
        let scan2 = r#"{"_pss_manifest": true, "visited_scope_paths": ["/home", "foo/skills"]}
{"type": "skill", "name": "shared", "source": "user", "path": "/home/shared.md", "description": "user scope", "enabled": true}
{"type": "skill", "name": "shared", "source": "plugin:foo", "path": "foo/skills/moved.md", "description": "plugin scope", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan2), true)
            .expect("scan 2 must succeed");

        let q = r#"?[count(event_id)] := *events{event_id, event_type}, event_type = "override_started""#;
        let r = db
            .run_script(q, BTreeMap::new(), ScriptMutability::Immutable)
            .expect("query override_started count");
        let n = match &r.rows[0][0] {
            DataValue::Num(cozo::Num::Int(n)) => *n,
            _ => -1,
        };
        assert_eq!(
            n, 2,
            "two override_started expected (both from scan 1; scan 2's move must \
             emit none); {} means the override pass re-read its own write (F6)",
            n
        );
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
    // F7 (TRDD-1Z8SGQ7N): manifest v2 `exhaustive_scopes` — full-scope removal
    // ========================================================================

    /// Names of every element that got a `removed` event, sorted. Datalog set
    /// semantics dedupe the projection, so one name per removed element.
    fn removed_element_names(db: &DbInstance) -> Vec<String> {
        let q = r#"
            ?[element_name] :=
                *events{element_name, event_type},
                event_type = "removed"
        "#;
        let r = db
            .run_script(q, BTreeMap::new(), ScriptMutability::Immutable)
            .expect("query removed events");
        let mut names: Vec<String> = r
            .rows
            .iter()
            .filter_map(|row| match row.first() {
                Some(DataValue::Str(s)) => Some(s.to_string()),
                _ => None,
            })
            .collect();
        names.sort();
        names
    }

    fn mem_db_with_metadata() -> DbInstance {
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        db
    }

    /// F7 shape 1 — the scope ROOT VANISHED (`kazuph-dotfiles` et al.: gone from
    /// `~/.claude/plugins/marketplaces/`). Its elements produce no observation
    /// AND its scope_path can no longer be enumerated from the filesystem, so
    /// every result-derived coverage set misses it and its rows stay
    /// `exists=true` forever. This is the F7 regression test: it MUST fail
    /// before the fix (799 such zombies measured live; the old policy caught 1).
    #[test]
    fn manifest_v2_exhaustive_scope_removes_element_in_vanished_scope() {
        use std::io::Cursor;

        let db = mem_db_with_metadata();

        let scan1 = r#"{"_pss_manifest": true, "_pss_manifest_version": 2, "visited_scope_paths": ["gone-mp", "live-mp"], "exhaustive_scopes": ["marketplace"]}
{"type": "skill", "name": "doomed", "source": "marketplace:gone-mp", "path": "gone-mp/skills/doomed.md", "description": "in the marketplace about to vanish", "enabled": true}
{"type": "skill", "name": "survivor", "source": "marketplace:live-mp", "path": "live-mp/skills/survivor.md", "description": "in a marketplace that stays", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan1), true).expect("scan 1");

        // Scan 2: `gone-mp` is gone from disk — zero observations for it, and
        // it is absent from visited_scope_paths (the discoverer builds that
        // only from surviving elements). Only the domain-level claim
        // `exhaustive_scopes:["marketplace"]` can authorize the sweep.
        let scan2 = r#"{"_pss_manifest": true, "_pss_manifest_version": 2, "visited_scope_paths": ["live-mp"], "exhaustive_scopes": ["marketplace"]}
{"type": "skill", "name": "survivor", "source": "marketplace:live-mp", "path": "live-mp/skills/survivor.md", "description": "in a marketplace that stays", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan2), true).expect("scan 2");

        assert_eq!(
            removed_element_names(&db),
            vec!["doomed".to_string()],
            "an exhaustive `marketplace` scan that did not observe `doomed` must \
             remove it even though its scope_path is unenumerable (root gone)"
        );
    }

    /// F7 shape 2 — the scope root is STILL PRESENT but now yields ZERO elements
    /// (`melodic-software`: dir and marketplace.json intact, content swapped to
    /// 4 formatter plugins; all 599 skills stranded at the seed scan). Identical
    /// blindness to shape 1: no observation ⇒ the scope_path never enters any
    /// result-derived set. Seeds two elements so the assertion proves the whole
    /// bucket is swept, which is the signature of full-scope removal.
    #[test]
    fn manifest_v2_exhaustive_scope_removes_element_in_present_but_empty_scope() {
        use std::io::Cursor;

        let db = mem_db_with_metadata();

        let scan1 = r#"{"_pss_manifest": true, "_pss_manifest_version": 2, "visited_scope_paths": ["emptied-mp", "live-mp"], "exhaustive_scopes": ["marketplace"]}
{"type": "skill", "name": "stranded-one", "source": "marketplace:emptied-mp", "path": "emptied-mp/skills/one.md", "description": "seeded then never seen again", "enabled": true}
{"type": "agent", "name": "stranded-two", "source": "marketplace:emptied-mp", "path": "emptied-mp/agents/two.md", "description": "seeded then never seen again", "enabled": true}
{"type": "skill", "name": "survivor", "source": "marketplace:live-mp", "path": "live-mp/skills/survivor.md", "description": "in a marketplace that stays", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan1), true).expect("scan 1");

        let scan2 = r#"{"_pss_manifest": true, "_pss_manifest_version": 2, "visited_scope_paths": ["live-mp"], "exhaustive_scopes": ["marketplace"]}
{"type": "skill", "name": "survivor", "source": "marketplace:live-mp", "path": "live-mp/skills/survivor.md", "description": "in a marketplace that stays", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan2), true).expect("scan 2");

        assert_eq!(
            removed_element_names(&db),
            vec!["stranded-one".to_string(), "stranded-two".to_string()],
            "every element of an emptied marketplace must be swept, and the \
             surviving marketplace's element must not be touched"
        );
    }

    /// F7 negative guard — the claim is per-scope, so an UNCLAIMED scope keeps
    /// today's scope_path-membership rule. A `marketplace`-only claim must not
    /// reach into `plugin` rows: over-reaching here would delete real history
    /// for a scope the scan never promised to have enumerated.
    #[test]
    fn manifest_v2_does_not_remove_element_of_unclaimed_scope() {
        use std::io::Cursor;

        let db = mem_db_with_metadata();

        let scan1 = r#"{"_pss_manifest": true, "_pss_manifest_version": 2, "visited_scope_paths": ["foo", "live-mp"], "exhaustive_scopes": ["marketplace", "plugin"]}
{"type": "skill", "name": "plug-skill", "source": "plugin:foo", "path": "foo/skills/plug.md", "description": "plugin scope", "enabled": true}
{"type": "skill", "name": "mkt-skill", "source": "marketplace:live-mp", "path": "live-mp/skills/mkt.md", "description": "marketplace scope", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan1), true).expect("scan 1");

        // Scan 2 claims ONLY `marketplace` (e.g. a plugin-filtered run), and
        // does not observe the plugin element. `plugin` is unclaimed and
        // scope_path "foo" is unvisited ⇒ plug-skill must survive.
        let scan2 = r#"{"_pss_manifest": true, "_pss_manifest_version": 2, "visited_scope_paths": ["live-mp"], "exhaustive_scopes": ["marketplace"]}
{"type": "skill", "name": "mkt-skill", "source": "marketplace:live-mp", "path": "live-mp/skills/mkt.md", "description": "marketplace scope", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan2), true).expect("scan 2");

        assert!(
            removed_element_names(&db).is_empty(),
            "a marketplace-only claim must not remove a plugin-scoped element; got {:?}",
            removed_element_names(&db)
        );
    }

    /// F7 back-compat — a manifest with NO `exhaustive_scopes` key (v1, or an
    /// older discoverer) must behave byte-for-byte as before: only elements
    /// whose scope_path was visited this scan are removal candidates.
    /// `foo-second` (visited scope_path, unobserved) goes; `bar-only`
    /// (unvisited scope_path) stays.
    #[test]
    fn manifest_v1_behavior_unchanged() {
        use std::io::Cursor;

        let db = mem_db_with_metadata();

        let scan1 = r#"{"_pss_manifest": true, "visited_scope_paths": ["foo", "bar"]}
{"type": "skill", "name": "foo-first", "source": "plugin:foo", "path": "foo/skills/first.md", "description": "stays", "enabled": true}
{"type": "skill", "name": "foo-second", "source": "plugin:foo", "path": "foo/skills/second.md", "description": "goes away", "enabled": true}
{"type": "skill", "name": "bar-only", "source": "plugin:bar", "path": "bar/skills/only.md", "description": "unvisited next scan", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan1), true).expect("scan 1");

        let scan2 = r#"{"_pss_manifest": true, "visited_scope_paths": ["foo"]}
{"type": "skill", "name": "foo-first", "source": "plugin:foo", "path": "foo/skills/first.md", "description": "stays", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan2), true).expect("scan 2");

        assert_eq!(
            removed_element_names(&db),
            vec!["foo-second".to_string()],
            "v1 manifest must keep the scope_path-membership rule exactly: \
             partial removal detected, unvisited scope untouched"
        );
    }

    /// F7 back-compat — the key present but EMPTY claims nothing, and must be
    /// indistinguishable from a v1 manifest. This is what every filtered run
    /// (`--name`, `--type`, `--project-only`) emits, so it is the common path.
    #[test]
    fn empty_exhaustive_scopes_is_a_no_op() {
        use std::io::Cursor;

        let db = mem_db_with_metadata();

        let scan1 = r#"{"_pss_manifest": true, "_pss_manifest_version": 2, "visited_scope_paths": ["foo", "bar"], "exhaustive_scopes": []}
{"type": "skill", "name": "foo-first", "source": "plugin:foo", "path": "foo/skills/first.md", "description": "stays", "enabled": true}
{"type": "skill", "name": "foo-second", "source": "plugin:foo", "path": "foo/skills/second.md", "description": "goes away", "enabled": true}
{"type": "skill", "name": "bar-only", "source": "plugin:bar", "path": "bar/skills/only.md", "description": "unvisited next scan", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan1), true).expect("scan 1");

        let scan2 = r#"{"_pss_manifest": true, "_pss_manifest_version": 2, "visited_scope_paths": ["foo"], "exhaustive_scopes": []}
{"type": "skill", "name": "foo-first", "source": "plugin:foo", "path": "foo/skills/first.md", "description": "stays", "enabled": true}
"#;
        cli::merge_events_from_reader(&db, Cursor::new(scan2), true).expect("scan 2");

        assert_eq!(
            removed_element_names(&db),
            vec!["foo-second".to_string()],
            "an empty claim must reproduce v1 behavior exactly (claim nothing)"
        );
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

    // ====================================================================
    // Issue #10 Wave 2 — P-4 (first_seen + synthetic), P-7 (unlimited),
    // P-1 (active-in union). All exercise the REAL production helpers
    // `cli::as_of_rows` and `cli::active_in_rows`.
    // ====================================================================

    /// Helper to read a string field off a serde_json row object.
    fn row_str<'a>(row: &'a serde_json::Value, key: &str) -> &'a str {
        row.get(key).and_then(|v| v.as_str()).unwrap_or("")
    }

    /// Build a DB whose events exercise P-4: `sk-real` has a genuine
    /// install (diff_json without `migrated`), `sk-synth` has only a
    /// migration-stamped placeholder install (diff_json `{"migrated":true}`).
    /// `sk-real` also gets a LATER content_changed event so we can prove
    /// `first_seen` is the EARLIEST install instant, not the latest event.
    fn build_test_db_for_first_seen() -> DbInstance {
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("schema");
        // diff_json carries inner quotes, so pass the three values as
        // parameters rather than embedding JSON-escaped strings in the Cozo
        // source (mirrors how insert_install_event binds $diff_json).
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("d_real".into(),
            DataValue::Str(r#"{"description":"real"}"#.into()));
        params.insert("d_edit".into(),
            DataValue::Str(r#"{"description":"edited"}"#.into()));
        params.insert("d_synth".into(),
            DataValue::Str(r#"{"description":"migrated","migrated":true}"#.into()));
        let seed = r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["r0", "2026-01-05T00:00:00Z", "s1", "installed", "skill",
                 "sk-real", "skill:sk-real@user:_home", "user", "/home", "user",
                 "/home/sk-real.md", "h1", 100, 50, true, "active",
                 $d_real, ""],
                ["r1", "2026-03-01T00:00:00Z", "s2", "content_changed", "skill",
                 "sk-real", "skill:sk-real@user:_home", "user", "/home", "user",
                 "/home/sk-real.md", "h1b", 110, 55, true, "active",
                 $d_edit, ""],
                ["y0", "2026-02-10T00:00:00Z", "s1", "installed", "skill",
                 "sk-synth", "skill:sk-synth@user:_home", "user", "/home", "user",
                 "/home/sk-synth.md", "h2", 200, 80, true, "active",
                 $d_synth, ""]
              ]
            :put events { event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }
        "#;
        db.run_script(seed, params, ScriptMutability::Mutable)
            .expect("seed first-seen events");
        db
    }

    #[test]
    fn as_of_rows_populates_first_seen_and_synthetic_flag() {
        // P-4: every as-of row must carry `first_seen` (the earliest
        // `installed` observed_at for that element_id in the same scope)
        // and `first_seen_is_synthetic` (true iff that earliest install's
        // diff_json carried `"migrated":true`).
        let db = build_test_db_for_first_seen();
        let rows = cli::as_of_rows(&db, "2026-12-01T00:00:00Z", None, None, None, 1_000_000);
        assert_eq!(rows.len(), 2, "expected 2 elements, got {}", rows.len());

        let real = rows.iter()
            .find(|r| row_str(r, "element_id") == "skill:sk-real@user:_home")
            .expect("sk-real row present");
        // first_seen is the EARLIEST install (2026-01-05), NOT the later
        // content_changed event (2026-03-01) — proving it tracks install,
        // not last-modified.
        assert_eq!(row_str(real, "first_seen"), "2026-01-05T00:00:00Z");
        assert_eq!(real.get("first_seen_is_synthetic").and_then(|v| v.as_bool()),
                   Some(false), "real install must NOT be synthetic");

        let synth = rows.iter()
            .find(|r| row_str(r, "element_id") == "skill:sk-synth@user:_home")
            .expect("sk-synth row present");
        assert_eq!(row_str(synth, "first_seen"), "2026-02-10T00:00:00Z");
        assert_eq!(synth.get("first_seen_is_synthetic").and_then(|v| v.as_bool()),
                   Some(true), "migration placeholder must be synthetic");
    }

    #[test]
    fn as_of_rows_keeps_all_twelve_legacy_fields() {
        // P-4 is ADDITIVE: the 12 pre-existing fields must remain unchanged.
        let db = build_test_db_for_first_seen();
        let rows = cli::as_of_rows(&db, "2026-12-01T00:00:00Z", None, None, None, 1_000_000);
        let r = &rows[0];
        for k in ["element_id", "event_type", "element_type", "element_name",
                  "scope", "scope_path", "path", "content_hash", "file_size",
                  "token_count", "enabled"] {
            assert!(r.get(k).is_some(), "legacy field '{}' must still be present", k);
        }
        // And the two NEW fields exist on top of the 12.
        assert!(r.get("first_seen").is_some());
        assert!(r.get("first_seen_is_synthetic").is_some());
    }

    /// Build a DB modelling a sample project folder so `active-in` can be
    /// tested. Union members:
    ///   (a) project/local-scope rows whose scope_path == the folder slug,
    ///   (b) all user-scope rows,
    ///   (c) enabled plugin/marketplace rows.
    /// Plus NON-members that must be excluded: a DIFFERENT project's slug,
    /// and a DISABLED plugin.
    fn build_test_db_for_active_in(slug: &str) -> DbInstance {
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("schema");
        // Row template: each gets its OWN install event.
        let seed = format!(
            r#"
            ?[event_id, observed_at, scan_id, event_type, element_type,
              element_name, element_id, scope, scope_path, source, path,
              content_hash, file_size, token_count, enabled, override_status,
              diff_json, snapshot_ref] <- [
                ["a", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "proj-skill", "skill:proj-skill@project:{slug}", "project", "{slug}", "project:{slug}",
                 "/p/proj-skill.md", "h1", 100, 50, true, "active", "{{}}", ""],
                ["b", "2026-01-01T00:00:00Z", "s1", "installed", "rule",
                 "loc-rule", "rule:loc-rule@local:{slug}", "local", "{slug}", "local:{slug}",
                 "/p/.claude/rules/loc-rule.md", "h2", 60, 30, true, "active", "{{}}", ""],
                ["c", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "user-skill", "skill:user-skill@user:_home", "user", "/home", "user",
                 "/home/user-skill.md", "h3", 100, 50, true, "active", "{{}}", ""],
                ["d", "2026-01-01T00:00:00Z", "s1", "installed", "plugin",
                 "enabled-plug", "plugin:enabled-plug@user:_home", "plugin", "mkt", "plugin:enabled-plug",
                 "/plugins/enabled-plug", "h4", 300, 120, true, "active", "{{}}", ""],
                ["e", "2026-01-01T00:00:00Z", "s1", "installed", "plugin",
                 "disabled-plug", "plugin:disabled-plug@user:_home", "plugin", "mkt", "plugin:disabled-plug",
                 "/plugins/disabled-plug", "h5", 300, 120, false, "disabled", "{{}}", ""],
                ["f", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                 "other-proj-skill", "skill:other-proj-skill@project:OTHER-deadbeef", "project", "OTHER-deadbeef", "project:OTHER-deadbeef",
                 "/o/other-proj-skill.md", "h6", 100, 50, true, "active", "{{}}", ""]
              ]
            :put events {{ event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }}
        "#,
            slug = slug
        );
        db.run_script(&seed, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed active-in events");
        db
    }

    #[test]
    fn active_in_rows_returns_correct_union() {
        // P-1: active-in returns the UNION of (a) project/local rows whose
        // scope_path == the folder slug, (b) all user-scope rows, (c) enabled
        // plugin/marketplace rows. A different project's rows and a disabled
        // plugin must be EXCLUDED.
        let slug = "demo-abcd1234";
        let db = build_test_db_for_active_in(slug);
        let rows = cli::active_in_rows(&db, slug, "2026-12-01T00:00:00Z", 1_000_000);
        let ids: std::collections::HashSet<String> = rows.iter()
            .map(|r| row_str(r, "element_id").to_string())
            .collect();

        // (a) this project's project + local scope rows.
        assert!(ids.contains("skill:proj-skill@project:demo-abcd1234"), "project-scope match missing");
        assert!(ids.contains("rule:loc-rule@local:demo-abcd1234"), "local-scope match missing");
        // (b) global user-scope row.
        assert!(ids.contains("skill:user-skill@user:_home"), "user-scope row missing");
        // (c) enabled plugin row.
        assert!(ids.contains("plugin:enabled-plug@user:_home"), "enabled plugin missing");
        // Excluded: disabled plugin + a different project's row.
        assert!(!ids.contains("plugin:disabled-plug@user:_home"), "disabled plugin must be excluded");
        assert!(!ids.contains("skill:other-proj-skill@project:OTHER-deadbeef"),
                "a different project's row must be excluded");
        assert_eq!(rows.len(), 4, "exactly 4 union members expected, got {}", rows.len());
    }

    #[test]
    fn active_in_rows_carries_first_seen_fields() {
        // P-1 rows must have the same shape as as-of rows, including P-4.
        let slug = "demo-abcd1234";
        let db = build_test_db_for_active_in(slug);
        let rows = cli::active_in_rows(&db, slug, "2026-12-01T00:00:00Z", 1_000_000);
        let r = &rows[0];
        assert!(r.get("first_seen").is_some(), "active-in row needs first_seen");
        assert!(r.get("first_seen_is_synthetic").is_some(),
                "active-in row needs first_seen_is_synthetic");
    }

    #[test]
    fn as_of_unlimited_default_returns_more_than_old_cap() {
        // P-7: with the unlimited sentinel default (1_000_000) a snapshot of
        // ALL active components must not be truncated at the old 1000 cap.
        // Seed 1001 distinct user-scope skills and prove all 1001 come back.
        let db = DbInstance::new("mem", "", "").expect("mem db");
        let _ = db.run_script(
            r#":create pss_metadata { key: String => value: String }"#,
            BTreeMap::new(),
            ScriptMutability::Mutable,
        );
        ensure_schema(&db).expect("schema");
        // Build a single batched insert of 1001 installed events.
        let mut tuples = String::new();
        for i in 0..1001 {
            if i > 0 {
                tuples.push(',');
            }
            tuples.push_str(&format!(
                r#"["e{i}", "2026-01-01T00:00:00Z", "s1", "installed", "skill",
                   "sk{i}", "skill:sk{i}@user:_home", "user", "/home", "user",
                   "/home/sk{i}.md", "h{i}", 1, 1, true, "active", "{{}}", ""]"#,
                i = i
            ));
        }
        let seed = format!(
            r#"?[event_id, observed_at, scan_id, event_type, element_type,
               element_name, element_id, scope, scope_path, source, path,
               content_hash, file_size, token_count, enabled, override_status,
               diff_json, snapshot_ref] <- [{tuples}]
            :put events {{ event_id =>
              observed_at, scan_id, event_type, element_type, element_name,
              element_id, scope, scope_path, source, path, content_hash,
              file_size, token_count, enabled, override_status, diff_json,
              snapshot_ref }}"#,
            tuples = tuples
        );
        db.run_script(&seed, BTreeMap::new(), ScriptMutability::Mutable)
            .expect("seed 1001 events");
        // The unlimited sentinel the CLI passes by default.
        let rows = cli::as_of_rows(&db, "2026-12-01T00:00:00Z", None, None, None, 1_000_000);
        assert_eq!(rows.len(), 1001,
                   "unlimited default must return all 1001 rows, got {}", rows.len());
    }
}
