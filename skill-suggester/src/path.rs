//! Externally-facing path & contract helpers + CozoDB access plumbing
//! (XUD7YUZH modularization step 10 — pure move, zero behavior change).
//! Banner section "Issue #10 P-2 / P-6 / P-9 — externally-facing path &
//! contract helpers" moved here byte-for-byte; every moved top-level item's
//! visibility widened to `pub(crate)` so it stays reachable from `main.rs`
//! via the crate-root re-export (`mod path;` + `pub(crate) use path::*;`).
//!
//! Referenced items from the crate root and sibling modules are imported
//! explicitly because child modules do not see the crate root's glob
//! re-exports.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use cozo::{DataValue, DbInstance, ScriptMutability};
use tracing::{debug, info, warn};

use crate::consts::{CACHE_DIR, DB_FILE};
use crate::scoring::make_entry_id;
use crate::types::SuggesterError;
use crate::{DomainRegistry, DomainRegistryEntry, SkillEntry, SkillIndex};
use crate::{load_ownership_columns, read_version};

// ============================================================================
// Issue #10 P-2 / P-6 / P-9 — externally-facing path & contract helpers.
// These intentionally do NOT touch CozoDB; they let external consumers stop
// reverse-engineering PSS's path-resolution and version contract.
// ============================================================================

/// Stable cross-version contract handle (P-9). Bumped only when the JSON shape
/// or semantics of PSS's external CLI contract change in a breaking way — NOT on
/// every release. Integrators key on `{cli_version, schema_version,
/// contract_version}` to decide whether their assumptions still hold.
pub(crate) const CONTRACT_VERSION: &str = "1";

/// P-2: resolve the canonical DB path PSS *would* use, mirroring
/// [`get_db_path`]'s resolution order (`--index` → `PSS_INDEX_PATH` → default
/// `~/.claude/cache/pss-skill-index.db`) but WITHOUT the existence gate.
///
/// `get_db_path` returns `None` when the file is absent (it is the runtime
/// "open the DB if present" path). A `db-path` consumer instead needs the path
/// it would create/use even before it exists — returning nothing there would
/// defeat the entire purpose of the subcommand. Hence this sibling helper
/// always returns the resolved `PathBuf`.
pub(crate) fn resolve_db_path_canonical(cli_index: Option<&str>) -> PathBuf {
    let env_index = std::env::var("PSS_INDEX_PATH").ok();
    resolve_db_path_canonical_gated(cli_index, env_index.as_deref())
}

/// The pure decision behind [`resolve_db_path_canonical`], with the env value
/// passed IN rather than read from the process environment, so it is testable
/// without `std::env::set_var` (process-global ⇒ racy under cargo's threaded
/// harness). F14 (TRDD-1Z8SGQ7N): mirrors what F12 did for `get_db_path` /
/// `resolve_db_path_gated` — pre-F14 the one existing test of this resolver
/// SKIPPED whenever PSS_INDEX_PATH was set, the exact coverage hole that let
/// F12 survive.
///
/// Deliberate properties (verbatim from the pre-F14 body — do NOT "fix"):
///
///   1. NO existence gate — this is the `db-path` subcommand's resolver; it
///      reports the path PSS *would* use even before the file exists (see the
///      doc-comment on the wrapper).
///   2. The `.db` ASYMMETRY — `--index x.db` → `x.db` itself, but
///      `PSS_INDEX_PATH=x.db` → the SIBLING `pss-skill-index.db`. Mirrored in
///      `scripts/pss_cozodb.py` L150-153 and enforced by
///      `tests/unit/test_pss_db_path_parity.py`.
///   3. `--index ""` FALLS THROUGH to env → home default (see the inline
///      comment in branch 1) — a documented divergence from the runtime
///      resolver, pinned by `f14_cli_empty_index_falls_through_to_env_then_home`.
pub(crate) fn resolve_db_path_canonical_gated(cli_index: Option<&str>, env_index: Option<&str>) -> PathBuf {
    // 1. Explicit --index.
    if let Some(path) = cli_index {
        if path.ends_with(".db") {
            return PathBuf::from(path);
        }
        // JSON path → sibling DB in the same directory.
        let json_path = PathBuf::from(path);
        if let Some(parent) = json_path.parent() {
            return parent.join(DB_FILE);
        }
        // `--index ""` reaches here (no `.db` suffix, and `Path::new("")`
        // has no parent) and FALLS THROUGH to env → home default. This
        // deliberately DIVERGES from the runtime resolver
        // (`resolve_db_path_gated` returns None for the same input) —
        // TRDD-1Z8SGQ7N F12 residual (a): a KNOWN, accepted divergence, only
        // reachable by explicitly passing an empty flag, and
        // `get_index_path("")` already errors on the JSON side. Keep it —
        // changing it must be a conscious decision, not a refactor accident.
    }

    // 2. PSS_INDEX_PATH env var → sibling DB in its directory. An empty value
    //    means "unset" (`std::env::var` yields Ok("") for an
    //    exported-but-empty var), so only a non-empty value engages this
    //    branch. NOTE: no `.ends_with(".db")` shortcut here — the env branch
    //    always takes the sibling (deliberate property 2 above).
    if let Some(path) = env_index {
        if !path.is_empty() {
            let json_path = PathBuf::from(path);
            if let Some(parent) = json_path.parent() {
                return parent.join(DB_FILE);
            }
        }
    }

    // 3. Default: ~/.claude/cache/pss-skill-index.db. If the home dir is
    //    somehow unresolvable, fall back to a relative path so the command
    //    still emits something deterministic rather than panicking.
    match dirs::home_dir() {
        Some(home) => home.join(".claude").join(CACHE_DIR).join(DB_FILE),
        None => PathBuf::from(".claude").join(CACHE_DIR).join(DB_FILE),
    }
}

/// P-2: build the `db-path` subcommand's output string.
/// Bare path on one line by default; `{"db_path":"<abs>"}` with `--format json`.
pub(crate) fn db_path_output(cli_index: Option<&str>, json: bool) -> String {
    let path = resolve_db_path_canonical(cli_index);
    let s = path.to_string_lossy().to_string();
    if json {
        serde_json::json!({ "db_path": s }).to_string()
    } else {
        s
    }
}

/// P-6: replicate Python's `pathlib.Path.resolve(strict=False)` byte-for-byte.
///
/// Python's `_slugify_project_path` hashes `str(project_path.resolve())`, which
/// on macOS canonicalizes through filesystem symlinks (`/tmp` → `/private/tmp`,
/// `/var` → `/private/var`). Rust's `std::fs::canonicalize` is the exact
/// equivalent for paths that EXIST, but it errors on any missing component —
/// whereas Python `resolve(strict=False)` canonicalizes the longest existing
/// ancestor and appends the missing tail literally. We mirror that:
///   1. canonicalize the whole path (fast path — the usual case: real projects);
///   2. on failure, walk up to the longest existing ancestor, canonicalize it,
///      then push the remaining components verbatim.
/// This keeps the slug identical to the discoverer's, so an `element_id` written
/// by Python is reproducible from `pss project-slug <abs>`.
pub(crate) fn python_resolve(input: &Path) -> PathBuf {
    if let Ok(canon) = std::fs::canonicalize(input) {
        return canon;
    }
    // Find the longest existing ancestor; canonicalize it; re-append the tail.
    let mut existing = input;
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        match existing.parent() {
            Some(parent) => {
                if let Some(name) = existing.file_name() {
                    tail.push(name.to_os_string());
                }
                existing = parent;
                if existing.exists() {
                    break;
                }
            }
            None => break, // reached the root with nothing existing
        }
    }
    let mut resolved = std::fs::canonicalize(existing)
        .unwrap_or_else(|_| existing.to_path_buf());
    for name in tail.into_iter().rev() {
        resolved.push(name);
    }
    resolved
}

/// Whether `cwd` is a usable basis for deciding project membership.
///
/// This gates the cross-project filter, so it answers "can we decide?", NOT
/// "is this a project?". The two failure directions are not symmetric, which is
/// the whole reason this is a named function rather than an inline expression:
/// when the basis is undecidable the filter must be DISABLED (keep everything,
/// degrading to the pre-v3.14.2 cross-project leak — visible and recoverable),
/// because applying it would condemn every project-scoped element at once and
/// look exactly like working software.
///
/// Undecidable, so the filter is disabled:
///   - empty / whitespace — `HookInput::cwd` is `#[serde(default)]`, so a
///     payload without the field lands here (the v3.14.2 defect).
///   - RELATIVE — a malformed payload. `python_resolve` would resolve it
///     against the BINARY's own working directory and answer confidently about
///     a location the caller never named. Measured on v3.14.3, which guarded
///     emptiness alone: `cwd: "relative/path"` condemned every project-scoped
///     element exactly as `cwd: ""` used to.
///
///   - the filesystem ROOT, `/`. This one is subtle and was wrong in v3.14.4.
///     `Path::starts_with` is component-wise, so `/` is a prefix of EVERY
///     absolute path — containment carries zero information there. The two
///     source families then disagree: slugged `project:<slug>` rows mismatch
///     and drop, while bare `project` / `project:agentskills` rows are judged
///     by containment, trivially match, and are KEPT. Dropping one shape and
///     keeping the other is not a decision, it is an artefact of spelling.
///     `/` is the ONLY path whose containment test is vacuous — hence
///     `parent().is_some()`, which excludes exactly it.
///
/// Decidable, so the filter stays ENABLED even though it owns nothing:
///   - an absolute path with a parent that is not a project (a deleted
///     directory, an unregistered checkout). The filter then correctly
///     concludes no project owns the caller and drops project-scoped elements
///     because none belong to them. Decidable-and-negative is a right answer,
///     not a failure — do NOT "fix" this by also requiring the path to EXIST.
///     Existence is not the test; a meaningful comparison basis is.
pub(crate) fn cwd_is_usable_basis(cwd: &str) -> bool {
    let trimmed = cwd.trim();
    if trimmed.is_empty() {
        return false;
    }
    let p = Path::new(trimmed);
    // `parent()` is None only at a root, which is the one absolute path where
    // the containment half of the predicate cannot discriminate anything.
    p.is_absolute() && p.parent().is_some()
}

/// The `enabledPlugins` key for an element's `source`, or `None` when the
/// source does not name an installed plugin.
///
/// The ONLY shape that carries both halves of the key is
/// `plugin:<marketplace>/<plugin-name>`, and the key spells them in the
/// opposite order: `<plugin-name>@<marketplace>`.
///
/// Built from `source` and NOT from the `plugin` / `origin` columns, which look
/// like the obvious source of truth and are not:
///   - `plugin` is `None` for a large share of real rows (measured on this
///     machine: every `plugin:ai-maestro-plugins/*` row bar two), so keying on
///     it silently skips the filter for exactly those elements.
///   - `origin` is the marketplace's REPO OWNER (`github.com/davepoon`), not
///     its local name (`buildwithclaude`). `agents-language-specialists@github.com/davepoon`
///     matches nothing in settings, so the filter would read as a clean no-op.
/// `source` carries the local marketplace name, which is what settings keys use.
///
/// `project:<slug>/plugin:<name>` is deliberately NOT handled: a project-local
/// plugin has no marketplace, so no `<name>@<marketplace>` key can exist for it
/// and there is nothing to look up. It falls through as enabled.
pub(crate) fn plugin_enablement_key(source: &str) -> Option<String> {
    let rest = source.strip_prefix("plugin:")?;
    let (marketplace, name) = rest.split_once('/')?;
    if marketplace.is_empty() || name.is_empty() {
        return None;
    }
    Some(format!("{name}@{marketplace}"))
}

/// Every plugin key whose EFFECTIVE enablement is `false`, resolved at USE time
/// with Claude Code's own precedence: user < project < project-local.
///
/// Read on every prompt rather than baked into the index, because enablement is
/// live state: `pss_discover.py` reads `~/.claude/settings.json` once at
/// index-build time and behind an opt-in flag that defaults OFF, so today a
/// toggle changes nothing until the next reindex — and by default not even then.
///
/// FAILS OPEN by construction. A missing, unreadable, or malformed settings file
/// contributes no keys, so the filter degrades to a no-op instead of condemning a
/// scope class. This is the same asymmetry `should_drop_as_foreign_project`
/// applies to an unusable `cwd`: an unknown that destroys the BASIS must not be
/// allowed to empty the suggestion set. A plugin key absent from every layer is
/// ENABLED — `enabledPlugins` lists only what has been explicitly toggled.
pub(crate) fn disabled_plugin_keys(cwd: &str) -> HashSet<String> {
    let mut layers: Vec<PathBuf> = Vec::new();
    if let Some(home) = dirs::home_dir() {
        layers.push(home.join(".claude").join("settings.json"));
    }
    // Project layers only when `cwd` is a usable comparison basis — the same
    // guard the cross-project filter uses, for the same reason: resolving a
    // relative or empty `cwd` would read some unrelated directory's settings
    // and apply another project's overrides here.
    if cwd_is_usable_basis(cwd) {
        let root = owning_project_root(Path::new(cwd.trim()));
        layers.push(root.join(".claude").join("settings.json"));
        layers.push(root.join(".claude").join("settings.local.json"));
    }

    let texts: Vec<String> = layers
        .iter()
        .map(|p| std::fs::read_to_string(p).unwrap_or_default())
        .collect();
    merge_enablement_layers(texts.iter().map(String::as_str))
}

/// The precedence resolution itself, over settings TEXTS in LOW-to-HIGH order —
/// split out from the file reading so the layering is unit-testable without
/// touching `$HOME` (an env-var swap that races every other test in the binary).
///
/// **Reads the VALUE; never tests for presence.** An explicit `true` in a higher
/// layer must override a `false` below it, so collecting "keys that say false"
/// per layer and unioning them would keep a plugin the project deliberately
/// re-enabled. Confirmed against real data: `~/agents/frank/.claude/settings.local.json`
/// carries 37 entries of BOTH polarities, not a disable-list.
///
/// An unparseable layer, a missing `enabledPlugins`, or an empty one contributes
/// nothing and leaves the layers below it standing — the fail-open case, and the
/// common one (7 agent workdirs on this machine ship an EMPTY `enabledPlugins`).
/// Sibling keys such as `permissions` and `crossSessionInbound` are ignored.
pub(crate) fn merge_enablement_layers<'a>(texts: impl IntoIterator<Item = &'a str>) -> HashSet<String> {
    let mut effective: HashMap<String, bool> = HashMap::new();
    for text in texts {
        let Ok(json) = serde_json::from_str::<serde_json::Value>(text) else {
            continue;
        };
        // `enabledPlugins` is hand-editable: anything that is not an object of
        // booleans is ignored key-by-key rather than aborting the layer.
        let Some(map) = json.get("enabledPlugins").and_then(|v| v.as_object()) else {
            continue;
        };
        for (key, value) in map {
            if let Some(flag) = value.as_bool() {
                effective.insert(key.clone(), flag);
            }
        }
    }
    effective
        .into_iter()
        .filter_map(|(key, enabled)| (!enabled).then_some(key))
        .collect()
}

/// The COMPLETE retain predicate — all four exclusion classes in one place,
/// so the whole decision is unit-testable and `run()`'s closure is a single
/// call rather than a conjunction assembled at the call site.
///
/// HONEST LIMIT, stated because the previous two versions of this comment
/// overclaimed: extracting this shrinks the untested boundary to one line
/// (the `retain(...)` call itself) but cannot eliminate it. Deleting that call
/// still leaves the suite green. Moving a boundary is not closing it, and only
/// an integration test that drives `run()` end-to-end would close this one.
/// What IS covered is every rule and their composition; what is not is the
/// single act of calling it.
pub(crate) fn candidate_is_invocable_here(
    cwd: &str,
    source: &str,
    path: &str,
    id: &str,
    noninvocable: &HashSet<String>,
    disabled_plugins: &HashSet<String>,
) -> bool {
    !source.starts_with("marketplace:")
        && !noninvocable.contains(id)
        && !should_drop_as_foreign_project(cwd, source, path)
        // Fourth class: the element is real and invocable in principle, but the
        // harness has not loaded its plugin, so naming it is unactionable — the
        // same defect as a cross-project leak on an orthogonal axis.
        && !plugin_enablement_key(source)
            .is_some_and(|key| disabled_plugins.contains(&key))
}

/// Resolve `cwd` to the root of the project that OWNS it.
///
/// This is the fix for the structural error the first four `cwd` guards were
/// all patching around. Discovery derives every slug from a project ROOT (the
/// registry in `~/.claude.json`, or the cwd's own `.claude/`), so comparing a
/// slug computed from a RAW cwd answers "is cwd exactly a project root?" — not
/// "which project owns cwd?". Those differ the moment anyone prompts from a
/// subdirectory: from `/Users/me/proj/src` the slug is `src-<hash(.../src)>`
/// while the element carries `proj-<hash(.../proj)>`, so a project's own
/// elements read as foreign and are dropped. Measured on v3.14.4:
/// `cwd=.../SVG-MATRIX` suggested that project's agents, `cwd=.../SVG-MATRIX/src`
/// suggested none.
///
/// Walking up to the nearest ancestor containing `.claude/` mirrors what
/// discovery actually scans (`pss_discover.py` adds `cwd/.claude` for the
/// current project), so the two ends agree by construction rather than by
/// coincidence. No `.claude/` anywhere up the chain means cwd is not inside a
/// project at all — return it unchanged, and the caller then correctly finds
/// that no project owns it.
/// `$HOME/.claude` is the USER scope, NOT a project marker — treating it as one
/// makes every non-project directory under `$HOME` "belong to" a `$HOME`
/// project. Measured here: `$HOME` is in the registry with 569 elements (the
/// hooks out of `~/.claude/settings.json`), so a cwd like `~/scratch` adopted
/// all of them. Skipping this ONE directory still lets a cwd of exactly `$HOME`
/// resolve correctly: the walk then finds no marker and returns cwd unchanged,
/// whose slug is `$HOME`'s own.
pub(crate) fn is_user_scope_claude_dir(dir: &Path) -> bool {
    std::env::var_os("HOME")
        .map(|h| dir == Path::new(&h))
        .unwrap_or(false)
}

pub(crate) fn owning_project_root(cwd: &Path) -> PathBuf {
    // Canonicalize BEFORE walking, never after. Discovery slugs a RESOLVED path,
    // so walking the raw path and resolving the result lets a symlinked cwd
    // terminate at a different root than the one the element was keyed under —
    // the same two-ends-must-agree defect as the trim inconsistency, in path
    // form. Resolve once here; the caller then resolves the (already-resolved)
    // root idempotently.
    let start = python_resolve(cwd);
    let mut cur = Some(start.as_path());
    while let Some(dir) = cur {
        if !is_user_scope_claude_dir(dir) && dir.join(".claude").is_dir() {
            return dir.to_path_buf();
        }
        cur = dir.parent();
    }
    start.clone()
}

/// The whole cross-project decision for ONE candidate: the `cwd` guard AND the
/// origin predicate, composed. `run()` calls exactly this, so the WIRING
/// between guard and predicate is covered by tests rather than living as an
/// untested expression at the call site.
///
/// That is the point of the composition. A previous version applied the guard
/// inline in `run()` and unit-tested the guard separately, which meant deleting
/// or inverting the guard left every test green while re-arming a 689-element
/// wipe. A test of the parts is not a test of the assembly.
///
/// **`cwd` is trimmed ONCE here and the trimmed value feeds BOTH the guard and
/// the slug/path computation.** They must not disagree: guarding on the trimmed
/// string while hashing the raw one lets `" /Users/me/proj"` pass the guard and
/// then hash to a slug that matches no project — condemning everything through
/// a door the guard held open.
pub(crate) fn should_drop_as_foreign_project(cwd: &str, source: &str, path: &str) -> bool {
    let cwd = cwd.trim();
    if !cwd_is_usable_basis(cwd) {
        return false; // undecidable basis → keep everything; never mass-condemn
    }
    // Compare against the OWNING PROJECT ROOT, never the raw cwd — see
    // `owning_project_root`. Using the raw cwd asks "is cwd exactly a project
    // root", which drops a project's own elements from any subdirectory.
    let here = owning_project_root(Path::new(cwd));
    is_foreign_project_element(source, path, &project_slug(&here), &python_resolve(&here))
}

/// True when `source` names a project OTHER than the one we are prompting from.
///
/// The index is CROSS-PROJECT by construction: `pss_reindex.py` invokes
/// `pss_discover.py` with `--all-projects`, so every registered project's
/// `project:`/`local:` elements live in one DB (measured on the live index
/// 2026-08-29: 689 `project:` rows spanning 20 distinct projects, plus 111
/// `local:` rows). Nothing downstream keys on an element's origin, so without
/// this predicate a prompt typed in project B is scored against B's elements
/// AND all 19 others'. Those are not merely irrelevant — the harness has not
/// loaded them, so a suggestion naming one is unactionable by construction.
///
/// Comparing against the LIVE cwd is also what makes a mid-session `/cd`
/// correct for free: every prompt carries its own `cwd`, so the first prompt
/// after the move already filters to the new project — no reindex, no
/// `CwdChanged` hook, no window in which the stale set is still scoreable.
///
/// Two per-project source shapes exist and they are NOT spelled the same way;
/// this is why the check cannot be one `starts_with`:
///   - `project:<slug>` and `project:<slug>/plugin:<name>` — a SLUG
///     (`<basename>-<8 hex>`), compared with `project_slug`.
///   - `local:<absolute path>` — a raw PATH, compared after `python_resolve`.
///
/// Everything else is deliberately kept: `user:`, `plugin:`, `built-in:` are
/// global, and `marketplace:` is already dropped by the caller.
///
/// Two shapes carry NO slug and mean "whatever the cwd was AT INDEX TIME", so
/// neither can be judged from the source string alone — bare `project` (that
/// project's `.claude/`) and `project:agentskills` (its `.agents/`, a literal
/// label, `pss_discover.py:827`). They are decided by `path` instead, and the
/// asymmetry matters in both directions:
///   - Keeping them unconditionally leaks the INDEX-TIME project's elements
///     into every other project — the same bug this function exists to fix,
///     surviving in the one case a slug comparison cannot see.
///   - Dropping them unconditionally is just as wrong, and this is the
///     non-obvious half: `pss_discover.py:946` seeds `seen_project_paths` with
///     `{cwd}` and skips it in the registry loop, so the cwd project is emitted
///     ONLY in bare form — there is no slugged duplicate to fall back on.
///     Dropping it would erase the current project's own elements outright.
/// A path test is the only honest discriminator, and it is exact: those
/// elements live under `<project>/.claude/` or `<project>/.agents/` by
/// construction, so containment in the live cwd IS project membership.
pub(crate) fn is_foreign_project_element(
    source: &str,
    path: &str,
    here_slug: &str,
    here_path: &Path,
) -> bool {
    // Unslugged, index-time-bound shapes → decide by path containment.
    let unslugged = source == "project"
        || source == "project:agentskills"
        || source.starts_with("project:agentskills/");
    if unslugged {
        // An empty path cannot be proven to belong here. Treat it as foreign:
        // a false drop costs one missing suggestion, a false keep silently
        // reintroduces the cross-project leak for every prompt.
        if path.is_empty() {
            return true;
        }
        return !python_resolve(Path::new(path)).starts_with(here_path);
    }
    if let Some(rest) = source.strip_prefix("project:") {
        // `project:<slug>/plugin:<name>` — the slug ends at the first '/'.
        return rest.split('/').next().unwrap_or("") != here_slug;
    }
    if let Some(p) = source.strip_prefix("local:") {
        return python_resolve(Path::new(p)) != here_path;
    }
    false
}

/// P-6: compute the scope-path slug EXACTLY as
/// `scripts/pss_discover.py::_slugify_project_path` does:
/// `"<basename>-<first 8 hex chars of sha256(resolved_abs_path)>"`.
///
/// The basename comes from the INPUT argument (Python uses `project_path.name`,
/// not the resolved path's name), while the hash is over the RESOLVED path —
/// matching Python precisely.
pub(crate) fn project_slug(input: &Path) -> String {
    use sha2::{Digest, Sha256};
    let resolved = python_resolve(input);
    let resolved_str = resolved.to_string_lossy();
    let digest = Sha256::digest(resolved_str.as_bytes());
    // hex-encode the first 4 bytes → 8 lowercase hex chars (== Python `[:8]`).
    let mut hex = String::with_capacity(8);
    for byte in &digest[..4] {
        hex.push_str(&format!("{:02x}", byte));
    }
    // Python `Path(arg).name`: "" for "/" and other root-only inputs.
    let basename = input
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    format!("{}-{}", basename, hex)
}

/// P-6: build the `project-slug` subcommand's output string.
/// Bare slug by default; `{"abs_path":"<in>","slug":"<out>"}` with `--format json`.
pub(crate) fn project_slug_output(input: &str, json: bool) -> String {
    let slug = project_slug(Path::new(input));
    if json {
        serde_json::json!({ "abs_path": input, "slug": slug }).to_string()
    } else {
        slug
    }
}

/// P-9: build the `--contract-version` output:
/// `{"cli_version":"<runtime VERSION>","schema_version":"<TEMPORAL_SCHEMA_VERSION>","contract_version":"1"}`.
///
/// `cli_version` MUST come from `read_version()` (the runtime VERSION file), NOT
/// `env!("CARGO_PKG_VERSION")` (compile-time). The project bumps the version
/// without rebuilding the binary on a docs-only release (per CLAUDE.md: "the Rust
/// binary reads the version at runtime from the VERSION file"). Using the
/// compile-time constant made the integrator-facing contract surface disagree with
/// `--version` after such a release — e.g. v3.8.2 (docs-only) left the unrebuilt
/// binary reporting `cli_version: 3.8.1` while `--version` correctly read 3.8.2.
pub(crate) fn contract_version_output() -> String {
    serde_json::json!({
        "cli_version": read_version(),
        "schema_version": crate::temporal::TEMPORAL_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
    })
    .to_string()
}

/// Open a CozoDB instance with SQLite backend at the given path.
pub(crate) fn open_db(path: &Path) -> Result<DbInstance, SuggesterError> {
    DbInstance::new("sqlite", path.to_str().unwrap_or(""), Default::default())
        .map_err(|e| SuggesterError::IndexParse(format!("CozoDB open failed: {}", e)))
}

/// F3 (TRDD-1Z8SGQ7N): build the flock path for a DB coordination file.
/// The suffix is appended to the FULL db filename — `pss-skill-index.db` +
/// `.lock` → `pss-skill-index.db.lock` — because that is the exact spelling
/// both Python sides use (pss_paths.get_db_lock_path for the hook's reader
/// LOCK_SH; pss_cozodb's WRITE_LOCK_SUFFIX for the skills writer's LOCK_EX).
/// A lock on any other spelling coordinates with nobody.
pub(crate) fn db_flock_path(db_path: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{}", db_path.display(), suffix))
}

/// F3 (TRDD-1Z8SGQ7N): take a BLOCKING exclusive flock on `<db><suffix>` and
/// return the open handle as an RAII guard (the OS lock releases on drop /
/// process exit). Failure is fatal by design: an unlocked merge-events write
/// is exactly the reproduced `database is locked` / partial-state corruption
/// this lock exists to prevent, so "proceed unlocked" would trade a visible
/// error for silent damage (project fail-fast rule). Blocking (not try_lock)
/// is intentional too — hook readers hold their LOCK_SH for at most one
/// bounded binary call and release on process exit, so the wait is short and
/// self-clearing, whereas bailing out would drop the reindex's stage 4.
pub(crate) fn acquire_db_flock(db_path: &Path, suffix: &str) -> std::fs::File {
    let lock_path = db_flock_path(db_path, suffix);
    use fs2::FileExt;
    // truncate(false): the lock file exists only to carry an fd for flock —
    // its CONTENT is never read or written, so we must not clobber whatever a
    // prior holder left in it (and clippy requires the choice be explicit).
    let f = match std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "merge-events failed: cannot open lock file {}: {}",
                lock_path.display(),
                e
            );
            std::process::exit(1);
        }
    };
    // Blocking exclusive flock; released when the returned File drops (RAII)
    // or the process exits. fs2::lock_exclusive maps to flock(LOCK_EX) on Unix
    // and LockFileEx on Windows.
    if let Err(e) = f.lock_exclusive() {
        eprintln!(
            "merge-events failed: cannot flock {}: {}",
            lock_path.display(),
            e
        );
        std::process::exit(1);
    }
    f
}

/// Helper: build inline Datalog data from a vec of (skill_name, value) pairs.
/// Returns a string like: [["skill1", "kw1"], ["skill1", "kw2"], ...]
pub(crate) fn build_inline_data(pairs: &[(String, String)]) -> String {
    pairs.iter()
        .map(|(name, val)| {
            format!("[\"{}\", \"{}\"]",
                name.replace('\\', "\\\\").replace('"', "\\\""),
                val.replace('\\', "\\\\").replace('"', "\\\""))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Batch-insert all skills from a SkillIndex into the CozoDB.
/// Uses batched inline data (500 rows per query) for efficiency.
///
/// Kept as a reference implementation alongside
/// `scripts/pss_cozodb.py` after the Phase C (v3.0.0) removal of
/// `run_build_db`. No Rust code path calls this function.
#[allow(dead_code)]
pub(crate) fn insert_skills_batch(
    db: &DbInstance,
    skills: &HashMap<String, SkillEntry>,
) -> Result<usize, SuggesterError> {
    let mut count = 0;

    // Collect normalized relation data for batch insert
    let mut kw_pairs: Vec<(String, String)> = Vec::new();
    let mut intent_pairs: Vec<(String, String)> = Vec::new();
    let mut tool_pairs: Vec<(String, String)> = Vec::new();
    let mut svc_pairs: Vec<(String, String)> = Vec::new();
    let mut fw_pairs: Vec<(String, String)> = Vec::new();
    let mut lang_pairs: Vec<(String, String)> = Vec::new();
    let mut plat_pairs: Vec<(String, String)> = Vec::new();
    let mut domain_pairs: Vec<(String, String)> = Vec::new();
    let mut ft_pairs: Vec<(String, String)> = Vec::new();
    let mut id_pairs: Vec<(String, String, String)> = Vec::new();

    // Insert main skills table in batches of 100 (using parameterized queries)
    let skill_entries: Vec<(&String, &SkillEntry)> = skills.iter().collect();
    for chunk in skill_entries.chunks(100) {
        for &(id_key, entry) in chunk {
            // HashMap key is already the entry ID after load_index() re-keying
            let entry_id = id_key.clone();
            let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
            params.insert("name".into(), DataValue::Str(entry.name.clone().into()));
            params.insert("source".into(), DataValue::Str(entry.source.clone().into()));
            params.insert("id".into(), DataValue::Str(entry_id.clone().into()));
            params.insert("path".into(), DataValue::Str(entry.path.clone().into()));
            params.insert("skill_type".into(), DataValue::Str(entry.skill_type.clone().into()));
            params.insert("description".into(), DataValue::Str(
                entry.description.chars().take(500).collect::<String>().into()));
            params.insert("tier".into(), DataValue::Str(entry.tier.clone().into()));
            params.insert("boost".into(), DataValue::from(entry.boost as i64));
            params.insert("category".into(), DataValue::Str(entry.category.clone().into()));
            params.insert("server_type".into(), DataValue::Str(entry.server_type.clone().into()));
            params.insert("server_command".into(), DataValue::Str(entry.server_command.clone().into()));
            params.insert("server_args_json".into(), DataValue::Str(
                serde_json::to_string(&entry.server_args).unwrap_or_default().into()));
            params.insert("language_ids_json".into(), DataValue::Str(
                serde_json::to_string(&entry.language_ids).unwrap_or_default().into()));
            params.insert("negative_kw_json".into(), DataValue::Str(
                serde_json::to_string(&entry.negative_keywords).unwrap_or_default().into()));
            params.insert("patterns_json".into(), DataValue::Str(
                serde_json::to_string(&entry.patterns).unwrap_or_default().into()));
            params.insert("directories_json".into(), DataValue::Str(
                serde_json::to_string(&entry.directories).unwrap_or_default().into()));
            params.insert("path_patterns_json".into(), DataValue::Str(
                serde_json::to_string(&entry.path_patterns).unwrap_or_default().into()));
            params.insert("use_cases_json".into(), DataValue::Str(
                serde_json::to_string(&entry.use_cases).unwrap_or_default().into()));
            params.insert("co_usage_json".into(), DataValue::Str(
                serde_json::to_string(&entry.co_usage).unwrap_or_default().into()));
            params.insert("alternatives_json".into(), DataValue::Str(
                serde_json::to_string(&entry.alternatives).unwrap_or_default().into()));
            params.insert("domain_gates_json".into(), DataValue::Str(
                serde_json::to_string(&entry.domain_gates).unwrap_or_default().into()));
            params.insert("file_types_json".into(), DataValue::Str(
                serde_json::to_string(&entry.file_types).unwrap_or_default().into()));
            // Store vector fields as JSON for single-query candidate loading
            params.insert("keywords_json".into(), DataValue::Str(
                serde_json::to_string(&entry.keywords).unwrap_or_default().into()));
            params.insert("intents_json".into(), DataValue::Str(
                serde_json::to_string(&entry.intents).unwrap_or_default().into()));
            params.insert("tools_json".into(), DataValue::Str(
                serde_json::to_string(&entry.tools).unwrap_or_default().into()));
            params.insert("services_json".into(), DataValue::Str(
                serde_json::to_string(&entry.services).unwrap_or_default().into()));
            params.insert("frameworks_json".into(), DataValue::Str(
                serde_json::to_string(&entry.frameworks).unwrap_or_default().into()));
            params.insert("languages_json".into(), DataValue::Str(
                serde_json::to_string(&entry.languages).unwrap_or_default().into()));
            params.insert("platforms_json".into(), DataValue::Str(
                serde_json::to_string(&entry.platforms).unwrap_or_default().into()));
            params.insert("domains_json".into(), DataValue::Str(
                serde_json::to_string(&entry.domains).unwrap_or_default().into()));
            params.insert("path_gates_json".into(), DataValue::Str(
                serde_json::to_string(&entry.path_gates).unwrap_or_default().into()));

            // Timestamps: preserve first_indexed_at across rebuilds (set to "now"
            // only on the very first insert); refresh last_updated_at on every
            // write. run_build_db snapshots old timestamps before wipe and passes
            // them through on the entry — so a non-empty first_indexed_at here
            // means "this element was already in the prior DB" and we preserve it.
            let now_rfc3339 = chrono::Utc::now().to_rfc3339_opts(
                chrono::SecondsFormat::Secs, true);
            let first_indexed_at = if entry.first_indexed_at.is_empty() {
                now_rfc3339.clone()
            } else {
                entry.first_indexed_at.clone()
            };
            params.insert("first_indexed_at".into(),
                DataValue::Str(first_indexed_at.into()));
            params.insert("last_updated_at".into(),
                DataValue::Str(now_rfc3339.into()));

            db.run_script(
                "?[name, id, path, skill_type, source, description, tier, boost, category, \
                 server_type, server_command, server_args_json, language_ids_json, \
                 negative_kw_json, patterns_json, directories_json, path_patterns_json, \
                 use_cases_json, co_usage_json, alternatives_json, domain_gates_json, file_types_json, \
                 keywords_json, intents_json, tools_json, services_json, frameworks_json, languages_json, platforms_json, domains_json, path_gates_json, \
                 first_indexed_at, last_updated_at] <- \
                 [[$name, $id, $path, $skill_type, $source, $description, $tier, $boost, $category, \
                   $server_type, $server_command, $server_args_json, $language_ids_json, \
                   $negative_kw_json, $patterns_json, $directories_json, $path_patterns_json, \
                   $use_cases_json, $co_usage_json, $alternatives_json, $domain_gates_json, $file_types_json, \
                   $keywords_json, $intents_json, $tools_json, $services_json, $frameworks_json, $languages_json, $platforms_json, $domains_json, $path_gates_json, \
                   $first_indexed_at, $last_updated_at]] \
                 :put skills { name, source => id, path, skill_type, description, tier, boost, category, \
                              server_type, server_command, server_args_json, language_ids_json, \
                              negative_kw_json, patterns_json, directories_json, path_patterns_json, \
                              use_cases_json, co_usage_json, alternatives_json, domain_gates_json, file_types_json, \
                              keywords_json, intents_json, tools_json, services_json, frameworks_json, languages_json, platforms_json, domains_json, path_gates_json, \
                              first_indexed_at, last_updated_at }",
                params,
                ScriptMutability::Mutable,
            ).map_err(|e| SuggesterError::IndexParse(
                format!("Insert skill '{}' failed: {}", entry.name, e)
            ))?;

            // Collect normalized data for batch insert (keyed by element name, not entry ID)
            for kw in &entry.keywords {
                kw_pairs.push((entry.name.clone(), kw.clone()));
            }
            for intent in &entry.intents {
                intent_pairs.push((entry.name.clone(), intent.clone()));
            }
            for tool in &entry.tools {
                tool_pairs.push((entry.name.clone(), tool.clone()));
            }
            for svc in &entry.services {
                svc_pairs.push((entry.name.clone(), svc.clone()));
            }
            for fw in &entry.frameworks {
                fw_pairs.push((entry.name.clone(), fw.clone()));
            }
            for lang in &entry.languages {
                lang_pairs.push((entry.name.clone(), lang.clone()));
            }
            for plat in &entry.platforms {
                plat_pairs.push((entry.name.clone(), plat.clone()));
            }
            for domain in &entry.domains {
                domain_pairs.push((entry.name.clone(), domain.clone()));
            }
            for ft in &entry.file_types {
                ft_pairs.push((entry.name.clone(), ft.clone()));
            }
            // Collect ID → (name, source) mapping for batch insert into skill_ids
            id_pairs.push((entry_id, entry.name.clone(), entry.source.clone()));

            count += 1;
        }
    }

    // Batch-insert normalized relations (500 rows per query to avoid query size limits)
    let batch_insert = |rel: &str, pairs: &[(String, String)]| -> Result<(), SuggesterError> {
        for chunk in pairs.chunks(500) {
            let data = build_inline_data(chunk);
            if data.is_empty() { continue; }
            let script = format!(
                "?[skill_name, value] <- [{}] :put {} {{ skill_name, value }}",
                data, rel
            );
            db.run_script(&script, Default::default(), ScriptMutability::Mutable)
                .map_err(|e| SuggesterError::IndexParse(
                    format!("Batch insert {} failed: {}", rel, e)
                ))?;
        }
        Ok(())
    };

    batch_insert("skill_keywords", &kw_pairs)?;
    batch_insert("skill_intents", &intent_pairs)?;
    batch_insert("skill_tools", &tool_pairs)?;
    batch_insert("skill_services", &svc_pairs)?;
    batch_insert("skill_frameworks", &fw_pairs)?;
    batch_insert("skill_languages", &lang_pairs)?;
    batch_insert("skill_platforms", &plat_pairs)?;
    batch_insert("skill_domains", &domain_pairs)?;
    batch_insert("skill_file_types", &ft_pairs)?;

    // Build unified kw_lookup from all inverted sources.
    // keyword_lower is the first key → O(log n) prefix scan at query time.
    {
        let mut lookup_pairs: Vec<(String, String)> = Vec::new();
        let all_sources: [&Vec<(String, String)>; 9] = [
            &kw_pairs, &intent_pairs, &tool_pairs, &svc_pairs,
            &fw_pairs, &lang_pairs, &plat_pairs, &domain_pairs, &ft_pairs,
        ];
        for source in &all_sources {
            for (skill_name, value) in *source {
                let lower = value.to_lowercase();
                if lower.len() >= 2 {
                    lookup_pairs.push((lower, skill_name.clone()));
                }
            }
        }
        // Deduplicate
        lookup_pairs.sort();
        lookup_pairs.dedup();
        // Custom insert: kw_lookup uses {keyword_lower, skill_name} not {skill_name, value}
        for chunk in lookup_pairs.chunks(500) {
            let data: String = chunk.iter()
                .map(|(kw, name)| format!(
                    "[\"{}\", \"{}\"]",
                    kw.replace('\\', "\\\\").replace('"', "\\\""),
                    name.replace('\\', "\\\\").replace('"', "\\\""),
                ))
                .collect::<Vec<_>>()
                .join(", ");
            if data.is_empty() { continue; }
            let script = format!(
                "?[keyword_lower, skill_name] <- [{}] :put kw_lookup {{ keyword_lower, skill_name }}",
                data
            );
            db.run_script(&script, Default::default(), ScriptMutability::Mutable)
                .map_err(|e| SuggesterError::IndexParse(
                    format!("Batch insert kw_lookup failed: {}", e)
                ))?;
        }
        info!("Built kw_lookup with {} entries", lookup_pairs.len());
    }

    // Batch-insert ID → (name, source) mappings into skill_ids lookup table
    for chunk in id_pairs.chunks(500) {
        let data: String = chunk.iter()
            .map(|(id, name, source)| {
                format!("[\"{}\", \"{}\", \"{}\"]",
                    id.replace('\\', "\\\\").replace('"', "\\\""),
                    name.replace('\\', "\\\\").replace('"', "\\\""),
                    source.replace('\\', "\\\\").replace('"', "\\\""))
            })
            .collect::<Vec<_>>()
            .join(", ");
        if data.is_empty() { continue; }
        let script = format!(
            "?[id, name, source] <- [{}] :put skill_ids {{ id => name, source }}",
            data
        );
        db.run_script(&script, Default::default(), ScriptMutability::Mutable)
            .map_err(|e| SuggesterError::IndexParse(
                format!("Batch insert skill_ids failed: {}", e)
            ))?;
    }

    Ok(count)
}

// Phase C (v3.0.0): `run_build_db` has been removed. The Python merge writer
// (`scripts/pss_merge_queue.py::_sync_cozodb`) populates CozoDB directly
// during each merge — see `scripts/pss_cozodb.py::atomic_write_cozodb`. The
// schema is DEFINED ONLY in `scripts/pss_cozodb.py::_create_db_schema`; the
// Rust `create_db_schema` copy was deleted (TRDD-AS90UUQ5) after it silently
// drifted from the live schema (missing `disable_model_invocation`), proving
// a dead "reference copy" guards nothing. The data helpers
// `insert_skills_batch` and `insert_domain_registry_batch` remain below with
// `#[allow(dead_code)]` as insert-shape references only.

/// Batch-insert domain registry data into CozoDB.
/// Populates domain_registry, domain_aliases, domain_keywords, domain_skills tables.
///
/// Kept as a reference implementation alongside
/// `scripts/pss_cozodb.py` after the Phase C (v3.0.0) removal of
/// `run_build_db`. No Rust code path calls this function.
#[allow(dead_code)]
pub(crate) fn insert_domain_registry_batch(
    db: &DbInstance,
    registry: &DomainRegistry,
) -> Result<usize, SuggesterError> {
    let mut count = 0;
    let mut alias_pairs: Vec<(String, String)> = Vec::new();
    let mut keyword_pairs: Vec<(String, String)> = Vec::new();
    let mut skill_pairs: Vec<(String, String)> = Vec::new();

    // Insert each domain into domain_registry (parameterized, one at a time)
    for (canonical, entry) in &registry.domains {
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        params.insert("canonical_name".into(), DataValue::Str(canonical.clone().into()));
        params.insert("has_generic".into(), DataValue::Bool(entry.has_generic));
        params.insert("skill_count".into(), DataValue::from(entry.skill_count as i64));

        db.run_script(
            "?[canonical_name, has_generic, skill_count] <- \
             [[$canonical_name, $has_generic, $skill_count]] \
             :put domain_registry { canonical_name => has_generic, skill_count }",
            params,
            ScriptMutability::Mutable,
        ).map_err(|e| SuggesterError::IndexParse(
            format!("Insert domain '{}' failed: {}", canonical, e)
        ))?;

        // Collect normalized data for batch insert
        for alias in &entry.aliases {
            alias_pairs.push((alias.clone(), canonical.clone()));
        }
        for kw in &entry.example_keywords {
            keyword_pairs.push((canonical.clone(), kw.clone()));
        }
        for skill in &entry.skills {
            skill_pairs.push((canonical.clone(), skill.clone()));
        }

        count += 1;
    }

    // Batch-insert normalized relations (500 rows per query)
    let batch_insert_2col = |rel: &str, col1: &str, col2: &str, pairs: &[(String, String)]| -> Result<(), SuggesterError> {
        for chunk in pairs.chunks(500) {
            let data = build_inline_data(chunk);
            if data.is_empty() { continue; }
            let script = format!(
                "?[{}, {}] <- [{}] :put {} {{ {}, {} }}",
                col1, col2, data, rel, col1, col2
            );
            db.run_script(&script, Default::default(), ScriptMutability::Mutable)
                .map_err(|e| SuggesterError::IndexParse(
                    format!("Batch insert {} failed: {}", rel, e)
                ))?;
        }
        Ok(())
    };

    batch_insert_2col("domain_aliases", "alias", "canonical_name", &alias_pairs)?;
    batch_insert_2col("domain_keywords", "canonical_name", "keyword", &keyword_pairs)?;
    batch_insert_2col("domain_skills", "canonical_name", "skill_name", &skill_pairs)?;

    Ok(count)
}

/// Load domain registry from CozoDB.
/// Reconstructs DomainRegistry from the domain_registry, domain_aliases,
/// domain_keywords, and domain_skills tables.
pub(crate) fn load_domain_registry_from_db(db: &DbInstance) -> Result<Option<DomainRegistry>, SuggesterError> {
    // Check if domain_registry table has any data
    let count_result = db.run_script(
        "?[count(canonical_name)] := *domain_registry{ canonical_name }",
        Default::default(),
        ScriptMutability::Immutable,
    );

    let domain_count = match count_result {
        Ok(ref result) => {
            result.rows.first()
                .and_then(|row| row.first())
                .and_then(|v| match v {
                    DataValue::Num(cozo::Num::Int(n)) => Some(*n as usize),
                    DataValue::Num(cozo::Num::Float(f)) => Some(*f as usize),
                    _ => None,
                })
                .unwrap_or(0)
        }
        Err(_) => return Ok(None), // Table doesn't exist or query failed — no registry
    };

    if domain_count == 0 {
        debug!("No domain registry data in CozoDB");
        return Ok(None);
    }

    // Load all domain entries
    let domains_result = db.run_script(
        "?[canonical_name, has_generic, skill_count] := *domain_registry{ canonical_name, has_generic, skill_count }",
        Default::default(),
        ScriptMutability::Immutable,
    ).map_err(|e| SuggesterError::IndexParse(format!("Load domain_registry failed: {}", e)))?;

    let mut domains: HashMap<String, DomainRegistryEntry> = HashMap::new();

    for row in &domains_result.rows {
        let canonical = match row.first() {
            Some(DataValue::Str(s)) => s.to_string(),
            _ => continue,
        };
        let has_generic = match row.get(1) {
            Some(DataValue::Bool(b)) => *b,
            _ => false,
        };
        let skill_count = match row.get(2) {
            Some(DataValue::Num(cozo::Num::Int(n))) => *n as usize,
            Some(DataValue::Num(cozo::Num::Float(f))) => *f as usize,
            _ => 0,
        };

        domains.insert(canonical.clone(), DomainRegistryEntry {
            canonical_name: canonical,
            aliases: Vec::new(),
            example_keywords: Vec::new(),
            has_generic,
            skill_count,
            skills: Vec::new(),
        });
    }

    // Load aliases: alias → canonical_name
    let aliases_result = db.run_script(
        "?[alias, canonical_name] := *domain_aliases{ alias, canonical_name }",
        Default::default(),
        ScriptMutability::Immutable,
    ).map_err(|e| SuggesterError::IndexParse(format!("Load domain_aliases failed: {}", e)))?;

    for row in &aliases_result.rows {
        if let (Some(DataValue::Str(alias)), Some(DataValue::Str(canonical))) = (row.first(), row.get(1)) {
            if let Some(entry) = domains.get_mut(&canonical.to_string()) {
                entry.aliases.push(alias.to_string());
            }
        }
    }

    // Load keywords: canonical_name → keyword
    let keywords_result = db.run_script(
        "?[canonical_name, keyword] := *domain_keywords{ canonical_name, keyword }",
        Default::default(),
        ScriptMutability::Immutable,
    ).map_err(|e| SuggesterError::IndexParse(format!("Load domain_keywords failed: {}", e)))?;

    for row in &keywords_result.rows {
        if let (Some(DataValue::Str(canonical)), Some(DataValue::Str(keyword))) = (row.first(), row.get(1)) {
            if let Some(entry) = domains.get_mut(&canonical.to_string()) {
                entry.example_keywords.push(keyword.to_string());
            }
        }
    }

    // Load skills: canonical_name → skill_name
    let skills_result = db.run_script(
        "?[canonical_name, skill_name] := *domain_skills{ canonical_name, skill_name }",
        Default::default(),
        ScriptMutability::Immutable,
    ).map_err(|e| SuggesterError::IndexParse(format!("Load domain_skills failed: {}", e)))?;

    for row in &skills_result.rows {
        if let (Some(DataValue::Str(canonical)), Some(DataValue::Str(skill))) = (row.first(), row.get(1)) {
            if let Some(entry) = domains.get_mut(&canonical.to_string()) {
                entry.skills.push(skill.to_string());
            }
        }
    }

    // Load registry version from metadata
    let version = db.run_script(
        "?[value] := *pss_metadata{ key: 'registry_version', value }",
        Default::default(),
        ScriptMutability::Immutable,
    ).ok()
        .and_then(|r| r.rows.first().cloned())
        .and_then(|row| row.first().cloned())
        .and_then(|v| match v {
            DataValue::Str(s) => Some(s.to_string()),
            _ => None,
        })
        .unwrap_or_else(|| "unknown".to_string());

    let registry = DomainRegistry {
        version,
        generated: String::new(),
        source_index: String::new(),
        domain_count: domains.len(),
        domains,
    };

    info!("Loaded domain registry from CozoDB: {} domains", registry.domains.len());
    Ok(Some(registry))
}

/// Load ALL skills from CozoDB as a SkillIndex.
/// Used as an alternative to JSON parsing — produces the exact same SkillIndex.
pub(crate) fn load_index_from_db(db: &DbInstance) -> Result<SkillIndex, SuggesterError> {
    // Single query: fetch all fields including JSON-serialized vector fields
    let main_result = db.run_script(
        "?[name, path, skill_type, source, description, tier, boost, category, \
         server_type, server_command, server_args_json, language_ids_json, \
         negative_kw_json, patterns_json, directories_json, path_patterns_json, \
         use_cases_json, co_usage_json, alternatives_json, domain_gates_json, file_types_json, \
         keywords_json, intents_json, tools_json, services_json, frameworks_json, languages_json, platforms_json, domains_json, path_gates_json, \
         first_indexed_at, last_updated_at] := \
         *skills{ name, path, skill_type, source, description, tier, boost, category, \
                  server_type, server_command, server_args_json, language_ids_json, \
                  negative_kw_json, patterns_json, directories_json, path_patterns_json, \
                  use_cases_json, co_usage_json, alternatives_json, domain_gates_json, file_types_json, \
                  keywords_json, intents_json, tools_json, services_json, frameworks_json, languages_json, platforms_json, domains_json, path_gates_json, \
                  first_indexed_at, last_updated_at }",
        Default::default(),
        ScriptMutability::Immutable,
    ).map_err(|e| SuggesterError::IndexParse(format!("Load all skills from DB failed: {}", e)))?;

    // Helper to extract string from DataValue
    let dv_str = |v: &DataValue| -> String {
        match v {
            DataValue::Str(s) => s.to_string(),
            _ => String::new(),
        }
    };
    let dv_i32 = |v: &DataValue| -> i32 {
        match v {
            DataValue::Num(cozo::Num::Int(n)) => *n as i32,
            DataValue::Num(cozo::Num::Float(f)) => *f as i32,
            _ => 0,
        }
    };

    let ownership = load_ownership_columns(db);
    let mut skills: HashMap<String, SkillEntry> = HashMap::new();
    for row in &main_result.rows {
        if row.len() < 32 { continue; }
        let name = dv_str(&row[0]);
        let source = dv_str(&row[3]);
        // Use entry ID as HashMap key (collision-safe: same name + different source = different ID)
        let entry_id = make_entry_id(&name, &source);
        // Keyed on (name, source) — the skills relation's own primary key, so
        // two same-named elements from different sources keep distinct owners.
        let (plugin, origin) = ownership
            .get(&(name.clone(), source.clone()))
            .cloned()
            .unwrap_or((None, None));
        let entry = SkillEntry {
            name: name.clone(),
            path: dv_str(&row[1]),
            skill_type: dv_str(&row[2]),
            source,
            description: dv_str(&row[4]),
            tier: dv_str(&row[5]),
            boost: dv_i32(&row[6]),
            category: dv_str(&row[7]),
            server_type: dv_str(&row[8]),
            server_command: dv_str(&row[9]),
            server_args: serde_json::from_str(&dv_str(&row[10])).unwrap_or_default(),
            language_ids: serde_json::from_str(&dv_str(&row[11])).unwrap_or_default(),
            negative_keywords: serde_json::from_str(&dv_str(&row[12])).unwrap_or_default(),
            patterns: serde_json::from_str(&dv_str(&row[13])).unwrap_or_default(),
            directories: serde_json::from_str(&dv_str(&row[14])).unwrap_or_default(),
            path_patterns: serde_json::from_str(&dv_str(&row[15])).unwrap_or_default(),
            use_cases: serde_json::from_str(&dv_str(&row[16])).unwrap_or_default(),
            co_usage: serde_json::from_str(&dv_str(&row[17])).unwrap_or_default(),
            alternatives: serde_json::from_str(&dv_str(&row[18])).unwrap_or_default(),
            domain_gates: serde_json::from_str(&dv_str(&row[19])).unwrap_or_default(),
            file_types: serde_json::from_str(&dv_str(&row[20])).unwrap_or_default(),
            keywords: serde_json::from_str(&dv_str(&row[21])).unwrap_or_default(),
            intents: serde_json::from_str(&dv_str(&row[22])).unwrap_or_default(),
            tools: serde_json::from_str(&dv_str(&row[23])).unwrap_or_default(),
            services: serde_json::from_str(&dv_str(&row[24])).unwrap_or_default(),
            frameworks: serde_json::from_str(&dv_str(&row[25])).unwrap_or_default(),
            languages: serde_json::from_str(&dv_str(&row[26])).unwrap_or_default(),
            platforms: serde_json::from_str(&dv_str(&row[27])).unwrap_or_default(),
            domains: serde_json::from_str(&dv_str(&row[28])).unwrap_or_default(),
            path_gates: serde_json::from_str(&dv_str(&row[29])).unwrap_or_default(),
            first_indexed_at: dv_str(&row[30]),
            last_updated_at: dv_str(&row[31]),
            plugin,
            origin,
        };
        skills.insert(entry_id, entry);
    }

    // Load version from metadata
    let version = db.run_script(
        "?[value] := *pss_metadata{ key: 'version', value }",
        Default::default(),
        ScriptMutability::Immutable,
    ).ok()
        .and_then(|r| r.rows.first().cloned())
        .and_then(|row| row.first().cloned())
        .and_then(|v| match v {
            DataValue::Str(s) => Some(s.to_string()),
            _ => None,
        })
        .unwrap_or_else(|| "unknown".to_string());

    let skills_count = skills.len();
    info!("Loaded {} skills from CozoDB", skills_count);

    let mut index = SkillIndex {
        version,
        generated: String::new(),
        method: "cozodb".to_string(),
        skills_count,
        skills,
        name_to_ids: HashMap::new(),
    };
    index.build_name_index();
    Ok(index)
}

/// Load only candidate entries matching prompt words via the kw_lookup inverted index.
/// Returns a SkillIndex with only the matching entries (typically 50-500 instead of 10K).
/// Falls back to load_index_from_db (full load) if kw_lookup is empty or query fails.
pub(crate) fn load_candidates_from_db(
    db: &DbInstance,
    prompt_words: &[String],
) -> Result<SkillIndex, SuggesterError> {
    if prompt_words.is_empty() {
        info!("No prompt words for pre-filtering, loading full index");
        return load_index_from_db(db);
    }

    // Build inline data for prompt words (already lowercased by caller)
    let words_data: String = prompt_words.iter()
        .map(|w| format!("[\"{}\"]", w.replace('\\', "\\\\").replace('"', "\\\"")))
        .collect::<Vec<_>>()
        .join(", ");

    // Single Datalog query: look up kw_lookup → join with skills table
    let query = format!(
        "words[w] <- [{}]\n\
         candidates[name] := *kw_lookup{{keyword_lower: w, skill_name: name}}, words[w]\n\
         ?[name, path, skill_type, source, description, tier, boost, category, \
          server_type, server_command, server_args_json, language_ids_json, \
          negative_kw_json, patterns_json, directories_json, path_patterns_json, \
          use_cases_json, co_usage_json, alternatives_json, domain_gates_json, file_types_json, \
          keywords_json, intents_json, tools_json, services_json, frameworks_json, languages_json, platforms_json, domains_json, path_gates_json, \
          first_indexed_at, last_updated_at] := \
          candidates[name], *skills{{ name, source, path, skill_type, description, tier, boost, category, \
                   server_type, server_command, server_args_json, language_ids_json, \
                   negative_kw_json, patterns_json, directories_json, path_patterns_json, \
                   use_cases_json, co_usage_json, alternatives_json, domain_gates_json, file_types_json, \
                   keywords_json, intents_json, tools_json, services_json, frameworks_json, languages_json, platforms_json, domains_json, path_gates_json, \
                   first_indexed_at, last_updated_at }}",
        words_data
    );

    let main_result = match db.run_script(&query, Default::default(), ScriptMutability::Immutable) {
        Ok(r) => r,
        Err(e) => {
            warn!("Candidate query failed: {}, falling back to full load", e);
            return load_index_from_db(db);
        }
    };

    if main_result.rows.is_empty() {
        info!("No candidates matched, falling back to full load");
        return load_index_from_db(db);
    }

    // Same deserialization as load_index_from_db
    let dv_str = |v: &DataValue| -> String {
        match v {
            DataValue::Str(s) => s.to_string(),
            _ => String::new(),
        }
    };
    let dv_i32 = |v: &DataValue| -> i32 {
        match v {
            DataValue::Num(cozo::Num::Int(n)) => *n as i32,
            DataValue::Num(cozo::Num::Float(f)) => *f as i32,
            _ => 0,
        }
    };

    let mut skills: HashMap<String, SkillEntry> = HashMap::new();
    // DI-6 (audit 20260514): the previous 19× `unwrap_or_default()` chain
    // silently absorbed corrupt JSON in any of the 19 _json columns —
    // turning a real bug-hiding parse error into "empty Vec" and producing
    // "no match" with no warning. Per fail-fast, we now count corrupt
    // columns, log per-skill (stderr), and emit an aggregate WARN at
    // end-of-load so the user actually SEES the corruption.
    let mut total_corrupt_columns: usize = 0;
    let mut rows_with_corruption: usize = 0;

    // Helper: parse JSON column, log on failure, return Default. Increments
    // *corrupt_count when the raw value was non-empty but didn't parse —
    // that's the case the silent unwrap_or_default used to hide.
    fn parse_json_col<T: serde::de::DeserializeOwned + Default>(
        raw: &str,
        col_name: &str,
        skill_name: &str,
        corrupt_count: &mut usize,
    ) -> T {
        match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(parse_err) => {
                // Empty string is legit empty (Cozo default for unwritten
                // _json columns); only non-empty + unparseable = corrupt.
                if !raw.is_empty() {
                    eprintln!(
                        "[pss-load] WARN: corrupt JSON in skills.{} for '{}' \
                         (parse error: {}). Treating column as empty.",
                        col_name, skill_name, parse_err
                    );
                    *corrupt_count += 1;
                }
                T::default()
            }
        }
    }

    let ownership = load_ownership_columns(db);
    for row in &main_result.rows {
        if row.len() < 32 { continue; }
        let name = dv_str(&row[0]);
        let source = dv_str(&row[3]);
        let entry_id = make_entry_id(&name, &source);
        let mut corrupt_in_row: usize = 0;
        let (plugin, origin) = ownership
            .get(&(name.clone(), source.clone()))
            .cloned()
            .unwrap_or((None, None));
        let entry = SkillEntry {
            name: name.clone(),
            path: dv_str(&row[1]),
            skill_type: dv_str(&row[2]),
            source,
            description: dv_str(&row[4]),
            tier: dv_str(&row[5]),
            boost: dv_i32(&row[6]),
            category: dv_str(&row[7]),
            server_type: dv_str(&row[8]),
            server_command: dv_str(&row[9]),
            server_args: parse_json_col(&dv_str(&row[10]), "server_args_json", &name, &mut corrupt_in_row),
            language_ids: parse_json_col(&dv_str(&row[11]), "language_ids_json", &name, &mut corrupt_in_row),
            negative_keywords: parse_json_col(&dv_str(&row[12]), "negative_kw_json", &name, &mut corrupt_in_row),
            patterns: parse_json_col(&dv_str(&row[13]), "patterns_json", &name, &mut corrupt_in_row),
            directories: parse_json_col(&dv_str(&row[14]), "directories_json", &name, &mut corrupt_in_row),
            path_patterns: parse_json_col(&dv_str(&row[15]), "path_patterns_json", &name, &mut corrupt_in_row),
            use_cases: parse_json_col(&dv_str(&row[16]), "use_cases_json", &name, &mut corrupt_in_row),
            co_usage: parse_json_col(&dv_str(&row[17]), "co_usage_json", &name, &mut corrupt_in_row),
            alternatives: parse_json_col(&dv_str(&row[18]), "alternatives_json", &name, &mut corrupt_in_row),
            domain_gates: parse_json_col(&dv_str(&row[19]), "domain_gates_json", &name, &mut corrupt_in_row),
            file_types: parse_json_col(&dv_str(&row[20]), "file_types_json", &name, &mut corrupt_in_row),
            keywords: parse_json_col(&dv_str(&row[21]), "keywords_json", &name, &mut corrupt_in_row),
            intents: parse_json_col(&dv_str(&row[22]), "intents_json", &name, &mut corrupt_in_row),
            tools: parse_json_col(&dv_str(&row[23]), "tools_json", &name, &mut corrupt_in_row),
            services: parse_json_col(&dv_str(&row[24]), "services_json", &name, &mut corrupt_in_row),
            frameworks: parse_json_col(&dv_str(&row[25]), "frameworks_json", &name, &mut corrupt_in_row),
            languages: parse_json_col(&dv_str(&row[26]), "languages_json", &name, &mut corrupt_in_row),
            platforms: parse_json_col(&dv_str(&row[27]), "platforms_json", &name, &mut corrupt_in_row),
            domains: parse_json_col(&dv_str(&row[28]), "domains_json", &name, &mut corrupt_in_row),
            path_gates: parse_json_col(&dv_str(&row[29]), "path_gates_json", &name, &mut corrupt_in_row),
            first_indexed_at: dv_str(&row[30]),
            last_updated_at: dv_str(&row[31]),
            plugin,
            origin,
        };
        if corrupt_in_row > 0 {
            rows_with_corruption += 1;
            total_corrupt_columns += corrupt_in_row;
        }
        skills.insert(entry_id, entry);
    }

    if total_corrupt_columns > 0 {
        eprintln!(
            "[pss-load] WARN: {} corrupt JSON column(s) across {} skill row(s) — \
             run /pss-reindex-skills to rebuild the index.",
            total_corrupt_columns, rows_with_corruption
        );
    }

    let skills_count = skills.len();
    info!("Pre-filtered to {} candidates from kw_lookup", skills_count);

    let mut index = SkillIndex {
        version: String::new(),
        generated: String::new(),
        method: "cozodb-prefiltered".to_string(),
        skills_count,
        skills,
        name_to_ids: HashMap::new(),
    };
    index.build_name_index();
    Ok(index)
}
