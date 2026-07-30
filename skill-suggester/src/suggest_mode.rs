//! Suggestion-mode switch — which element type PSS offers on every prompt.
//!
//! PSS historically hard-filtered hook output to `skill` entries. Skills have
//! since become numerous and narrowly specialized, which makes picking the
//! right one algorithmically unreliable; an AGENT, by contrast, declares the
//! skills it needs in its own frontmatter (the harness loads them into the
//! agent's context automatically), and main Claude acts on a suggested agent
//! more often than on a suggested skill. So `agents` is the default mode and
//! `skills` is opt-in — see `commands/pss-suggest-*.md`.
//!
//! # Why the state lives in a plain file, read by Rust
//!
//! `bin/pss-hook-dispatch.sh` execs this binary DIRECTLY on every
//! `UserPromptSubmit` (it deliberately bypasses Python to save ~130 ms), so a
//! Python-side toggle would simply never be consulted on the hot path. The
//! mode therefore has to be readable by the Rust binary with no DB open:
//!
//!   * NOT CozoDB — a reindex rebuilds the DB wholesale via staging +
//!     `os.replace`, which would silently discard the user's setting, and the
//!     hook must never contend on a DB lock (v3.5.0 SIGABRT history).
//!   * NOT an env var — a slash command cannot durably set one for the
//!     already-running session, let alone the next one.
//!   * A one-line file next to the DB: one `read_to_string` (sub-millisecond,
//!     no lock), and it survives reindex because nothing else writes there.
//!
//! Both the read and the write live HERE, in Rust, rather than splitting the
//! write into Python: `scripts/pss_paths.py` and this crate already resolve the
//! data dir by different rules (see the note at the top of `pss_paths.py`), and
//! a writer that disagreed with the reader about the directory would reproduce
//! that class of bug — the setting would be written where nothing reads it.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use tracing::warn;

/// Basename of the state file, stored beside the CozoDB in the PSS data dir.
pub(crate) const SUGGEST_MODE_FILE: &str = "pss-suggest-mode";

/// Which element type the `UserPromptSubmit` hook may suggest.
///
/// Deliberately THREE states, not a pair of booleans: the four slash commands
/// (`agents-on`/`agents-off`/`skills-on`/`skills-off`) drive one another, and
/// two independent flags would admit a meaningless "both on" state and leave
/// "both off" undefined. One enum makes every transition total.
// `pub`, not `pub(crate)`: it appears in the signatures of the already-`pub`
// `ContextItem::format_as_context` / `HookOutput::with_suggestions`, and a
// narrower type there trips the `private_interfaces` lint. This is a binary
// crate, so `pub` widens no externally-consumable API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionMode {
    /// Suggest agents (the default).
    Agents,
    /// Suggest skills (the pre-v3.11 behavior).
    Skills,
    /// Suggest nothing — the hook stays silent.
    None,
}

impl SuggestionMode {
    /// What a fresh install gets. This is a DEFINED DEFAULT, not a
    /// fail-fast fallback: an absent file is the normal state of every new
    /// install, not an error condition.
    pub(crate) const DEFAULT: SuggestionMode = SuggestionMode::Agents;

    /// Canonical on-disk spelling.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            SuggestionMode::Agents => "agents",
            SuggestionMode::Skills => "skills",
            SuggestionMode::None => "none",
        }
    }

    /// Parse the on-disk / CLI spelling. Accepts the singular form too, since
    /// the commands are named `pss-suggest-agents-on` but a human typing the
    /// verb by hand reaches for `--set agent` about as often.
    pub(crate) fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "agents" | "agent" => Some(SuggestionMode::Agents),
            "skills" | "skill" => Some(SuggestionMode::Skills),
            "none" | "off" => Some(SuggestionMode::None),
            _ => None,
        }
    }

    /// The `item_type` the hook keeps, or `None` when suggestions are silenced.
    pub(crate) fn element_type(self) -> Option<&'static str> {
        match self {
            SuggestionMode::Agents => Some("agent"),
            SuggestionMode::Skills => Some("skill"),
            SuggestionMode::None => Option::None,
        }
    }

    /// Tag wrapping the hook's `additionalContext`. Mode-aware on purpose: the
    /// tag is part of what tells the model WHAT it is being offered, so
    /// announcing agents inside `<pss-skills>` would actively mislead it.
    pub(crate) fn context_tag(self) -> &'static str {
        match self {
            SuggestionMode::Agents => "pss-agents",
            // `none` never renders a block; keep the historical tag so the
            // skills path is byte-identical to pre-v3.11 output.
            SuggestionMode::Skills | SuggestionMode::None => "pss-skills",
        }
    }
}

/// Absolute path of the state file: a sibling of the resolved CozoDB.
///
/// Resolved through `resolve_db_path_canonical` so the mode file always lands
/// in the same directory the rest of PSS considers its data dir — including
/// when `--index` or `$PSS_INDEX_PATH` redirects it, which is what makes the
/// tests hermetic.
pub(crate) fn mode_file_path(cli_index: Option<&str>) -> PathBuf {
    let db = crate::resolve_db_path_canonical(cli_index);
    db.parent()
        .map(|dir| dir.join(SUGGEST_MODE_FILE))
        .unwrap_or_else(|| PathBuf::from(SUGGEST_MODE_FILE))
}

/// Read the active mode. Never fails: this runs on the hot path of every user
/// prompt, where dying would take the whole hook down over a cosmetic setting.
pub(crate) fn read_mode(cli_index: Option<&str>) -> SuggestionMode {
    read_mode_at(&mode_file_path(cli_index))
}

/// [`read_mode`] against an explicit path — the testable core.
///
/// Degrades to [`SuggestionMode::DEFAULT`] on absence (normal), and WARNS then
/// degrades on unreadable/garbage content. That is not a silent fallback: a
/// corrupt file is announced on stderr, which PSS's debug logging surfaces,
/// while still leaving the user with a working prompt hook.
pub(crate) fn read_mode_at(path: &Path) -> SuggestionMode {
    let raw = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            // The overwhelmingly common case: nobody has run a
            // /pss-suggest-* command yet. Not worth a warning.
            return SuggestionMode::DEFAULT;
        }
        Err(err) => {
            warn!(
                "suggest-mode: cannot read {} ({err}) — falling back to '{}'",
                path.display(),
                SuggestionMode::DEFAULT.as_str()
            );
            return SuggestionMode::DEFAULT;
        }
    };

    match SuggestionMode::parse(&raw) {
        Some(mode) => mode,
        None => {
            warn!(
                "suggest-mode: {} contains {:?}, which is not agents|skills|none — \
                 falling back to '{}'",
                path.display(),
                raw.trim(),
                SuggestionMode::DEFAULT.as_str()
            );
            SuggestionMode::DEFAULT
        }
    }
}

/// Persist the mode, returning the path written. Fails fast — a `--set` that
/// cannot write must NOT report success, or the user is left believing they
/// switched modes when the next prompt will behave the old way.
pub(crate) fn write_mode(cli_index: Option<&str>, mode: SuggestionMode) -> std::io::Result<PathBuf> {
    let path = mode_file_path(cli_index);
    write_mode_at(&path, mode)?;
    Ok(path)
}

/// [`write_mode`] against an explicit path — the testable core.
///
/// Writes to a pid-suffixed temp file and `rename`s it into place. The rename
/// is atomic on POSIX, so a hook firing concurrently with the write reads
/// either the whole old value or the whole new one — never a truncated file
/// that would trip the corrupt-content path above.
pub(crate) fn write_mode_at(path: &Path, mode: SuggestionMode) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_file_name(format!(
        "{}.tmp.{}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(SUGGEST_MODE_FILE),
        std::process::id()
    ));

    // Scoped so the handle is closed before the rename — Windows refuses to
    // rename over/from an open handle, and PSS ships a windows-x86_64 binary.
    {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(mode.as_str().as_bytes())?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }

    match fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(err) => {
            // Don't leave the temp file behind on a failed rename.
            let _ = fs::remove_file(&tmp);
            Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "pss-suggest-mode-{}-{}",
            tag,
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn default_is_agents_so_a_fresh_install_suggests_agents() {
        assert_eq!(SuggestionMode::DEFAULT, SuggestionMode::Agents);
    }

    #[test]
    fn absent_file_reads_as_the_default_without_erroring() {
        let path = tmpdir("absent").join("definitely-not-created");
        assert_eq!(read_mode_at(&path), SuggestionMode::DEFAULT);
    }

    #[test]
    fn round_trips_every_mode_through_the_file() {
        let dir = tmpdir("roundtrip");
        for mode in [
            SuggestionMode::Agents,
            SuggestionMode::Skills,
            SuggestionMode::None,
        ] {
            let path = dir.join(format!("mode-{}", mode.as_str()));
            write_mode_at(&path, mode).unwrap();
            assert_eq!(read_mode_at(&path), mode, "round-trip failed for {mode:?}");
        }
    }

    #[test]
    fn corrupt_content_degrades_to_default_instead_of_panicking() {
        let path = tmpdir("corrupt").join("garbage");
        fs::write(&path, "not-a-mode\n").unwrap();
        assert_eq!(read_mode_at(&path), SuggestionMode::DEFAULT);
    }

    #[test]
    fn parse_accepts_singular_and_off_aliases_and_rejects_junk() {
        assert_eq!(SuggestionMode::parse("agent"), Some(SuggestionMode::Agents));
        assert_eq!(SuggestionMode::parse(" SKILLS \n"), Some(SuggestionMode::Skills));
        assert_eq!(SuggestionMode::parse("off"), Some(SuggestionMode::None));
        assert_eq!(SuggestionMode::parse("both"), Option::None);
    }

    #[test]
    fn element_type_selects_the_hook_filter_and_none_silences_it() {
        assert_eq!(SuggestionMode::Agents.element_type(), Some("agent"));
        assert_eq!(SuggestionMode::Skills.element_type(), Some("skill"));
        assert_eq!(SuggestionMode::None.element_type(), Option::None);
    }

    #[test]
    fn agents_mode_uses_its_own_tag_and_skills_keeps_the_historical_one() {
        assert_eq!(SuggestionMode::Agents.context_tag(), "pss-agents");
        assert_eq!(SuggestionMode::Skills.context_tag(), "pss-skills");
    }

    #[test]
    fn write_leaves_no_temp_file_behind() {
        let dir = tmpdir("notemp");
        let path = dir.join("pss-suggest-mode");
        write_mode_at(&path, SuggestionMode::Skills).unwrap();
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp."))
            .collect();
        assert!(leftovers.is_empty(), "temp files left behind: {leftovers:?}");
    }

    #[test]
    fn overwriting_replaces_rather_than_appends() {
        let dir = tmpdir("overwrite");
        let path = dir.join("pss-suggest-mode");
        write_mode_at(&path, SuggestionMode::Skills).unwrap();
        write_mode_at(&path, SuggestionMode::Agents).unwrap();
        assert_eq!(read_mode_at(&path), SuggestionMode::Agents);
        assert_eq!(fs::read_to_string(&path).unwrap().trim(), "agents");
    }

    #[test]
    fn mode_file_sits_next_to_the_resolved_db() {
        let dir = tmpdir("sibling");
        let db = dir.join("pss-skill-index.db");
        let resolved = mode_file_path(Some(db.to_str().unwrap()));
        assert_eq!(resolved.parent(), Some(dir.as_path()));
        assert_eq!(
            resolved.file_name().and_then(|n| n.to_str()),
            Some(SUGGEST_MODE_FILE)
        );
    }
}
