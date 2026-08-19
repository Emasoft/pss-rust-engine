//! Suggestion dedupe window — don't repeat an identical suggestion set.
//!
//! Fleet request (ai-maestro hub, 2026-08-18): on heavily-automated sessions
//! the hook emitted the SAME agent set on dozens of consecutive prompts —
//! pure noise-tokens, because a set the model just saw and ignored carries no
//! new information. This module suppresses a hook emission when the exact
//! same (session, mode, names) set was already emitted within the TTL.
//!
//! Design mirrors `suggest_mode.rs` for the same reasons spelled out there:
//! the state must be readable/writable by the Rust binary alone (the shell
//! shim bypasses Python on the hot path), must not touch the CozoDB (lock
//! contention + reindex wipes), and lives as a one-line file beside the DB.
//!
//! Fail-open is the invariant: any read/parse/write problem means "not a
//! duplicate" — a repeated suggestion is mildly noisy, a silently swallowed
//! first-time suggestion is a real loss.

use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::warn;

/// Basename of the state file, a sibling of the CozoDB (like `pss-suggest-mode`).
pub(crate) const DEDUPE_STATE_FILE: &str = "pss-last-suggestion";

/// How long an identical set stays suppressed. After the TTL the set is
/// emitted once more (a periodic reminder), then suppressed again. 10 minutes
/// is deliberately shorter than the 15-minute janitor-heartbeat cadence so a
/// time-based window never permanently silences a genuinely recurring need.
pub(crate) const DEDUPE_TTL_SECS: u64 = 600;

/// Absolute path of the state file — resolved through the same canonical DB
/// path as the rest of PSS so `--index` / `$PSS_INDEX_PATH` redirects (and the
/// hermetic tests) land it in the same directory.
pub(crate) fn state_file_path(cli_index: Option<&str>) -> PathBuf {
    let db = crate::resolve_db_path_canonical(cli_index);
    db.parent()
        .map(|dir| dir.join(DEDUPE_STATE_FILE))
        .unwrap_or_else(|| PathBuf::from(DEDUPE_STATE_FILE))
}

/// One stable fingerprint per (session, mode, suggestion names). Names are
/// hashed in emission order — `find_matches` orders deterministically by
/// score, so an unchanged set hashes identically without sorting.
fn fingerprint(session_id: &str, mode: &str, names: &[&str]) -> u64 {
    let mut h = DefaultHasher::new();
    session_id.hash(&mut h);
    mode.hash(&mut h);
    names.hash(&mut h);
    h.finish()
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Returns `true` when this exact set was already emitted for this session
/// within [`DEDUPE_TTL_SECS`] — the caller should then emit nothing. On
/// `false` the new fingerprint is recorded so the NEXT identical emission is
/// the one suppressed.
///
/// Not refreshed on suppression, on purpose: refreshing the timestamp while
/// suppressing would extend the window forever for a steadily-repeating
/// prompt stream and the set would never be shown again.
pub(crate) fn should_suppress_and_record(
    cli_index: Option<&str>,
    session_id: &str,
    mode: &str,
    names: &[&str],
) -> bool {
    should_suppress_and_record_at(&state_file_path(cli_index), session_id, mode, names)
}

/// [`should_suppress_and_record`] against an explicit path — the testable core.
pub(crate) fn should_suppress_and_record_at(
    path: &Path,
    session_id: &str,
    mode: &str,
    names: &[&str],
) -> bool {
    let fp = fingerprint(session_id, mode, names);
    let now = now_unix_secs();

    // Read the previous record. Any failure → not a duplicate (fail-open).
    if let Ok(raw) = fs::read_to_string(path) {
        let mut fields = raw.trim().split('\t');
        let (ver, prev_fp, prev_ts) = (fields.next(), fields.next(), fields.next());
        if ver == Some("v1") {
            if let (Some(prev_fp), Some(prev_ts)) = (
                prev_fp.and_then(|s| s.parse::<u64>().ok()),
                prev_ts.and_then(|s| s.parse::<u64>().ok()),
            ) {
                if prev_fp == fp && now.saturating_sub(prev_ts) < DEDUPE_TTL_SECS {
                    return true;
                }
            }
        }
    }

    // New (or expired) set — record it. A write failure only costs dedupe,
    // never the suggestion itself.
    if let Err(err) = write_record_at(path, fp, now) {
        warn!(
            "suggest-dedupe: cannot write {} ({err}) — dedupe disabled for this prompt",
            path.display()
        );
    }
    false
}

/// Atomic single-line write: temp file + rename, same pattern (and Windows
/// caveat) as `suggest_mode::write_mode_at`.
fn write_record_at(path: &Path, fp: u64, ts: u64) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_file_name(format!(
        "{}.tmp.{}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(DEDUPE_STATE_FILE),
        std::process::id()
    ));
    {
        let mut file = fs::File::create(&tmp)?;
        writeln!(file, "v1\t{fp}\t{ts}")?;
        file.sync_all()?;
    }
    match fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(err) => {
            let _ = fs::remove_file(&tmp);
            Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpfile(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "pss-suggest-dedupe-{}-{}",
            tag,
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir.join(DEDUPE_STATE_FILE)
    }

    #[test]
    fn first_emission_is_never_suppressed() {
        let path = tmpfile("first");
        let _ = fs::remove_file(&path);
        assert!(!should_suppress_and_record_at(&path, "s1", "agents", &["a", "b"]));
    }

    #[test]
    fn identical_set_within_ttl_is_suppressed() {
        let path = tmpfile("dup");
        let _ = fs::remove_file(&path);
        let names = ["python-test-writer", "llm-ext-fixer"];
        assert!(!should_suppress_and_record_at(&path, "s1", "agents", &names));
        assert!(should_suppress_and_record_at(&path, "s1", "agents", &names));
        // Still suppressed — suppression does not refresh the record.
        assert!(should_suppress_and_record_at(&path, "s1", "agents", &names));
    }

    #[test]
    fn different_set_session_or_mode_is_not_suppressed() {
        let path = tmpfile("diff");
        let _ = fs::remove_file(&path);
        assert!(!should_suppress_and_record_at(&path, "s1", "agents", &["a"]));
        assert!(!should_suppress_and_record_at(&path, "s1", "agents", &["b"]));
        assert!(!should_suppress_and_record_at(&path, "s2", "agents", &["b"]));
        assert!(!should_suppress_and_record_at(&path, "s2", "skills", &["b"]));
    }

    #[test]
    fn expired_record_is_not_suppressed() {
        let path = tmpfile("expired");
        let fp = fingerprint("s1", "agents", &["a"]);
        let old_ts = now_unix_secs() - DEDUPE_TTL_SECS - 1;
        write_record_at(&path, fp, old_ts).unwrap();
        assert!(!should_suppress_and_record_at(&path, "s1", "agents", &["a"]));
    }

    #[test]
    fn garbage_state_file_fails_open() {
        let path = tmpfile("garbage");
        fs::write(&path, "not a valid record at all\n").unwrap();
        assert!(!should_suppress_and_record_at(&path, "s1", "agents", &["a"]));
        // And the garbage got replaced by a valid record.
        assert!(should_suppress_and_record_at(&path, "s1", "agents", &["a"]));
    }
}
