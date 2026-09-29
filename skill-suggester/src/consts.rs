// ============================================================================
// Constants
// ============================================================================

/// Default index file location (JSON)
pub(crate) const INDEX_FILE: &str = "skill-index.json";

/// Default CozoDB index file location (SQLite-backed)
pub(crate) const DB_FILE: &str = "pss-skill-index.db";

/// Default domain registry file location
pub(crate) const REGISTRY_FILE: &str = "domain-registry.json";

/// Cache directory name under ~/.claude/
pub(crate) const CACHE_DIR: &str = "cache";

/// Maximum number of suggestions to keep after matching (internal buffer)
/// Set higher than --top default (10) to allow co-usage boosting to surface related skills
pub(crate) const MAX_SUGGESTIONS: usize = 50;

/// Absolute anchor for relative score floor (W5 innovation).
/// When one skill scores very high (e.g., framework match = 20000), pure
/// relative scoring (score/max_score) crushes genuinely matched skills below
/// the min_score filter. The absolute floor ensures any skill scoring at least
/// ABSOLUTE_ANCHOR/2 raw points always passes, regardless of the top scorer.
pub(crate) const ABSOLUTE_ANCHOR: i32 = 1000;

/// PSS file extension for per-skill matcher files
#[allow(dead_code)]  // Used for documentation and future file detection
pub(crate) const PSS_EXTENSION: &str = ".pss";

/// Log file name for activation logging
pub(crate) const ACTIVATION_LOG_FILE: &str = "pss-activations.jsonl";

/// Log directory under ~/.claude/
pub(crate) const LOG_DIR: &str = "logs";

/// Maximum prompt length to store in logs (for privacy)
pub(crate) const MAX_LOG_PROMPT_LENGTH: usize = 100;

/// Maximum number of log entries before rotation (keep logs manageable)
pub(crate) const MAX_LOG_ENTRIES: usize = 10000;
