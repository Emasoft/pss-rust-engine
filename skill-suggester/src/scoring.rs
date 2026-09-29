// ============================================================================
// Entry ID Generation (deterministic FNV-1a hash → base36)
// ============================================================================

/// Generate a deterministic 13-char ID from an entry's name + source.
/// Uses FNV-1a 64-bit hash encoded as base36 (a-z0-9).
/// Hashing both name and source ensures unique IDs even when different
/// plugins provide elements with the same name.
pub(crate) fn make_entry_id(name: &str, source: &str) -> String {
    // FNV-1a 64-bit hash over name + separator + source
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in name.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    // 0xFF separator prevents collisions between "ab"+"cd" and "abc"+"d"
    hash ^= 0xFF_u64;
    hash = hash.wrapping_mul(0x100000001b3);
    for byte in source.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    // Encode as base36, zero-padded to 13 chars (64-bit needs up to 13 base36 digits)
    let mut result = String::with_capacity(13);
    let mut val = hash;
    let chars: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    loop {
        result.push(chars[(val % 36) as usize] as char);
        val /= 36;
        if val == 0 { break; }
    }
    // Pad to 13 chars, reverse for consistent ordering
    while result.len() < 13 {
        result.push('0');
    }
    result.chars().rev().collect()
}

// ============================================================================
// Scoring Weights — 4-tier logarithmic scale
// ============================================================================
// Each tier is 10x the previous:
//   Common terms (test, build, run):       10 - 90 points
//   Phrases ("trace this function"):      100 - 900 points
//   Tools (bun, docker, ffmpeg):        1,000 - 9,000 points
//   Frameworks (react, flutter, fastapi): 10,000 - 90,000 points
// Domains and languages are FILTERS (domain gates), not scoring factors.

/// Scoring weights for different match types.
/// Base values are in the PHRASE tier (100-900).
/// Low-signal words get divided by LOW_SIGNAL_DIVISOR (10) to drop to common tier.
pub(crate) struct MatchWeights {
    /// Skill in matching directory (between phrase and tool tiers)
    pub(crate) directory: i32,
    /// Prompt mentions file path pattern (between phrase and tool tiers)
    pub(crate) path: i32,
    /// Action verb matches skill intent (phrase tier)
    pub(crate) intent: i32,
    /// Regex pattern matches (phrase tier)
    pub(crate) pattern: i32,
    /// Keyword match (phrase tier for specific phrases)
    pub(crate) keyword: i32,
    /// First keyword bonus (phrase tier)
    pub(crate) first_match: i32,
    /// Keyword in original prompt, not just expanded synonym (phrase tier)
    pub(crate) original_bonus: i32,
    /// Tool name exact match (tool tier, T3: 1K-9K)
    pub(crate) tool_match: i32,
    /// Framework name exact match (framework tier, T4: 10K-90K)
    pub(crate) framework_match: i32,
    /// Service/API name exact match (service tier, T5: 100K-900K)
    pub(crate) service_match: i32,
    /// Maximum capped score (service tier max)
    pub(crate) capped_max: i32,
}

/// Divisor for low-signal words — drops phrase-tier weights to common tier.
/// 100 / 10 = 10 (common), 300 / 10 = 30 (common), etc.
pub(crate) const LOW_SIGNAL_DIVISOR: i32 = 10;

/// Maximum score for all-low-signal matches — prevents common words
/// from ever reaching MEDIUM confidence regardless of how many match.
pub(crate) const ALL_LOW_SIGNAL_CAP: i32 = 90;

impl Default for MatchWeights {
    fn default() -> Self {
        Self {
            directory: 500,       // Between phrase and tool tiers
            path: 500,            // Between phrase and tool tiers
            intent: 150,          // Phrase tier (low-signal: 150/10 = 15)
            pattern: 200,         // Phrase tier (patterns are always specific)
            keyword: 100,         // Phrase tier (low-signal: 100/10 = 10)
            first_match: 300,     // Phrase tier (low-signal: 300/10 = 30)
            original_bonus: 100,  // Phrase tier (low-signal: 100/10 = 10)
            tool_match: 2000,       // Tool tier (T3: 1K-9K)
            framework_match: 20000, // Framework tier (T4: 10K-90K)
            service_match: 200000,  // Service/API tier (T5: 100K-900K)
            capped_max: 900000,     // Service tier max
        }
    }
}

/// Confidence thresholds
pub(crate) struct ConfidenceThresholds {
    /// Score >= this is HIGH confidence (tool tier — one tool match or many phrases)
    pub(crate) high: i32,
    /// Score >= this (but < high) is MEDIUM confidence (phrase tier — one phrase match)
    pub(crate) medium: i32,
}

impl Default for ConfidenceThresholds {
    fn default() -> Self {
        Self {
            high: 1000,
            medium: 100,
        }
    }
}
