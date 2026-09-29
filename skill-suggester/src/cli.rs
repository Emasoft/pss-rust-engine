use clap::Parser;

// ============================================================================
// CLI Arguments
// ============================================================================

/// Perfect Skill Suggester (PSS) - High-accuracy skill activation for Claude Code
#[derive(Parser, Debug)]
#[command(name = "pss")]
#[command(about = "High-accuracy skill suggester for Claude Code")]
pub(crate) struct Cli {
    /// Run in incomplete mode for Pass 2 co-usage analysis.
    /// In this mode, co_usage fields are ignored and only keyword
    /// similarity is used to find candidate skills.
    #[arg(long, default_value_t = false)]
    pub(crate) incomplete_mode: bool,

    /// Issue #10 P-9: print the external CLI contract
    /// `{"cli_version","schema_version","contract_version"}` and exit 0,
    /// BEFORE any normal dispatch. A stable handle for integrators across PSS
    /// upgrades (like `--version`, but it carries the temporal schema version
    /// and the contract version too).
    #[arg(long, default_value_t = false)]
    pub(crate) contract_version: bool,

    /// Return only the top N candidates (default: 4, reduced from 10 to save context)
    #[arg(long, default_value_t = 4)]
    pub(crate) top: usize,

    /// Minimum score threshold - skip suggestions below this normalized score (default: 0.5)
    /// Score is normalized to 0.0-1.0 range. Helps filter low-confidence matches.
    #[arg(long, default_value_t = 0.5)]
    pub(crate) min_score: f64,

    /// Output format: "hook" (default) or "json" (raw skill list)
    #[arg(long, default_value = "hook")]
    pub(crate) format: String,

    /// Load and merge .pss files (per-skill matcher files) into the index.
    /// By default, only skill-index.json is used (PSS files are transient).
    #[arg(long, default_value_t = false)]
    pub(crate) load_pss: bool,

    /// Path to skill-index.json. Overrides the default (~/.claude/cache/skill-index.json).
    /// Required on WASM targets where home directory is unavailable.
    /// Can also be set via PSS_INDEX_PATH environment variable.
    #[arg(long)]
    pub(crate) index: Option<String>,

    /// Path to domain-registry.json. Overrides the default (~/.claude/cache/domain-registry.json).
    /// When provided, domain gates are enforced as hard pre-filters.
    /// Can also be set via PSS_REGISTRY_PATH environment variable.
    #[arg(long)]
    pub(crate) registry: Option<String>,

    /// Generate .agent.toml profile for an agent. Accepts an agent name (resolved
    /// via index) or a path to the agent's .md file. Parses frontmatter + body to
    /// extract name, description, duties, tools, domains, then scores against the
    /// skill index and writes <name>.agent.toml to the current directory.
    #[arg(long)]
    pub(crate) agent: Option<String>,

    /// Run Pass 1 batch enrichment: read JSONL from stdin (one element per line),
    /// enrich each with deterministic keywords/category/intents, output enriched JSONL.
    /// Replaces Sonnet agent calls for 10K-scale indexing.
    #[arg(long, default_value_t = false)]
    pub(crate) pass1_batch: bool,

    /// Index a single element file: read .md, parse frontmatter+body,
    /// enrich with Pass 1 pipeline (keywords, activities, languages, frameworks),
    /// output enriched JSON to stdout.
    #[arg(long, value_name = "PATH")]
    pub(crate) index_file: Option<String>,

    /// Extract the previous user message from a JSONL transcript file.
    /// Uses mmap + backward scan — zero-copy, constant memory, ~3ms on 500MB files.
    /// Outputs the 2nd most recent user message text (skips current prompt).
    /// Returns empty string if not found.  Used by the Python hook to avoid
    /// Python I/O overhead on large transcripts.
    #[arg(long, value_name = "PATH")]
    pub(crate) extract_prev_msg: Option<String>,

    /// Query/inspect subcommand (search, list, inspect, compare, stats, vocab, coverage, resolve).
    /// When omitted, the binary runs in hook mode (reads JSON from stdin).
    #[command(subcommand)]
    pub(crate) command: Option<Commands>,
}

/// Canonical output format for every output-producing subcommand
/// (COR-6, v3.7). `Table` is the default; `Json` is opt-in.
///
/// `Csv`, `Tsv`, and `Markdown` variants are reserved — most subcommands
/// currently emit a "TODO: <format> format" stub when these are requested.
/// Add the per-format formatter alongside the table / JSON branches in each
/// `cmd_*` function as the need arises.
///
/// Legacy `--json` boolean flags remain on the subcommands as deprecated
/// aliases for `--format json`. The compatibility helper
/// [`resolve_format`] folds the two so every `cmd_*` only branches on a
/// single `OutputFormat` value.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "lower")]
pub(crate) enum OutputFormat {
    /// Human-readable Unicode-bordered table (default).
    #[default]
    Table,
    /// Machine-readable JSON.
    Json,
    /// Comma-separated values. TODO: most subcommands stub this.
    Csv,
    /// Tab-separated values. TODO: most subcommands stub this.
    Tsv,
    /// Markdown table / fenced code block. TODO: most subcommands stub this.
    Markdown,
}

impl OutputFormat {
    /// Stable lowercase form used both for human-readable error messages
    /// and for the legacy `--format` string compatibility shim.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            OutputFormat::Table => "table",
            OutputFormat::Json => "json",
            OutputFormat::Csv => "csv",
            OutputFormat::Tsv => "tsv",
            OutputFormat::Markdown => "markdown",
        }
    }

    /// Returns true iff this format is currently a no-op stub. The
    /// `cmd_*` functions use this to print a single `TODO: <fmt> format`
    /// line instead of crashing or returning an empty body.
    #[allow(dead_code)] // reserved API; not yet branched on in every cmd
    pub(crate) fn is_stub(&self) -> bool {
        matches!(self, OutputFormat::Csv | OutputFormat::Tsv | OutputFormat::Markdown)
    }

    /// Print a single canonical stub line for an unsupported format.
    /// Use from inside a `cmd_*` function when [`Self::is_stub`] returns true
    /// so users always get the same wording.
    pub(crate) fn print_stub(&self, subcommand: &str) {
        println!(
            "TODO: {} format for `{}` (Phase 3 reserves the variant; implementation pending)",
            self.as_str(),
            subcommand,
        );
    }
}

/// Fold the legacy `--json` boolean flag and the new `--format` enum into
/// a single canonical [`OutputFormat`].
///
/// Behavior:
/// - If `json` (legacy `--json` flag) is `true`, force [`OutputFormat::Json`].
///   This preserves backward compatibility for users / scripts still passing
///   `--json` on subcommands where the new `--format` flag is also exposed.
/// - Otherwise, return `format` verbatim.
pub(crate) fn resolve_format(json: bool, format: OutputFormat) -> OutputFormat {
    if json {
        OutputFormat::Json
    } else {
        format
    }
}

/// Parse a legacy `--format <str>` parameter. Returns [`OutputFormat::Json`]
/// for "json", [`OutputFormat::Table`] for everything else.  Used by the
/// pre-v3.7 subcommands that still store `format: String` (because the
/// `format` value is read from a position-aware match arm and migrating
/// all of them in one pass is too much surface for one patch).
///
/// New subcommands should NOT use this helper — they should use the
/// canonical `format: OutputFormat` field directly.
#[allow(dead_code)] // reserved for future migrations of subcommands that
                     // still parse `format: String` internally.
pub(crate) fn parse_legacy_format(s: &str) -> OutputFormat {
    match s.to_ascii_lowercase().as_str() {
        "json" => OutputFormat::Json,
        "csv" => OutputFormat::Csv,
        "tsv" => OutputFormat::Tsv,
        "markdown" | "md" => OutputFormat::Markdown,
        // "table" or anything unknown — default to table (most user-friendly).
        _ => OutputFormat::Table,
    }
}

/// Query/inspect subcommands for exploring the skill index.
/// These use CozoDB Datalog queries when available, with JSON fallback.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum Commands {
    /// Full-text search across name, description, and keywords
    Search {
        /// Search query string (case-insensitive substring match)
        query: String,

        /// Filter by entry type: skill, agent, command, rule, mcp, lsp
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,

        /// Filter by domain (e.g. security, ai-ml, devops)
        #[arg(long)]
        domain: Option<String>,

        /// Filter by programming language (e.g. python, typescript, rust)
        #[arg(long)]
        language: Option<String>,

        /// Filter by framework (e.g. react, django, flutter)
        #[arg(long)]
        framework: Option<String>,

        /// Filter by tool (e.g. docker, ffmpeg, terraform)
        #[arg(long)]
        tool: Option<String>,

        /// Filter by category
        #[arg(long)]
        category: Option<String>,

        /// Filter by file type/extension (e.g. pdf, svg, xlsx)
        #[arg(long)]
        file_type: Option<String>,

        /// Filter by keyword
        #[arg(long)]
        keyword: Option<String>,

        /// Filter by platform (e.g. ios, linux, universal)
        #[arg(long)]
        platform: Option<String>,

        /// Maximum number of results (default: 20)
        #[arg(long, default_value_t = 20)]
        top: usize,

        /// Output format: table (default), json, csv, tsv, markdown.
        /// COR-6 (v3.7): standardized across every subcommand.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.  Kept for backward
        /// compatibility with pre-v3.7 scripts.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// List entries with optional filtering and sorting
    List {
        /// Filter by entry type: skill, agent, command, rule, mcp, lsp
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,

        /// Filter by domain
        #[arg(long)]
        domain: Option<String>,

        /// Filter by programming language
        #[arg(long)]
        language: Option<String>,

        /// Filter by framework
        #[arg(long)]
        framework: Option<String>,

        /// Filter by tool
        #[arg(long)]
        tool: Option<String>,

        /// Filter by category
        #[arg(long)]
        category: Option<String>,

        /// Filter by file type/extension
        #[arg(long)]
        file_type: Option<String>,

        /// Filter by keyword
        #[arg(long)]
        keyword: Option<String>,

        /// Filter by platform
        #[arg(long)]
        platform: Option<String>,

        /// UX-9 (audit 20260514): filter to entries whose `source`
        /// column starts with the given prefix. Useful for "all
        /// plugin:*", "all marketplace:*", or specific
        /// "plugin:emasoft-plugins/*" subsets. Combines (AND) with the
        /// other filters.
        #[arg(long, value_name = "PREFIX")]
        source_prefix: Option<String>,

        /// Sort order: name (default) or category
        #[arg(long, default_value = "name")]
        sort: String,

        /// Maximum number of results (default: 50)
        #[arg(long, default_value_t = 50)]
        top: usize,

        /// Output format: table (default), json, csv, tsv, markdown.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Show full details of a named entry (accepts name or ID)
    Inspect {
        /// Name or 13-char ID of the entry to inspect
        name: String,

        /// Output format: table (default), json, csv, tsv, markdown.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Side-by-side comparison of two entries (accepts names or IDs)
    Compare {
        /// First entry (name or ID)
        name1: String,

        /// Second entry (name or ID)
        name2: String,

        /// Output format: table (default), json, csv, tsv, markdown.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Show index statistics (counts by type, domain, category, language, etc.)
    Stats {
        /// Output format: table (default), json, csv, tsv, markdown.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// List all distinct values for a field (the "menu" of available options).
    /// Valid fields: languages, frameworks, tools, services, domains, keywords, intents,
    /// platforms, file-types, categories, types
    Vocab {
        /// Field name to enumerate
        field: String,

        /// Filter by entry type when listing field values
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,

        /// Maximum number of values to return (default: 50)
        #[arg(long, default_value_t = 50)]
        top: usize,

        /// Output format: table (default), json, csv, tsv, markdown.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Per-type coverage breakdown: what languages/domains/frameworks are covered
    Coverage {
        /// Entry type to analyze: skill, agent, command, rule, mcp, lsp
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,

        /// Output format: table (default), json, csv, tsv, markdown.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Resolve entry IDs to file paths (for reading actual skill/agent files)
    Resolve {
        /// One or more entry IDs (13-char deterministic hashes) or names
        ids: Vec<String>,

        /// Output format: table (default), json, csv, tsv, markdown.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Get lightweight metadata (description, type, plugin, keywords) for elements.
    /// Designed for tooltips, UI panels, and token-efficient lookups.
    #[command(name = "get-description")]
    GetDescription {
        /// Element name(s).  Single name, or comma-separated names in --batch mode.
        names: String,

        /// Treat `names` as a comma-separated list and return an array of results.
        #[arg(long)]
        batch: bool,

        /// Output format: table (default), json, csv, tsv, markdown.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Index rule files from ~/.claude/rules/ and .claude/rules/ into the DB.
    /// Rules are not suggestable (auto-injected) but are needed for agent profiling
    /// and get-description lookups.  Extracts name from filename and description
    /// from the first non-heading, non-empty content line.
    #[command(name = "index-rules")]
    IndexRules {
        /// Project root directory (for finding .claude/rules/).
        /// Defaults to current working directory.
        #[arg(long, value_name = "PATH")]
        project_root: Option<String>,

        /// Output format: table (default), json, csv, tsv, markdown.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// List all indexed rules with their descriptions.
    #[command(name = "list-rules")]
    ListRules {
        /// Filter by scope: user or project
        #[arg(long)]
        scope: Option<String>,

        /// Output format: table (default), json, csv, tsv, markdown.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Export a JSON snapshot of the CozoDB for debugging / git diff workflows.
    /// As of v2.11.0 (Phase B), the runtime hook no longer reads JSON — this
    /// subcommand exists purely so power users can still `git diff` the
    /// index. The export is written atomically to --path (default:
    /// $CLAUDE_PLUGIN_DATA/skill-index.export.json).
    Export {
        /// Export format. Only "json" is supported today.
        #[arg(long = "json", default_value_t = true)]
        json: bool,

        /// Destination path for the JSON export. Default is
        /// $CLAUDE_PLUGIN_DATA/skill-index.export.json (or
        /// ~/.claude/cache/skill-index.export.json if env var is unset).
        #[arg(long, value_name = "PATH")]
        path: Option<String>,
    },

    /// Print the total skill count from the CozoDB.
    /// Default output is a single integer on stdout; `--json` yields `{"count": N}`.
    /// Exits non-zero if the DB is missing or unreadable.
    Count {
        /// Output as JSON: {"count": N}.  Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// COR-6 (v3.7): output format (table = bare integer, json = {"count": N}).
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Fetch a single entry by name. With --source, disambiguates when the
    /// same name exists in multiple sources (e.g. user + plugin).
    Get {
        /// Element name (exact match; use find-by-name for substring).
        name: String,

        /// Restrict to a specific source (e.g. "user", "plugin:owner/name").
        #[arg(long)]
        source: Option<String>,

        /// Output as JSON (default is human-readable).  Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Health probe. Exit 0=DB populated; 1=empty/corrupt; 2=missing.
    /// Silent by default; use --verbose for a diagnostic line.
    Health {
        /// Print a one-line diagnostic to stdout.
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },

    /// List entries whose `first_indexed_at` is >= the given datetime.
    /// Accepts RFC 3339, date-only (YYYY-MM-DD → midnight UTC), or relative
    /// shorthand (`1d`, `2w`, `24h`, `30m`, `120s`).
    #[command(name = "list-added-since")]
    ListAddedSince {
        /// Datetime: RFC 3339, "YYYY-MM-DD", or relative ("1d", "2w", "24h").
        when: String,

        /// Maximum number of rows (default: 50).
        #[arg(long, default_value_t = 50)]
        limit: usize,

        /// Output as JSON.  Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// List entries whose `first_indexed_at` falls within [start, end] (inclusive).
    #[command(name = "list-added-between")]
    ListAddedBetween {
        /// Start datetime: RFC 3339, "YYYY-MM-DD", or relative.
        start: String,

        /// End datetime: RFC 3339, "YYYY-MM-DD", or relative.
        end: String,

        /// Maximum number of rows (default: 50).
        #[arg(long, default_value_t = 50)]
        limit: usize,

        /// Output as JSON.  Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// List entries whose `last_updated_at` is >= the given datetime.
    /// Useful for "what changed in the last reindex?" queries.
    #[command(name = "list-updated-since")]
    ListUpdatedSince {
        /// Datetime: RFC 3339, "YYYY-MM-DD", or relative ("1d", "2w", "24h").
        when: String,

        /// Maximum number of rows (default: 50).
        #[arg(long, default_value_t = 50)]
        limit: usize,

        /// Output as JSON.  Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Find entries whose name contains the given substring (case-insensitive).
    #[command(name = "find-by-name")]
    FindByName {
        /// Substring to match (case-insensitive). With `--regex` this is
        /// instead a Rust regex pattern, anchored partially (matches
        /// anywhere unless the pattern includes `^` / `$`).
        substring: String,

        /// Maximum number of rows (default: 50).
        #[arg(long, default_value_t = 50)]
        limit: usize,

        /// Output as JSON.  Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// UX-5 (audit 20260514): treat `substring` as a Rust regex
        /// pattern instead of a case-insensitive substring. Regex is
        /// applied to the lowercased name. Invalid pattern → exit 2
        /// with a parse-error line on stderr (fail-fast).
        #[arg(long, default_value_t = false)]
        regex: bool,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Find entries with an exact keyword match via skill_keywords.
    #[command(name = "find-by-keyword")]
    FindByKeyword {
        /// Keyword to match (case-insensitive, exact match).
        keyword: String,

        /// Maximum number of rows (default: 50).
        #[arg(long, default_value_t = 50)]
        limit: usize,

        /// Output as JSON.  Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Find entries gated by (or tagged with) the given domain.
    #[command(name = "find-by-domain")]
    FindByDomain {
        /// Domain value (e.g. "devops", "security", "web").
        domain: String,

        /// Maximum number of rows (default: 50).
        #[arg(long, default_value_t = 50)]
        limit: usize,

        /// Output as JSON.  Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Find entries targeting the given programming language.
    #[command(name = "find-by-language")]
    FindByLanguage {
        /// Language value (e.g. "python", "rust", "typescript").
        language: String,

        /// Maximum number of rows (default: 50).
        #[arg(long, default_value_t = 50)]
        limit: usize,

        /// Output as JSON.  Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// UX-8 (audit 20260514): find entries targeting the given framework
    /// (e.g. "react", "django", "fastapi", "rails").
    #[command(name = "find-by-framework")]
    FindByFramework {
        /// Framework value.
        framework: String,

        /// Maximum number of rows (default: 50).
        #[arg(long, default_value_t = 50)]
        limit: usize,

        /// Output as JSON.  Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// UX-8 (audit 20260514): find entries that integrate with the given
    /// external tool (e.g. "git", "docker", "kubernetes", "jq").
    #[command(name = "find-by-tool")]
    FindByTool {
        /// Tool value.
        tool: String,

        /// Maximum number of rows (default: 50).
        #[arg(long, default_value_t = 50)]
        limit: usize,

        /// Output as JSON.  Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// UX-8 (audit 20260514): find entries targeting the given platform
    /// (e.g. "linux", "macos", "windows", "browser", "aws", "gcp").
    #[command(name = "find-by-platform")]
    FindByPlatform {
        /// Platform value.
        platform: String,

        /// Maximum number of rows (default: 50).
        #[arg(long, default_value_t = 50)]
        limit: usize,

        /// Output as JSON.  Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    // ========================================================================
    // Temporal history index — TRDD-152e697f. v3.3.0+.
    // These query the event-sourced tables (events, elements_state, scan_runs)
    // populated by `pss reindex`. All output is JSON.
    // ========================================================================
    /// List every element that was installed and active at the given date.
    /// Reads from elements_state with an as-of cutoff against events.
    /// Each row also carries (P-4): `first_seen` — the earliest `installed`
    /// event's timestamp for that element_id in its scope (the install
    /// instant) — and `first_seen_is_synthetic` — true iff that earliest
    /// install is the v1→v2 migration placeholder rather than a real observed
    /// install. (The same real install instant is also exposed per-event by
    /// `pss timeline <ELEMENT_ID>` and `pss installed-between <S> <E>`.)
    #[command(name = "as-of")]
    AsOf {
        /// RFC3339 date or shorthand: "2026-03-14", "2026-03-14T12:00:00Z", "yesterday", "now"
        date: String,
        /// Optional element-type filter (skill / agent / command / rule / mcp / lsp / hook / plugin / channel / monitor / output-style / theme / marketplace).
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
        /// Optional scope filter (local / project / user / plugin / marketplace).
        #[arg(long)]
        scope: Option<String>,
        /// Optional scope_path filter (absolute path).
        #[arg(long, value_name = "PATH")]
        scope_path: Option<String>,
        /// Maximum rows. Default is UNLIMITED (P-7): a snapshot of ALL active
        /// components must never silently truncate. The 1_000_000 sentinel is
        /// "no practical cap"; pass `--limit N` to bound the result.
        #[arg(long, default_value_t = 1_000_000)]
        limit: usize,
    },

    /// List every component ACTIVE in a project folder at a point in time —
    /// the UNION of (a) project/local-scope elements whose scope_path is the
    /// folder's slug, (b) all global user-scope elements, and (c) plugin /
    /// marketplace elements currently enabled. Same row shape as `as-of`
    /// (including `first_seen` / `first_seen_is_synthetic`). Default is
    /// UNLIMITED (P-7) — pass `--limit N` to bound the result.
    ///
    /// NOTE (honesty): faithful PER-PROJECT plugin enablement at a PAST instant
    /// is not recorded yet (issue #10 P-8 is going-forward). So (c) reflects
    /// CURRENT/global enablement; per-project historical enablement fidelity is
    /// limited until enablement observation accrues. The (a)/(b) members ARE
    /// resolved as-of the requested time from the event history.
    #[command(name = "active-in")]
    ActiveIn {
        /// Absolute project folder path. Its scope-path slug is computed with
        /// the SAME algorithm as `pss project-slug` and matched against the
        /// project/local-scope rows' scope_path.
        abs_path: String,
        /// Point in time. RFC3339 date or shorthand ("2026-03-14", "now",
        /// "yesterday"). Default: now. Parsed exactly like `as-of`'s date.
        #[arg(long, value_name = "DATE", default_value = "now")]
        as_of: String,
        /// Maximum rows. Default is UNLIMITED (P-7); pass `--limit N` to bound.
        #[arg(long, default_value_t = 1_000_000)]
        limit: usize,

        /// COR-6 (v3.7): output format. Only `json` is meaningful for this
        /// verb; any non-table format yields the JSON array.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Show the full event timeline for one element.
    Timeline {
        /// element_id (e.g. "skill:my-skill@user:")
        element_id: String,
        /// Maximum rows (default: 200).
        #[arg(long, default_value_t = 200)]
        limit: usize,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// First-seen and last-seen (or null) timestamps for one element.
    Lifespan {
        element_id: String,
    },

    /// All events whose `event_type` is content_changed / size_changed /
    /// frontmatter_changed / description_changed / path_changed within
    /// the closed `[start, end]` window.
    #[command(name = "changed-between")]
    ChangedBetween {
        /// RFC3339 start (inclusive)
        start: String,
        /// RFC3339 end (inclusive)
        end: String,
        /// Optional element-type filter.
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
        /// Maximum rows (default: 1000).
        #[arg(long, default_value_t = 1000)]
        limit: usize,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// All `removed` events since the given date (inclusive).
    #[command(name = "removed-since")]
    RemovedSince {
        /// RFC3339 cutoff
        date: String,
        /// Maximum rows.
        #[arg(long, default_value_t = 1000)]
        limit: usize,
    },

    /// List recent scan runs (most recent first).
    #[command(name = "scan-log")]
    ScanLog {
        /// Maximum rows (default: 20).
        #[arg(long, default_value_t = 20)]
        limit: usize,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Statistics about the temporal database: event count, blob count,
    /// blob bytes, oldest event, retention window.
    #[command(name = "db-stats")]
    DbStats {
        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Run a full reindex (discover → enrich → emit events). Suitable
    /// for cron via the janitor `pss-reindex-due` detector.
    Reindex {
        /// Print the event set without writing.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },

    /// Drop events older than the configured retention window
    /// (default: 9 months). Idempotent.
    #[command(name = "prune-history")]
    PruneHistory {
        /// Print which rows would be dropped without committing.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },

    /// Re-key events/elements_state from the old lossy element_id scheme
    /// to the lossless one (F4, TRDD-1Z8SGQ7N). Prints `{"changed": N}`.
    /// Idempotent and gated — also auto-runs inside `merge-events`, so
    /// this verb is only the manual/validation lever. Aborts without
    /// writing anything if the old scheme had merged distinct elements.
    #[command(name = "migrate-element-ids")]
    MigrateElementIds {},

    /// Get or set which element type the prompt hook suggests.
    ///
    /// `agents` (default) offers agents, `skills` offers skills, `none`
    /// silences the hook. The two content modes are mutually exclusive by
    /// construction — this is one setting, not two toggles. Driven by the
    /// /pss-suggest-agents-on|off and /pss-suggest-skills-on|off commands.
    #[command(name = "suggest-mode")]
    SuggestMode {
        /// Set the mode: `agents`, `skills`, or `none`. Omit to print the
        /// current value.
        #[arg(long, value_name = "MODE")]
        set: Option<String>,
    },

    /// Get or set the retention window (default: 9 months).
    Retention {
        /// Set the retention window. Accepts ISO 8601 duration ("P9M",
        /// "P30D") or shorthand ("9m", "30d", "1y"). Omit to print
        /// the current value.
        #[arg(long, value_name = "DURATION")]
        set: Option<String>,
    },

    // ---------- Phase-2 secondary lifecycle queries (TRDD §9.1) ----------

    /// Snapshot of an element at a point in time.
    /// `pss show <ELEMENT_ID> --as-of <DATE>`
    Show {
        /// Element id (`<type>:<name>@<scope>:<scope_path_slug>`).
        element_id: String,
        /// Date or shorthand (`now`, `yesterday`, `YYYY-MM-DD`, RFC3339).
        #[arg(long, value_name = "DATE", default_value = "now")]
        as_of: String,
    },

    /// File size of an element at a point in time.
    /// `pss size-at <ELEMENT_ID> --as-of <DATE>`
    #[command(name = "size-at")]
    SizeAt {
        element_id: String,
        #[arg(long, value_name = "DATE", default_value = "now")]
        as_of: String,
    },

    /// Token count of an element at a point in time (cl100k approximation).
    /// `pss tokens-at <ELEMENT_ID> --as-of <DATE>`
    #[command(name = "tokens-at")]
    TokensAt {
        element_id: String,
        #[arg(long, value_name = "DATE", default_value = "now")]
        as_of: String,
    },

    /// Diff snapshots of an element between two dates.
    /// `pss diff <ELEMENT_ID> <DATE1> <DATE2>`
    Diff {
        element_id: String,
        date1: String,
        date2: String,
    },

    /// Every install event in a time window.
    /// `pss installed-between <START> <END>`
    #[command(name = "installed-between")]
    InstalledBetween {
        start: String,
        end: String,
        /// Optional element-type filter.
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
        #[arg(long, default_value_t = 500)]
        limit: usize,
    },

    /// Every removal event in a time window.
    /// `pss removed-between <START> <END>`
    #[command(name = "removed-between")]
    RemovedBetween {
        start: String,
        end: String,
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
        #[arg(long, default_value_t = 500)]
        limit: usize,
    },

    /// Elements that ever existed but aren't currently present.
    /// Synonym: `never-current`. `pss currently-missing-but-once-was`
    #[command(name = "currently-missing-but-once-was")]
    CurrentlyMissing {
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
        #[arg(long, default_value_t = 500)]
        limit: usize,
    },

    /// Alias for `currently-missing-but-once-was`.
    #[command(name = "never-current")]
    NeverCurrent {
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
        #[arg(long, default_value_t = 500)]
        limit: usize,
    },

    /// Find the same name living at multiple scopes simultaneously.
    /// `pss multi-scope <NAME>`
    #[command(name = "multi-scope")]
    MultiScope {
        name: String,
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
    },

    /// Override start/end events for one element.
    /// `pss override-history <ELEMENT_ID>`
    #[command(name = "override-history")]
    OverrideHistory {
        element_id: String,
        #[arg(long, default_value_t = 200)]
        limit: usize,
    },

    /// Enable/disable events for one element.
    /// `pss enable-history <ELEMENT_ID>`
    #[command(name = "enable-history")]
    EnableHistory {
        element_id: String,
        #[arg(long, default_value_t = 200)]
        limit: usize,
    },

    /// Scope_moved events for a name.
    /// `pss scope-moves <NAME>`
    #[command(name = "scope-moves")]
    ScopeMoves {
        name: String,
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
        #[arg(long, default_value_t = 200)]
        limit: usize,
    },

    /// All marketplace_added / marketplace_removed events.
    /// `pss marketplace-history`
    #[command(name = "marketplace-history")]
    MarketplaceHistory {
        #[arg(long, default_value_t = 500)]
        limit: usize,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// All events for one plugin name (across versions / scopes).
    /// `pss plugin-history <PLUGIN_NAME>`
    ///
    /// DI-9 (audit 20260514): plugins are stored as the composite
    /// `<name>@<marketplace>` (e.g. `perfect-skill-suggester@emasoft-plugins`).
    /// You can pass either form:
    ///   • the full `<name>@<marketplace>` for an exact match, or
    ///   • just `<name>` to find the plugin across all marketplaces.
    #[command(name = "plugin-history")]
    PluginHistory {
        /// Either `<plugin>@<marketplace>` (exact) or `<plugin>`
        /// (matches across all marketplaces).
        plugin_name: String,
        #[arg(long, default_value_t = 500)]
        limit: usize,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Read JSONL observations from stdin and emit temporal events.
    /// One JSON object per line; each line is one observation produced by
    /// `pss_discover.py --jsonl`. For every observation, this subcommand
    /// reads `elements_state` for the corresponding element_id, calls
    /// `compare_and_emit()`, persists any resulting events into the
    /// `events` table, and refreshes `elements_state`. After the stream
    /// ends, it issues a single `removed` event per element_id that is
    /// `exists=true` in `elements_state`, was NOT observed in this scan,
    /// and that this scan covered — meaning its scope is listed in the
    /// manifest's `exhaustive_scopes` (F7, manifest v2), or its scope_path
    /// is one of the visited scope_paths (the manifest-v1 rule, kept for
    /// scopes no claim covers). Finally, it records a `scan_runs` row.
    ///
    /// This is the only writer of the `events` table during normal
    /// reindex flow. Wired into `pss_reindex.py` after the existing
    /// merge-queue stage so the JSONL stream is consumed twice (once by
    /// the legacy skills-table writer, once here).
    #[command(name = "merge-events")]
    MergeEvents {
        /// Read JSONL from stdin (default).
        #[arg(long, default_value_t = true)]
        batch_stdin: bool,
        /// Suppress per-line stats; print only the final scan summary.
        #[arg(long, default_value_t = false)]
        quiet: bool,
    },

    // ─── Phase 3 Tier A — new query subcommands (audit 20260514) ────────
    /// List every currently-active element whose `source` is `plugin:<name>`.
    /// E.g. `pss by-plugin perfect-skill-suggester` lists every skill / agent /
    /// command / hook PSS itself provides.
    #[command(name = "by-plugin")]
    ByPlugin {
        /// Plugin name (the `<name>` part of `plugin:<name>` in events.source).
        name: String,
        /// Optional element-type filter (e.g. `--type skill`).
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
        #[arg(long, default_value_t = 500)]
        limit: usize,
    },

    /// F-2 (audit 20260514): list every currently-active element whose
    /// `source` starts with `marketplace:<name>` — i.e. installed from
    /// the given marketplace.
    #[command(name = "by-marketplace")]
    ByMarketplace {
        /// Marketplace name (the `<name>` part of `marketplace:<name>` in events.source).
        name: String,
        /// Optional element-type filter (e.g. `--type plugin`).
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
        #[arg(long, default_value_t = 500)]
        limit: usize,
    },

    /// F-6 (audit 20260514): show elements present in `scope1` but not
    /// `scope2`, and vice versa. Useful for "what does the user scope
    /// have that the project scope doesn't?" or "what plugin elements
    /// am I missing in my project?". Both scopes are matched against
    /// the `scope` column (user/project/local/plugin/marketplace).
    #[command(name = "scope-diff")]
    ScopeDiff {
        /// First scope to compare.
        scope1: String,
        /// Second scope to compare.
        scope2: String,
        /// Optional element-type filter (e.g. `--type skill`).
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
        #[arg(long, default_value_t = 500)]
        limit: usize,
    },

    /// F-17 (audit 20260514): list every event from a specific scan_id.
    /// Useful for "what happened during this reindex?" — you can read
    /// the scan_id from `pss scan-log` or from any event row.
    #[command(name = "changes-in-batch")]
    ChangesInBatch {
        /// ULID-format scan_id (from scan_runs / events.scan_id).
        scan_id: String,
        #[arg(long, default_value_t = 500)]
        limit: usize,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// F-18 (audit 20260514): emit every event from the most recent
    /// scan. Shortcut for `changes-in-batch $(pss scan-log | latest)`.
    #[command(name = "last-changes")]
    LastChanges {
        #[arg(long, default_value_t = 500)]
        limit: usize,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// F-19 (audit 20260514): count elements per scope (and per type
    /// within each scope). Output is a JSON object keyed by scope.
    #[command(name = "stats-by-scope")]
    StatsByScope {
        /// Optional element-type filter; without it counts every type.
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// F-12 (audit 20260514): focused "what versions has this element
    /// gone through?" history. Where `pss timeline` shows every event
    /// (including size/enabled noise), version-history filters to the
    /// signal events: installed, content_changed, description_changed,
    /// removed. Each row carries the content hash and (where present)
    /// the diff JSON so callers can reconstruct the version chain.
    ///
    /// Builds on DI-1 wave 1 (description_changed) shipped in v3.6.6
    /// and DI-2 (override events) shipped in v3.6.4.
    #[command(name = "version-history")]
    VersionHistory {
        /// Element ID (use `pss show <name>` to discover it).
        element_id: String,
        #[arg(long, default_value_t = 500)]
        limit: usize,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Count events by event_type in a recent time window.
    /// `pss changes-summary --window 7d` → installed: 12, content_changed: 4, removed: 1.
    /// Accepts the same date shorthand as `as-of` / `list-added-since`.
    #[command(name = "changes-summary")]
    ChangesSummary {
        /// Time window before now (default: 24h).
        #[arg(long, default_value = "24h")]
        window: String,
        /// Optional element-type filter.
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// List the scopes where the given element name is currently enabled
    /// (one row per scope/scope_path where exists=true AND enabled=true).
    #[command(name = "enabled-where")]
    EnabledWhere {
        /// Element name (or composite `<plugin>@<marketplace>` for plugins).
        name: String,
        /// Optional element-type filter.
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
    },

    /// Diff two point-in-time snapshots — returns `only_at_date1`,
    /// `only_at_date2`, and `common_count`. Useful for "what got installed
    /// vs removed between last week and now?" audits. (Tier A F-5.)
    #[command(name = "compare-snapshots")]
    CompareSnapshots {
        /// First snapshot date (RFC 3339, YYYY-MM-DD, relative shorthand, or 'now'/'yesterday').
        date1: String,
        /// Second snapshot date (same formats).
        date2: String,
        /// Optional element-type filter.
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
        /// Cap total elements scanned per snapshot (default 5000).
        #[arg(long, default_value_t = 5000)]
        limit: usize,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Find (element_type, element_name) pairs that appear in 2+ scopes.
    /// Helps catch accidental duplicates between user/project/plugin
    /// installations of the same skill or agent — the audit's "dedup
    /// candidates" pattern (Tier A F-8).
    #[command(name = "dedup-candidates")]
    DedupCandidates {
        /// Minimum number of scopes for a name to be reported (default 2).
        #[arg(long, default_value_t = 2)]
        min_count: usize,
        /// Optional element-type filter (e.g. `--type skill`).
        #[arg(long, value_name = "TYPE")]
        r#type: Option<String>,
        #[arg(long, default_value_t = 200)]
        limit: usize,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// One-line overview of the PSS index — total element count, per-type
    /// breakdown, and per-scope source breakdown.
    ///
    /// COR-6 / v3.7 task 3.2.  Reads from the temporal `elements_state` table
    /// joined against `events` for the most recent event of each element.
    /// Output is human-friendly by default; pass `--format json` for a
    /// structured `{total, by_type, by_source}` envelope.
    Summary {
        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Directory-tree view of the PSS index, grouped by source (user / project
    /// / plugin / marketplace) then by element type.
    ///
    /// COR-6 / v3.7 task 3.3.  Uses Unicode box-drawing characters
    /// (├ │ └ ─) to render the hierarchy.  `--format json` returns the same
    /// data as nested objects (`{source: {type: count, ...}, ...}`).
    Tree {
        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    // ========================================================================
    // Issue #10 wave 1 — external integration helpers (no DB required).
    // ========================================================================
    /// Print the canonical resolved CozoDB path PSS would use (honoring
    /// `--index` / `PSS_INDEX_PATH` / the default `~/.claude/cache/pss-skill-index.db`).
    /// Lets external consumers stop reverse-engineering PSS path resolution.
    /// Default: the bare path on one line. `--format json`: `{"db_path":"<abs>"}`.
    #[command(name = "db-path")]
    DbPath {
        /// COR-6 (v3.7): output format. Only `table` (bare path) and `json`
        /// are meaningful; stub formats fall back to the JSON envelope.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Print the resolved pss-nlp binary path (TRDD-YC51I1C0 phase 2, P10):
    /// mirrors `find_pss_nlp_binary()` without any side effects, so the
    /// Python↔Rust parity test can assert both languages agree on the search
    /// order by driving the REAL resolvers. Bare path on one line; empty
    /// output (exit 0) when nothing is found — mirroring the "negation
    /// detection silently skipped" miss mode.
    #[command(name = "nlp-binary-path")]
    NlpBinaryPath,

    /// Compute the scope-path slug for an absolute project path, EXACTLY as the
    /// Python discoverer (`pss_discover.py::_slugify_project_path`) does:
    /// `"<basename>-<8-char-sha256>"` over the filesystem-resolved path. Use it
    /// to reconstruct an element_id's scope_path slug from an absolute path.
    /// Default: the bare slug. `--format json`: `{"abs_path":"<in>","slug":"<out>"}`.
    #[command(name = "project-slug")]
    ProjectSlug {
        /// Absolute project path to slugify.
        abs_path: String,

        /// COR-6 (v3.7): output format. Only `table` (bare slug) and `json`
        /// are meaningful; stub formats fall back to the JSON envelope.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Generate an agent of one of the three archetypes.
    ///
    /// Skills are referenced by bare name and stay standalone files — nothing is
    /// ever inlined into the emitted agent, so a skill can still be shared,
    /// edited and updated in one place.
    ///
    /// - `all-in-one`  — every step skill preloaded; skills run in this agent.
    /// - `one-for-all` — a step MENU preloaded; each step runs in its own subagent.
    /// - `plugin-omni` — exactly one skill (the plugin menu); broad, not a tree.
    #[command(name = "make-agent")]
    MakeAgent {
        /// What the agent should specialize in — free text, or a path to a .md
        /// file describing it. PSS scores this against the index to pick the
        /// skills, so a specific description yields a better agent than a vague
        /// one. Ignored for plugin-omni, which takes every skill of its plugin.
        description: Option<String>,

        /// Which archetype: normal | all-in-one | one-for-all | plugin-omni
        /// (aliases: allin1, 1xall, omni).
        #[arg(long, visible_alias = "type")]
        kind: String,

        /// Name of the generated agent (also its filename stem). Derived from
        /// the description, or from a description file's frontmatter, if omitted.
        #[arg(long)]
        name: Option<String>,

        /// Comma-separated skill names to reference. Overrides description-based
        /// selection.
        #[arg(long)]
        skills: Option<String>,

        /// How many skills to select from a description. Default 8.
        #[arg(long, default_value_t = 8)]
        top: usize,

        /// Do not give the agent any MCP servers.
        #[arg(long, default_value_t = false)]
        no_mcp: bool,

        /// Do not give the agent any skills.
        #[arg(long, default_value_t = false)]
        no_skill: bool,

        /// Do not point the agent at any complementary agents.
        #[arg(long, default_value_t = false)]
        no_agent: bool,

        /// Reasoning effort pin: low | medium | high | xhigh | max.
        #[arg(long)]
        effort: Option<String>,

        /// Reference every skill belonging to this plugin (plugin-omni's usual input).
        #[arg(long)]
        plugin: Option<String>,

        /// One-line `description:` for the emitted frontmatter. Defaults to the
        /// positional description, trimmed to its first sentence.
        #[arg(long, visible_alias = "description")]
        summary: Option<String>,

        /// Optional `model:` pin for the emitted agent.
        #[arg(long)]
        model: Option<String>,

        /// Output directory. Files land under `<dir>/agents/` and `<dir>/skills/`.
        #[arg(long, default_value = ".")]
        output: String,

        /// Print what would be written without touching the filesystem.
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Route read-only one-for-all steps through the built-in `Explore`
        /// instead of the generated micro-agent. Off by default: measured, the
        /// saving is ~1.6k tokens per step (both environments load
        /// `~/.claude/rules/*`; only the project CLAUDE.md differs), which does
        /// not pay for a second environment that cannot write files.
        #[arg(long, default_value_t = false)]
        explore: bool,

        /// COR-6 (v3.7): output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Deprecated alias for `--format json`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}
