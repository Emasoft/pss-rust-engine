// Main entry point + top-level CLI dispatch (`main` body, `run`) and the
// hook-input text helpers (`strip_system_reminders`, `is_skip_prompt`,
// `augment_prompt_if_short`). Moved out of main.rs verbatim (XUD7YUZH
// modularization step 14); the crate-root `fn main()` in main.rs is now a
// two-line shim calling `main_dispatch::main()`. Re-exported at the crate
// root via `pub(crate) use`.

use clap::{CommandFactory, FromArgMatches};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, error, info, warn};

use crate::*;

// ============================================================================
// Main Entry Point
// ============================================================================

pub(crate) fn main() {
    // Initialize tracing if RUST_LOG is set
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    // Parse CLI arguments with runtime version injected from VERSION file
    // Box::leak converts String to &'static str, which clap's .version() requires
    let version: &'static str = Box::leak(read_version().into_boxed_str());
    let matches = Cli::command().version(version).get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(2);
    });

    if cli.incomplete_mode {
        info!("Running in INCOMPLETE MODE - co_usage data will be ignored");
    }

    // Issue #10 P-9: --contract-version is a global flag (like --version). It
    // prints the external contract object and exits 0 BEFORE any normal
    // dispatch — and works with or without a subcommand present, so a bare
    // `pss --contract-version` answers without reading stdin or opening a DB.
    if cli.contract_version {
        println!("{}", contract_version_output());
        return;
    }

    // Dispatch: query subcommands vs build-db vs pass1-batch vs agent-profile vs hook
    if let Some(ref cmd) = cli.command {
        // Issue #10 P-2 / P-6: db-path and project-slug are pure path helpers —
        // they must NOT route through run_query_command (which opens the CozoDB
        // and errors when it is absent). Intercept here, like Health above,
        // print the result, and exit 0.
        if let Commands::DbPath { format, json } = cmd {
            let want_json = *json || !matches!(format, OutputFormat::Table);
            println!("{}", db_path_output(cli.index.as_deref(), want_json));
            return;
        }
        if let Commands::ProjectSlug { abs_path, format, json } = cmd {
            let want_json = *json || !matches!(format, OutputFormat::Table);
            println!("{}", project_slug_output(abs_path, want_json));
            return;
        }
        // nlp-binary-path is a pure path probe like db-path/project-slug — no
        // CozoDB. Empty line = nothing found (negation detection would be
        // skipped), still exit 0.
        if let Commands::NlpBinaryPath = cmd {
            match find_pss_nlp_binary() {
                Some(path) => println!("{}", path.display()),
                None => println!(),
            }
            return;
        }
        // suggest-mode is a pure file get/set — like db-path/project-slug it
        // must NOT route through run_query_command, which opens the CozoDB and
        // errors when it is absent. Switching modes has to work on a fresh
        // install that has never been indexed.
        if let Commands::SuggestMode { set } = cmd {
            match set {
                Some(raw) => {
                    let Some(mode) = SuggestionMode::parse(raw) else {
                        eprintln!(
                            "error: unknown suggest-mode {raw:?} — expected agents, skills, or none"
                        );
                        std::process::exit(2);
                    };
                    // Fail fast and LOUDLY: a --set that silently failed to
                    // persist would leave the user certain they had switched
                    // modes while every later prompt used the old one.
                    match suggest_mode::write_mode(cli.index.as_deref(), mode) {
                        Ok(path) => println!(
                            "{{\"mode\":\"{}\",\"path\":\"{}\"}}",
                            mode.as_str(),
                            path.display()
                        ),
                        Err(err) => {
                            eprintln!("error: cannot write suggest-mode: {err}");
                            std::process::exit(1);
                        }
                    }
                }
                None => {
                    let mode = suggest_mode::read_mode(cli.index.as_deref());
                    println!(
                        "{{\"mode\":\"{}\",\"path\":\"{}\"}}",
                        mode.as_str(),
                        suggest_mode::mode_file_path(cli.index.as_deref()).display()
                    );
                }
            }
            return;
        }
        // Health has special exit-code semantics (0=populated, 1=empty/corrupt,
        // 2=missing) and must not route through the generic error-to-exit-1
        // path below. Intercept before `run_query_command`.
        if let Commands::Health { verbose } = cmd {
            // Health probe semantics: if the caller passed --index explicitly,
            // that exact path is authoritative (no fallback to defaults).
            // This lets `pss --index <bogus> health` return exit 2 without
            // silently picking up the user's real DB.
            let db_path = if let Some(explicit) = &cli.index {
                let p = if explicit.ends_with(".db") {
                    PathBuf::from(explicit)
                } else {
                    PathBuf::from(explicit).parent().map(|d| d.join(DB_FILE))
                        .unwrap_or_else(|| PathBuf::from(explicit))
                };
                if p.exists() { Some(p) } else { None }
            } else {
                get_db_path(None)
            };
            let db = db_path.as_ref().and_then(|p| open_db(p).ok());
            cmd_health(db.as_ref(), *verbose); // diverges — never returns
        }

        // F3 (TRDD-1Z8SGQ7N): merge-events is the ONLY in-place writer of the
        // live DB (the Python skills writer builds a staging file and
        // atomically renames it into place). Reproduced 2026-07-16 on the
        // first real reindex after the DB paths were unified: with no
        // coordination it raced the suggestion hook's readers on the SQLite
        // file and died with `database is locked`, leaving `events` advanced
        // but `elements_state` lagging. Intercepted BEFORE run_query_command
        // because the locks must be acquired BEFORE the DB file is opened: a
        // concurrent Python writer os.replace()s a fresh inode over the live
        // path, so an FD opened first would keep writing the orphaned old
        // file after the lock cleared — lock held, data still lost.
        //
        // Two flocks, fixed order, so the whole model stays deadlock-free:
        //   1. `<db>.write.lock` (EX) — writer-vs-writer: excludes a
        //      concurrent Python skills-write (it holds the same file for its
        //      whole staging+rename, pss_cozodb.atomic_write_cozodb).
        //   2. `<db>.lock`       (EX) — writer-vs-reader: the hook holds
        //      LOCK_SH here around each binary call (pss_hook._db_shared_lock
        //      via pss_paths.get_db_lock_path).
        // Python writers take only #1, hook readers take only #2, and this
        // arm takes #1 then #2 — no party ever acquires in the opposite
        // order. flock(2) matches Python's fcntl.flock on the same files; on
        // Windows the hook side is a documented no-op, so there the locks
        // only serialize concurrent merge-events — an improvement, never a
        // regression. Query subcommands stay lock-free: they are read-only
        // and the staging+rename design already keeps them safe.
        if let Commands::MergeEvents { batch_stdin: _, quiet } = cmd {
            let db_path = match get_db_path(cli.index.as_deref()) {
                Some(p) => p,
                None => {
                    eprintln!(
                        "merge-events failed: no CozoDB index found (run /pss-reindex-skills first)"
                    );
                    std::process::exit(1);
                }
            };
            let _write_guard = acquire_db_flock(&db_path, ".write.lock");
            let _reader_guard = acquire_db_flock(&db_path, ".lock");
            let db = match open_db(&db_path) {
                Ok(db) => db,
                Err(e) => {
                    eprintln!("merge-events failed: {}", e);
                    std::process::exit(1);
                }
            };
            if let Err(e) = temporal::ensure_schema(&db) {
                eprintln!("merge-events failed: ensure_schema: {}", e);
                std::process::exit(1);
            }
            if let Err(e) = temporal::cli::cmd_merge_events(&db, *quiet) {
                eprintln!("merge-events failed: {}", e);
                std::process::exit(1);
            }
            return;
        }

        // F4 (TRDD-1Z8SGQ7N): migrate-element-ids is a WRITER — it rewrites
        // every events/elements_state/element_descriptions row — so it takes
        // the SAME two flocks in the SAME order as MergeEvents above (order
        // is the deadlock invariant). Without them it would race the
        // suggestion hook's LOCK_SH readers exactly like F3's reproduction:
        // cozo-ce hits "database is locked", `.unwrap()`s, and either this
        // migration or the user's hook process dies with SIGABRT. Its write
        // window is LONGER than merge-events' (a full-table rewrite), so the
        // race is more likely, not less. Intercepted BEFORE run_query_command
        // for the same open-after-lock reason documented on MergeEvents.
        // F10 (TRDD-1Z8SGQ7N): prune-history is a WRITER — it deletes events
        // older than the retention window — but it was dispatched on the
        // unlocked run_query_command path, the same latent F3-class race that
        // bit merge-events (a Rust writer racing the suggestion hook's LOCK_SH
        // readers → cozo-ce "database is locked" → `.unwrap()` → SIGABRT).
        // Same intercept, same two flocks, same order (the deadlock
        // invariant). --dry-run only reads, but it takes the locks too: one
        // code path beats a conditional lock, and the EX hold for its short
        // scan just briefly queues hook readers.
        if let Commands::PruneHistory { dry_run } = cmd {
            let db_path = match get_db_path(cli.index.as_deref()) {
                Some(p) => p,
                None => {
                    eprintln!(
                        "prune-history failed: no CozoDB index found (run /pss-reindex-skills first)"
                    );
                    std::process::exit(1);
                }
            };
            let _write_guard = acquire_db_flock(&db_path, ".write.lock");
            let _reader_guard = acquire_db_flock(&db_path, ".lock");
            let db = match open_db(&db_path) {
                Ok(db) => db,
                Err(e) => {
                    eprintln!("prune-history failed: {}", e);
                    std::process::exit(1);
                }
            };
            temporal::cli::cmd_prune_history(&db, *dry_run);
            return;
        }

        if let Commands::MigrateElementIds {} = cmd {
            let db_path = match get_db_path(cli.index.as_deref()) {
                Some(p) => p,
                None => {
                    eprintln!(
                        "migrate-element-ids failed: no CozoDB index found (run /pss-reindex-skills first)"
                    );
                    std::process::exit(1);
                }
            };
            let _write_guard = acquire_db_flock(&db_path, ".write.lock");
            let _reader_guard = acquire_db_flock(&db_path, ".lock");
            let db = match open_db(&db_path) {
                Ok(db) => db,
                Err(e) => {
                    eprintln!("migrate-element-ids failed: {}", e);
                    std::process::exit(1);
                }
            };
            // Surface the un-merge abort verbatim — a silently half-migrated
            // history is worse than a loud failure.
            if let Err(e) = temporal::cli::cmd_migrate_element_ids(&db) {
                eprintln!("migrate-element-ids failed: {}", e);
                std::process::exit(1);
            }
            return;
        }

        // Query subcommands use stderr for errors and exit(1) on failure
        if let Err(e) = run_query_command(&cli, cmd) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // Fast path: extract-prev-msg bypasses all index/DB loading
    if let Some(ref transcript_path) = cli.extract_prev_msg {
        let text = extract_prev_user_message(transcript_path);
        print!("{}", text);  // No newline — Python reads exact output
        return;
    }

    let result = if let Some(ref file_path) = cli.index_file {
        info!("Running in INDEX FILE mode: {}", file_path);
        run_index_file(file_path)
    } else if cli.pass1_batch {
        info!("Running in PASS1 BATCH mode");
        run_pass1_batch()
    } else if let Some(ref agent_ref) = cli.agent {
        info!("Running in AGENT PROFILE mode: {}", agent_ref);
        run_agent_profile(&cli, agent_ref)
    } else {
        run(&cli)
    };

    if let Err(e) = result {
        error!("Error: {}", e);
        // Output empty response on error (non-blocking)
        let output = HookOutput::empty();
        println!("{}", serde_json::to_string(&output).unwrap_or_default());
        std::process::exit(0); // Exit 0 to not block Claude
    }
}

pub(crate) fn run(cli: &Cli) -> Result<(), SuggesterError> {
    // PERF-1 (audit 20260514): gated profiling. Set PSS_PERF_DEBUG=1 to emit
    // per-section timings to stderr — used to identify the cost split between
    // stdin/parse, typo/synonym lazy_static init, DB open, candidate load,
    // and find_matches. Off by default (zero overhead unless env var is set).
    let perf_debug = std::env::var_os("PSS_PERF_DEBUG").is_some();
    let perf_t0 = Instant::now();
    let perf_log = |label: &str, t0: Instant| {
        if perf_debug {
            eprintln!("[pss-perf] +{:>5}ms  {}", t0.elapsed().as_millis(), label);
        }
    };

    // Read input from stdin
    let mut input_json = String::new();
    io::stdin().read_to_string(&mut input_json)?;
    perf_log("stdin read", perf_t0);

    debug!("Received input: {}", input_json);

    // Parse input
    let mut input: HookInput = serde_json::from_str(&input_json)?;
    // Normalise `cwd` ONCE, here, so every consumer agrees which project we are
    // in. Two of them read it — `scan_project_context` for context inference and
    // the cross-project filter for membership — and if they disagree about
    // surrounding whitespace they disagree about the project itself: the filter
    // would keep an element the context scoring had already ranked as though it
    // came from somewhere else. Trimming at the single point of entry removes
    // that whole class rather than patching each reader.
    if input.cwd != input.cwd.trim() {
        input.cwd = input.cwd.trim().to_string();
    }
    perf_log("json parse", perf_t0);

    // PERF-1 (audit 20260514): strip <system-reminder> blocks BEFORE skip
    // detection so legitimate <system-reminder> tags aren't matched as skip
    // signals. Idempotent — safe to call on already-cleaned prompts from
    // the legacy Python wrapper path.
    let cleaned = strip_system_reminders(&input.prompt);
    if cleaned != input.prompt {
        input.prompt = cleaned;
    }

    // PERF-1: augment short prompts with the previous user message from the
    // transcript (Python used to do this before invoking the binary). Cap at
    // 4 000 chars total to match the Python pipeline's MAX_PROMPT_CHARS.
    if !input.transcript_path.is_empty() {
        let augmented = augment_prompt_if_short(&input.prompt, &input.transcript_path, 4000);
        if augmented != input.prompt {
            input.prompt = augmented;
        }
    }

    // Skip processing for certain prompts.
    if is_skip_prompt(&input.prompt) {
        debug!("Skipping prompt: {}", &input.prompt[..input.prompt.len().min(50)]);
        let output = HookOutput::empty();
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }

    info!(
        "Processing prompt: {}",
        &input.prompt[..input.prompt.len().min(50)]
    );

    // Start timing for activation logging
    let start_time = Instant::now();

    // Open CozoDB first — if available, load index from DB instead of parsing JSON
    let db = get_db_path(cli.index.as_deref()).and_then(|p| {
        match open_db(&p) {
            Ok(db) => {
                info!("Using CozoDB at {:?}", p);
                Some(db)
            }
            Err(e) => {
                warn!("Failed to open CozoDB: {}, falling back to JSON", e);
                None
            }
        }
    });
    perf_log("db open", perf_t0);

    // Extract prompt words early for CozoDB pre-filtering.
    // Apply typo correction + synonym expansion so pre-filter catches the same terms as scoring.
    let corrected_for_filter = correct_typos(&input.prompt);
    perf_log("correct_typos (first call — lazy_static init)", perf_t0);
    let expanded_for_filter = expand_synonyms(&corrected_for_filter);
    perf_log("expand_synonyms (first call — lazy_static init)", perf_t0);
    let filter_words: Vec<String> = expanded_for_filter
        .split_whitespace()
        .filter(|w| w.len() >= 2)
        .take(50) // cap to prevent huge CozoDB queries
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| !w.is_empty())
        .collect::<HashSet<_>>() // deduplicate
        .into_iter()
        .collect();

    // Load skill index: CozoDB pre-filtered (fast) → CozoDB full → JSON fallback
    let mut index = if let Some(ref db) = db {
        match load_candidates_from_db(db, &filter_words) {
            Ok(idx) => {
                info!("Loaded {} candidates from CozoDB (pre-filtered)", idx.skills.len());
                idx
            }
            Err(e) => {
                warn!("CozoDB candidate load failed: {}, falling back to JSON", e);
                let index_path = get_index_path(cli.index.as_deref())?;
                match load_index(&index_path) {
                    Ok(idx) => idx,
                    Err(SuggesterError::IndexNotFound(path)) => {
                        warn!("Skill index not found at {:?}, returning empty", path);
                        let output = HookOutput::empty();
                        println!("{}", serde_json::to_string(&output)?);
                        return Ok(());
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    } else {
        let index_path = get_index_path(cli.index.as_deref())?;
        debug!("Loading index from: {:?}", index_path);
        match load_index(&index_path) {
            Ok(idx) => idx,
            Err(SuggesterError::IndexNotFound(path)) => {
                warn!("Skill index not found at {:?}, returning empty", path);
                let output = HookOutput::empty();
                println!("{}", serde_json::to_string(&output)?);
                return Ok(());
            }
            Err(e) => return Err(e),
        }
    };
    perf_log(&format!("load_candidates_from_db ({} skills)", index.skills.len()), perf_t0);
    info!("Loaded {} skills from index", index.skills.len());

    // Load and merge PSS files only if --load-pss flag is passed
    // By default, only use skill-index.json (PSS files are now transient)
    if cli.load_pss {
        load_pss_files(&mut index);
    }

    // SUGGESTION BOUNDARY: the hook may only offer elements the user can
    // actually invoke, so catalog-only rows leave the candidate pool here.
    //
    // `marketplace:<mp>` rows come from the recursive scan of
    // ~/.claude/plugins/marketplaces/<mp>/ — a CATALOG of installable plugins,
    // not an install. An installed plugin's elements are indexed separately
    // under `plugin:<marketplace>/<plugin>` (path under ~/.claude/plugins/cache/).
    // Measured on the live index: 12503 of 15854 rows are marketplace-scope and
    // NOT ONE of them lives under an installed-plugin dir, so a catalog row is
    // unreachable at runtime. Suggesting them wasted the model's attention on
    // things like `claude` (a stray agents/CLAUDE.md) or `missing-name-agent`
    // (a unit-test fixture vendored inside a marketplace checkout).
    //
    // Applied BEFORE scoring, not to the result list: find_matches() truncates
    // to MAX_SUGGESTIONS, and the catalog holds dozens of near-identical copies
    // of every popular element (26 `code-reviewer` rows, 24 of them catalog).
    // Measured post-hoc, 42 of the 50 returned matches were catalog dupes that
    // had crowded the installed originals out of the window entirely — the
    // filter then emptied the block instead of revealing them. Removing the
    // rows up front lets the invocable copies compete for those 50 slots.
    //
    // Hook-only (`--format hook`). Profiling / search / list deliberately need
    // the FULL corpus (see pss_discover.py section "5b"). The predicate is
    // prompt-independent, so namespacing built from this index below is still
    // decided identically on every prompt.
    if cli.format == "hook" {
        let before = index.skills.len();
        // Same boundary, second class of un-invocable element: a skill whose
        // frontmatter says `disable-model-invocation: true` can only be run by the
        // USER from its slash command. Dropped here, BEFORE scoring, for the very
        // reason spelled out above — filtering the result list instead would let
        // these occupy slots and then blank them.
        let noninvocable = db
            .as_ref()
            .map(load_noninvocable_ids)
            .unwrap_or_default();
        // Third class: an element belonging to a DIFFERENT project. See
        // `is_foreign_project_element` for why the index contains any, and why
        // comparing against the live `input.cwd` is what fixes a mid-session
        // `/cd` without a hook. Computed once, outside the closure.
        //
        // `HookInput::cwd` is `#[serde(default)]`, so a payload without the
        // field yields "". That is NOT a project we can compare against: the
        // degenerate slug matches nothing, so the filter would drop EVERY
        // project-scoped element — all 689 of them, in every project, on every
        // prompt — while looking like it worked. So an empty cwd disables this
        // predicate rather than applying it.
        //
        // This is deliberately the OPPOSITE of the per-element unprovable-path
        // rule inside the predicate, and the asymmetry is the point: an unknown
        // that is local to one element costs one suggestion when it fails
        // closed, whereas an unknown that destroys the comparison BASIS costs an
        // entire scope class. Both directions minimise the blast radius of the
        // thing we do not know. Degrading to the pre-v3.14.2 behaviour (a
        // cross-project leak) is recoverable and visible; silently emptying a
        // scope is neither.
        // "Usable basis" is ABSOLUTE-and-non-empty, not merely non-empty. A
        // RELATIVE `cwd` is a malformed payload, and it is the dangerous shape:
        // `python_resolve` would silently resolve it against the BINARY's own
        // working directory, yielding a confident answer about a location the
        // caller never named — and every project-scoped element would be
        // condemned against that phantom root. Measured on v3.14.3, which
        // guarded emptiness alone: `cwd: "relative/path"` dropped every
        // project-scoped element exactly as `cwd: ""` used to.
        //
        // An absolute path that is NOT a project (`/`, a deleted directory) is
        // deliberately left ENABLED: the filter then correctly finds that no
        // project owns the caller, and drops project-scoped elements because
        // none of them belong to them. That is the right answer, not a failure —
        // the distinction is decidable-and-negative versus undecidable.
        if !cwd_is_usable_basis(&input.cwd) {
            debug!(
                "cross-project filter DISABLED: the hook payload carried no `cwd`, \
                 so project membership is undecidable; keeping every project-scoped \
                 element rather than dropping all of them"
            );
        }
        // Fourth class: an element whose PLUGIN the user has disabled. Resolved
        // here, once per prompt, from the live settings files rather than from
        // the index — that is what makes a toggle take effect with no reindex.
        let disabled_plugins = disabled_plugin_keys(&input.cwd);
        debug!(
            "plugin-enablement filter: {} disabled plugin key(s) in effect",
            disabled_plugins.len()
        );
        index.skills.retain(|id, e| {
            candidate_is_invocable_here(
                &input.cwd,
                &e.source,
                &e.path,
                id,
                &noninvocable,
                &disabled_plugins,
            )
        });
        // MANDATORY after any retain: `name_to_ids` maps a name to entry IDs and
        // get_by_name() takes the FIRST id. Leaving it stale would resolve a
        // co-usage lookup to a just-removed catalog id and return None even
        // though an installed entry with that name survived — a silent loss of
        // exactly the elements this filter exists to surface.
        index.build_name_index();
        debug!(
            "Invocability filter: dropped {} catalog-scope + user-only candidate(s), {} remain",
            before - index.skills.len(),
            index.skills.len()
        );
    }

    // Load domain registry: try CozoDB first, then JSON file fallback
    let registry = if let Some(ref db) = db {
        // Try loading from CozoDB (domain tables inside pss-skill-index.db)
        match load_domain_registry_from_db(db) {
            Ok(Some(reg)) => Some(reg),
            _ => {
                debug!("No domain registry in CozoDB, trying JSON file fallback");
                match get_registry_path(cli.registry.as_deref()) {
                    Some(reg_path) => load_domain_registry(&reg_path).ok().flatten(),
                    None => None,
                }
            }
        }
    } else {
        // No DB available — load from JSON file
        match get_registry_path(cli.registry.as_deref()) {
            Some(reg_path) => {
                debug!("Loading domain registry from: {:?}", reg_path);
                match load_domain_registry(&reg_path) {
                    Ok(Some(reg)) => {
                        info!("Loaded domain registry: {} domains", reg.domains.len());
                        Some(reg)
                    }
                    Ok(None) => {
                        debug!("Domain registry not found at {:?}, gate filtering disabled", reg_path);
                        None
                    }
                    Err(e) => {
                        warn!("Failed to load domain registry: {}, gate filtering disabled", e);
                        None
                    }
                }
            }
            None => {
                debug!("No domain registry path configured, gate filtering disabled");
                None
            }
        }
    };

    // ========================================================================
    // STEP 1: Prepare full context BEFORE any domain checking.
    // Typo correction, synonym expansion, project metadata, and conversation
    // context must all be assembled first so the domain check runs against
    // the complete picture — not the raw prompt alone.
    // ========================================================================

    perf_log("registry load", perf_t0);
    // 1a. Scan project directory for languages/frameworks/tools from config files.
    // This runs in Rust (not in the Python hook) because project contents can
    // change at any time and must be detected fresh on every invocation.
    let project_scan = scan_project_context(&input.cwd);
    perf_log("scan_project_context", perf_t0);
    if !project_scan.languages.is_empty() || !project_scan.tools.is_empty() {
        debug!(
            "Project scan: languages={:?}, frameworks={:?}, platforms={:?}, tools={:?}, file_types={:?}",
            project_scan.languages, project_scan.frameworks,
            project_scan.platforms, project_scan.tools, project_scan.file_types
        );
    }

    // Create project context by merging hook input with fresh disk scan results.
    // The hook may provide conversation-derived context (e.g., domains mentioned in chat)
    // that the Rust scan cannot detect, while the scan provides ground-truth project data.
    let mut context = ProjectContext::from_hook_input(&input);
    context.merge_scan(&project_scan);
    if !context.is_empty() {
        debug!(
            "Merged project context: platforms={:?}, frameworks={:?}, languages={:?}",
            context.platforms, context.frameworks, context.languages
        );
    }

    // 1b. Apply typo corrections (e.g., "pyhton" → "python") so domain keywords match
    let corrected_prompt = correct_typos(&input.prompt);
    if corrected_prompt != input.prompt.to_lowercase() {
        debug!("Typo-corrected input: {}", corrected_prompt);
    }

    // 1c. Expand synonyms (e.g., "k8s" → "kubernetes") so domain keywords match
    let expanded_prompt = expand_synonyms(&corrected_prompt);
    if expanded_prompt != corrected_prompt {
        debug!("Synonym-expanded prompt: {}", expanded_prompt);
    }

    // 1d. Build context signals by merging:
    //   - Rust project scan results (fresh, from config files on disk — computed in 1a)
    //   - Python hook context (may include conversation history, session metadata)
    // Both sources are merged because the hook may provide context the Rust scan
    // cannot detect (e.g., domains from recent conversation messages, tools mentioned
    // in chat). The Rust scan provides ground-truth project structure context.
    let mut context_signals: Vec<String> = Vec::new();
    // Rust scan results first (ground truth from disk, computed in step 1a)
    context_signals.extend(project_scan.languages.iter().cloned());
    context_signals.extend(project_scan.frameworks.iter().cloned());
    context_signals.extend(project_scan.platforms.iter().cloned());
    context_signals.extend(project_scan.tools.iter().cloned());
    context_signals.extend(project_scan.file_types.iter().cloned());
    // Hook-provided context (may overlap with scan — dedup handles this)
    context_signals.extend(input.context_languages.iter().cloned());
    context_signals.extend(input.context_frameworks.iter().cloned());
    context_signals.extend(input.context_platforms.iter().cloned());
    context_signals.extend(input.context_tools.iter().cloned());
    context_signals.extend(input.context_domains.iter().cloned());
    context_signals.extend(input.context_file_types.iter().cloned());
    // Deduplicate context signals (Rust scan and hook may report the same items)
    dedup_vec(&mut context_signals);

    // The full_context_text combines the corrected+expanded prompt with all context
    // signals into a single lowercased string for domain keyword scanning.
    // This ensures the domain check considers everything: the user's words (corrected),
    // their synonyms (expanded), and the project/conversation metadata.
    let full_context_text = {
        let mut parts: Vec<String> = vec![expanded_prompt.clone()];
        for sig in &context_signals {
            parts.push(sig.to_lowercase());
        }
        parts.join(" ")
    };

    // ========================================================================
    // STEP 2: Global domain gate early-exit.
    // Build a flat set of ALL domain keywords from ALL skills' gates, scan the
    // full context once, short-circuit on first match. If 0 matches AND all
    // skills are gated → exit immediately, skip all scoring.
    // ========================================================================
    if let Some(reg) = registry.as_ref() {
        let all_skills_gated = !index.skills.is_empty()
            && index.skills.values().all(|e| !e.domain_gates.is_empty());

        if all_skills_gated {
            // Collect every keyword that could satisfy any gate in any skill.
            // For "generic" gates: add the registry's example_keywords for that domain
            // (because generic passes when the domain is detected via any registry keyword).
            let mut all_gate_keywords: HashSet<String> = HashSet::new();

            for entry in index.skills.values() {
                for (gate_name, gate_keywords) in &entry.domain_gates {
                    let has_generic = gate_keywords.iter().any(|kw| kw.eq_ignore_ascii_case("generic"));

                    if has_generic {
                        let canonical = find_canonical_domain(gate_name, reg);
                        if let Some(domain_entry) = reg.domains.get(&canonical) {
                            for kw in &domain_entry.example_keywords {
                                if !kw.eq_ignore_ascii_case("generic") {
                                    all_gate_keywords.insert(kw.to_lowercase());
                                }
                            }
                        }
                    }

                    for kw in gate_keywords {
                        if !kw.eq_ignore_ascii_case("generic") {
                            all_gate_keywords.insert(kw.to_lowercase());
                        }
                    }
                }
            }

            // Single scan of full context: does ANY keyword appear?
            // Short-circuits on first match — O(1) best case, O(K) worst case.
            let any_match = all_gate_keywords.iter().any(|kw| {
                full_context_text.contains(kw.as_str())
            });

            if !any_match {
                info!(
                    "Domain gate early-exit: 0/{} gate keywords matched in prompt+context. \
                     All {} skills are gated. Skipping scoring entirely.",
                    all_gate_keywords.len(),
                    index.skills.len()
                );

                let processing_ms = start_time.elapsed().as_millis() as u64;
                let session_id = if input.session_id.is_empty() { None } else { Some(input.session_id.as_str()) };
                log_activation(
                    &input.prompt,
                    session_id,
                    Some(&input.cwd),
                    1,
                    &[],
                    Some(processing_ms),
                );

                let output = HookOutput::empty();
                println!("{}", serde_json::to_string(&output)?);
                return Ok(());
            }

            debug!(
                "Domain gate pre-check passed: at least one keyword matched from {} total gate keywords",
                all_gate_keywords.len()
            );
        }
    }

    // ========================================================================
    // STEP 3: Domain detection (uses corrected+expanded prompt + context).
    // Runs once and is shared across all sub-tasks.
    // ========================================================================
    let detected_domains: DetectedDomains = match &registry {
        Some(reg) => {
            let detected = detect_domains_from_prompt_with_context(
                &expanded_prompt,
                reg,
                &context_signals,
            );
            if !detected.is_empty() {
                info!(
                    "Detected {} domains in prompt: {:?}",
                    detected.len(),
                    detected.keys().collect::<Vec<_>>()
                );
            }
            detected
        }
        None => HashMap::new(),
    };

    // ========================================================================
    // STEP 4: Task decomposition and scoring.
    // ========================================================================
    let sub_tasks = decompose_tasks(&corrected_prompt);
    let is_multi_task = sub_tasks.len() > 1;

    if is_multi_task {
        info!("Decomposed prompt into {} sub-tasks", sub_tasks.len());
        for (i, task) in sub_tasks.iter().enumerate() {
            debug!("  Sub-task {}: {}", i + 1, &task[..task.len().min(50)]);
        }
    }

    perf_log("decompose_tasks + setup", perf_t0);
    // Score all skills with the unchanged find_matches() algorithm
    // (index was loaded from CozoDB or JSON above — scoring is identical either way)
    let matches = if is_multi_task {
        let all_matches: Vec<Vec<MatchedSkill>> = sub_tasks
            .iter()
            .map(|task| {
                let task_expanded = expand_synonyms(task);
                find_matches(task, &task_expanded, &index, &input.cwd, &context, cli.incomplete_mode, &detected_domains, registry.as_ref())
            })
            .collect();

        aggregate_subtask_matches(all_matches)
    } else {
        find_matches(&corrected_prompt, &expanded_prompt, &index, &input.cwd, &context, cli.incomplete_mode, &detected_domains, registry.as_ref())
    };
    perf_log(&format!("find_matches ({} matches)", matches.len()), perf_t0);

    if matches.is_empty() {
        debug!("No matches found");

        // Log activation even for no matches (helps with analysis)
        let processing_ms = start_time.elapsed().as_millis() as u64;
        let session_id = if input.session_id.is_empty() { None } else { Some(input.session_id.as_str()) };
        log_activation(
            &input.prompt,
            session_id,
            Some(&input.cwd),
            sub_tasks.len(),
            &[],  // Empty matches
            Some(processing_ms),
        );

        let output = HookOutput::empty();
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }

    // Get max score for relative scoring
    let max_score = matches.iter().map(|m| m.score).max().unwrap_or(1);

    // Namespace tiers are decided against the WHOLE index, not just the items
    // being emitted, so an element renders the same way on every prompt
    // regardless of which neighbours happened to score alongside it.
    let name_counts = NameCounts::build(index.skills.values().map(|e| {
        (
            e.name.clone(),
            e.plugin.clone(),
            marketplace_of(&e.source).map(str::to_string),
            e.path.clone(),
        )
    }));

    // One physical file can be indexed under several sources — a user-scope
    // agent is also seen as project-scope, and an installed plugin's element is
    // also seen in its marketplace checkout. Those rows are distinct index
    // entries but the SAME element, and emitting each of them offered the
    // identical name twice with nothing to tell them apart (measured: two
    // `python-test-writer` rows, one path, both surfaced). Namespacing cannot
    // fix that — the duplicates agree on their namespace precisely because they
    // are the same element — so collapse on resolved identity before emitting.
    let mut emitted: HashSet<(Option<String>, String, String)> = HashSet::new();
    let context_items: Vec<ContextItem> = matches
        .iter()
        .filter(|m| emitted.insert((m.plugin.clone(), m.name.clone(), m.path.clone())))
        .map(|m| {
            // Add commitment reminder for HIGH confidence (from reliable)
            let commitment = if m.confidence == Confidence::High {
                Some("Before implementing: Evaluate YES/NO - Will this skill solve the user's actual problem?".to_string())
            } else {
                None
            };

            ContextItem {
                item_type: m.skill_type.clone(),
                // The namespaced form IS the display name: a bare name is not
                // enough to tell the model which of several same-named elements
                // it is being offered, nor where it came from.
                name: namespaced_name(
                    &m.name,
                    m.plugin.as_deref(),
                    m.marketplace.as_deref(),
                    m.origin.as_deref(),
                    &name_counts,
                ),
                path: m.path.clone(),
                description: m.description.clone(),
                score: calculate_relative_score(m.score, max_score),
                confidence: m.confidence.as_str().to_string(),
                match_count: m.evidence.len(),
                evidence: m.evidence.clone(),
                commitment,
            }
        })
        .collect();

    // Log suggestions to stderr for debugging
    for item in &context_items {
        let conf_color = match item.confidence.as_str() {
            "HIGH" => item.confidence.green(),
            "MEDIUM" => item.confidence.yellow(),
            _ => item.confidence.red(),
        };
        info!(
            "{} {} [{}] - {} matches (score: {:.2}, confidence: {})",
            match item.item_type.as_str() {
                "skill" => "📚".green(),
                "agent" => "🤖".blue(),
                "command" => "⚡".yellow(),
                "rule" => "📏".cyan(),
                "mcp" => "🔌".magenta(),
                "lsp" => "🔤".white(),
                _ => "❓".white(),
            },
            item.name.bold(),
            item.item_type,
            item.match_count,
            item.score,
            conf_color
        );
    }

    // Hook mode: emit ONE element type — whichever the active suggestion mode
    // selects (agents by default, skills opt-in, nothing when silenced). Mixing
    // types would spend ~50 lines of the model's budget on every user message.
    // JSON mode (agent profiler): include ALL element types for .agent.toml generation.
    //
    // Read once, outside the closure: this is per-prompt hot-path code, and
    // re-reading the file per candidate would turn one stat into hundreds.
    let active_mode = if cli.format == "hook" {
        suggest_mode::read_mode(cli.index.as_deref())
    } else {
        SuggestionMode::DEFAULT // unused — JSON mode keeps every type below
    };
    let filtered_items: Vec<_> = context_items
        .into_iter()
        .filter(|item| {
            let t = item.item_type.as_str();
            if cli.format == "hook" {
                match active_mode.element_type() {
                    Some(wanted) => t == wanted,
                    // `none` — the user silenced suggestions; emit nothing.
                    Option::None => false,
                }
            } else {
                // JSON/profiler mode: all actionable types for .agent.toml profiles
                t == "skill" || t == "agent" || t == "command" || t == "rule" || t == "mcp" || t == "lsp" || t.is_empty()
            }
        })
        .collect();

    // Apply filters: require evidence, min-score, then --top limit
    let limited_items: Vec<_> = filtered_items
        .into_iter()
        .filter(|item| !item.evidence.is_empty())  // Must have at least 1 keyword match
        .filter(|item| item.score >= cli.min_score)
        .take(cli.top)
        .collect();

    // Log activation with matches and timing
    let processing_ms = start_time.elapsed().as_millis() as u64;
    let session_id = if input.session_id.is_empty() { None } else { Some(input.session_id.as_str()) };
    log_activation(
        &input.prompt,
        session_id,
        Some(&input.cwd),
        sub_tasks.len(),
        &matches,
        Some(processing_ms),
    );

    // Output based on --format option
    match cli.format.as_str() {
        "json" => {
            // Raw JSON format for Pass 2 agents - just skill metadata
            #[derive(Serialize)]
            struct CandidateSkill {
                name: String,
                path: String,
                pss_path: String,  // Path to .pss file for reading
                score: f64,
                raw_score: i32,    // FM-W3: debug field for raw score analysis
                confidence: String,
                keywords_matched: Vec<String>,
            }

            // FM-W3: Build a raw score lookup from the original matches
            // for debugging purposes (raw_score field in JSON output)
            let raw_score_map: std::collections::HashMap<String, i32> = matches
                .iter()
                .map(|m| (m.name.clone(), m.score))
                .collect();

            let candidates: Vec<CandidateSkill> = limited_items
                .iter()
                .map(|item| {
                    // PSS files are transient, not persisted next to SKILL.md
                    let pss_path = String::new();

                    CandidateSkill {
                        name: item.name.clone(),
                        path: item.path.clone(),
                        pss_path,
                        score: item.score,
                        raw_score: *raw_score_map.get(&item.name).unwrap_or(&0),
                        confidence: item.confidence.clone(),
                        keywords_matched: item.evidence.clone(),
                    }
                })
                .collect();

            println!("{}", serde_json::to_string_pretty(&candidates)?);
        }
        _ => {
            // Default hook format for Claude Code integration.
            //
            // Dedupe window (fleet request 2026-08-18): the exact set the
            // model just saw carries no new information — emit nothing when
            // it repeats within the TTL. Empty sets never touch the state
            // file, so a quiet prompt can't "use up" the window.
            let mut items = limited_items;
            // Dedupe ONLY when a session_id is present: real CC hook input
            // always carries one, while test harnesses and manual invocations
            // don't — and deduping across unrelated session-less invocations
            // through the shared state file is wrong (it turned the CI e2e
            // suite red on v3.13.1: Phase 5 scored a prompt, Phase 6 re-sent
            // the same prompt seconds later and got an empty emission).
            if cli.format == "hook" && !items.is_empty() && !input.session_id.is_empty() {
                let names: Vec<&str> = items.iter().map(|i| i.name.as_str()).collect();
                if suggest_dedupe::should_suppress_and_record(
                    cli.index.as_deref(),
                    &input.session_id,
                    active_mode.as_str(),
                    &names,
                ) {
                    debug!("suggest-dedupe: identical set within TTL — suppressing emission");
                    items.clear();
                }
            }
            let output = HookOutput::with_suggestions(items, active_mode);
            println!("{}", serde_json::to_string(&output)?);
        }
    }

    Ok(())
}

/// Check if prompt should be skipped (simple words, task notifications)
/// Strip `<system-reminder>...</system-reminder>` blocks from prompt text.
///
/// PERF-1 (audit 20260514): ported from `scripts/pss_hook.py:_strip_system_reminders`.
/// Linear-time find/skip loop (no regex backtracking) — critical for 200 KB
/// prompts where `re.DOTALL` would add >1s of overhead. The function is
/// idempotent: applying it twice produces the same result, so it's safe to
/// call from both the legacy Python wrapper path and a direct-binary path.
pub(crate) fn strip_system_reminders(text: &str) -> String {
    const OPEN: &str = "<system-reminder>";
    const CLOSE: &str = "</system-reminder>";
    let mut out = String::with_capacity(text.len());
    let mut pos = 0;
    while pos < text.len() {
        match text[pos..].find(OPEN) {
            None => {
                out.push_str(&text[pos..]);
                break;
            }
            Some(rel_start) => {
                let start = pos + rel_start;
                if start > pos {
                    out.push_str(&text[pos..start]);
                }
                let after_open = start + OPEN.len();
                match text[after_open..].find(CLOSE) {
                    None => break, // unclosed tag — discard rest (contains system content)
                    Some(rel_end) => {
                        pos = after_open + rel_end + CLOSE.len();
                    }
                }
            }
        }
    }
    out.trim().to_string()
}

/// Decide whether a prompt should skip skill suggestion entirely.
///
/// PERF-1 (audit 20260514): ported from `scripts/pss_hook.py:should_skip_prompt`.
/// Matches the Python implementation including slash-command prefix, system
/// tag detection, release-notes pattern, and the simple-word allowlist. Called
/// AFTER `strip_system_reminders` so we don't see legitimate <system-reminder>
/// tags as triggers.
pub(crate) fn is_skip_prompt(prompt: &str) -> bool {
    if prompt.is_empty() {
        return true;
    }
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return true;
    }

    // Slash commands like /plugin, /help, /exit
    if trimmed.starts_with('/') || trimmed.starts_with("<command-name>/") {
        return true;
    }

    // Automation-shaped prompts (fleet request, 2026-08-18): machine-generated
    // traffic — cron marker fires and cross-session peer messages — is not a
    // user ask, so a suggestion on it is pure noise. Two detectable shapes:
    //
    //  * Marker fire: the FIRST LINE is exactly `[token]` (lowercase ASCII
    //    letters/digits/hyphens), e.g. a `[janitor-heartbeat]` cron prompt.
    //    Whole-line on purpose: "[bug] parser crashes" stays suggestible.
    //  * Peer envelope: cross-session messages arrive wrapped in a
    //    `<cross-session-message …>` tag.
    let first_line = trimmed.lines().next().unwrap_or("").trim_end();
    if let Some(inner) = first_line
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
    {
        if !inner.is_empty()
            && inner
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            return true;
        }
    }
    if trimmed.contains("<cross-session-message") {
        return true;
    }

    // System-generated content tags (system-reminder is already stripped
    // earlier; these are the remaining tags Python skips).
    if trimmed.contains("<task-notification>")
        || trimmed.contains("<local-command-caveat>")
        || trimmed.contains("<local-command-stdout>")
    {
        return true;
    }

    // /release-notes paste: starts with "Version " and contains "\n• "
    // within the first 500 chars.
    if trimmed.starts_with("Version ") {
        let head_end = trimmed.len().min(500);
        if trimmed[..head_end].contains("\n• ") {
            return true;
        }
    }

    // Simple one-word / short-phrase confirmations.
    let lower = trimmed.to_lowercase();
    matches!(lower.as_str(),
        "continue" | "yes" | "no" | "ok" | "okay" | "thanks" | "sure"
        | "done" | "stop" | "y" | "n" | "yep" | "nope" | "thx" | "ty"
        | "next" | "go" | "proceed" | "k" | "yea" | "yeah" | "nah"
        | "good" | "great" | "perfect" | "fine" | "cool" | "nice"
        // Two-word phrases
        | "got it" | "thank you" | "thanks!" | "ok thanks" | "okay thanks"
        | "sounds good" | "go ahead" | "do it" | "looks good" | "that works"
        | "yes please" | "no thanks" | "i see" | "i understand"
        | "makes sense" | "all good" | "thank you!"
    )
}

/// Augment a short prompt with the previous user message from the transcript.
///
/// PERF-1 (audit 20260514): ported from `scripts/pss_hook.py:augment_prompt_with_context`.
/// Only augments when the current prompt has fewer than 30 non-trivial chars
/// (alphanumeric only) — longer prompts already carry enough signal, and
/// prepending unrelated history pollutes scoring. Total output capped at
/// `max_prompt_chars` to keep stdin small.
pub(crate) fn augment_prompt_if_short(prompt: &str, transcript_path: &str, max_prompt_chars: usize) -> String {
    let stripped = prompt.trim();
    let stripped = if stripped.len() > max_prompt_chars {
        &stripped[..max_prompt_chars]
    } else {
        stripped
    };

    // Count alphanumeric chars; skip augmentation if prompt has ≥30.
    let non_trivial: usize = stripped.chars().filter(|c| c.is_alphanumeric()).count();
    if non_trivial >= 30 {
        return stripped.to_string();
    }

    if transcript_path.is_empty() {
        return stripped.to_string();
    }

    let prev_msg = extract_prev_user_message(transcript_path);
    if prev_msg.is_empty() {
        return stripped.to_string();
    }

    // Cap prev_msg so total stays under max_prompt_chars (with 1-char separator).
    let budget = max_prompt_chars
        .saturating_sub(stripped.len())
        .saturating_sub(1);
    if budget <= 200 {
        return stripped.to_string();
    }
    let prev_capped = if prev_msg.len() > budget {
        &prev_msg[..budget]
    } else {
        prev_msg.as_str()
    };
    format!("{} {}", prev_capped, stripped)
}
