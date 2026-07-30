//! Generation of the three agent archetypes: ALL-IN-ONE, ONE-FOR-ALL, PLUGIN-OMNI.
//!
//! The invariant this module exists to protect: **skill content is never inlined
//! into an agent.** An agent references skills by bare name; the skill stays a
//! standalone file that can be shared, edited and updated in one place. Copying a
//! skill FILE into a generated plugin (for portability) is not inlining — the skill
//! is still a skill.
//!
//! ## Why the archetypes differ, in one table
//!
//! | kind | orchestrator `skills:` | where a skill executes |
//! |---|---|---|
//! | ALL-IN-ONE  | every step skill + the verification skill | inline, same agent |
//! | ONE-FOR-ALL | the generated step MENU only               | one skill per subagent |
//! | PLUGIN-OMNI | the generated plugin MENU only             | via the Skill tool, on demand |
//!
//! ONE-FOR-ALL deliberately does NOT preload step bodies. `skills:` injects the FULL
//! body of every listed skill at subagent startup (verified empirically: an agent
//! preloading two skills quoted one of them byte-for-byte having made zero tool
//! calls). Preloading step bodies into the router would pay each body on every router
//! turn AND again inside each micro-agent — making ONE-FOR-ALL cost more than
//! ALL-IN-ONE, which is the opposite of its purpose. The router only needs each
//! step's name, when-to-use, arguments and report path; that is what the menu carries.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Which archetype to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Archetype {
    /// A plain subagent — no menu, no router, no micro-agent. The escape hatch
    /// for when none of the orchestration shapes is warranted.
    Normal,
    AllInOne,
    OneForAll,
    PluginOmni,
}

impl Archetype {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "normal" | "plain" | "standard" => Some(Self::Normal),
            "all-in-one" | "allinone" | "allin1" | "aio" => Some(Self::AllInOne),
            "one-for-all" | "oneforall" | "1xall" | "ofa" => Some(Self::OneForAll),
            "plugin-omni" | "pluginomni" | "omni" => Some(Self::PluginOmni),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::AllInOne => "all-in-one",
            Self::OneForAll => "one-for-all",
            Self::PluginOmni => "plugin-omni",
        }
    }
}

/// Which element types the generated agent may reference.
///
/// These are exclusions rather than inclusions because the useful default is
/// "give the agent what the description calls for", and the flags exist for the
/// cases where a caller knows a whole class is unwanted.
#[derive(Debug, Clone, Copy, Default)]
pub struct ElementFilters {
    pub no_skill: bool,
    pub no_agent: bool,
    pub no_mcp: bool,
}

/// Turn a free-text specialization into a kebab-case agent name.
///
/// Only used when the caller gave a description but no `--name`. Stops at the
/// first four meaningful words: a name derived from a whole paragraph is worse
/// than no name at all.
pub fn derive_name(description: &str) -> String {
    const SKIP: &[&str] = &[
        "a", "an", "the", "for", "that", "which", "with", "and", "or", "to", "of",
        "in", "on", "is", "it", "this", "agent", "specialized", "specialises",
        "specializes", "specialist",
    ];
    let words: Vec<String> = description
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_ascii_lowercase())
        .filter(|w| !SKIP.contains(&w.as_str()))
        .take(4)
        .collect();
    if words.is_empty() {
        "generated-agent".to_string()
    } else {
        format!("{}-agent", words.join("-"))
    }
}

/// Execution environment for a ONE-FOR-ALL step.
///
/// **`Custom` is the default, and the measurement is why.** Two probes, same
/// model, same prompt, zero tool calls each:
///
/// | environment | project CLAUDE.md | `~/.claude/rules/*` | tokens |
/// |---|---|---|---|
/// | built-in `Explore` | ABSENT | LOADED | 66,954 |
/// | custom minimal agent | LOADED | LOADED | 68,563 |
///
/// Explore saves **~1.6k**, not the ~54k the CLAUDE.md hierarchy would suggest —
/// because it skips only the *project* CLAUDE.md, while the rules files (the bulk
/// of that total) load in both. Against that 2%, Explore costs you a second
/// environment to reason about and cannot write files at all (its system prompt
/// forbids creating them, including under /tmp — not a tool gap a prompt can
/// override). One environment for every step is simpler and within noise.
///
/// The enum stays because the trade-off is a measurement, and measurements
/// change; `--explore` opts back in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicroEnv {
    /// Built-in `Explore`: skips CLAUDE.md, cannot write, carries MCP schemas.
    Explore,
    /// The generated minimal custom agent: can write, no MCP, pays CLAUDE.md.
    Custom,
}

impl MicroEnv {
    pub fn agent_name(self, agent: &str) -> String {
        match self {
            Self::Explore => "Explore".to_string(),
            Self::Custom => format!("{}-micro", agent),
        }
    }
}

/// A skill the generator was asked to reference, resolved to its file on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillRef {
    pub name: String,
    /// Absolute path to the skill's `SKILL.md`. Empty when the caller could not
    /// resolve it — which the gate treats as fatal, never as "probably fine".
    pub path: String,
    pub description: String,
}

/// Why a skill cannot be preloaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    Unresolved,
    UserOnly,
    Bundled,
    Unreadable(String),
}

impl RejectReason {
    pub fn explain(&self) -> String {
        match self {
            Self::Unresolved => {
                "not found in the index — a listed-but-missing skill is SILENTLY \
                 skipped at startup with only a debug-log warning, so the agent would \
                 lose it with no error"
                    .to_string()
            }
            Self::UserOnly => {
                "sets `disable-model-invocation: true` — preloading draws from the \
                 same set Claude may invoke, so this can never be preloaded"
                    .to_string()
            }
            Self::Bundled => {
                "is a bundled user-only skill (`/verify`, `/code-review`) and cannot \
                 be preloaded"
                    .to_string()
            }
            Self::Unreadable(e) => format!("could not be read from disk: {}", e),
        }
    }
}

/// Outcome of the preload gate.
#[derive(Debug, Default)]
pub struct GateResult {
    pub ok: Vec<SkillRef>,
    pub rejected: Vec<(String, RejectReason)>,
}

/// Bundled skills that are user-invocable only and therefore never preloadable.
const BUNDLED_USER_ONLY: &[&str] = &["verify", "code-review"];

/// Extract the raw YAML frontmatter block from a markdown document.
///
/// Returns `None` when the file does not open with a `---` fence. Deliberately
/// tolerant of a UTF-8 BOM and CRLF, both of which appear in skills authored on
/// Windows and would otherwise make a perfectly good skill look frontmatter-less.
pub fn frontmatter_block(content: &str) -> Option<&str> {
    let body = content.strip_prefix('\u{feff}').unwrap_or(content);
    let body = body.trim_start_matches(['\r', '\n']);
    let rest = body.strip_prefix("---")?;
    let rest = rest.strip_prefix('\r').unwrap_or(rest);
    let rest = rest.strip_prefix('\n')?;
    // The closing fence must be its own line.
    for (idx, line) in rest.match_indices("---") {
        let starts_line = idx == 0 || rest[..idx].ends_with('\n');
        let ends_line = rest[idx + line.len()..]
            .chars()
            .next()
            .is_none_or(|c| c == '\n' || c == '\r');
        if starts_line && ends_line {
            return Some(&rest[..idx]);
        }
    }
    None
}

/// Read a scalar boolean field out of a frontmatter block.
///
/// Hand-rolled rather than pulling in a YAML parser: the gate needs exactly one
/// boolean, and the frontmatter it reads is authored by third parties whose files
/// must not be able to fail the whole generation because of an unrelated YAML
/// quirk elsewhere in the block.
fn frontmatter_bool(block: &str, key: &str) -> Option<bool> {
    for line in block.lines() {
        let line = line.trim_end();
        if line.starts_with([' ', '\t']) {
            continue; // nested — not a top-level field
        }
        let (k, v) = line.split_once(':')?;
        if k.trim() != key {
            continue;
        }
        let v = v.trim().trim_matches(['"', '\'']).to_ascii_lowercase();
        return match v.as_str() {
            "true" | "yes" | "on" => Some(true),
            "false" | "no" | "off" => Some(false),
            _ => None,
        };
    }
    None
}

/// Decide whether every skill may legally be preloaded into an agent's `skills:`.
///
/// The check reads each `SKILL.md` **from disk** rather than consulting the index,
/// because the index stores none of `disable-model-invocation`, `context`, `agent`
/// or `user-invocable` (verified against a live `pss inspect`). Reading from disk
/// also means a stale index cannot wave through a skill that has since been made
/// user-only.
pub fn gate_preloadable(skills: &[SkillRef]) -> GateResult {
    let mut out = GateResult::default();
    for skill in skills {
        if BUNDLED_USER_ONLY.contains(&skill.name.as_str()) {
            out.rejected
                .push((skill.name.clone(), RejectReason::Bundled));
            continue;
        }
        if skill.path.trim().is_empty() {
            out.rejected
                .push((skill.name.clone(), RejectReason::Unresolved));
            continue;
        }
        let content = match fs::read_to_string(&skill.path) {
            Ok(c) => c,
            Err(e) => {
                out.rejected
                    .push((skill.name.clone(), RejectReason::Unreadable(e.to_string())));
                continue;
            }
        };
        let user_only = frontmatter_block(&content)
            .and_then(|fm| frontmatter_bool(fm, "disable-model-invocation"))
            .unwrap_or(false);
        if user_only {
            out.rejected
                .push((skill.name.clone(), RejectReason::UserOnly));
        } else {
            out.ok.push(skill.clone());
        }
    }
    out
}

/// A ONE-FOR-ALL step and the environment chosen to run it.
#[derive(Debug, Clone)]
pub struct StepPlan {
    pub skill: SkillRef,
    pub mutating: bool,
    pub environment: MicroEnv,
}

/// Tokens that indicate a skill's procedure modifies files.
///
/// A step that only reads can run in an environment that cannot write, which is
/// what makes the cheap path available; anything that edits must go to an
/// environment with Edit. Misclassifying a mutating step as read-only would hand it
/// to an agent that physically cannot do the job, so the test is deliberately
/// generous: any hint of mutation wins.
const MUTATING_HINTS: &[&str] = &[
    "edit", "write", "modify", "patch", "refactor", "fix", "rename", "delete",
    "remove", "create", "update", "apply", "format", "generate", "install",
    "commit", "migrate",
];

/// Classify one step from its skill's declared tools and prose.
pub fn classify_step(skill: &SkillRef, allow_explore: bool) -> StepPlan {
    let mutating = detect_mutating(skill);
    let env = if mutating || !allow_explore {
        MicroEnv::Custom
    } else {
        MicroEnv::Explore
    };
    StepPlan {
        skill: skill.clone(),
        mutating,
        environment: env,
    }
}

fn detect_mutating(skill: &SkillRef) -> bool {
    let content = fs::read_to_string(&skill.path).unwrap_or_default();
    // An explicit allowed-tools list is authoritative when present: a skill that
    // declares Write/Edit mutates regardless of how its prose is worded.
    if let Some(fm) = frontmatter_block(&content) {
        for line in fm.lines() {
            if line.trim_start().starts_with("allowed-tools") {
                let lower = line.to_ascii_lowercase();
                if lower.contains("write") || lower.contains("edit") {
                    return true;
                }
                // Declared, and it does not include a writing tool.
                return false;
            }
        }
    }
    let haystack = format!("{} {}", skill.name, skill.description).to_ascii_lowercase();
    MUTATING_HINTS.iter().any(|h| haystack.contains(h))
}

/// A file the generator wants to create, held in memory until every one is ready.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedFile {
    pub path: PathBuf,
    pub contents: String,
}

/// Write a file atomically: full contents to a sibling temp, then rename.
///
/// Same shape as the index export path. A half-written agent definition is worse
/// than none at all — Claude Code would load it and behave unpredictably — so the
/// file must appear complete or not appear.
pub fn write_atomic(path: &Path, contents: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut tmp = path.to_path_buf();
    let stem = path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "out".to_string());
    tmp.set_file_name(format!(".{}.tmp", stem));
    fs::write(&tmp, contents)?;
    match fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Quote a value for a single-line YAML scalar.
pub fn yaml_scalar(raw: &str) -> String {
    let flat = raw.replace(['\r', '\n'], " ");
    let flat = flat.trim();
    let escaped = flat.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{}\"", escaped)
}

/// Render a YAML block sequence, or nothing when the list is empty.
fn yaml_list(key: &str, items: &[String]) -> String {
    if items.is_empty() {
        return String::new();
    }
    let mut out = format!("{}:\n", key);
    for item in items {
        out.push_str(&format!("  - {}\n", item));
    }
    out
}

/// The verification skill every generated agent is held to.
pub const VERIFICATION_SKILL: &str = "verification-before-completion";

/// Build the orchestrator's `skills:` list for an archetype.
///
/// This is the single place the preload policy lives, so the three archetypes
/// cannot drift apart silently.
pub fn orchestrator_skills(
    kind: Archetype,
    steps: &[SkillRef],
    menu_skill: &str,
    verification_skill: &str,
) -> Vec<String> {
    match kind {
        // A plain agent still gets the verification skill: it is the one thing
        // every generated agent is held to regardless of shape.
        Archetype::Normal => {
            let mut v: Vec<String> = steps.iter().map(|s| s.name.clone()).collect();
            v.push(verification_skill.to_string());
            v
        }
        // Skills execute in this same agent, so preloading them is the point.
        Archetype::AllInOne => {
            let mut v: Vec<String> = steps.iter().map(|s| s.name.clone()).collect();
            v.push(verification_skill.to_string());
            v
        }
        // Menu only. Step bodies load once, inside the micro-agent that runs them.
        Archetype::OneForAll => vec![menu_skill.to_string(), verification_skill.to_string()],
        // Exactly one. The verification rules are inlined into the body instead, so
        // the frontmatter stays a single entry as specified.
        Archetype::PluginOmni => vec![menu_skill.to_string()],
    }
}

/// What a `Custom` step costs *over* an `Explore` one, for the cost table.
///
/// This is a MEASURED DIFFERENTIAL, not the size of the CLAUDE.md hierarchy. An
/// earlier version of this table used 53,800 — the full hierarchy — and so told
/// authors that routing a step through Explore saved ~54k. It saves ~1.6k: both
/// environments load `~/.claude/rules/*`, and only the project CLAUDE.md differs.
/// Printing the larger number would recommend a two-environment split on a
/// saving that does not exist.
#[derive(Debug, Clone, Copy)]
pub struct CostModel {
    pub custom_premium_tokens: usize,
}

impl Default for CostModel {
    fn default() -> Self {
        // 68,563 (custom, 0 tool calls) − 66,954 (Explore, 0 tool calls).
        Self {
            custom_premium_tokens: 1_609,
        }
    }
}

/// Render the per-step environment table.
///
/// Shows what each step costs ABOVE the cheapest available environment, so the
/// author sees the real trade-off rather than a headline number. The premium is
/// small by design — see `CostModel`.
pub fn cost_table(plan: &[StepPlan], cost: CostModel) -> String {
    let mut out = String::from("  step  environment   vs. Explore  reason\n");
    let mut total = 0usize;
    for (i, step) in plan.iter().enumerate() {
        let (env, tokens, reason) = match step.environment {
            MicroEnv::Explore => ("Explore", 0usize, "read-only; cannot write files"),
            MicroEnv::Custom => (
                "custom micro",
                cost.custom_premium_tokens,
                if step.mutating {
                    "mutates files; needs Edit"
                } else {
                    "default; one environment for every step"
                },
            ),
        };
        total += tokens;
        out.push_str(&format!(
            "  {:<4}  {:<12}  {:>11}  {}\n",
            i + 1,
            env,
            format!("+{}", tokens),
            reason
        ));
    }
    out.push_str(&format!(
        "  {:<4}  {:<12}  {:>11}  measured differential, not the CLAUDE.md total\n",
        "",
        "TOTAL",
        format!("+{}", total)
    ));
    out
}

/// Everything the caller resolved before emission can begin.
#[derive(Debug, Clone)]
pub struct GenSpec {
    pub kind: Archetype,
    pub name: String,
    pub description: String,
    pub model: Option<String>,
    pub skills: Vec<SkillRef>,
    /// Source plugin, for PLUGIN-OMNI.
    pub plugin: Option<String>,
    /// Whether non-mutating steps may use the built-in `Explore`.
    pub allow_explore: bool,
    /// Directory the bundle is written under.
    pub out_dir: PathBuf,
    /// Optional `effort:` pin.
    pub effort: Option<String>,
    /// Complementary agents the description selected. Emitted only as prose —
    /// an agent references another agent by launching it, not by a frontmatter
    /// field, so listing them anywhere else would be inventing a schema.
    pub agents: Vec<String>,
    /// MCP servers the description selected.
    pub mcp: Vec<String>,
    /// Which element classes are excluded.
    pub filters: ElementFilters,
}

impl GenSpec {
    fn menu_skill(&self) -> String {
        match self.kind {
            Archetype::PluginOmni => format!(
                "{}-the-skills-menu",
                self.plugin.as_deref().unwrap_or(&self.name)
            ),
            _ => format!("{}-step-menu", self.name),
        }
    }
}

/// The complete emission: every file, held in memory until all are ready.
#[derive(Debug, Default)]
pub struct Emission {
    pub files: Vec<EmittedFile>,
    pub plan: Vec<StepPlan>,
    pub warnings: Vec<String>,
}

impl Emission {
    /// Write every file, or none.
    ///
    /// Each file is written atomically, and the whole set is checked for path
    /// collisions first — a generator that stops half way leaves a directory that
    /// looks like a valid agent bundle but is missing the skill its agent names.
    pub fn commit(&self) -> Result<(), String> {
        let paths: Vec<String> = self
            .files
            .iter()
            .map(|f| f.path.to_string_lossy().to_string())
            .collect();
        assert_unique(&paths)?;
        for f in &self.files {
            write_atomic(&f.path, &f.contents)
                .map_err(|e| format!("writing {}: {}", f.path.display(), e))?;
        }
        Ok(())
    }
}

/// The verification rules, inlined as prose for PLUGIN-OMNI.
///
/// Only this archetype inlines them, and only because its frontmatter is
/// specified to carry exactly one skill (the plugin menu). The other two get the
/// real `verification-before-completion` skill in `skills:` instead — which is
/// the better mechanism, because the skill can be updated once for everybody.
const VERIFICATION_INLINE: &str = "\
## Before you claim anything is done

Verification is not optional and it is not a summary of your intent.

1. **Run it.** A change that has not been executed is a hypothesis. Run the
   test, the build, the command — whatever proves the behavior.
2. **Read the actual output.** Not the exit code alone, not what you expected
   to see. If a test passes, confirm it ran and was not skipped.
3. **Report what happened, including the parts that failed.** \"Tests pass\"
   when two were skipped is a false report. Say which, and why.
4. **If you could not verify something, say so explicitly** and name what would
   verify it. An unverified claim presented as fact is the failure mode this
   section exists to prevent.
";

/// Emit the orchestrator agent, its menu skill (if any) and the micro-agent.
pub fn emit(spec: &GenSpec) -> Emission {
    let mut em = Emission::default();
    let out = spec.out_dir.clone();

    // `--no-skill` is applied BEFORE the gate: gating a set the caller has
    // already excluded would emit warnings about skills that were never going
    // to be referenced.
    let usable = if spec.filters.no_skill {
        Vec::new()
    } else {
        let gate = gate_preloadable(&spec.skills);
        for (name, why) in &gate.rejected {
            em.warnings.push(format!("skill `{}` {}", name, why.explain()));
        }
        gate.ok
    };

    if spec.kind == Archetype::OneForAll {
        em.plan = usable
            .iter()
            .map(|s| classify_step(s, spec.allow_explore))
            .collect();
    }

    let menu = spec.menu_skill();
    let preload = orchestrator_skills(spec.kind, &usable, &menu, VERIFICATION_SKILL);

    em.files.push(EmittedFile {
        path: out.join("agents").join(format!("{}.md", spec.name)),
        contents: render_agent(spec, &usable, &preload, &em.plan),
    });

    // Normal and all-in-one reference their skills directly; only the two
    // menu-driven archetypes need a menu file.
    if matches!(spec.kind, Archetype::OneForAll | Archetype::PluginOmni) {
        em.files.push(EmittedFile {
            path: out.join("skills").join(&menu).join("SKILL.md"),
            contents: render_menu(spec, &usable, &menu, &em.plan),
        });
    }

    // Only ONE-FOR-ALL spawns subagents, and only a mutating step needs an
    // environment that can write. Emitting the micro-agent unconditionally would
    // ship a file nothing references.
    if em.plan.iter().any(|s| s.environment == MicroEnv::Custom) {
        em.files.push(EmittedFile {
            path: out
                .join("agents")
                .join(format!("{}-micro.md", spec.name)),
            contents: render_micro(&spec.name),
        });
    }

    em.files.push(EmittedFile {
        path: out.join("pss-agent-deps.json"),
        contents: render_deps(spec, &usable),
    });

    em
}

fn render_agent(
    spec: &GenSpec,
    usable: &[SkillRef],
    preload: &[String],
    plan: &[StepPlan],
) -> String {
    let mut s = String::from("---\n");
    s.push_str(&format!("name: {}\n", spec.name));
    s.push_str(&format!("description: {}\n", yaml_scalar(&spec.description)));
    if let Some(m) = &spec.model {
        s.push_str(&format!("model: {}\n", m));
    }
    if let Some(e) = &spec.effort {
        s.push_str(&format!("effort: {}\n", e));
    }
    s.push_str(&yaml_list("skills", preload));
    let tools = match spec.kind {
        // The router launches subagents, so it needs the Agent tool and little else.
        Archetype::OneForAll => vec!["Agent", "Read", "Bash", "Skill"],
        _ => vec!["Bash", "Read", "Write", "Edit", "Glob", "Grep", "Skill"],
    };
    // An explicit `tools:` list is an ALLOWLIST — it is what keeps MCP tool
    // schemas out of the agent's context, and every archetype emits one. So
    // `--no-mcp` does not need to subtract anything here; what it additionally
    // suppresses is the `mcpServers:` declaration below.
    s.push_str(&yaml_list(
        "tools",
        &tools.iter().map(|t| t.to_string()).collect::<Vec<_>>(),
    ));
    if !spec.filters.no_mcp && !spec.mcp.is_empty() {
        s.push_str(&yaml_list("mcpServers", &spec.mcp));
    }
    s.push_str("---\n\n");

    match spec.kind {
        Archetype::Normal => {
            s.push_str(&format!("# {}\n\n{}\n\n", spec.name, spec.description));
            if !usable.is_empty() {
                s.push_str(
                    "The skills below are already loaded — invoke one with the Skill tool \
                     when the work calls for it. Do not re-read a skill file; you already \
                     have it.\n\n",
                );
                for sk in usable {
                    s.push_str(&format!("- **`{}`** — {}\n", sk.name, sk.description));
                }
                s.push('\n');
            }
            if !spec.filters.no_agent && !spec.agents.is_empty() {
                s.push_str(
                    "When a task fits one of these better than you, launch it with the \
                     Agent tool rather than doing the work yourself:\n\n",
                );
                for a in &spec.agents {
                    s.push_str(&format!("- `{}`\n", a));
                }
                s.push('\n');
            }
            s.push_str(
                "Report what you actually did, including anything you could not verify.\n",
            );
        }
        Archetype::AllInOne => {
            s.push_str(&format!("# {}\n\n", spec.name));
            s.push_str(
                "Every skill below is already loaded — invoke it with the Skill tool at the \
                 point in the procedure where it applies. Do not re-read a skill file; you \
                 already have it.\n\n## The procedure\n\n",
            );
            for (i, sk) in usable.iter().enumerate() {
                s.push_str(&format!(
                    "{}. **`{}`** — {}\n   Use it when that is the step in front of you. \
                     Skip it when the work does not call for it; a step run out of order is \
                     worse than a step skipped.\n",
                    i + 1,
                    sk.name,
                    sk.description
                ));
            }
            s.push_str(
                "\nWhen two steps could both apply, do the one whose precondition is \
                 already true. When none applies, say so rather than forcing the closest fit.\n",
            );
        }
        Archetype::OneForAll => {
            s.push_str(&format!("# {}\n\n", spec.name));
            s.push_str(&format!(
                "You are a router. You do **not** perform the steps yourself — each node \
                 below is one skill, run by its own subagent with a fresh context.\n\n\
                 `{}` is loaded and carries every step's name, when-to-use, inputs \
                 and report path. Consult it, pick the next node, launch it, read the report \
                 path it returns — never its content into this context unless you need it.\n\n\
                 ## Launching a step\n\n\
                 Call the Agent tool with the step's environment (below), and a prompt of the \
                 form: *\"Load the `<skill>` skill. Do only that step. Write your report to \
                 `<path>`. Reply with the path and nothing else.\"*\n\n\
                 ## The decision tree\n\n",
                spec.menu_skill()
            ));
            for (i, st) in plan.iter().enumerate() {
                s.push_str(&format!(
                    "{}. `{}` → agent `{}`{}\n",
                    i + 1,
                    st.skill.name,
                    st.environment.agent_name(&spec.name),
                    if st.mutating { " (writes files)" } else { "" }
                ));
            }
            s.push_str(
                "\nRun a node only when its precondition holds. If a step's report says it \
                 could not complete, do not advance past it — resolve or report the blocker.\n",
            );
        }
        Archetype::PluginOmni => {
            s.push_str(&format!("# {}\n\n", spec.name));
            s.push_str(&format!(
                "The `{}` skill is loaded: it lists every skill in the `{}` plugin with a \
                 description and when to reach for it. Treat it as a menu, not a script — \
                 read the situation, pick what fits, and invoke it with the Skill tool.\n\n\
                 There is no fixed order here. Several skills may apply to one request, or \
                 none may; say so rather than forcing one.\n\n",
                spec.menu_skill(),
                spec.plugin.as_deref().unwrap_or("this")
            ));
            s.push_str(VERIFICATION_INLINE);
        }
    }
    s
}

fn render_menu(spec: &GenSpec, usable: &[SkillRef], menu: &str, plan: &[StepPlan]) -> String {
    let mut s = String::from("---\n");
    s.push_str(&format!("name: {}\n", menu));
    s.push_str(&format!(
        "description: {}\n---\n\n",
        yaml_scalar(&match spec.kind {
            Archetype::PluginOmni => format!(
                "Every skill in the {} plugin, with when to use each.",
                spec.plugin.as_deref().unwrap_or("bundled")
            ),
            _ => format!("The steps {} routes between, with when to use each.", spec.name),
        })
    ));
    s.push_str(&format!("# {}\n\n", menu));

    // The menu carries names and when-to-use, never bodies. That is the whole
    // reason it is cheap enough to preload — and the reason a skill stays a
    // skill instead of becoming a copy inside an agent.
    for (i, sk) in usable.iter().enumerate() {
        s.push_str(&format!("## `{}`\n\n{}\n\n", sk.name, sk.description));
        // Only a one-for-all step runs in its own subagent, so only it has an
        // environment and a report path. A plugin-omni skill is invoked inline
        // via the Skill tool — telling it to write a report would be an
        // instruction with nobody to carry it out.
        if let Some(st) = plan.get(i) {
            s.push_str(&format!(
                "- run by: `{}`{}\n- report to: `reports/{}/<timestamp>-{}.md`\n",
                st.environment.agent_name(&spec.name),
                if st.mutating { " — writes files" } else { "" },
                spec.name,
                sk.name
            ));
        }
        s.push('\n');
    }
    s
}

fn render_micro(agent: &str) -> String {
    format!(
        "---\nname: {}-micro\ndescription: {}\ntools:\n  - Bash\n  - Read\n  - Write\n  - Edit\n  - Skill\n---\n\n\
         Load the skill you were told to load. Do that one step and nothing more.\n\n\
         Write your report to the path you were given. Reply with that path and nothing \
         else — returning the report itself would spend the caller's context on something \
         it can read from disk when it needs to.\n",
        agent,
        yaml_scalar("Runs one skill and writes its report to the given path.")
    )
}

fn render_deps(spec: &GenSpec, usable: &[SkillRef]) -> String {
    let mut s = String::from("{\n");
    s.push_str(&format!("  \"agent\": {},\n", yaml_scalar(&spec.name)));
    s.push_str(&format!("  \"archetype\": \"{}\",\n", spec.kind.as_str()));
    s.push_str("  \"skills\": [\n");
    for (i, sk) in usable.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"name\": {}, \"source\": {}}}{}\n",
            yaml_scalar(&sk.name),
            yaml_scalar(&sk.path),
            if i + 1 < usable.len() { "," } else { "" }
        ));
    }
    s.push_str("  ]\n}\n");
    s
}

/// Names that must not collide inside one generated output directory.
pub fn assert_unique(names: &[String]) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for n in names {
        if !seen.insert(n) {
            return Err(format!("duplicate generated name: {}", n));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmpdir(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!("pss-archetype-test-{}-{}", tag, std::process::id()));
        fs::create_dir_all(&base).unwrap();
        base
    }

    fn skill_file(dir: &Path, name: &str, frontmatter: &str, body: &str) -> SkillRef {
        let path = dir.join(format!("{}.md", name));
        let mut f = fs::File::create(&path).unwrap();
        write!(f, "---\nname: {}\n{}---\n\n{}\n", name, frontmatter, body).unwrap();
        SkillRef {
            name: name.to_string(),
            path: path.to_string_lossy().to_string(),
            description: format!("description of {}", name),
        }
    }

    #[test]
    fn archetype_parses_aliases_and_rejects_junk() {
        assert_eq!(Archetype::parse("all-in-one"), Some(Archetype::AllInOne));
        assert_eq!(Archetype::parse("ONE_FOR_ALL"), Some(Archetype::OneForAll));
        assert_eq!(Archetype::parse("omni"), Some(Archetype::PluginOmni));
        assert_eq!(Archetype::parse("nonsense"), None);
    }

    #[test]
    fn frontmatter_block_handles_bom_and_crlf() {
        let plain = "---\nname: a\n---\nbody";
        assert_eq!(frontmatter_block(plain), Some("name: a\n"));
        let crlf = "\u{feff}---\r\nname: a\r\n---\r\nbody";
        assert!(frontmatter_block(crlf).unwrap().contains("name: a"));
        assert_eq!(frontmatter_block("no frontmatter here"), None);
    }

    #[test]
    fn gate_rejects_user_only_skill() {
        let dir = tmpdir("gate-useronly");
        let s = skill_file(&dir, "deployer", "disable-model-invocation: true\n", "Deploy it.");
        let res = gate_preloadable(&[s]);
        assert!(res.ok.is_empty());
        assert_eq!(res.rejected.len(), 1);
        assert_eq!(res.rejected[0].1, RejectReason::UserOnly);
    }

    #[test]
    fn gate_accepts_ordinary_skill_and_rejects_unresolved() {
        let dir = tmpdir("gate-mixed");
        let good = skill_file(&dir, "tester", "", "Run the tests.");
        let missing = SkillRef {
            name: "ghost".into(),
            path: String::new(),
            description: String::new(),
        };
        let res = gate_preloadable(&[good, missing]);
        assert_eq!(res.ok.len(), 1);
        assert_eq!(res.rejected[0].1, RejectReason::Unresolved);
    }

    #[test]
    fn gate_rejects_bundled_user_only_names() {
        let res = gate_preloadable(&[SkillRef {
            name: "code-review".into(),
            path: "/nonexistent".into(),
            description: String::new(),
        }]);
        assert_eq!(res.rejected[0].1, RejectReason::Bundled);
    }

    #[test]
    fn one_for_all_preloads_menu_not_step_bodies() {
        let steps = vec![
            SkillRef { name: "a".into(), path: String::new(), description: String::new() },
            SkillRef { name: "b".into(), path: String::new(), description: String::new() },
        ];
        let ofa = orchestrator_skills(Archetype::OneForAll, &steps, "m-menu", VERIFICATION_SKILL);
        assert_eq!(ofa, vec!["m-menu".to_string(), VERIFICATION_SKILL.to_string()]);
        assert!(!ofa.contains(&"a".to_string()), "step bodies must not be preloaded");

        let aio = orchestrator_skills(Archetype::AllInOne, &steps, "m-menu", VERIFICATION_SKILL);
        assert!(aio.contains(&"a".to_string()) && aio.contains(&"b".to_string()));

        let omni = orchestrator_skills(Archetype::PluginOmni, &steps, "p-menu", VERIFICATION_SKILL);
        assert_eq!(omni.len(), 1, "plugin-omni carries exactly one skill");
    }

    #[test]
    fn allowed_tools_beats_prose_for_mutation() {
        let dir = tmpdir("classify");
        // Prose says "fix" but the declared tools are read-only: tools win.
        let readonly = skill_file(&dir, "fix-advisor", "allowed-tools: Read Grep\n", "Suggest a fix.");
        assert!(!classify_step(&readonly, true).mutating);
        assert_eq!(classify_step(&readonly, true).environment, MicroEnv::Explore);

        let writer = skill_file(&dir, "calm-name", "allowed-tools: Read Edit\n", "Adjust things.");
        assert!(classify_step(&writer, true).mutating);
        assert_eq!(classify_step(&writer, true).environment, MicroEnv::Custom);
    }

    /// The default: every step in one environment, Explore opted into.
    #[test]
    fn explore_is_opt_in_not_the_default() {
        let dir = tmpdir("no-explore");
        let readonly = skill_file(&dir, "auditor", "allowed-tools: Read\n", "Audit.");
        assert_eq!(classify_step(&readonly, false).environment, MicroEnv::Custom);
        assert_eq!(classify_step(&readonly, true).environment, MicroEnv::Explore);
    }

    #[test]
    fn write_atomic_leaves_no_temp_and_writes_contents() {
        let dir = tmpdir("atomic");
        let target = dir.join("nested").join("agent.md");
        write_atomic(&target, "hello").unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "hello");
        let leftovers: Vec<_> = fs::read_dir(target.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with('.'))
            .collect();
        assert!(leftovers.is_empty(), "temp file must not survive");
    }

    #[test]
    fn yaml_scalar_flattens_and_escapes() {
        assert_eq!(yaml_scalar("a \"b\"\nc"), "\"a \\\"b\\\" c\"");
    }

    #[test]
    fn yaml_list_is_empty_for_no_items() {
        assert_eq!(yaml_list("skills", &[]), "");
        assert_eq!(yaml_list("skills", &["a".into()]), "skills:\n  - a\n");
    }

    #[test]
    fn cost_table_totals_only_custom_envs() {
        let mk = |name: &str, env: MicroEnv, mutating: bool| StepPlan {
            skill: SkillRef { name: name.into(), path: String::new(), description: String::new() },
            mutating,
            environment: env,
        };
        let plan = vec![
            mk("a", MicroEnv::Explore, false),
            mk("b", MicroEnv::Custom, true),
        ];
        let table = cost_table(&plan, CostModel::default());
        assert!(table.contains("Explore"));
        assert!(table.contains("custom micro"));
        // The premium is the MEASURED Explore-vs-custom differential (~1.6k),
        // not the size of the CLAUDE.md hierarchy. A table showing ~54k here
        // would recommend a two-environment split that buys nothing.
        assert!(table.contains("+1609"), "got:\n{}", table);
        assert!(!table.contains("53"), "stale CLAUDE.md-sized figure:\n{}", table);
    }

    fn spec_with(kind: Archetype, dir: &Path, bodies: &[(&str, &str)]) -> GenSpec {
        let skills = bodies
            .iter()
            .map(|(n, body)| skill_file(dir, n, "", body))
            .collect();
        GenSpec {
            kind,
            name: "demo".into(),
            description: "A demo agent.".into(),
            model: None,
            skills,
            plugin: Some("demo-plugin".into()),
            allow_explore: true,
            out_dir: dir.join("out"),
            effort: None,
            agents: Vec::new(),
            mcp: Vec::new(),
            filters: ElementFilters::default(),
        }
    }

    /// The load-bearing invariant of the whole feature.
    #[test]
    fn no_emitted_file_contains_skill_body_text() {
        let dir = tmpdir("no-inline");
        // A sentence that appears ONLY in the skill body, nowhere in its name or
        // description — so finding it in the output can only mean inlining.
        let canary = "ZZQX-canary-sentence-from-the-skill-body";
        let spec = spec_with(
            Archetype::AllInOne,
            &dir,
            &[("alpha", canary), ("beta", canary)],
        );
        for kind in [Archetype::AllInOne, Archetype::OneForAll, Archetype::PluginOmni] {
            let em = emit(&GenSpec { kind, ..spec.clone() });
            assert!(!em.files.is_empty());
            for f in &em.files {
                assert!(
                    !f.contents.contains(canary),
                    "{:?} inlined skill body into {}",
                    kind,
                    f.path.display()
                );
            }
        }
    }

    /// Regression: the one-for-all body once carried a literal `{}-step-menu`
    /// because the placeholder sat in a plain string literal instead of a
    /// `format!`. It compiled, it tested green on every assertion we had, and it
    /// told the router to consult a skill that does not exist — only reading the
    /// real generated file caught it.
    #[test]
    fn no_emitted_file_contains_an_unsubstituted_placeholder() {
        let dir = tmpdir("placeholder");
        let spec = spec_with(Archetype::AllInOne, &dir, &[("alpha", "body")]);
        for kind in [Archetype::AllInOne, Archetype::OneForAll, Archetype::PluginOmni] {
            for f in emit(&GenSpec { kind, ..spec.clone() }).files {
                assert!(
                    !f.contents.contains("{}"),
                    "{:?} emitted an unsubstituted placeholder in {}",
                    kind,
                    f.path.display()
                );
            }
        }
    }

    #[test]
    fn parse_accepts_the_short_type_aliases() {
        assert_eq!(Archetype::parse("allin1"), Some(Archetype::AllInOne));
        assert_eq!(Archetype::parse("1xall"), Some(Archetype::OneForAll));
        assert_eq!(Archetype::parse("omni"), Some(Archetype::PluginOmni));
        assert_eq!(Archetype::parse("normal"), Some(Archetype::Normal));
    }

    #[test]
    fn derive_name_skips_filler_and_bounds_length() {
        assert_eq!(
            derive_name("An agent that is specialized in auditing Rust memory safety"),
            "auditing-rust-memory-safety-agent"
        );
        assert_eq!(derive_name("   "), "generated-agent");
    }

    #[test]
    fn no_skill_emits_no_skills_and_no_gate_warnings() {
        let dir = tmpdir("no-skill");
        let bad = skill_file(&dir, "manual", "disable-model-invocation: true\n", "b");
        let em = emit(&GenSpec {
            skills: vec![bad],
            filters: ElementFilters { no_skill: true, ..Default::default() },
            ..spec_with(Archetype::Normal, &dir, &[])
        });
        let agent = &em.files[0].contents;
        assert!(!agent.contains("- manual"));
        // The gate never ran, so it cannot warn about a skill the caller excluded.
        assert!(em.warnings.is_empty(), "got: {:?}", em.warnings);
    }

    #[test]
    fn no_mcp_suppresses_mcp_servers_and_tools_stay_an_allowlist() {
        let dir = tmpdir("no-mcp");
        let base = spec_with(Archetype::Normal, &dir, &[("alpha", "b")]);

        let with_mcp = emit(&GenSpec { mcp: vec!["ctx7".into()], ..base.clone() });
        assert!(with_mcp.files[0].contents.contains("mcpServers:\n  - ctx7\n"));

        let without = emit(&GenSpec {
            mcp: vec!["ctx7".into()],
            filters: ElementFilters { no_mcp: true, ..Default::default() },
            ..base
        });
        let agent = &without.files[0].contents;
        assert!(!agent.contains("mcpServers"));
        // The explicit tools: allowlist is what actually keeps MCP tool schemas
        // out of the agent's context; --no-mcp only drops the server declaration.
        assert!(agent.contains("tools:\n"));
    }

    #[test]
    fn no_agent_suppresses_the_complementary_agent_list() {
        let dir = tmpdir("no-agent");
        let base = spec_with(Archetype::Normal, &dir, &[("alpha", "b")]);
        let with = emit(&GenSpec { agents: vec!["helper".into()], ..base.clone() });
        assert!(with.files[0].contents.contains("`helper`"));
        let without = emit(&GenSpec {
            agents: vec!["helper".into()],
            filters: ElementFilters { no_agent: true, ..Default::default() },
            ..base
        });
        assert!(!without.files[0].contents.contains("`helper`"));
    }

    #[test]
    fn normal_emits_one_file_and_no_menu() {
        let dir = tmpdir("normal");
        let em = emit(&spec_with(Archetype::Normal, &dir, &[("alpha", "b")]));
        assert!(!em.files.iter().any(|f| f.path.ends_with("SKILL.md")));
        assert!(!em.files.iter().any(|f| f.path.ends_with("-micro.md")));
        assert!(em.plan.is_empty());
        assert!(em.files[0].contents.contains("skills:\n  - alpha\n"));
    }

    #[test]
    fn effort_and_model_reach_the_frontmatter() {
        let dir = tmpdir("effort");
        let em = emit(&GenSpec {
            model: Some("opus".into()),
            effort: Some("xhigh".into()),
            ..spec_with(Archetype::Normal, &dir, &[("alpha", "b")])
        });
        let agent = &em.files[0].contents;
        assert!(agent.contains("model: opus\n"));
        assert!(agent.contains("effort: xhigh\n"));
    }

    #[test]
    fn emitted_agents_reference_skills_by_bare_name() {
        let dir = tmpdir("bare-name");
        let spec = spec_with(Archetype::AllInOne, &dir, &[("alpha", "body")]);
        let em = emit(&spec);
        let agent = &em.files[0].contents;
        assert!(agent.contains("skills:\n  - alpha\n"), "got:\n{}", agent);
        assert!(
            !agent.contains(&dir.to_string_lossy().to_string()),
            "an absolute source path leaked into the agent"
        );
    }

    #[test]
    fn one_for_all_emits_menu_and_router_never_preloads_steps() {
        let dir = tmpdir("ofa-emit");
        let spec = spec_with(
            Archetype::OneForAll,
            &dir,
            &[("alpha", "body"), ("beta", "body")],
        );
        let em = emit(&spec);
        let agent = &em.files[0].contents;
        assert!(agent.contains("demo-step-menu"));
        assert!(!agent.contains("\n  - alpha\n"), "step preloaded into router");
        assert!(
            em.files.iter().any(|f| f.path.ends_with("demo-step-menu/SKILL.md")),
            "menu skill not emitted"
        );
        assert_eq!(em.plan.len(), 2);
    }

    #[test]
    fn plugin_omni_carries_one_skill_and_inlines_verification() {
        let dir = tmpdir("omni-emit");
        let spec = spec_with(Archetype::PluginOmni, &dir, &[("alpha", "body")]);
        let em = emit(&spec);
        let agent = &em.files[0].contents;
        assert!(agent.contains("demo-plugin-the-skills-menu"));
        assert_eq!(
            agent.matches("\n  - ").count(),
            1 + 7, // one skill + the seven default tools
            "plugin-omni must preload exactly one skill:\n{}",
            agent
        );
        assert!(agent.contains("Before you claim anything is done"));

        // A plugin-omni skill runs inline via the Skill tool — there is no
        // subagent, so a report path in its menu is an order nobody executes.
        let menu = em
            .files
            .iter()
            .find(|f| f.path.ends_with("SKILL.md"))
            .expect("menu not emitted");
        assert!(!menu.contents.contains("report to:"));
        assert!(!menu.contents.contains("run by:"));
    }

    #[test]
    fn one_for_all_menu_does_carry_env_and_report_path() {
        let dir = tmpdir("ofa-menu");
        let spec = spec_with(Archetype::OneForAll, &dir, &[("alpha", "body")]);
        let menu = emit(&spec)
            .files
            .into_iter()
            .find(|f| f.path.ends_with("SKILL.md"))
            .expect("menu not emitted");
        assert!(menu.contents.contains("run by:"));
        assert!(menu.contents.contains("report to:"));
    }

    #[test]
    fn micro_agent_emitted_only_when_a_step_mutates() {
        let dir = tmpdir("micro-cond");
        let readonly = skill_file(&dir, "auditor", "allowed-tools: Read\n", "Audit.");
        let writer = skill_file(&dir, "patcher", "allowed-tools: Edit\n", "Patch.");
        let base = GenSpec {
            kind: Archetype::OneForAll,
            name: "demo".into(),
            description: "d".into(),
            model: None,
            skills: vec![readonly.clone()],
            plugin: None,
            allow_explore: true,
            out_dir: dir.join("out"),
            effort: None,
            agents: Vec::new(),
            mcp: Vec::new(),
            filters: ElementFilters::default(),
        };
        let em = emit(&base);
        assert!(
            !em.files.iter().any(|f| f.path.ends_with("demo-micro.md")),
            "micro-agent emitted with no mutating step"
        );
        let em2 = emit(&GenSpec { skills: vec![readonly, writer], ..base });
        assert!(em2.files.iter().any(|f| f.path.ends_with("demo-micro.md")));
    }

    #[test]
    fn deps_manifest_covers_every_usable_skill() {
        let dir = tmpdir("deps");
        let spec = spec_with(Archetype::AllInOne, &dir, &[("alpha", "b"), ("beta", "b")]);
        let em = emit(&spec);
        let deps = em
            .files
            .iter()
            .find(|f| f.path.ends_with("pss-agent-deps.json"))
            .expect("manifest missing");
        assert!(deps.contents.contains("\"alpha\""));
        assert!(deps.contents.contains("\"beta\""));
        assert!(deps.contents.contains("all-in-one"));
    }

    #[test]
    fn rejected_skill_is_warned_and_left_out_of_preload() {
        let dir = tmpdir("emit-gate");
        let good = skill_file(&dir, "alpha", "", "body");
        let bad = skill_file(&dir, "manual", "disable-model-invocation: true\n", "body");
        let em = emit(&GenSpec {
            kind: Archetype::AllInOne,
            name: "demo".into(),
            description: "d".into(),
            model: None,
            skills: vec![good, bad],
            plugin: None,
            allow_explore: true,
            out_dir: dir.join("out"),
            effort: None,
            agents: Vec::new(),
            mcp: Vec::new(),
            filters: ElementFilters::default(),
        });
        assert_eq!(em.warnings.len(), 1);
        assert!(em.warnings[0].contains("manual"));
        assert!(!em.files[0].contents.contains("- manual\n"));
    }

    #[test]
    fn commit_writes_every_file_and_rejects_collisions() {
        let dir = tmpdir("commit");
        let out = dir.join("out");
        let mut em = Emission::default();
        em.files.push(EmittedFile {
            path: out.join("a.md"),
            contents: "one".into(),
        });
        em.commit().unwrap();
        assert_eq!(fs::read_to_string(out.join("a.md")).unwrap(), "one");

        em.files.push(EmittedFile {
            path: out.join("a.md"),
            contents: "two".into(),
        });
        assert!(em.commit().is_err(), "duplicate path must be refused");
    }

    #[test]
    fn assert_unique_catches_collisions() {
        assert!(assert_unique(&["a".into(), "b".into()]).is_ok());
        assert!(assert_unique(&["a".into(), "a".into()]).is_err());
    }
}
