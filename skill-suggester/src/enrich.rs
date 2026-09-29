// Pass 1 batch enrichment: deterministic keyword/category/intent generation
// + Cue→Activity classification (XUD7YUZH modularization step 8). Moved out
// of main.rs verbatim; re-exported at the crate root via `pub(crate) use`.

use std::fs;
use std::io::{self, BufRead, Write};

use crate::agent_meta;
use crate::data::LOW_SIGNAL_WORDS;
use crate::{
    ACTIVITY_REGISTRY, ALL_LOW_SIGNAL_CAP, LOW_SIGNAL_DIVISOR, MatchWeights, SuggesterError,
    ActivityDef, infer_domains_from_text, stem_word,
};

// ============================================================================
// Pass 1 Batch Enrichment — deterministic keyword/category/intent generation
// ============================================================================

/// Stopwords to filter out during keyword generation.
/// These add no discriminative value for skill matching.
pub(crate) fn is_pass1_stopword(w: &str) -> bool {
    matches!(
        w,
        "the" | "a" | "an" | "and" | "or" | "but" | "in" | "on" | "at" | "to" | "for"
            | "of" | "with" | "by" | "from" | "as" | "is" | "was" | "are" | "be" | "been"
            | "being" | "have" | "has" | "had" | "do" | "does" | "did" | "will" | "would"
            | "could" | "should" | "may" | "might" | "can" | "shall" | "this" | "that"
            | "these" | "those" | "it" | "its" | "you" | "your" | "use" | "using" | "used"
            | "all" | "each" | "every" | "any" | "both" | "more" | "most" | "other"
            | "some" | "such" | "than" | "too" | "very" | "just" | "also" | "not" | "no"
            | "so" | "if" | "then" | "when" | "how" | "what" | "which" | "who" | "whom"
            | "where" | "why" | "new" | "best" | "modern" | "advanced" | "expert"
    )
}

/// Generate keywords from element name + description (deterministic, no LLM).
/// Splits on separators, stems, deduplicates, caps at 15.
pub(crate) fn generate_pass1_keywords(name: &str, description: &str) -> Vec<String> {
    let mut keywords: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Helper: add a keyword if not already seen and not a stopword
    let mut add = |word: String| {
        if word.len() > 1 && !is_pass1_stopword(&word) && seen.insert(word.clone()) {
            keywords.push(word);
        }
    };

    // 1. Split name on separators: "go-developer" → ["go", "developer"]
    for part in name.split(|c: char| c == '-' || c == '_' || c == ' ' || c == '.') {
        let lower = part.to_lowercase();
        if lower.is_empty() {
            continue;
        }
        let stemmed = stem_word(&lower);
        add(lower.clone());
        if stemmed != lower {
            add(stemmed);
        }
    }

    // 2. Split description on whitespace, strip punctuation, stem
    for word in description.split_whitespace() {
        let lower: String = word
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        if lower.len() <= 2 {
            continue;
        }
        let stemmed = stem_word(&lower);
        add(lower.clone());
        if stemmed != lower {
            add(stemmed);
        }
    }

    // 3. Add compound name as-is for exact matching (e.g., "go-developer")
    let compound = name.to_lowercase();
    if compound.len() > 1 && !is_pass1_stopword(&compound) && !seen.contains(&compound) {
        seen.insert(compound.clone());
        keywords.push(compound);
    }

    // 4. Cap at 15 keywords
    keywords.truncate(15);
    keywords
}

// ============================================================================
// Activity Classification System — Cue→Activity Scoring (same tiers as hook mode)
// ============================================================================
// Replaces the 16-category flat taxonomy with a multi-activity scored system.
// Uses the same 4-tier logarithmic weights as the hook mode scorer:
//   Tool cues:      2,000 pts (e.g., "ruff" → linting)
//   Framework cues: 20,000 pts (e.g., "flutter" → mobile-development)
//   Keyword cues:   100 pts (phrase tier), 10 pts (common tier via LOW_SIGNAL_DIVISOR)
// Each entry gets scored against ALL activities; top-N are assigned.

/// Score an entry's text against a single activity definition using the same
/// logarithmic tier weights as the hook mode scorer.
/// Returns the total score (0 if no cues matched).
pub(crate) fn score_entry_against_activity(
    entry_words: &[&str],
    entry_name_parts: &[&str],
    activity: &ActivityDef,
    weights: &MatchWeights,
) -> i32 {
    let mut score: i32 = 0;
    let mut has_non_low_signal = false;

    // Helper: match a cue name against entry words.
    // Short cues (< 4 chars, e.g. "age", "d3", "k6") require exact word match
    // to avoid false positives like "coverage".contains("age").
    // Longer cues allow substring match (e.g. "docker" in "dockerfile" is valid).
    let cue_matches = |cue: &str, words: &[&str]| -> bool {
        if cue.len() < 4 {
            // Short cue: exact word match only
            words.iter().any(|w| *w == cue)
        } else {
            // Longer cue: exact match or substring
            words.iter().any(|w| *w == cue || w.contains(cue))
        }
    };

    // Framework-tier matching (20000 pts) — highest confidence cue
    for fw in activity.frameworks {
        if cue_matches(fw, entry_words) {
            score += weights.framework_match; // 20000
            has_non_low_signal = true;
        }
    }

    // Tool-tier matching (2000 pts) — high confidence cue
    // Same common-tool dampening as hook mode (COMMON_TOOLS at line ~6067)
    static INDEXER_COMMON_TOOLS: &[&str] = &[
        "python", "python3", "bash", "git", "npm", "pip", "node",
        "pnpm", "yarn", "cargo", "docker", "make", "rust", "go",
    ];
    for tool in activity.tools {
        if cue_matches(tool, entry_words) {
            let is_common = INDEXER_COMMON_TOOLS.iter().any(|t| *t == *tool);
            let tool_score = if is_common {
                weights.tool_match / 5 // 400 for common tools
            } else {
                weights.tool_match // 2000 for specific tools
            };
            score += tool_score;
            has_non_low_signal = true;
        }
    }

    // Keyword/phrase-tier matching (100 pts, /10 for low-signal)
    let mut kw_count = 0;
    for kw in activity.keywords {
        // Prefix-match: entry word can prefix-match keyword only if word is >= 4 chars
        // This prevents short words like "and" from matching "android"
        let matched = entry_words.iter().any(|w| {
            *w == *kw
                || w.starts_with(kw)
                || (w.len() >= 4 && kw.starts_with(w))
        });
        if matched {
            let is_low = LOW_SIGNAL_WORDS.contains(kw);
            let base = if is_low {
                weights.keyword / LOW_SIGNAL_DIVISOR // 10
            } else {
                weights.keyword // 100
            };
            // First-match bonus (same as hook mode)
            score += if kw_count == 0 {
                base + weights.first_match / (if is_low { LOW_SIGNAL_DIVISOR } else { 1 })
            } else {
                base
            };
            if !is_low { has_non_low_signal = true; }
            kw_count += 1;
        }
    }

    // Activity name in entry name — strong signal (like whole_name_match in hook mode)
    let activity_name_words: Vec<&str> = activity.name.split('-').collect();
    let name_overlap = activity_name_words
        .iter()
        .filter(|aw| entry_name_parts.iter().any(|np| np == *aw || np.starts_with(*aw) || aw.starts_with(np)))
        .count();
    if name_overlap > 0 {
        // 2000 + 1000*(n-1): same scaling as whole_name_match in hook mode
        score += weights.tool_match + (name_overlap as i32 - 1) * 1000;
        has_non_low_signal = true;
    }

    // Same ALL_LOW_SIGNAL_CAP as hook mode: if only common words matched, cap at 90
    if !has_non_low_signal {
        score = score.min(ALL_LOW_SIGNAL_CAP);
    }

    score
}

/// Classify an entry into multiple activities using the reversed logarithmic scorer.
/// Returns top-N (activity_name, score) pairs, sorted descending by score.
/// Replaces the old assign_pass1_category() single-category assignment.
pub(crate) fn classify_entry_activities(
    name: &str,
    description: &str,
    use_context: &str,
) -> Vec<(String, i32)> {
    // Combine all text into words for matching
    let combined = format!("{} {} {}", name.to_lowercase(), description.to_lowercase(), use_context.to_lowercase());
    let entry_words: Vec<&str> = combined.split_whitespace().collect();
    let name_lower = name.to_lowercase();
    let name_parts: Vec<&str> = name_lower.split(|c: char| c == '-' || c == '_' || c == ':').collect();
    let weights = MatchWeights::default();

    // Score against every activity definition
    let mut scores: Vec<(String, i32)> = ACTIVITY_REGISTRY
        .iter()
        .map(|activity| {
            let score = score_entry_against_activity(&entry_words, &name_parts, activity, &weights);
            (activity.name.to_string(), score)
        })
        .filter(|(_, score)| *score > 0)
        .collect();

    // Sort descending by score, take top 5
    scores.sort_by(|a, b| b.1.cmp(&a.1));
    scores.truncate(5);

    // Fallback if no activity scored
    if scores.is_empty() {
        scores.push(("general-development".to_string(), 10));
    }

    scores
}

/// DEPRECATED: Old 16-category flat taxonomy. Kept for reference only.
/// Use classify_entry_activities() instead, which uses the same logarithmic
/// tier weights as the hook mode scorer for accurate classification.
#[allow(dead_code)]
/// Assign a category from the 16-category taxonomy based on keyword signals.
/// Priority-ordered: first match wins (same logic as the scoring engine).
pub(crate) fn assign_pass1_category(keywords: &[String]) -> &'static str {
    // Category signal table — priority order (highest priority first)
    let category_signals: &[(&str, &[&str])] = &[
        ("mobile", &["ios", "android", "swift", "swiftui", "flutter", "react-native", "kotlin", "xcode", "mobile", "app-store"]),
        ("plugin-dev", &["plugin", "hook", "skill", "mcp", "extension", "marketplace"]),
        ("security", &["security", "secur", "audit", "vulnerability", "owasp", "pentest", "encrypt", "auth", "jwt", "oauth"]),
        ("devops-cicd", &["docker", "kubernetes", "k8s", "ci", "cd", "pipeline", "deploy", "github-actions", "jenkins", "helm"]),
        ("infrastructure", &["terraform", "aws", "azure", "gcp", "cloud", "infrastructure", "iac", "serverless", "lambda"]),
        // debugging MUST be before testing: agents like "sleuth" have both "debug" and "test"
        // in their keywords (from use_context), and first-match-wins means debugging must win
        ("debugging", &["debug", "debugger", "profil", "trace", "log", "breakpoint", "inspect", "diagnos", "bug", "investigat", "root-cause"]),
        ("testing", &["test", "jest", "pytest", "cypress", "playwright", "e2e", "tdd", "spec", "assert", "coverage"]),
        ("data-ml", &["data", "ml", "machine-learning", "pandas", "numpy", "sklearn", "dataset", "featur", "model"]),
        ("ai-llm", &["llm", "ai", "gpt", "prompt", "rag", "langchain", "openai", "anthropic", "embedding", "vector"]),
        ("web-frontend", &["react", "vue", "angular", "css", "html", "frontend", "ui", "component", "tailwind", "nextjs", "svelte"]),
        ("web-backend", &["api", "rest", "graphql", "backend", "server", "express", "django", "rails", "fastapi", "endpoint"]),
        ("visualization", &["chart", "graph", "plot", "d3", "visual", "dashboard", "diagram"]),
        ("cli-tools", &["cli", "command", "terminal", "shell", "bash", "script", "automation"]),
        ("code-quality", &["lint", "format", "refactor", "review", "clean", "style", "prettier", "eslint", "ruff"]),
        ("research", &["research", "paper", "academic", "analysis", "study", "survey", "literature"]),
        ("project-mgmt", &["project", "management", "plan", "organiz", "roadmap", "sprint", "agile", "kanban"]),
    ];

    for (category, signals) in category_signals {
        for kw in keywords {
            for signal in *signals {
                // Prefix match: "secur" matches "security", "secure", etc.
                if kw.starts_with(signal) || signal.starts_with(kw.as_str()) {
                    return category;
                }
            }
        }
    }
    "cli-tools" // fallback for unclassifiable elements
}

/// Infer default intents from the primary activity name.
/// Handles both old 16-category names (backward compat) and new ~130 activity names.
/// Uses substring matching so parent activities cover their children:
/// "unit-testing" matches the "testing" arm, "container-deployment" matches "deployment".
pub(crate) fn infer_pass1_intents(activity: &str) -> Vec<String> {
    // Match against activity groups using contains() for hierarchical coverage
    let intents: Vec<&str> = if activity.contains("testing") || activity == "test-automation" {
        vec!["test", "validate", "verify", "assert"]
    } else if activity.contains("debugging") || activity == "root-cause-analysis"
        || activity == "memory-debugging" || activity == "log-analysis"
        || activity == "network-debugging"
    {
        vec!["debug", "diagnose", "trace", "inspect"]
    } else if activity.contains("deployment") || activity == "ci-cd"
        || activity == "kubernetes-ops" || activity == "serverless-deployment"
    {
        vec!["deploy", "release", "ship", "configure"]
    } else if activity == "linting" {
        vec!["lint", "check", "validate", "enforce"]
    } else if activity == "formatting" {
        vec!["format", "style", "prettify", "normalize"]
    } else if activity == "type-checking" {
        vec!["check", "validate", "annotate", "infer"]
    } else if activity == "refactoring" {
        vec!["refactor", "restructure", "simplify", "extract"]
    } else if activity == "code-review" {
        vec!["review", "approve", "comment", "suggest"]
    } else if activity == "profiling" {
        vec!["profile", "measure", "optimize", "benchmark"]
    } else if activity == "tracing" {
        vec!["trace", "correlate", "instrument", "observe"]
    } else if activity.contains("security") || activity == "penetration-testing"
        || activity == "dependency-scanning" || activity == "code-scanning"
    {
        vec!["audit", "secure", "scan", "review"]
    } else if activity == "authentication" {
        vec!["authenticate", "login", "authorize", "verify"]
    } else if activity == "authorization" {
        vec!["authorize", "permit", "restrict", "enforce"]
    } else if activity == "encryption" || activity == "secret-management" {
        vec!["encrypt", "protect", "rotate", "manage"]
    } else if activity.contains("mobile") || activity == "desktop-development" {
        vec!["develop", "build", "deploy", "test"]
    } else if activity == "plugin-development" {
        vec!["create", "extend", "configure", "develop"]
    } else if activity == "infrastructure-provisioning" {
        vec!["provision", "configure", "manage", "scale"]
    } else if activity.contains("data") || activity == "machine-learning"
        || activity == "deep-learning"
    {
        vec!["analyze", "train", "predict", "transform"]
    } else if activity == "nlp" || activity == "computer-vision" {
        vec!["process", "classify", "detect", "extract"]
    } else if activity == "llm-integration" {
        vec!["generate", "prompt", "embed", "retrieve"]
    } else if activity.contains("frontend") || activity == "ui-development" {
        vec!["build", "style", "render", "animate"]
    } else if activity.contains("backend") || activity == "api-development" {
        vec!["serve", "route", "query", "authenticate"]
    } else if activity == "data-visualization" {
        vec!["visualize", "chart", "plot", "display"]
    } else if activity == "cli-development" || activity == "automation" {
        vec!["run", "execute", "automate", "script"]
    } else if activity == "documentation" {
        vec!["document", "describe", "explain", "annotate"]
    } else if activity == "research" {
        vec!["research", "analyze", "synthesize", "cite"]
    } else if activity == "project-management" {
        vec!["plan", "track", "organize", "prioritize"]
    } else if activity == "git-workflow" {
        vec!["commit", "branch", "merge", "review"]
    } else if activity == "code-generation" {
        vec!["generate", "scaffold", "create", "template"]
    } else if activity == "monitoring" || activity == "logging" {
        vec!["monitor", "alert", "observe", "diagnose"]
    } else if activity == "package-management" || activity == "bundling" {
        vec!["install", "build", "bundle", "publish"]
    } else if activity == "release-management" {
        vec!["release", "version", "tag", "publish"]
    } else if activity == "configuration-management" {
        vec!["configure", "manage", "toggle", "provision"]
    } else if activity == "localization" {
        vec!["translate", "localize", "adapt", "internationalize"]
    } else if activity == "media-processing" || activity == "pdf-processing" {
        vec!["process", "convert", "transform", "generate"]
    } else if activity == "seo" || activity == "content-creation" {
        vec!["write", "optimize", "publish", "promote"]
    } else if activity == "accessibility-audit" {
        vec!["audit", "check", "remediate", "comply"]
    } else if activity == "system-design" || activity == "api-design"
        || activity == "database-design" || activity == "design-patterns"
    {
        vec!["design", "architect", "model", "diagram"]
    } else if activity == "migration-planning" {
        vec!["migrate", "upgrade", "port", "modernize"]
    } else if activity == "web-development" || activity == "database-development" {
        vec!["develop", "build", "configure", "query"]
    } else {
        // Fallback for any unhandled activity name
        vec!["develop", "implement", "configure"]
    };

    intents.into_iter().map(String::from).collect()
}

/// Extract programming languages mentioned in keywords.
/// Returns a list of recognized language names.
pub(crate) fn extract_pass1_languages(keywords: &[String]) -> Vec<String> {
    // Language mapping sourced from LOC Subject Headings "Computer program language"
    // classification (467 entries), filtered to modern/relevant languages.
    // Source: https://id.loc.gov/authorities/subjects (2026-03-19)
    let lang_map: &[(&str, &str)] = &[
        // Tier 1: Major modern languages (LOC + industry)
        ("python", "python"), ("py", "python"),
        ("rust", "rust"),
        ("go", "go"), ("golang", "go"),
        ("java", "java"),
        ("javascript", "javascript"), ("js", "javascript"),
        ("typescript", "typescript"), ("ts", "typescript"),
        ("ruby", "ruby"), ("rb", "ruby"),
        ("swift", "swift"),
        ("kotlin", "kotlin"),
        ("dart", "dart"), ("flutter", "dart"),
        ("c++", "c++"), ("cpp", "c++"),
        ("csharp", "c#"), ("c#", "c#"),
        ("php", "php"),
        ("scala", "scala"),
        ("elixir", "elixir"),
        ("lua", "lua"),
        ("sql", "sql"),
        ("html", "html"),
        ("css", "css"),
        ("shell", "shell"), ("bash", "shell"),
        ("haskell", "haskell"),
        ("perl", "perl"),
        ("r", "r"),
        // Tier 2: LOC-sourced languages with active communities
        ("julia", "julia"),
        ("groovy", "groovy"),
        ("objective-c", "objective-c"), ("objc", "objective-c"),
        ("clojure", "clojure"),
        ("erlang", "erlang"),
        ("ocaml", "ocaml"),
        ("fortran", "fortran"),
        ("cobol", "cobol"),
        ("pascal", "pascal"),
        ("prolog", "prolog"),
        ("lisp", "lisp"),
        ("scheme", "scheme"),
        ("racket", "racket"),
        ("smalltalk", "smalltalk"),
        ("tcl", "tcl"),
        ("ada", "ada"),
        ("abap", "abap"),
        // Tier 3: LOC-sourced niche/emerging languages
        ("coffeescript", "coffeescript"),
        ("elm", "elm"),
        ("purescript", "purescript"),
        ("nim", "nim"),
        ("zig", "zig"),
        ("crystal", "crystal"),
        ("solidity", "solidity"),
        ("cuda", "cuda"),
        ("opencl", "opencl"),
        ("wasm", "webassembly"), ("webassembly", "webassembly"),
        ("powershell", "powershell"),
        ("vhdl", "vhdl"),
        ("verilog", "verilog"), ("systemverilog", "verilog"),
        ("matlab", "matlab"),
        ("visual-basic", "visual-basic"), ("vb", "visual-basic"), ("vba", "visual-basic"),
        ("awk", "awk"),
        ("sed", "sed"),
    ];

    let mut langs: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for kw in keywords {
        for (pattern, lang) in lang_map {
            if kw == pattern && seen.insert(lang.to_string()) {
                langs.push(lang.to_string());
            }
        }
    }
    langs
}

/// Extract frameworks mentioned in keywords.
pub(crate) fn extract_pass1_frameworks(keywords: &[String]) -> Vec<String> {
    // Tier 4 (10K-90K): Comprehensive development frameworks and platforms
    let fw_set: &[&str] = &[
        // JS/TS frontend frameworks
        "react", "vue", "angular", "svelte", "nextjs", "next", "nuxt", "express",
        "remix", "gatsby", "astro", "sveltekit", "solidjs", "qwik", "hono", "elysia",
        "fresh", "ember", "backbone", "preact", "htmx", "alpine",
        // JS/TS backend frameworks
        "koa", "fastify", "nest", "nestjs", "graphql", "trpc",
        // JS/TS runtimes & bundlers
        "bun", "deno", "vite", "turbopack", "rspack", "esbuild", "rollup", "webpack", "parcel",
        // Python frameworks
        "django", "flask", "fastapi", "starlette", "tornado", "sanic", "litestar",
        // Ruby frameworks
        "rails", "sinatra",
        // Java/JVM frameworks
        "spring", "spring-boot", "quarkus", "micronaut", "graalvm",
        // PHP frameworks
        "laravel", "symfony",
        // Elixir/Erlang frameworks
        "phoenix",
        // Go frameworks
        "gin", "echo", "fiber",
        // Rust frameworks
        "axum", "actix", "rocket",
        // CSS/UI frameworks & component libraries
        "tailwind", "bootstrap", "shadcn", "daisyui", "bulma", "chakra-ui", "mantine",
        "radix", "headless-ui", "mui", "material-ui", "ant-design",
        // Desktop/native frameworks
        "electron", "tauri", "maui", "avalonia", "qt",
        // Game engines & frameworks
        "unity", "unreal", "godot", "bevy", "phaser", "raylib", "love2d", "pygame",
        "pixi", "babylonjs", "threejs",
        // Graphics & rendering
        "skia", "skia-sharp", "skia-graphite", "graphite", "vulkan", "metal", "opengl",
        "directx", "webgpu", "wgpu", "sdl2", "sdl3", "dawn",
        // ML/AI frameworks
        "tensorflow", "pytorch", "keras", "jax", "scikit-learn", "sklearn",
        "huggingface", "transformers", "fastai", "onnx", "mlflow",
        "langchain", "llamaindex", "wandb", "ray", "dask",
        // AI image generation
        "stable-diffusion", "diffusers", "comfyui",
        // Mobile frameworks
        "flutter", "swiftui", "jetpack", "compose", "xamarin",
        "react-native", "kotlin-multiplatform", "ionic", "capacitor", "expo",
        // ORM/DB frameworks
        "prisma", "drizzle", "typeorm", "sequelize", "sqlalchemy", "diesel",
        // Testing frameworks
        "playwright", "cypress", "jest", "pytest", "vitest", "mocha",
        "selenium", "rspec", "junit", "testng", "jasmine",
        // Data engineering & orchestration frameworks
        "kafka", "spark", "flink", "airflow", "dbt", "snowflake",
        "delta-lake", "iceberg", "polars",
        "prefect", "temporal", "dagster", "kestra", "zenml", "kubeflow", "mage",
        "windmill",
        // Task queue & message broker frameworks
        "celery", "dramatiq", "taskiq", "rabbitmq", "redis",
        // DevOps/infra frameworks
        "docker", "kubernetes", "terraform", "pulumi", "ansible",
        "cloudformation", "helm", "argocd",
        // WebAssembly
        "wasm", "webassembly", "wasmtime",
        // Meta-frameworks & tools
        "wasp", "storybook",
    ];

    keywords
        .iter()
        .filter(|kw| fw_set.contains(&kw.as_str()))
        .cloned()
        .collect()
}

/// Extract platform signals from keywords.
pub(crate) fn extract_pass1_platforms(keywords: &[String]) -> Vec<String> {
    // Platform mapping: modern deployment targets.
    // LOC historical platforms (Atari, Commodore, IBM 360) excluded — not relevant.
    let plat_map: &[(&str, &str)] = &[
        // Operating systems / mobile
        ("ios", "ios"), ("iphone", "ios"), ("ipad", "ios"),
        ("android", "android"),
        ("macos", "macos"), ("mac", "macos"), ("darwin", "macos"),
        ("windows", "windows"), ("win32", "windows"), ("win64", "windows"),
        ("linux", "linux"), ("ubuntu", "linux"), ("debian", "linux"),
        // Deployment targets
        ("web", "web"), ("browser", "web"),
        ("mobile", "mobile"),
        ("desktop", "desktop"), ("electron", "desktop"),
        ("embedded", "embedded"), ("iot", "embedded"), ("arduino", "embedded"),
        ("raspberry", "embedded"),
        // Cloud platforms
        ("cloud", "cloud"),
        ("aws", "aws"), ("lambda", "aws"),
        ("azure", "azure"),
        ("gcp", "gcp"), ("firebase", "gcp"),
        ("docker", "docker"), ("kubernetes", "docker"),
        ("wasm", "wasm"), ("webassembly", "wasm"),
        ("serverless", "serverless"), ("edge", "edge"),
        // LOC-sourced: FPGA/hardware
        ("fpga", "fpga"), ("vhdl", "fpga"), ("verilog", "fpga"),
    ];

    let mut platforms: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for kw in keywords {
        for (pattern, platform) in plat_map {
            if kw == pattern && seen.insert(platform.to_string()) {
                platforms.push(platform.to_string());
            }
        }
    }
    platforms
}

/// Extract tool signals from keywords (Tier 3: 1K-9K).
/// Tools are specific named software: build tools, package managers, runtimes,
/// linters, CLI utilities, database clients, editors.
pub(crate) fn extract_pass1_tools(keywords: &[String]) -> Vec<String> {
    let tool_set: &[&str] = &[
        // Build tools
        "webpack", "vite", "esbuild", "rollup", "parcel", "turbopack", "swc",
        "tsup", "unbuild", "gulp", "grunt", "make", "cmake", "bazel", "meson",
        "rspack", "nx", "turborepo", "lerna",
        // Package managers
        "npm", "yarn", "pnpm", "pip", "cargo", "brew", "homebrew", "cocoapods",
        "maven", "gradle", "nuget", "composer", "uv", "rye", "poetry", "pipenv",
        // Runtimes
        "bun", "deno", "node", "nodejs",
        // DevOps tools
        "docker", "terraform", "ansible", "vagrant", "helm", "pulumi",
        "nginx", "caddy", "traefik", "argocd", "kustomize", "istio",
        "envoy", "consul", "vault",
        // Testing tools
        "jest", "vitest", "mocha", "karma", "selenium", "puppeteer",
        "playwright", "cypress", "pytest", "junit", "rspec", "testng",
        // Linters & formatters
        "eslint", "prettier", "ruff", "black", "mypy", "pyright", "clippy",
        "rubocop", "stylelint", "biome", "oxlint", "tslint", "shellcheck",
        "flake8", "pylint", "isort",
        // CLI tools
        "git", "gh", "curl", "wget", "ffmpeg", "imagemagick", "pandoc",
        "jq", "ripgrep", "fd", "fzf", "tmux", "wrangler",
        // Database tools
        "redis", "sqlite", "postgres", "postgresql", "mysql", "mongodb",
        "dynamodb", "couchdb", "cassandra", "supabase", "neon", "planetscale",
        // Data & orchestration tools
        "kafka", "spark", "flink", "airflow", "dbt", "snowflake",
        "bigquery", "duckdb", "polars", "pandas", "clickhouse",
        "prefect", "temporal", "dagster", "kestra", "zenml", "kubeflow", "mage",
        "windmill",
        // Task queue & message broker tools
        "celery", "dramatiq", "taskiq", "rabbitmq",
        // Graphics & rendering tools
        "skia", "skia-sharp", "skia-graphite", "graphite", "vulkan", "opengl", "metal",
        "webgpu", "wgpu", "sdl2", "sdl3", "dawn", "directx",
        // AI/ML tools
        "mlflow", "wandb", "onnx", "tensorrt", "triton", "comfyui",
        // Editors/IDEs
        "vim", "neovim", "vscode", "emacs", "xcode",
        // Container/orchestration tools
        "kubernetes", "k8s", "minikube", "skaffold", "podman",
    ];

    keywords
        .iter()
        .filter(|kw| tool_set.contains(&kw.as_str()))
        .cloned()
        .collect()
}

/// Extract service/API signals from keywords (Tier 5: 100K-900K).
/// Services are external platforms, cloud providers, SaaS APIs, and hosted services.
pub(crate) fn extract_pass1_services(keywords: &[String]) -> Vec<String> {
    let svc_set: &[&str] = &[
        // Cloud providers
        "aws", "azure", "gcp", "digitalocean", "heroku", "vercel", "netlify",
        "cloudflare", "fly", "railway", "render",
        // AI/ML APIs
        "openai", "anthropic", "claude", "gemini", "huggingface", "replicate",
        "ollama", "cohere", "mistral", "groq",
        // Dev platforms
        "github", "gitlab", "bitbucket", "jira", "confluence", "linear", "notion",
        // Data services (BaaS)
        "supabase", "firebase", "planetscale", "neon", "upstash", "convex",
        "appwrite", "pocketbase",
        // Auth services
        "auth0", "clerk", "okta", "keycloak", "cognito",
        // Communication services
        "slack", "discord", "twilio", "sendgrid", "resend", "postmark",
        // Payment services
        "stripe", "paypal", "square", "braintree",
        // Monitoring services
        "datadog", "sentry", "grafana", "prometheus", "newrelic",
        // CDN/Storage services
        "cloudinary", "s3", "r2", "minio", "uploadthing",
        // Search services
        "elasticsearch", "algolia", "meilisearch", "typesense",
        // CMS services
        "sanity", "contentful", "strapi", "wordpress",
        // CI/CD services
        "circleci", "travisci", "jenkins",
        // Managed orchestration services
        "composer", "mwaa", "sagemaker",
    ];

    keywords
        .iter()
        .filter(|kw| svc_set.contains(&kw.as_str()))
        .cloned()
        .collect()
}

/// Detect negation patterns in text and split words into positive/negative buckets.
/// Returns (positive_words, negative_words) extracted from the text.
///
/// Negation patterns recognized:
/// - "not compatible with X", "incompatible with X"
/// - "does not work with X", "doesn't work with X"
/// - "but not X", "except X", "excluding X"
/// - "instead of X", "not for X", "not intended for X"
/// - "should not be used for X", "do not use for X"
pub(crate) fn extract_negation_keywords(text: &str) -> (Vec<String>, Vec<String>) {
    let mut positive: Vec<String> = Vec::new();
    let mut negative: Vec<String> = Vec::new();

    if text.is_empty() {
        return (positive, negative);
    }

    // Split text into rough sentences (by period, newline, or bullet point)
    let sentences: Vec<&str> = text
        .split(|c: char| c == '.' || c == '\n' || c == ';')
        .collect();

    // Negation markers — when found, subsequent content words go to negative_keywords
    let negation_markers: &[&str] = &[
        "not compatible",
        "incompatible",
        "does not work",
        "doesn't work",
        "do not work",
        "not work with",
        "but not",
        "except for",
        "except",
        "excluding",
        "instead of",
        "not for",
        "not intended",
        "not designed",
        "not suitable",
        "should not",
        "do not use",
        "don't use",
        "not recommended",
        "won't work",
        "will not work",
        "cannot be used",
        "can't be used",
        "not supported",
        "unsupported",
    ];

    for sentence in &sentences {
        let lower = sentence.to_lowercase();
        let trimmed = lower.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Check if this sentence contains a negation marker
        let has_negation = negation_markers.iter().any(|marker| trimmed.contains(marker));

        // Extract content words from this sentence
        for word in trimmed.split_whitespace() {
            let clean: String = word
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .collect();
            if clean.len() <= 2 || is_pass1_stopword(&clean) {
                continue;
            }
            // Skip the negation marker words themselves
            if matches!(
                clean.as_str(),
                "not" | "does" | "doesn" | "don" | "won" | "cannot" | "can"
                    | "should" | "recommended" | "compatible" | "incompatible"
                    | "intended" | "designed" | "suitable" | "supported" | "unsupported"
                    | "work" | "works" | "working" | "except" | "excluding" | "instead"
            ) {
                continue;
            }

            if has_negation {
                negative.push(clean);
            } else {
                positive.push(clean);
            }
        }
    }

    (positive, negative)
}

/// Run Pass 1 batch enrichment: read JSONL from stdin, enrich each element,
/// output enriched JSONL to stdout. Zero LLM calls — pure deterministic.
pub(crate) fn run_pass1_batch() -> Result<(), SuggesterError> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    let mut count: usize = 0;
    let mut errors: usize = 0;

    for line_result in stdin.lock().lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Warning: read error: {}", e);
                errors += 1;
                continue;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Parse input JSONL line
        let input: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Warning: invalid JSON: {}", e);
                errors += 1;
                continue;
            }
        };

        let name = input["name"].as_str().unwrap_or("");
        let elem_type = input["type"].as_str().unwrap_or("skill");
        let source = input["source"].as_str().unwrap_or("");
        let path = input["path"].as_str().unwrap_or("");
        let description = input["description"].as_str().unwrap_or("");
        // use_context: "when to use" / "use cases" section from the body (if present)
        let use_context = input["use_context"].as_str().unwrap_or("");
        // AgentSkills open standard metadata (agentskills.io spec)
        let agentskills_meta = &input["agentskills_metadata"];
        let compatibility = input["compatibility"].as_str().unwrap_or("");

        if name.is_empty() {
            eprintln!("Warning: skipping element with empty name");
            errors += 1;
            continue;
        }

        // Agents are indexed from their FULL definition, not just the fields
        // the discoverer forwards. An agent's frontmatter is where it declares
        // the skills the harness preloads for it and the tools it may use, and
        // those are precisely the words a user's prompt contains — an agent
        // that preloads `playwright` must be reachable from "browser test"
        // even though its own prose may never say it. Skills are unaffected:
        // their SKILL.md body already arrives summarized in `use_context`.
        //
        // Reading the file here (rather than widening the Python discoverer)
        // keeps one parser for agent definitions, in the binary that already
        // owns enrichment. A file that vanished between discovery and
        // enrichment simply yields no extra terms — it must not abort the run.
        let agent_meta = if elem_type == "agent" && !path.is_empty() {
            fs::read_to_string(path)
                .ok()
                .map(|content| agent_meta::parse_agent_definition(&content))
        } else {
            None
        };
        let agent_terms = agent_meta
            .as_ref()
            .map(|meta| meta.search_terms())
            .unwrap_or_default();

        // Combine description + use_context for richer keyword extraction
        let mut combined_text = if use_context.is_empty() {
            description.to_string()
        } else {
            format!("{} {}", description, use_context)
        };
        if !agent_terms.is_empty() {
            combined_text.push(' ');
            combined_text.push_str(&agent_terms.join(" "));
        }

        // Deterministic enrichment from name + description + use_context
        let keywords = generate_pass1_keywords(name, &combined_text);
        // Activity classification using reversed logarithmic scorer (same tiers as hook mode)
        let activities = classify_entry_activities(name, description, use_context);
        // Primary category = top-scoring activity (backward compat for existing consumers)
        let category = activities.first().map(|(a, _)| a.as_str()).unwrap_or("general-development");
        let intents = infer_pass1_intents(category);
        let mut languages = extract_pass1_languages(&keywords);
        let mut frameworks = extract_pass1_frameworks(&keywords);
        let mut platforms = extract_pass1_platforms(&keywords);
        let tools = extract_pass1_tools(&keywords);
        let services = extract_pass1_services(&keywords);

        // Negation-aware extraction from description + use_context
        // Words near negation markers ("not compatible with X") go to negative_keywords
        let (extra_positive, negative_keywords) = extract_negation_keywords(&combined_text);

        // Merge extra positive words into keywords (dedup via seen set)
        let mut all_keywords = keywords;
        let kw_set: std::collections::HashSet<String> =
            all_keywords.iter().cloned().collect();
        for w in extra_positive {
            let stemmed = stem_word(&w);
            if !kw_set.contains(&w) && !is_pass1_stopword(&w) && w.len() > 2 {
                all_keywords.push(w);
            }
            if !kw_set.contains(&stemmed) && !is_pass1_stopword(&stemmed) && stemmed.len() > 2 {
                all_keywords.push(stemmed);
            }
        }
        all_keywords.truncate(20); // Allow slightly more with use_context

        // Dedup negative keywords
        let neg_set: std::collections::HashSet<String> =
            negative_keywords.into_iter().collect();
        let negative_kw: Vec<String> = neg_set.iter().cloned().collect();

        // Remove negative keywords AND their stems from positive keywords
        // e.g., "not compatible with Google" removes both "google" and "googl"
        if !neg_set.is_empty() {
            let neg_stems: std::collections::HashSet<String> =
                neg_set.iter().map(|w| stem_word(w)).collect();
            all_keywords.retain(|kw| !neg_set.contains(kw) && !neg_stems.contains(kw));
        }

        // Extract use_cases from use_context bullet points (lines starting with - or *)
        let use_cases: Vec<String> = if use_context.is_empty() {
            vec![]
        } else {
            use_context
                .lines()
                .filter(|line| {
                    let t = line.trim();
                    t.starts_with("- ") || t.starts_with("* ") || t.starts_with("• ")
                })
                .map(|line| {
                    line.trim()
                        .trim_start_matches(|c: char| c == '-' || c == '*' || c == '•')
                        .trim()
                        .to_string()
                })
                .filter(|s| !s.is_empty())
                .take(5) // Cap at 5 use cases
                .collect()
        };

        // Determine tier from source: marketplace → community, user/project → built-in
        let tier = if source.starts_with("marketplace:") {
            "community"
        } else if source == "project" || source.starts_with("project:") {
            "project"
        } else {
            "built-in"
        };

        // Generate domain_gates from extracted languages, frameworks, platforms, and category.
        // Domain gates are boolean pre-filters: a skill with a gate only matches prompts
        // where the gate's domain is detected. Without gates, the skill matches any prompt.
        let mut domain_gates: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();

        // Gate: programming_language — skills locked to specific languages
        if !languages.is_empty() {
            domain_gates.insert(
                "programming_language".to_string(),
                serde_json::json!(languages),
            );
        }

        // Gate: target_platform — skills locked to specific platforms
        if !platforms.is_empty() {
            domain_gates.insert(
                "target_platform".to_string(),
                serde_json::json!(platforms),
            );
        }

        // Gate: framework — skills locked to specific frameworks
        if !frameworks.is_empty() {
            domain_gates.insert(
                "framework".to_string(),
                serde_json::json!(frameworks),
            );
        }

        // AgentSkills metadata: if present, use as authoritative domain signals
        // (more reliable than text inference — skill author explicitly declared these)
        if agentskills_meta.is_object() {
            let meta = agentskills_meta.as_object().unwrap();
            // language/framework/platform from metadata → add to extracted lists + gates
            for (meta_key, gate_name, target) in [
                ("language", "programming_language", &mut languages),
                ("framework", "framework", &mut frameworks),
                ("platform", "target_platform", &mut platforms),
            ] {
                if let Some(val) = meta.get(meta_key).and_then(|v| v.as_str()) {
                    let lower = val.to_lowercase();
                    if !target.contains(&lower) {
                        target.push(lower.clone());
                    }
                    // Ensure corresponding domain gate exists
                    if !domain_gates.contains_key(gate_name) {
                        domain_gates.insert(
                            gate_name.to_string(),
                            serde_json::json!([lower]),
                        );
                    }
                }
            }
            // tags from metadata → add as extra keywords
            if let Some(tags) = meta.get("tags").and_then(|v| v.as_str()) {
                for tag in tags.split(',').map(|t| t.trim().to_lowercase()) {
                    if !tag.is_empty() && tag.len() > 2 && !all_keywords.contains(&tag) {
                        all_keywords.push(tag);
                    }
                }
            }
        }
        // compatibility field → add tool/platform keywords
        if !compatibility.is_empty() {
            for word in compatibility.split_whitespace() {
                let lower = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                if lower.len() > 2 && !all_keywords.contains(&lower) && !is_pass1_stopword(&lower) {
                    all_keywords.push(lower);
                }
            }
        }

        // Infer domains from name + description using the shared synonym taxonomy
        let domains = infer_domains_from_text(&format!("{} {} {}", name, description, use_context));

        // Extract rule path_gates from frontmatter `paths:` field (rules only).
        // The caller passes the source file path — we re-read it and parse frontmatter
        // to extract the `paths:` glob list. Negligible cost because rule count is small (<20).
        let path_gates: Vec<String> = if elem_type == "rule" && !path.is_empty() {
            fs::read_to_string(path)
                .ok()
                .map(|content| {
                    let fm = crate::parse_frontmatter(&content);
                    crate::extract_rule_paths(&fm)
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Serialized OUTSIDE the json! literal below: that block is already
        // large, and nesting another json! inside it exceeds serde_json's
        // macro recursion limit (raising it crate-wide to work around one call
        // site would be the wrong lever).
        let agent_metadata_json = match agent_meta.as_ref() {
            Some(meta) => serde_json::to_value(meta).unwrap_or(serde_json::Value::Null),
            None => serde_json::Value::Null,
        };

        // Build enriched output object
        let mut output = serde_json::json!({
            "name": name,
            "type": elem_type,
            "source": source,
            "path": path,
            "description": description,
            "keywords": all_keywords,
            "negative_keywords": negative_kw,
            "category": category,
            "activities": activities.iter().map(|(name, score)| serde_json::json!({"name": name, "score": score})).collect::<Vec<_>>(),
            "intents": intents,
            "tier": tier,
            "boost": 0,
            "platforms": platforms,
            "frameworks": frameworks,
            "languages": languages,
            "domains": domains,
            "tools": tools,
            "services": services,
            "file_types": [],
            "patterns": [],
            "directories": [],
            "path_patterns": [],
            "use_cases": use_cases,
            "secondary_categories": [],
            "domain_gates": domain_gates,
            "path_gates": path_gates,
            // Agent-only block (null for every other element type) — the
            // declared capability surface, kept structured so downstream
            // consumers can filter on it instead of re-parsing the .md.
            "agent_metadata": agent_metadata_json,
            // A `disable-model-invocation: true` skill is user-only: the model
            // cannot invoke it, so suggesting it wastes a slot on something
            // Claude is not allowed to call. Carried through enrichment as a
            // named field so the hook can drop these BEFORE scoring — see the
            // rebuild note directly below, which is why it must be copied here.
            "disable_model_invocation": input
                .get("disable_model_invocation")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        });

        // Ownership for the `<plugin>:<name>` suggestion namespace. Enrichment
        // REBUILDS the record from named fields instead of mutating the input,
        // so anything not copied here is dropped between discover and the
        // merge writer — the DB column then exists and is empty for every row.
        // Only inserted when present, so a standalone element stays absent
        // (the merge side reads absent as "") rather than gaining a null owner.
        if let Some(v) = input.get("plugin").and_then(|v| v.as_str()) {
            if !v.is_empty() {
                output["plugin"] = serde_json::Value::String(v.to_string());
            }
        }
        if let Some(v) = input.get("origin").and_then(|v| v.as_str()) {
            if !v.is_empty() {
                output["origin"] = serde_json::Value::String(v.to_string());
            }
        }

        // Write enriched JSONL line to stdout
        if let Err(e) = writeln!(out, "{}", serde_json::to_string(&output).unwrap_or_default()) {
            // Broken pipe (downstream consumer closed) — exit cleanly
            if e.kind() == io::ErrorKind::BrokenPipe {
                break;
            }
            return Err(SuggesterError::StdinRead(e));
        }
        count += 1;
    }

    eprintln!("Pass1 batch: enriched {} elements ({} errors)", count, errors);
    Ok(())
}
