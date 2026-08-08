//! Static data tables extracted out of `main.rs` (W1-STEP1 — pure move, zero
//! behavior change). Every item below kept its original content byte-for-byte
//! and only had its visibility widened to `pub(crate)` so it stays reachable
//! from `main.rs` via the crate-root re-export:
//!
//! ```ignore
//! mod data;
//! pub(crate) use data::*;
//! ```
//!
//! That glob re-export puts every item back into the crate-root namespace, so
//! every existing consumer (`correct_typos`, `is_abbreviation_match`,
//! `decompose_tasks`, `find_matches`, `expand_synonyms`,
//! `infer_domains_from_text`, `score_entry_against_activity`,
//! `classify_entry_activities`, `mod tests`, and `temporal.rs`) keeps
//! compiling unedited. See `reports/pss-improve-mainrs-plan/` for the full
//! modularization plan this step implements.

use lazy_static::lazy_static;
use regex::Regex;
use std::collections::{HashMap, HashSet};

// ============================================================================
// Typo Tolerance (from Claude-Rio patterns)
// ============================================================================

lazy_static! {
    /// Common typos and their corrections (from Claude-Rio typo-tolerant pattern)
    pub(crate) static ref TYPO_CORRECTIONS: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();

        // Common programming language typos
        m.insert("typscript", "typescript");
        m.insert("typescrpt", "typescript");
        m.insert("tyepscript", "typescript");
        m.insert("javscript", "javascript");
        m.insert("javascipt", "javascript");
        m.insert("javasript", "javascript");
        m.insert("pyhton", "python");
        m.insert("pythn", "python");
        m.insert("ptyhon", "python");
        m.insert("rusr", "rust");
        m.insert("ruts", "rust");

        // DevOps/Cloud typos
        m.insert("kuberntes", "kubernetes");
        m.insert("kuberentes", "kubernetes");
        m.insert("kubenretes", "kubernetes");
        m.insert("k8", "kubernetes");
        m.insert("dokcer", "docker");
        m.insert("dcoker", "docker");
        m.insert("doker", "docker");
        m.insert("tf", "terraform");
        m.insert("k8s", "kubernetes");
        m.insert("pg", "postgres");
        m.insert("mongo", "mongodb");
        m.insert("gh", "github");
        m.insert("containr", "container");
        m.insert("contaner", "container");

        // Git/GitHub typos
        m.insert("githb", "github");
        m.insert("gihub", "github");
        m.insert("gihtub", "github");
        m.insert("gtihub", "github");
        m.insert("comit", "commit");
        m.insert("commti", "commit");
        m.insert("brach", "branch");
        m.insert("brnach", "branch");
        m.insert("mege", "merge");
        m.insert("mreged", "merged");
        m.insert("rebas", "rebase");

        // CI/CD typos
        m.insert("pipline", "pipeline");
        m.insert("pipleine", "pipeline");
        m.insert("dpeloy", "deploy");
        m.insert("deplyo", "deploy");
        m.insert("dploy", "deploy");
        m.insert("realease", "release");
        m.insert("relase", "release");

        // Testing typos
        m.insert("tset", "test");
        m.insert("tets", "test");
        m.insert("tesst", "test");
        m.insert("uint", "unit");
        m.insert("intgration", "integration");
        m.insert("integartion", "integration");

        // Database typos
        m.insert("databse", "database");
        m.insert("databsae", "database");
        m.insert("postgrse", "postgres");
        m.insert("postgrs", "postgres");
        m.insert("sqll", "sql");
        m.insert("qurey", "query");
        m.insert("qeury", "query");

        // API typos
        m.insert("endpont", "endpoint");
        m.insert("endpiont", "endpoint");
        m.insert("reuqest", "request");
        m.insert("reqeust", "request");
        m.insert("repsone", "response");
        m.insert("respone", "response");

        // General coding typos
        m.insert("funciton", "function");
        m.insert("fucntion", "function");
        m.insert("functoin", "function");
        m.insert("calss", "class");
        m.insert("clas", "class");
        m.insert("metohd", "method");
        m.insert("mehod", "method");
        m.insert("varaible", "variable");
        m.insert("variabel", "variable");
        m.insert("improt", "import");
        m.insert("imoprt", "import");
        m.insert("exprot", "export");
        m.insert("exoprt", "export");

        // Framework typos
        m.insert("raect", "react");
        m.insert("reat", "react");
        m.insert("angualr", "angular");
        m.insert("agular", "angular");
        m.insert("nextjs", "next.js");
        m.insert("nodjes", "nodejs");
        m.insert("noed", "node");

        // Error/Debug typos
        m.insert("erorr", "error");
        m.insert("eroor", "error");
        m.insert("errro", "error");
        m.insert("dbug", "debug");
        m.insert("deubg", "debug");
        m.insert("bgu", "bug");
        m.insert("fixe", "fix");

        // Config typos
        m.insert("cofig", "config");
        m.insert("confg", "config");
        m.insert("configuation", "configuration");
        m.insert("configuartion", "configuration");
        m.insert("settigns", "settings");
        m.insert("setings", "settings");

        // Cloud provider typos
        m.insert("awss", "aws");
        m.insert("s3s", "s3");
        m.insert("gpc", "gcp");
        m.insert("azrue", "azure");
        m.insert("azuer", "azure");

        // MCP/Claude typos
        m.insert("mpc", "mcp");
        m.insert("cladue", "claude");
        m.insert("cluade", "claude");
        m.insert("antropic", "anthropic");
        m.insert("antrhoic", "anthropic");

        m
    };
}

// ============================================================================
// Abbreviation table (short form <-> long form, Phase 2.5 matching)
// ============================================================================

/// Common tech abbreviation pairs (short form → long form).
/// Used in Phase 2.5 to match abbreviations against their full forms.
/// Both directions are checked: "config" matches "configuration" and vice versa.
pub(crate) const ABBREVIATIONS: &[(&str, &str)] = &[
    ("config", "configuration"),
    ("repo", "repository"),
    ("env", "environment"),
    ("auth", "authentication"),
    ("authn", "authentication"),
    ("authz", "authorization"),
    ("admin", "administration"),
    ("app", "application"),
    ("args", "arguments"),
    ("async", "asynchronous"),
    ("auto", "automatic"),
    ("bg", "background"),
    ("bin", "binary"),
    ("bool", "boolean"),
    ("calc", "calculate"),
    ("cert", "certificate"),
    ("cfg", "configuration"),
    ("char", "character"),
    ("cmd", "command"),
    ("cmp", "compare"),
    ("concat", "concatenate"),
    ("cond", "condition"),
    ("conn", "connection"),
    ("const", "constant"),
    ("ctrl", "control"),
    ("ctx", "context"),
    ("db", "database"),
    ("decl", "declaration"),
    ("def", "definition"),
    ("del", "delete"),
    ("dep", "dependency"),
    ("deps", "dependencies"),
    ("desc", "description"),
    ("dest", "destination"),
    ("dev", "development"),
    ("dict", "dictionary"),
    ("diff", "difference"),
    ("dir", "directory"),
    ("dirs", "directories"),
    ("dist", "distribution"),
    ("doc", "documentation"),
    ("docs", "documentation"),
    ("elem", "element"),
    ("err", "error"),
    ("eval", "evaluate"),
    ("exec", "execute"),
    ("expr", "expression"),
    ("ext", "extension"),
    ("fmt", "format"),
    ("fn", "function"),
    ("func", "function"),
    ("gen", "generate"),
    ("hw", "hardware"),
    ("impl", "implementation"),
    ("import", "import"),
    ("info", "information"),
    ("init", "initialize"),
    ("iter", "iterator"),
    ("lang", "language"),
    ("len", "length"),
    ("lib", "library"),
    ("libs", "libraries"),
    ("ln", "link"),
    ("loc", "location"),
    ("max", "maximum"),
    ("mem", "memory"),
    ("mgmt", "management"),
    ("min", "minimum"),
    ("misc", "miscellaneous"),
    ("mod", "module"),
    ("msg", "message"),
    ("nav", "navigation"),
    ("num", "number"),
    ("obj", "object"),
    ("ops", "operations"),
    ("opt", "option"),
    ("org", "organization"),
    ("os", "operating_system"),
    ("param", "parameter"),
    ("params", "parameters"),
    ("perf", "performance"),
    ("pkg", "package"),
    ("pref", "preference"),
    ("prev", "previous"),
    ("proc", "process"),
    ("prod", "production"),
    ("prog", "program"),
    ("prop", "property"),
    ("props", "properties"),
    ("proto", "protocol"),
    ("pub", "public"),
    ("qty", "quantity"),
    ("recv", "receive"),
    ("ref", "reference"),
    ("regex", "regular_expression"),
    ("req", "request"),
    ("res", "response"),
    ("ret", "return"),
    ("rm", "remove"),
    ("sec", "security"),
    ("sel", "select"),
    ("sep", "separator"),
    ("seq", "sequence"),
    ("sig", "signature"),
    ("spec", "specification"),
    ("specs", "specifications"),
    ("src", "source"),
    ("srv", "server"),
    ("str", "string"),
    ("struct", "structure"),
    ("sub", "subscribe"),
    ("svc", "service"),
    ("sw", "software"),
    ("sync", "synchronize"),
    ("sys", "system"),
    ("temp", "temporary"),
    ("tmp", "temporary"),
    ("val", "value"),
    ("var", "variable"),
    ("vars", "variables"),
    ("ver", "version"),
];

// ============================================================================
// Task Decomposition data (from LimorAI - break complex prompts into sub-tasks)
// ============================================================================

lazy_static! {
    /// Patterns for task decomposition - detect multi-task prompts
    /// NOTE: We handle sentence-based decomposition separately (not via regex)
    /// because Rust's regex crate doesn't support lookahead assertions
    pub(crate) static ref TASK_SEPARATORS: Vec<Regex> = vec![
        // "X and then Y" - sequential tasks
        Regex::new(r"(?i)\s+and\s+then\s+").unwrap(),
        // "X then Y" - sequential tasks
        Regex::new(r"(?i)\s+then\s+").unwrap(),
        // "X; Y" - semicolon separation
        Regex::new(r"\s*;\s*").unwrap(),
        // "first X, then Y" - explicit ordering
        Regex::new(r"(?i),?\s*then\s+").unwrap(),
        // "X, and Y" - comma with and
        Regex::new(r",\s+and\s+").unwrap(),
        // "X also Y" - additional task
        Regex::new(r"(?i)\s+also\s+").unwrap(),
        // "X as well as Y" - additional task
        Regex::new(r"(?i)\s+as\s+well\s+as\s+").unwrap(),
        // "X plus Y" - additional task
        Regex::new(r"(?i)\s+plus\s+").unwrap(),
        // "X additionally Y" - additional task
        Regex::new(r"(?i)\s+additionally\s+").unwrap(),
    ];

    /// Regex for splitting on sentence boundaries (period + space + optional capital)
    pub(crate) static ref SENTENCE_BOUNDARY: Regex = Regex::new(r"\.\s+").unwrap();

    /// Action verbs that indicate task starts
    pub(crate) static ref ACTION_VERBS: Vec<&'static str> = vec![
        "help", "create", "build", "write", "fix", "debug", "deploy", "test",
        "run", "check", "configure", "set", "add", "remove", "update", "install",
        "generate", "implement", "refactor", "optimize", "analyze", "review",
        "setup", "migrate", "convert", "delete", "modify", "explain", "show",
        "find", "search", "list", "get", "make", "start", "stop", "restart",
    ];

    /// Low-signal words: common procedural verbs that appear in almost every dev
    /// conversation as secondary instructions ("...and then test it", "run the build",
    /// "check the output"). When these are the ONLY matching words from the user prompt,
    /// scores are reduced to avoid false-positive skill suggestions.
    pub(crate) static ref LOW_SIGNAL_WORDS: HashSet<&'static str> = {
        let mut s = HashSet::new();
        // Procedural verbs — used as steps in any workflow, not topical
        s.insert("test"); s.insert("run"); s.insert("check");
        s.insert("build"); s.insert("fix"); s.insert("start");
        s.insert("stop"); s.insert("show"); s.insert("get");
        s.insert("make"); s.insert("set"); s.insert("add");
        s.insert("list"); s.insert("find"); s.insert("update");
        // Generic creation/action verbs — omnipresent in dev conversations
        s.insert("create"); s.insert("write"); s.insert("use");
        s.insert("need"); s.insert("want"); s.insert("work");
        s.insert("help"); s.insert("try");
        // Omnipresent nouns — appear in every dev context regardless of topic
        s.insert("code"); s.insert("file"); s.insert("project");
        // Meta-terms — everything in a skill-suggester is skill/agent/plugin related
        s.insert("skill"); s.insert("agent"); s.insert("command");
        s.insert("plugin"); s.insert("hook");
        // W8/W11 additions: common but non-discriminative words that inflate scores
        s.insert("also"); s.insert("then"); s.insert("look");
        s.insert("thing"); s.insert("stuff"); s.insert("properly");
        s.insert("proper"); s.insert("output"); s.insert("input");
        s.insert("process"); s.insert("handle"); s.insert("manage");
        s.insert("provide"); s.insert("support"); s.insert("enable");
        s.insert("ensure"); s.insert("tool"); s.insert("system");
        s
    };
}

// ============================================================================
// Synonym Expansion regex table (70+ patterns from LimorAI)
// ============================================================================

lazy_static! {
    // Compiled regex patterns for synonym expansion
    pub(crate) static ref RE_PR: Regex = Regex::new(r"(?i)\bpr\b").unwrap();
    pub(crate) static ref RE_DB: Regex = Regex::new(r"(?i)\b(db|database|postgres|postgresql|sql)\b").unwrap();
    pub(crate) static ref RE_DEPLOY: Regex = Regex::new(r"(?i)\b(deploy|deployment|deploying|release)\b").unwrap();
    pub(crate) static ref RE_TEST: Regex = Regex::new(r"(?i)\b(test|testing|tests|spec)\b").unwrap();
    pub(crate) static ref RE_GIT: Regex = Regex::new(r"(?i)\b(git|github|repo|repository)\b").unwrap();
    pub(crate) static ref RE_TROUBLE: Regex = Regex::new(r"(?i)(troubleshoot|debug|error|problem|fail|bug)").unwrap();
    pub(crate) static ref RE_CONTEXT: Regex = Regex::new(r"(?i)(context|memory|optimi)").unwrap();
    pub(crate) static ref RE_RAG: Regex = Regex::new(r"(?i)\b(rag|retrieval|vector|embeddings?)\b").unwrap();
    pub(crate) static ref RE_PROMPT: Regex = Regex::new(r"(?i)(prompt.engineer|system.prompt|llm.prompt)").unwrap();
    pub(crate) static ref RE_API: Regex = Regex::new(r"(?i)(api.design|rest.api|graphql|openapi)").unwrap();
    pub(crate) static ref RE_API_FIRST: Regex = Regex::new(r"(?i)(api.first|check.api|validate.api|api.source)").unwrap();
    pub(crate) static ref RE_TRACE: Regex = Regex::new(r"(?i)(tracing|distributed.trace|opentelemetry|jaeger)").unwrap();
    pub(crate) static ref RE_GRAFANA: Regex = Regex::new(r"(?i)(grafana|prometheus|metrics|dashboard.monitor)").unwrap();
    pub(crate) static ref RE_SQL_OPT: Regex = Regex::new(r"(?i)(sql.optimi|query.optimi|index.optimi|explain.analyze)").unwrap();
    pub(crate) static ref RE_FEEDBACK: Regex = Regex::new(r"(?i)\b(feedback|review|rating|thumbs)\b").unwrap();
    pub(crate) static ref RE_AI: Regex = Regex::new(r"(?i)\b(ai|llm|gemini|vertex|model)\b").unwrap();
    pub(crate) static ref RE_VALIDATE: Regex = Regex::new(r"(?i)\b(validate|validation|verify|check|confirm)\b").unwrap();
    pub(crate) static ref RE_MCP: Regex = Regex::new(r"(?i)\b(mcp|tool.server)\b").unwrap();
    pub(crate) static ref RE_SACRED: Regex = Regex::new(r"(?i)\b(sacred|golden.?rule|commandment|compliance)\b").unwrap();
    pub(crate) static ref RE_HEBREW: Regex = Regex::new(r"(?i)\b(hebrew|עברית|rtl|israeli)\b").unwrap();
    pub(crate) static ref RE_BEECOM: Regex = Regex::new(r"(?i)\b(beecom|pos|orders?|products?|restaurant)\b").unwrap();
    pub(crate) static ref RE_SHIFT: Regex = Regex::new(r"(?i)\b(shift|schedule|labor|employee.hours)\b").unwrap();
    pub(crate) static ref RE_REVENUE: Regex = Regex::new(r"(?i)\b(revenue|sales|income)\b").unwrap();
    pub(crate) static ref RE_SESSION: Regex = Regex::new(r"(?i)\b(session|workflow|start.session|end.session|checkpoint)\b").unwrap();
    pub(crate) static ref RE_PERPLEXITY: Regex = Regex::new(r"(?i)\b(perplexity|research|search.online|web.search)\b").unwrap();
    pub(crate) static ref RE_BLUEPRINT: Regex = Regex::new(r"(?i)\b(blueprint|architecture|feature.context|how.does.*work)\b").unwrap();
    pub(crate) static ref RE_PARITY: Regex = Regex::new(r"(?i)\b(parity|environment.match|localhost.vs|staging.vs)\b").unwrap();
    pub(crate) static ref RE_CACHE: Regex = Regex::new(r"(?i)\b(cache|caching|cached|ttl|invalidate)\b").unwrap();
    pub(crate) static ref RE_WHATSAPP: Regex = Regex::new(r"(?i)\b(whatsapp|messaging|chat.bot|webhook)\b").unwrap();
    pub(crate) static ref RE_SYNC: Regex = Regex::new(r"(?i)\b(sync|syncing|migration|migrate|backfill)\b").unwrap();
    pub(crate) static ref RE_SEMANTIC: Regex = Regex::new(r"(?i)\b(semantic|query.router|tier|embedding)\b").unwrap();
    pub(crate) static ref RE_VISUAL: Regex = Regex::new(r"(?i)\b(visual|screenshot|regression|baseline|ui.test)\b").unwrap();
    // Only expand for explicit skill-creation phrases, NOT standalone "skill"
    // "skill" alone is too common in Claude Code conversations to be a useful signal
    pub(crate) static ref RE_SKILL: Regex = Regex::new(r"(?i)\b(create.skill|add.skill|update.skill|write.skill|develop.skill|retrospective)\b").unwrap();
    pub(crate) static ref RE_CI: Regex = Regex::new(r"(?i)\b(ci|cd|pipeline|workflow|action)\b").unwrap();
    pub(crate) static ref RE_DOCKER: Regex = Regex::new(r"(?i)\b(docker|container|dockerfile|compose|kubernetes|k8s)\b").unwrap();
    pub(crate) static ref RE_AWS: Regex = Regex::new(r"(?i)\b(aws|s3|ec2|lambda|cloudformation)\b").unwrap();
    pub(crate) static ref RE_GCP: Regex = Regex::new(r"(?i)\b(gcp|gcloud|cloud.run|bigquery|pubsub)\b").unwrap();
    pub(crate) static ref RE_AZURE: Regex = Regex::new(r"(?i)\b(azure|blob|functions|cosmos)\b").unwrap();
    pub(crate) static ref RE_SECURITY: Regex = Regex::new(r"(?i)\b(security|auth|oauth|jwt|encryption)\b").unwrap();
    pub(crate) static ref RE_PERF: Regex = Regex::new(r"(?i)\b(performance|slow|latency|optimize|profil)\b").unwrap();
    // Language abbreviations -> canonical language name, for domain-gate satisfaction.
    // WORD-BOUNDARY ONLY, never `contains`: a bare substring test for "ts" matches
    // "tests", "artifacts" and "components", which would inject "typescript" into
    // prompts that never mentioned it. `\b` also makes a file extension match, so
    // "fix main.ts" and "run foo.py" resolve to the right language too.
    pub(crate) static ref RE_LANG_TS: Regex = Regex::new(r"(?i)\bts\b").unwrap();
    pub(crate) static ref RE_LANG_JS: Regex = Regex::new(r"(?i)\bjs\b").unwrap();
    pub(crate) static ref RE_LANG_PY: Regex = Regex::new(r"(?i)\bpy\b").unwrap();
    pub(crate) static ref RE_LANG_RS: Regex = Regex::new(r"(?i)\brs\b").unwrap();
}

// ============================================================================
// Domain taxonomy (Library of Congress Subject Headings-derived synonym groups)
// ============================================================================

/// Shared domain taxonomy: each domain is a group of synonyms meaning the same thing.
/// Based on Library of Congress Subject Headings (LCSH) classification, enriched with
/// modern industry terms. Used by both find_matches() (runtime prompt inference) and
/// the enrichment pipeline (index-time skill domain tagging).
/// Source: https://id.loc.gov/authorities/subjects (extracted 2026-03-19)
pub(crate) const DOMAIN_TAXONOMY: &[(&str, &[&str])] = &[
    // =========================================================================
    // PROGRAMMING (LOC: Computer science + Software engineering)
    // The default fallback — most skills are programming-related
    // =========================================================================
    ("programming", &[
        "programming", "coding", "software development", "computer science",
        "software engineering", "programm", "coder",
        // LOC: Software engineering
        "software measurement", "software prototyping", "antipatterns",
        "agile software development", "extreme programming",
        "software refactoring", "software reengineering",
        "software maintenance", "software documentation",
        "software architecture", "software patterns", "software frameworks",
        "software configuration management", "software visualization",
        "software product line", "component software",
        "cross-platform software development", "model-driven software",
        "aspect-oriented programming", "scrum",
        // LOC: Core CS
        "data structures", "algorithms", "source code",
        "parallel programming", "functional programming",
        "object-oriented programming", "generic programming",
        "constraint programming", "scripting languages",
        "formal methods", "modeling languages",
        // LOC: Software (additional)
        "software architecture", "representational state transfer",
        "application program interface", "enterprise service bus",
        "open source software", "free computer software",
        // LOC: Computer programs (conceptual terms)
        "compiler", "interpreter", "linker", "subroutine",
        "device driver", "emulator", "coroutine",
        "program transformation", "mutation testing",
        "plug-in", "text editor",
        "computer programming", "assembly language",
        "programming language",
    ]),

    // =========================================================================
    // SOFTWARE SUB-DOMAINS (LOC: Computer security, AI, Database, etc.)
    // =========================================================================

    // LOC: Computer security + Penetration testing + Intrusion detection + Malware
    // ACM CCS: Security and privacy (79 concepts)
    ("security", &[
        "security", "vulnerability", "penetration testing", "pentest",
        "owasp", "cve", "exploit", "hardening", "threat model",
        "security audit", "security scan", "secret detection",
        "authentication", "authorization", "encryption",
        // LOC: Computer security
        "computer security", "intrusion detection", "anomaly detection",
        "firewalls", "public key infrastructure", "cyber intelligence",
        "behavioral cybersecurity", "group signatures",
        "data encryption", "privacy-preserving",
        // LOC: Software security
        "malware", "computer virus", "rootkit", "ransomware", "spyware",
        "software protection", "access control",
        // ACM CCS: Security and privacy
        "key management", "digital signatures", "cryptanalysis",
        "multi-factor authentication", "digital rights management",
        "social engineering attacks", "spoofing", "phishing",
        "denial-of-service", "vulnerability scanner",
        "browser security", "web application security",
        "software reverse engineering", "software security engineering",
    ]),

    // LOC: Software testing + Debugging + Formal verification + Quality control
    // ACM CCS: Software testing and debugging, Software verification and validation
    ("testing", &[
        "testing", "test automation", "unit test", "integration test",
        "end-to-end test", "e2e test", "test driven", "tdd",
        "test coverage", "test suite", "test runner",
        // LOC terms
        "debugging", "structured walkthrough", "formal verification",
        "self-stabilization", "software testing",
        // LOC: Software quality
        "software verification", "software validation",
        "quality control", "software reliability",
        "capability maturity model",
        // ACM CCS: Software testing and debugging
        "acceptance testing", "fault tree analysis",
        "software defect analysis", "process validation",
        "automated static analysis", "dynamic analysis",
        "model checking", "pair programming",
    ]),

    // DevOps (LOC: Software container technologies + Software-defined networking)
    ("devops", &[
        "devops", "ci/cd", "continuous integration", "continuous deployment",
        "infrastructure as code", "terraform", "ansible", "kubernetes",
        "docker", "containerization", "deployment pipeline",
        "container orchestration", "configuration management",
        "monitoring", "observability",
        // LOC: Software (from software search)
        "software container", "software-defined networking",
        "software deployment", "software configuration",
        // CI / GitOps tooling — common prompt phrases the user types
        // when they're talking about DevOps without saying the word
        // "devops" or "ci/cd". Without these the unit tests
        // `test_find_matches_with_synonyms` and `test_confidence_levels`
        // panic (devops-expert skill gets excluded by domain inference
        // when the prompt is "help me set up github actions").
        "github actions", "gitlab ci", "circleci", "jenkins",
        "travis", "argo cd", "argocd", "argo workflows", "flux cd",
        "github workflow", "ci pipeline", "cd pipeline",
        "deploy pipeline", "release pipeline", "build pipeline",
        "helm chart", "helm", "k8s", "podman", "containerd",
        "docker compose", "compose file",
    ]),

    // LOC: Web site development + Document Object Model + Ajax
    ("frontend", &[
        "frontend", "front-end", "ui component", "user interface",
        "css", "html", "dom", "responsive design", "web design",
        // LOC terms
        "web site development", "document object model",
        "ajax", "web application",
        // LOC: Computer programs
        "browser", "html editor",
        // SPA framework names — when these appear alone in a prompt,
        // the user is talking about the frontend even if they never type
        // "frontend" explicitly. See TRDD-014bcc92 — without these
        // entries, a prompt like "build react component with hooks"
        // doesn't get the frontend domain inferred and downstream
        // filters drop frontend-tagged skills.
        "react", "vue", "vue.js", "angular", "svelte", "preact",
        "solid", "solid.js", "ember", "ember.js", "qwik", "lit",
        "next.js", "nuxt", "remix", "astro", "gatsby",
        // Frontend toolkit terms commonly seen without "frontend"
        "jsx", "tsx", "shadcn", "tailwind", "tailwindcss",
        "storybook", "vite", "esbuild", "rollup",
    ]),

    // LOC: Database management + Service-oriented architecture
    ("backend", &[
        "backend", "back-end", "server-side", "api development",
        "rest api", "graphql api", "microservice",
        // LOC terms
        "service-oriented architecture", "web services",
    ]),

    // LOC: Database management + Database design + Querying
    // ACM CCS: Information systems → Data management systems (291 concepts)
    ("database", &[
        "database", "database management", "database design",
        "database security", "database searching", "querying",
        "relational database", "sql", "nosql",
        // LOC terms
        "federated database", "web databases",
        "multidimensional database", "data integration",
        "materialized views", "data recovery",
        // ACM CCS: Data management systems
        "query optimization", "query planning",
        "transaction processing", "data locking",
        "key-value store", "mapreduce",
        "column based storage", "data warehouse",
        "extraction, transformation and loading", "etl",
        "data cleaning", "entity resolution",
        "object-relational mapping", "orm",
        "message queue", "service bus",
        "information retrieval", "search engine",
        "restful web services", "web services",
    ]),

    // LOC: Mobile apps + Mobile communication systems
    ("mobile", &[
        "mobile development", "mobile app", "ios development",
        "android development", "react native", "flutter",
        "mobile ui", "app store",
        // LOC terms
        "mobile apps", "mobile communication",
    ]),

    // LOC: Artificial intelligence + Neural networks + NLP
    ("data-ml", &[
        "machine learning", "deep learning", "data science",
        "neural network", "model training", "data analysis",
        "data pipeline", "data engineering",
        // LOC terms
        "artificial intelligence", "back propagation",
        "natural language processing", "natural language generation",
        "expert systems", "generative artificial intelligence",
        "human face recognition", "gesture recognition",
        "computer vision", "distributed artificial intelligence",
        "genetic programming", "evolutionary programming",
        "truth maintenance", "human computation",
        // LOC: Software AI
        "intelligent agents", "intelligent personal assistant",
        "chatbot", "multiagent systems", "mobile agents",
        // ACM CCS: Machine learning + AI + Computing methodologies
        "supervised learning", "unsupervised learning",
        "reinforcement learning", "transfer learning",
        "classification", "clustering", "regression",
        "object detection", "object recognition", "image segmentation",
        "speech recognition", "machine translation",
        "recommender systems", "sentiment analysis",
        "information extraction", "question answering",
        "knowledge representation", "ontology engineering",
        "topic modeling", "dimensionality reduction",
        "anomaly detection", "multi-agent systems",
    ]),

    // LOC: Cloud computing + Ubiquitous computing + Distributed systems
    ("cloud", &[
        "cloud computing", "cloud infrastructure", "serverless",
        "cloud function", "cloud deployment", "iaas", "paas",
        // LOC terms
        "ubiquitous computing", "distributed shared memory",
        "high performance computing", "granular computing",
    ]),

    // LOC: Computer network architectures + Network protocols
    ("networking", &[
        "networking", "computer network", "network protocol",
        "network architecture", "network security",
        // LOC terms
        "directory services", "network publishing",
        "network slicing", "network file system",
        "network management", "network time protocol",
    ]),

    // LOC: Blockchains (Databases) + Smart contracts
    ("blockchain", &[
        "blockchain", "smart contract", "solidity", "ethereum",
        "web3", "defi", "nft", "cryptocurrency",
        // LOC terms
        "blockchains",
    ]),

    // LOC: Video games + Level design + Game theory
    ("game-dev", &[
        "game development", "game engine", "unity", "unreal",
        "game design", "game programming", "sprite", "physics engine",
        // LOC terms
        "video games", "level design", "computer game",
    ]),

    // =========================================================================
    // NON-PROGRAMMING DOMAINS (LOC subject classifications)
    // Skills matching these are excluded when the prompt is about programming
    // =========================================================================

    ("video-production", &[
        "video editing", "video production", "video processing",
        "film editing", "film production", "filmmaking",
        "video editor", "video edit",
    ]),
    ("audio-production", &[
        "music production", "audio editing", "audio production",
        "sound design", "sound editing", "music composition",
        "audio editor", "audio edit", "music producer",
    ]),
    ("photography", &[
        "photo editing", "photography", "image editing",
        "photo retouching", "photo manipulation",
        "photograph", "photo editor", "photo edit",
    ]),
    // LOC: Graphic design (Typography) + Graphic arts
    ("graphic-design", &[
        "graphic design", "visual design", "illustration",
        "digital illustration", "graphic designer", "illustrat",
        // LOC terms
        "graphic arts", "graphic methods", "graphic statics",
        "presentation graphics",
    ]),
    // LOC: Computer graphics + Rendering + Real-time rendering + 3D
    ("3d-graphics", &[
        "3d modeling", "3d rendering", "3d animation",
        "3d design", "3d modelling",
        // LOC terms
        "real-time rendering", "interactive computer graphics",
        "avatars", "virtual reality", "x3d",
    ]),
    ("motion-graphics", &[
        "motion graphics", "motion design", "visual effects",
    ]),
    // LOC: Computer graphics + Graphics processing units + WebGL + SVG
    ("computer-graphics", &[
        "computer graphics", "graphics processing",
        "rendering", "rasterization", "ray tracing",
        "shader", "fragment shader", "vertex shader",
        "opengl", "vulkan", "directx", "metal graphics",
        // LOC terms
        "color computer graphics", "bit-mapped graphics",
        "icons", "layers", "graphical user interface",
        "webgl", "svg", "canvas",
        "graphics processing unit", "gpu programming", "gpu computing",
        "image synthesis", "texture mapping", "anti-aliasing",
        "screen space", "framebuffer", "pixel shader",
    ]),
    ("copywriting", &[
        "copywriting", "content writing", "blog writing",
        "article writing", "copywriter", "content writer", "blog writer",
    ]),
    ("creative-writing", &[
        "creative writing", "fiction writing", "screenplay",
        "screenwriting", "novel writing", "ghostwriting",
        "fiction writer", "novelist", "screenwriter", "ghostwriter",
    ]),
    ("poetry", &["poetry writing", "poem writing", "poetic composition"]),
    ("journalism", &[
        "journalism", "news writing", "investigative reporting",
        "journalist", "news reporter",
    ]),
    ("translation", &["translation", "language translation", "translating", "translator"]),
    ("marketing", &[
        "digital marketing", "social media marketing",
        "email marketing", "marketing strategy",
        "marketer", "marketing campaign",
    ]),
    ("advertising", &["advertising", "ad campaign", "ad copy", "advertiser"]),
    ("branding", &["branding", "brand identity", "brand strategy"]),
    ("seo", &["search engine optimization"]),
    ("education", &[
        "lesson plan", "curriculum design", "e-learning",
        "course creation", "educational content",
        "lesson planning", "curriculum development",
    ]),
    ("tutoring", &["tutoring", "private tutoring", "academic tutoring", "tutor"]),
    ("chemistry", &["chemical analysis", "chemical reaction", "chemist"]),
    ("biology", &["biological research", "microbiology", "biologist"]),
    ("physics", &["quantum mechanics", "astrophysics", "particle physics", "physicist"]),
    ("geology", &["geological survey", "mineralogy", "geologist"]),
    ("astronomy", &["astrophotography", "astronomical observation", "astronomer"]),
    ("medicine", &[
        "medical research", "clinical trial", "pharmaceutical",
        "clinical research", "medical diagnosis",
        "physician", "clinician", "pharmacist",
    ]),
    ("genomics", &["genomics", "proteomics", "bioinformatics", "gene sequencing"]),
    ("legal", &[
        "legal writing", "contract drafting", "patent writing",
        "legal research", "legal analysis",
        "lawyer", "attorney", "paralegal",
    ]),
    ("accounting", &[
        "bookkeeping", "tax preparation", "financial accounting",
        "accountant", "bookkeeper",
    ]),
    ("cooking", &["cooking", "culinary", "chef", "cookbook"]),
    ("nutrition", &["nutrition", "dietetics", "meal planning", "nutritionist", "dietitian"]),
    ("fitness", &["fitness training", "workout routine", "exercise program", "personal trainer"]),
    ("real-estate", &["real estate", "property management", "realtor"]),
    ("interior-design", &["interior design", "space planning", "interior designer"]),
    ("architecture", &[
        "architecture design", "architectural design", "building design", "architect",
    ]),
    ("event-planning", &["event planning", "event management", "event planner", "event organizer"]),
    ("geography", &["geography", "cartography", "geographic analysis", "geographer", "cartographer"]),
    ("linguistics", &[
        "linguistics", "linguistic analysis", "phonetics",
        "morphology analysis", "linguist", "phonetician",
    ]),
    ("music-theory", &["music theory", "harmony theory", "counterpoint", "music theorist"]),
    ("fine-art", &["fine art", "art history", "art historian"]),
    ("painting", &[
        "painting technique", "oil painting", "watercolor painting",
        "acrylic painting", "painter",
    ]),
    ("sculpture", &["sculpture", "sculpting", "ceramics", "sculptor"]),
];

// ============================================================================
// Activity Classification System data — Cue→Activity Scoring
// (same 4-tier logarithmic weights as the hook mode scorer)
// ============================================================================

/// Activity definition — a category with tiered cues for inference.
pub(crate) struct ActivityDef {
    /// Activity identifier, e.g. "linting", "unit-testing", "container-deployment"
    pub(crate) name: &'static str,
    /// Parent activity for hierarchy, e.g. "testing" is parent of "unit-testing"
    #[allow(dead_code)]
    pub(crate) parent: Option<&'static str>,
    /// Tool-tier cues (2000 pts) — tool names that strongly imply this activity
    pub(crate) tools: &'static [&'static str],
    /// Framework-tier cues (20000 pts) — framework names that strongly imply this activity
    pub(crate) frameworks: &'static [&'static str],
    /// Phrase-tier cues (100 pts each, /10 if low-signal) — descriptive keywords
    pub(crate) keywords: &'static [&'static str],
}

/// Static registry of all activity definitions (~130 activities).
/// Organized hierarchically: parent activities group related leaf activities.
pub(crate) static ACTIVITY_REGISTRY: &[ActivityDef] = &[
    // ── Development (10x) ──────────────────────────────────────────────────
    ActivityDef { name: "implementation", parent: None,
        tools: &[],
        frameworks: &[],
        keywords: &["implement", "develop", "build", "create", "feature", "functionality"] },
    ActivityDef { name: "api-development", parent: Some("implementation"),
        tools: &["postman", "insomnia", "swagger", "openapi", "hoppscotch"],
        frameworks: &["express", "fastapi", "flask", "django-rest", "spring-boot", "gin", "actix", "axum", "hono", "koa", "nest"],
        keywords: &["api", "endpoint", "rest", "graphql", "grpc", "route", "handler", "middleware", "request", "response"] },
    ActivityDef { name: "ui-development", parent: Some("implementation"),
        tools: &["storybook", "figma", "chromatic"],
        frameworks: &["react", "vue", "angular", "svelte", "solid", "qwik", "htmx", "alpine"],
        keywords: &["ui", "interface", "layout", "component", "widget", "render", "dom", "jsx", "tsx", "template"] },
    ActivityDef { name: "cli-development", parent: Some("implementation"),
        tools: &["clap", "commander", "yargs", "inquirer", "oclif", "cobra", "click"],
        frameworks: &[],
        keywords: &["cli", "command-line", "terminal", "argparse", "flag", "subcommand", "prompt", "interactive"] },
    ActivityDef { name: "mobile-development", parent: Some("implementation"),
        tools: &["xcode", "android-studio", "expo", "fastlane", "cocoapods", "gradle"],
        frameworks: &["react-native", "flutter", "swiftui", "uikit", "jetpack-compose", "ionic", "capacitor", "maui"],
        keywords: &["mobile", "app", "native", "ios", "android", "tablet", "smartphone", "touch", "gesture"] },
    ActivityDef { name: "game-development", parent: Some("implementation"),
        tools: &["unity", "unreal", "godot", "bevy", "pygame", "love2d", "phaser", "pixi"],
        frameworks: &[],
        keywords: &["game", "engine", "physics", "sprite", "render", "collision", "shader", "mesh", "scene"] },
    ActivityDef { name: "library-development", parent: Some("implementation"),
        tools: &[],
        frameworks: &[],
        keywords: &["library", "package", "module", "crate", "sdk", "wrapper", "binding", "publish", "registry"] },
    ActivityDef { name: "plugin-development", parent: Some("implementation"),
        tools: &[],
        frameworks: &[],
        keywords: &["plugin", "extension", "hook", "addon", "marketplace", "integration", "middleware"] },
    ActivityDef { name: "web-development", parent: Some("implementation"),
        tools: &[],
        frameworks: &["nextjs", "nuxt", "remix", "astro", "gatsby", "sveltekit", "fresh"],
        keywords: &["web", "website", "webapp", "browser", "html", "css", "responsive", "progressive"] },
    ActivityDef { name: "frontend-development", parent: Some("web-development"),
        tools: &["storybook", "chromatic", "bit"],
        frameworks: &["react", "vue", "angular", "svelte", "solid", "preact", "lit"],
        keywords: &["frontend", "client-side", "spa", "ssr", "ssg", "hydration", "state-management"] },
    ActivityDef { name: "backend-development", parent: Some("web-development"),
        tools: &[],
        frameworks: &["express", "django", "rails", "fastapi", "spring", "laravel", "phoenix", "gin", "actix", "nest"],
        keywords: &["backend", "server-side", "microservice", "service", "api-server", "worker"] },
    ActivityDef { name: "database-development", parent: Some("implementation"),
        tools: &["prisma", "drizzle", "typeorm", "sequelize", "knex", "sqlalchemy", "diesel", "sea-orm"],
        frameworks: &[],
        keywords: &["database", "schema", "migration", "query", "orm", "sql", "nosql", "table", "index", "relation"] },
    ActivityDef { name: "embedded-development", parent: Some("implementation"),
        tools: &["platformio", "arduino", "stm32cube", "esp-idf"],
        frameworks: &[],
        keywords: &["embedded", "firmware", "microcontroller", "iot", "rtos", "gpio", "uart", "spi", "i2c"] },
    ActivityDef { name: "blockchain-development", parent: Some("implementation"),
        tools: &["hardhat", "foundry", "truffle", "brownie", "anchor"],
        frameworks: &["ethers", "web3", "wagmi", "viem"],
        keywords: &["blockchain", "smart-contract", "web3", "solidity", "defi", "nft", "token", "wallet", "chain"] },
    ActivityDef { name: "desktop-development", parent: Some("implementation"),
        tools: &["electron", "tauri", "wails"],
        frameworks: &["swiftui", "wpf", "gtk", "qt", "tkinter"],
        keywords: &["desktop", "native-app", "window", "menu", "tray", "dialog", "cross-platform"] },

    // ── Quality (20x) ──────────────────────────────────────────────────────
    ActivityDef { name: "testing", parent: None,
        tools: &[],
        frameworks: &[],
        keywords: &["test", "spec", "assert", "coverage", "fixture", "mock", "stub", "suite"] },
    ActivityDef { name: "unit-testing", parent: Some("testing"),
        tools: &["jest", "pytest", "vitest", "rspec", "junit", "nunit", "xunit", "googletest", "catch2", "mocha", "ava"],
        frameworks: &[],
        keywords: &["unit-test", "unit", "isolated", "function-test", "method-test"] },
    ActivityDef { name: "integration-testing", parent: Some("testing"),
        tools: &["supertest", "testcontainers"],
        frameworks: &[],
        keywords: &["integration", "service-test", "api-test", "contract-test", "pact"] },
    ActivityDef { name: "e2e-testing", parent: Some("testing"),
        tools: &["cypress", "playwright", "selenium", "puppeteer", "webdriverio", "detox", "appium", "maestro"],
        frameworks: &[],
        keywords: &["e2e", "end-to-end", "acceptance", "browser-test", "ui-test", "functional-test"] },
    ActivityDef { name: "snapshot-testing", parent: Some("testing"),
        tools: &["percy", "chromatic", "loki", "backstop"],
        frameworks: &[],
        keywords: &["snapshot", "visual-regression", "screenshot-test", "pixel-diff"] },
    ActivityDef { name: "fuzz-testing", parent: Some("testing"),
        tools: &["afl", "libfuzzer", "cargo-fuzz", "atheris", "jazzer"],
        frameworks: &["hypothesis", "proptest", "quickcheck"],
        keywords: &["fuzz", "property-based", "generative-test", "mutation-test"] },
    ActivityDef { name: "load-testing", parent: Some("testing"),
        tools: &["k6", "artillery", "locust", "gatling", "jmeter", "wrk", "ab", "vegeta"],
        frameworks: &[],
        keywords: &["load-test", "stress-test", "benchmark", "throughput", "latency-test", "soak-test"] },
    ActivityDef { name: "test-automation", parent: Some("testing"),
        tools: &["github-actions", "gitlab-ci", "jenkins"],
        frameworks: &[],
        keywords: &["test-runner", "ci-test", "test-pipeline", "test-suite", "test-report", "test-coverage"] },
    ActivityDef { name: "linting", parent: None,
        tools: &["eslint", "ruff", "pylint", "clippy", "flake8", "rubocop", "golangci-lint", "ktlint",
                  "stylelint", "htmlhint", "shellcheck", "hadolint", "markdownlint", "yamllint",
                  "swiftlint", "detekt", "checkstyle", "pmd", "spotbugs", "biome"],
        frameworks: &[],
        keywords: &["lint", "linter", "static-analysis", "code-style", "rule-violation", "code-smell"] },
    ActivityDef { name: "formatting", parent: None,
        tools: &["prettier", "black", "gofmt", "rustfmt", "clang-format", "autopep8", "yapf",
                  "swift-format", "ktfmt", "google-java-format", "shfmt", "biome"],
        frameworks: &[],
        keywords: &["format", "formatter", "prettify", "indent", "whitespace", "code-style", "auto-format"] },
    ActivityDef { name: "code-review", parent: None,
        tools: &["reviewbot", "danger", "coderabbit", "codacy", "codeclimate"],
        frameworks: &[],
        keywords: &["review", "pull-request", "pr-review", "code-quality", "peer-review", "approve", "request-changes"] },
    ActivityDef { name: "type-checking", parent: None,
        tools: &["mypy", "pyright", "tsc", "flow", "sorbet", "steep"],
        frameworks: &[],
        keywords: &["type-check", "type-safe", "type-annotation", "typing", "type-error", "type-inference"] },
    ActivityDef { name: "refactoring", parent: None,
        tools: &["rope", "jscodeshift", "ts-morph"],
        frameworks: &[],
        keywords: &["refactor", "restructure", "clean-code", "technical-debt", "extract-method", "rename", "simplify"] },
    ActivityDef { name: "documentation", parent: None,
        tools: &["sphinx", "typedoc", "javadoc", "rustdoc", "doxygen", "mkdocs", "docusaurus", "vitepress", "jsdoc"],
        frameworks: &[],
        keywords: &["document", "docstring", "readme", "api-docs", "documentation", "wiki", "changelog", "guide"] },
    ActivityDef { name: "accessibility-audit", parent: None,
        tools: &["axe", "lighthouse", "pa11y", "wave"],
        frameworks: &[],
        keywords: &["accessibility", "a11y", "wcag", "aria", "screen-reader", "keyboard-nav", "contrast"] },

    // ── Operations (30x) ───────────────────────────────────────────────────
    ActivityDef { name: "deployment", parent: None,
        tools: &[],
        frameworks: &[],
        keywords: &["deploy", "release", "ship", "rollout", "publish", "promote"] },
    ActivityDef { name: "container-deployment", parent: Some("deployment"),
        tools: &["docker", "podman", "buildah", "docker-compose", "containerd", "nerdctl"],
        frameworks: &[],
        keywords: &["container", "containerize", "dockerfile", "image", "registry", "layer", "multi-stage"] },
    ActivityDef { name: "cloud-deployment", parent: Some("deployment"),
        tools: &["vercel", "netlify", "heroku", "fly", "render", "railway", "amplify", "firebase-hosting"],
        frameworks: &[],
        keywords: &["cloud-deploy", "cloud-hosting", "platform-as-service", "paas", "auto-deploy"] },
    ActivityDef { name: "kubernetes-ops", parent: Some("deployment"),
        tools: &["kubectl", "helm", "kustomize", "argocd", "flux", "skaffold", "tilt", "k3s", "minikube", "kind"],
        frameworks: &[],
        keywords: &["kubernetes", "k8s", "pod", "cluster", "namespace", "ingress", "service-mesh", "istio"] },
    ActivityDef { name: "serverless-deployment", parent: Some("deployment"),
        tools: &["serverless", "sam", "claudia", "architect", "sst"],
        frameworks: &["aws-lambda", "cloudflare-workers", "deno-deploy", "azure-functions", "google-cloud-functions"],
        keywords: &["serverless", "lambda", "function-as-service", "faas", "edge-function", "cold-start"] },
    ActivityDef { name: "ci-cd", parent: None,
        tools: &["github-actions", "gitlab-ci", "jenkins", "circleci", "travis", "drone", "buildkite",
                  "tekton", "concourse", "woodpecker", "semaphore", "bitbucket-pipelines"],
        frameworks: &[],
        keywords: &["ci", "cd", "pipeline", "continuous-integration", "continuous-delivery", "workflow", "build-automation"] },
    ActivityDef { name: "infrastructure-provisioning", parent: None,
        tools: &["terraform", "pulumi", "ansible", "cloudformation", "cdktf", "crossplane", "chef", "puppet", "salt"],
        frameworks: &[],
        keywords: &["infrastructure", "iac", "provision", "infrastructure-as-code", "resource", "stack"] },
    ActivityDef { name: "monitoring", parent: None,
        tools: &["datadog", "grafana", "prometheus", "sentry", "newrelic", "elastic-apm", "honeycomb", "pagerduty", "opsgenie"],
        frameworks: &[],
        keywords: &["monitor", "observability", "metrics", "alerting", "dashboard", "uptime", "health-check", "incident"] },
    ActivityDef { name: "logging", parent: None,
        tools: &["elk", "splunk", "loki", "fluentd", "logstash", "vector", "papertrail", "logtail"],
        frameworks: &[],
        keywords: &["logging", "log-aggregation", "log-analysis", "structured-logging", "log-level", "log-rotation"] },
    ActivityDef { name: "configuration-management", parent: None,
        tools: &["dotenv", "consul", "etcd", "vault", "configmap"],
        frameworks: &[],
        keywords: &["config", "env-vars", "settings", "dotenv", "configuration", "feature-flag", "toggle"] },
    ActivityDef { name: "package-management", parent: None,
        tools: &["npm", "yarn", "pnpm", "bun", "pip", "uv", "cargo", "maven", "gradle", "gem", "brew",
                  "composer", "nuget", "cocoapods", "swift-package-manager", "go-mod"],
        frameworks: &[],
        keywords: &["package", "dependency", "version", "install", "upgrade", "lockfile", "registry", "publish"] },
    ActivityDef { name: "bundling", parent: None,
        tools: &["webpack", "vite", "esbuild", "rollup", "parcel", "turbopack", "swc", "tsup", "unbuild", "bun"],
        frameworks: &[],
        keywords: &["bundle", "bundler", "build-tool", "transpile", "minify", "tree-shake", "code-split", "sourcemap"] },
    ActivityDef { name: "release-management", parent: None,
        tools: &["semantic-release", "changesets", "lerna", "release-it", "goreleaser", "standard-version"],
        frameworks: &[],
        keywords: &["release", "changelog", "semver", "versioning", "tag", "publish", "distribution"] },

    // ── Investigation (40x) ────────────────────────────────────────────────
    ActivityDef { name: "debugging", parent: None,
        tools: &["gdb", "lldb", "pdb", "chrome-devtools", "vs-debugger", "delve", "node-inspect"],
        frameworks: &[],
        keywords: &["debug", "debugger", "breakpoint", "step-through", "stack-trace", "backtrace", "watchpoint",
                     "bug", "investigate", "root-cause", "bisect"] },
    ActivityDef { name: "root-cause-analysis", parent: Some("debugging"),
        tools: &[],
        frameworks: &[],
        keywords: &["root-cause", "investigate", "diagnose", "postmortem", "incident-review",
                     "failure-analysis", "regression", "reproduce"] },
    ActivityDef { name: "profiling", parent: None,
        tools: &["perf", "instruments", "py-spy", "clinic", "flamegraph", "dotnet-trace",
                  "async-profiler", "yourkit", "gperftools", "vtune", "coz"],
        frameworks: &[],
        keywords: &["profile", "profiler", "performance", "bottleneck", "flame-graph", "cpu-time",
                     "hot-path", "benchmark", "optimize"] },
    ActivityDef { name: "memory-debugging", parent: Some("debugging"),
        tools: &["valgrind", "heaptrack", "leaks", "addresssanitizer", "msan", "drmemory"],
        frameworks: &[],
        keywords: &["memory-leak", "heap", "allocation", "garbage-collection", "use-after-free",
                     "buffer-overflow", "memory-safety", "stack-overflow"] },
    ActivityDef { name: "log-analysis", parent: Some("debugging"),
        tools: &[],
        frameworks: &[],
        keywords: &["log-analysis", "parse-logs", "error-pattern", "log-grep", "log-tail", "log-search"] },
    ActivityDef { name: "network-debugging", parent: Some("debugging"),
        tools: &["wireshark", "tcpdump", "curl", "httpie", "charles", "fiddler", "mitmproxy", "ngrep"],
        frameworks: &[],
        keywords: &["network-debug", "packet", "dns", "ssl", "tls", "http-trace", "latency", "timeout", "connection"] },
    ActivityDef { name: "tracing", parent: None,
        tools: &["jaeger", "zipkin", "otel", "opentelemetry", "lightstep", "tempo"],
        frameworks: &[],
        keywords: &["trace", "distributed-tracing", "span", "opentelemetry", "correlation-id", "propagation"] },

    // ── Architecture (50x) ─────────────────────────────────────────────────
    ActivityDef { name: "system-design", parent: None,
        tools: &["mermaid", "plantuml", "drawio", "excalidraw", "lucidchart"],
        frameworks: &[],
        keywords: &["architecture", "system-design", "scalability", "microservice", "monolith",
                     "event-driven", "cqrs", "domain-driven", "hexagonal"] },
    ActivityDef { name: "api-design", parent: Some("system-design"),
        tools: &["swagger", "openapi", "stoplight", "redocly", "graphql-codegen"],
        frameworks: &[],
        keywords: &["api-design", "schema-design", "openapi", "swagger", "specification", "contract-first", "versioning"] },
    ActivityDef { name: "database-design", parent: Some("system-design"),
        tools: &["dbdiagram", "pgmodeler", "dbeaver"],
        frameworks: &[],
        keywords: &["schema-design", "erd", "normalization", "indexing", "partitioning", "sharding", "replication"] },
    ActivityDef { name: "design-patterns", parent: Some("system-design"),
        tools: &[],
        frameworks: &[],
        keywords: &["design-pattern", "factory", "observer", "strategy", "solid", "dependency-injection",
                     "singleton", "adapter", "decorator", "mediator"] },
    ActivityDef { name: "migration-planning", parent: None,
        tools: &["goose", "flyway", "alembic", "knex-migrate", "dbmate"],
        frameworks: &[],
        keywords: &["migration", "upgrade", "modernize", "legacy", "rewrite", "port", "migrate", "backward-compatible"] },

    // ── Data (60x) ─────────────────────────────────────────────────────────
    ActivityDef { name: "data-processing", parent: None,
        tools: &["pandas", "spark", "dbt", "airflow", "dagster", "prefect", "beam", "flink", "polars"],
        frameworks: &[],
        keywords: &["data", "etl", "pipeline", "transform", "ingest", "extract", "batch", "stream"] },
    ActivityDef { name: "data-analysis", parent: Some("data-processing"),
        tools: &["numpy", "scipy", "jupyter", "rstudio", "stata", "matlab", "mathematica"],
        frameworks: &[],
        keywords: &["analyze", "statistics", "insight", "metric", "correlation", "regression", "hypothesis"] },
    ActivityDef { name: "data-visualization", parent: Some("data-processing"),
        tools: &["matplotlib", "d3", "plotly", "grafana", "tableau", "metabase", "superset",
                  "seaborn", "altair", "recharts", "nivo", "echarts", "chart-js", "vega"],
        frameworks: &[],
        keywords: &["visualize", "chart", "graph", "plot", "dashboard", "diagram", "heatmap", "histogram"] },
    ActivityDef { name: "data-cleaning", parent: Some("data-processing"),
        tools: &["openrefine", "trifacta", "great-expectations"],
        frameworks: &[],
        keywords: &["clean", "preprocess", "normalize", "deduplicate", "impute", "outlier", "validation"] },
    ActivityDef { name: "machine-learning", parent: None,
        tools: &["sklearn", "xgboost", "lightgbm", "catboost", "mlflow", "wandb", "optuna", "ray"],
        frameworks: &["tensorflow", "pytorch", "keras", "jax"],
        keywords: &["ml", "model", "train", "predict", "feature-engineering", "classification",
                     "regression", "clustering", "ensemble", "hyperparameter"] },
    ActivityDef { name: "deep-learning", parent: Some("machine-learning"),
        tools: &["tensorboard", "wandb", "weights-and-biases"],
        frameworks: &["pytorch", "tensorflow", "keras", "jax", "flax", "haiku", "lightning"],
        keywords: &["neural-network", "cnn", "rnn", "transformer", "attention", "backpropagation",
                     "gpu", "cuda", "distributed-training", "fine-tune"] },
    ActivityDef { name: "nlp", parent: Some("machine-learning"),
        tools: &["spacy", "nltk", "huggingface", "gensim", "fasttext", "stanza"],
        frameworks: &["transformers"],
        keywords: &["nlp", "text-processing", "tokenize", "embedding", "sentiment",
                     "ner", "pos-tagging", "text-classification", "summarization"] },
    ActivityDef { name: "computer-vision", parent: Some("machine-learning"),
        tools: &["opencv", "yolo", "detectron", "mediapipe", "tesseract"],
        frameworks: &["torchvision"],
        keywords: &["vision", "image-processing", "object-detection", "segmentation",
                     "ocr", "face-detection", "pose-estimation", "image-classification"] },
    ActivityDef { name: "llm-integration", parent: None,
        tools: &["langchain", "llamaindex", "openai", "anthropic", "ollama", "vllm",
                  "chromadb", "pinecone", "weaviate", "qdrant", "milvus", "faiss"],
        frameworks: &[],
        keywords: &["llm", "prompt-engineering", "rag", "fine-tune", "embedding",
                     "retrieval-augmented", "vector-store", "chain-of-thought", "agent", "chat-model"] },

    // ── Security (70x) ─────────────────────────────────────────────────────
    ActivityDef { name: "security-audit", parent: None,
        tools: &["trivy", "grype", "nessus", "qualys"],
        frameworks: &[],
        keywords: &["security", "vulnerability", "cve", "owasp", "threat-model",
                     "risk-assessment", "compliance", "hardening"] },
    ActivityDef { name: "authentication", parent: Some("security-audit"),
        tools: &["auth0", "clerk", "nextauth", "passport", "keycloak", "okta", "firebase-auth", "supabase-auth"],
        frameworks: &[],
        keywords: &["auth", "login", "sso", "oauth", "jwt", "session", "mfa", "2fa",
                     "identity", "sign-in", "sign-up", "password"] },
    ActivityDef { name: "authorization", parent: Some("security-audit"),
        tools: &["casbin", "opa", "cerbos", "permit"],
        frameworks: &[],
        keywords: &["rbac", "permissions", "access-control", "policy", "role", "privilege", "scope", "claim"] },
    ActivityDef { name: "encryption", parent: Some("security-audit"),
        tools: &["openssl", "age", "sops", "gpg"],
        frameworks: &[],
        keywords: &["encrypt", "decrypt", "hash", "tls", "ssl", "certificate", "pki", "signing", "cipher"] },
    ActivityDef { name: "secret-management", parent: Some("security-audit"),
        tools: &["vault", "doppler", "aws-secrets-manager", "infisical", "dotenvx", "truffleHog"],
        frameworks: &[],
        keywords: &["secrets", "vault", "credentials", "api-key", "token-rotation", "key-management"] },
    ActivityDef { name: "penetration-testing", parent: Some("security-audit"),
        tools: &["burp", "nmap", "metasploit", "zap", "nuclei", "sqlmap", "gobuster", "hydra"],
        frameworks: &[],
        keywords: &["pentest", "exploit", "ctf", "red-team", "payload", "injection", "brute-force"] },
    ActivityDef { name: "dependency-scanning", parent: Some("security-audit"),
        tools: &["snyk", "dependabot", "renovate", "socket", "mend", "fossa"],
        frameworks: &[],
        keywords: &["supply-chain", "dependency-audit", "cve-scan", "vulnerability-scan", "sbom", "license-check"] },
    ActivityDef { name: "code-scanning", parent: Some("security-audit"),
        tools: &["semgrep", "sonarqube", "codeql", "checkmarx", "veracode", "fortify", "bandit", "brakeman"],
        frameworks: &[],
        keywords: &["sast", "dast", "code-scan", "taint-analysis", "security-lint", "vuln-detect"] },

    // ── Content (80x) ──────────────────────────────────────────────────────
    ActivityDef { name: "content-creation", parent: None,
        tools: &[],
        frameworks: &[],
        keywords: &["write", "blog", "article", "copy", "content", "post", "newsletter", "editorial"] },
    ActivityDef { name: "seo", parent: Some("content-creation"),
        tools: &["ahrefs", "semrush", "screaming-frog", "google-search-console"],
        frameworks: &[],
        keywords: &["seo", "search-engine", "meta-tag", "sitemap", "ranking", "keyword-research", "backlink"] },
    ActivityDef { name: "localization", parent: None,
        tools: &["crowdin", "lokalise", "transifex", "weblate", "phrase", "i18next"],
        frameworks: &[],
        keywords: &["i18n", "l10n", "translate", "locale", "internationalization", "localization", "rtl", "pluralization"] },
    ActivityDef { name: "media-processing", parent: None,
        tools: &["ffmpeg", "imagemagick", "sharp", "jimp", "sox", "handbrake", "gimp"],
        frameworks: &[],
        keywords: &["image", "video", "audio", "transcode", "compress", "thumbnail", "resize", "watermark", "convert"] },
    ActivityDef { name: "pdf-processing", parent: Some("media-processing"),
        tools: &["puppeteer", "wkhtmltopdf", "pdfkit", "reportlab", "fpdf", "weasyprint", "prince"],
        frameworks: &[],
        keywords: &["pdf", "document", "report-generation", "print-layout", "page-break"] },

    // ── Management (90x) ───────────────────────────────────────────────────
    ActivityDef { name: "project-management", parent: None,
        tools: &["jira", "linear", "asana", "trello", "notion", "clickup", "shortcut", "monday"],
        frameworks: &[],
        keywords: &["project", "sprint", "backlog", "roadmap", "kanban", "scrum", "epic", "story", "ticket"] },
    ActivityDef { name: "git-workflow", parent: None,
        tools: &["git", "gh", "gitea", "gitlab"],
        frameworks: &[],
        keywords: &["git", "branch", "merge", "rebase", "commit", "pull-request", "cherry-pick", "stash", "conflict"] },
    ActivityDef { name: "code-generation", parent: None,
        tools: &["plop", "hygen", "yeoman", "cookiecutter", "create-react-app", "create-next-app"],
        frameworks: &[],
        keywords: &["generate", "scaffold", "boilerplate", "template", "starter", "init", "create-app"] },
    ActivityDef { name: "research", parent: None,
        tools: &["arxiv", "scholar", "zotero", "mendeley", "paperpile"],
        frameworks: &[],
        keywords: &["research", "paper", "literature", "survey", "analysis", "study", "experiment", "hypothesis"] },
    ActivityDef { name: "automation", parent: None,
        tools: &["make", "just", "task", "nox", "tox", "invoke"],
        frameworks: &[],
        keywords: &["automate", "script", "workflow", "cron", "scheduled", "batch", "pipeline", "taskfile"] },

    // ── Fallback ───────────────────────────────────────────────────────────
    ActivityDef { name: "general-development", parent: None,
        tools: &[],
        frameworks: &[],
        keywords: &["develop", "code", "program", "software", "engineering"] },
];
