use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn files(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
            .map(|d| d.flatten().map(|e| e.path()).collect())
            .unwrap_or_default();
        entries.sort();
        for e in entries {
            files(&e, out);
        }
    } else if path.is_file() {
        out.push(path.to_path_buf());
    }
}

fn path_deps(root: &Path, krate: &str, seen: &mut BTreeSet<String>) {
    let Ok(text) = std::fs::read_to_string(root.join("crates").join(krate).join("Cargo.toml")) else {
        return;
    };
    for line in text.lines() {
        let name = line.split('=').next().unwrap_or("").trim().split('.').next().unwrap_or("");
        if name.starts_with("victory-") && name != krate && line.contains('=') && seen.insert(name.to_owned()) {
            path_deps(root, name, seen);
        }
    }
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let mut inputs = vec![
        root.join("crates/victory-shader/src"),
        root.join("crates/victory-shader/Cargo.toml"),
        root.join("crates/victory-pipeline/src"),
        root.join("crates/victory-gpu/src/shader_cache.rs"),
        root.join("Cargo.lock"),
    ];
    let mut deps = BTreeSet::new();
    path_deps(&root, "victory-shader", &mut deps);
    inputs.extend(deps.iter().map(|d| root.join("crates").join(d).join("src")));
    let mut all = Vec::new();
    for i in &inputs {
        println!("cargo:rerun-if-changed={}", i.display());
        files(i, &mut all);
    }
    let mut rel: Vec<(String, PathBuf)> = all
        .into_iter()
        .filter_map(|f| {
            let c = f.canonicalize().ok()?;
            let r = c.strip_prefix(root.canonicalize().ok()?).ok()?.to_string_lossy().replace('\\', "/");
            Some((r, f))
        })
        .collect();
    rel.sort();
    rel.dedup_by(|a, b| a.0 == b.0);
    let mut h = blake3::Hasher::new();
    for (r, f) in &rel {
        h.update(r.as_bytes());
        h.update(&std::fs::read(f).unwrap_or_default());
    }
    println!("cargo:rustc-env=VICTORY_TRANSLATOR_ID={}", h.finalize().to_hex());
}
