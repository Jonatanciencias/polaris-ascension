use serde::{de::DeserializeOwned, Deserialize, Serialize};
#[cfg(feature = "python-module")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python-module")]
use pyo3::prelude::*;
#[cfg(feature = "python-module")]
use pyo3::types::PyModule;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchRequest {
    kernels: Vec<String>,
    vector_widths: Vec<u32>,
    unroll_k_values: Vec<u32>,
    local_sizes: Vec<[u32; 2]>,
    top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Candidate {
    candidate_id: String,
    kernel: String,
    vector_width: u32,
    unroll_k: u32,
    local_size: [u32; 2],
    estimated_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ReplayEntry {
    candidate_id: String,
    session: u32,
    run: u32,
    seed: u64,
    execution_tag: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SidecarInfo {
    sidecar_name: String,
    version: String,
    capabilities: Vec<String>,
}

fn parse_json<T: DeserializeOwned>(text: &str, context: &str) -> Result<T, String> {
    serde_json::from_str(text).map_err(|err| {
        format!("{context}: invalid JSON payload ({err})")
    })
}

fn to_pretty_json<T: Serialize>(value: &T, context: &str) -> Result<String, String> {
    serde_json::to_string_pretty(value).map_err(|err| {
        format!("{context}: serialization failed ({err})")
    })
}

fn kernel_bias(kernel: &str) -> f64 {
    if kernel.contains("tile20_v3") || kernel.contains("v3") {
        10.0
    } else if kernel.contains("tile24") {
        8.0
    } else if kernel.contains("tile20") {
        6.5
    } else {
        5.0
    }
}

fn vector_bonus(vector_width: u32) -> f64 {
    match vector_width {
        1 => 0.0,
        2 => 0.8,
        4 => 1.6,
        8 => 2.1,
        x => 1.0 + f64::from(x).log2() * 0.25,
    }
}

fn estimate_score(kernel: &str, vector_width: u32, unroll_k: u32, local_size: [u32; 2]) -> f64 {
    let base = kernel_bias(kernel);
    let vec_term = vector_bonus(vector_width);
    let unroll_term = f64::from(unroll_k) * 0.12;

    let local_x = local_size[0];
    let local_y = local_size[1];
    let area = local_x.saturating_mul(local_y);
    let area_penalty = f64::from(area.abs_diff(100)) * 0.015;
    let warp_penalty = if area > 256 { f64::from(area - 256) * 0.01 } else { 0.0 };

    base + vec_term + unroll_term - area_penalty - warp_penalty
}

fn enumerate_candidates(request: &SearchRequest) -> Result<Vec<Candidate>, String> {
    if request.kernels.is_empty() {
        return Err("enumerate_candidates: kernels must not be empty".to_string());
    }
    if request.vector_widths.is_empty() {
        return Err("enumerate_candidates: vector_widths must not be empty".to_string());
    }
    if request.unroll_k_values.is_empty() {
        return Err("enumerate_candidates: unroll_k_values must not be empty".to_string());
    }
    if request.local_sizes.is_empty() {
        return Err("enumerate_candidates: local_sizes must not be empty".to_string());
    }

    let mut out = Vec::new();
    for kernel in &request.kernels {
        for &vector_width in &request.vector_widths {
            for &unroll_k in &request.unroll_k_values {
                for &local_size in &request.local_sizes {
                    let candidate_id = format!(
                        "{}_vw{}_u{}_l{}x{}",
                        kernel, vector_width, unroll_k, local_size[0], local_size[1]
                    );
                    let score = estimate_score(kernel, vector_width, unroll_k, local_size);
                    out.push(Candidate {
                        candidate_id,
                        kernel: kernel.clone(),
                        vector_width,
                        unroll_k,
                        local_size,
                        estimated_score: score,
                    });
                }
            }
        }
    }

    out.sort_by(|a, b| {
        b.estimated_score
            .total_cmp(&a.estimated_score)
            .then_with(|| a.candidate_id.cmp(&b.candidate_id))
    });

    let top_k = request.top_k.unwrap_or(out.len());
    let keep = top_k.min(out.len());
    out.truncate(keep);
    Ok(out)
}

fn build_replay_plan(
    candidates: &[Candidate],
    sessions: u32,
    runs: u32,
    base_seed: u64,
) -> Result<Vec<ReplayEntry>, String> {
    if candidates.is_empty() {
        return Err("build_replay_plan: candidates list must not be empty".to_string());
    }
    if sessions == 0 || runs == 0 {
        return Err("build_replay_plan: sessions and runs must be > 0".to_string());
    }

    let mut out = Vec::new();
    for (idx, candidate) in candidates.iter().enumerate() {
        let idx_base = u64::try_from(idx).unwrap_or(0) * 100_000;
        for session in 0..sessions {
            for run in 0..runs {
                let seed = base_seed + idx_base + u64::from(session) * 1_000 + u64::from(run);
                out.push(ReplayEntry {
                    candidate_id: candidate.candidate_id.clone(),
                    session,
                    run,
                    seed,
                    execution_tag: format!("{}_s{}_r{}", candidate.candidate_id, session, run),
                });
            }
        }
    }
    Ok(out)
}

#[cfg(feature = "python-module")]
#[pyfunction]
fn enumerate_candidates_json(request_json: &str) -> PyResult<String> {
    let request: SearchRequest =
        parse_json(request_json, "enumerate_candidates_json")
            .map_err(PyValueError::new_err)?;
    let candidates = enumerate_candidates(&request).map_err(PyValueError::new_err)?;
    to_pretty_json(&candidates, "enumerate_candidates_json").map_err(PyValueError::new_err)
}

#[cfg(feature = "python-module")]
#[pyfunction]
#[pyo3(signature = (candidates_json, sessions=5, runs=10, base_seed=42))]
fn build_replay_plan_json(
    candidates_json: &str,
    sessions: u32,
    runs: u32,
    base_seed: u64,
) -> PyResult<String> {
    let candidates: Vec<Candidate> =
        parse_json(candidates_json, "build_replay_plan_json")
            .map_err(PyValueError::new_err)?;
    let out = build_replay_plan(&candidates, sessions, runs, base_seed)
        .map_err(PyValueError::new_err)?;
    to_pretty_json(&out, "build_replay_plan_json").map_err(PyValueError::new_err)
}

#[cfg(feature = "python-module")]
#[pyfunction]
fn sidecar_info_json() -> PyResult<String> {
    let info = SidecarInfo {
        sidecar_name: "t2_rust_sidecar".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: vec![
            "candidate_enumeration".to_string(),
            "deterministic_replay_plan".to_string(),
        ],
    };
    to_pretty_json(&info, "sidecar_info_json").map_err(PyValueError::new_err)
}

#[cfg(feature = "python-module")]
#[pymodule]
fn t2_rust_sidecar(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(enumerate_candidates_json, m)?)?;
    m.add_function(wrap_pyfunction!(build_replay_plan_json, m)?)?;
    m.add_function(wrap_pyfunction!(sidecar_info_json, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> SearchRequest {
        SearchRequest {
            kernels: vec!["tile20_v3".to_string(), "tile24".to_string()],
            vector_widths: vec![4, 8],
            unroll_k_values: vec![0, 4],
            local_sizes: vec![[10, 10], [12, 12]],
            top_k: Some(4),
        }
    }

    #[test]
    fn enumerate_candidates_is_deterministic() {
        let req = sample_request();
        let a = enumerate_candidates(&req).expect("first run should pass");
        let b = enumerate_candidates(&req).expect("second run should pass");
        assert_eq!(a.len(), 4);
        assert_eq!(a, b);
    }

    #[test]
    fn replay_plan_has_expected_cardinality() {
        let req = sample_request();
        let candidates = enumerate_candidates(&req).expect("candidate generation should pass");
        let plan = build_replay_plan(&candidates, 3, 2, 42).expect("plan should build");
        assert_eq!(plan.len(), candidates.len() * 3 * 2);
    }
}
