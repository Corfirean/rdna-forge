use forge_core::{
    match_profile_for_device, CapabilityResult, DiagnosticReport, ForgeError, ForgeResult,
    SupportStatus,
};
use forge_device::discover_all;
use forge_runtime::{
    default_untested_capabilities, probe_all_backends, probe_amf_capabilities,
    run_live_capability_checks, AmfAcceleration, AmfCodecDirection,
};
use serde::{Deserialize, Serialize};

/// Anonymized GitHub hardware compatibility report ready for `.github/ISSUE_TEMPLATE/hardware_report.md`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareReportOutput {
    pub issue_title: String,
    pub markdown: String,
    pub sanitized: bool,
}

fn replace_case_insensitive_literal(input: &str, needle: &str, replacement: &str) -> String {
    if needle.is_empty() {
        return input.to_string();
    }
    let lower_haystack = input.to_ascii_lowercase();
    let lower_needle = needle.to_ascii_lowercase();
    if !lower_haystack.contains(&lower_needle) {
        return input.to_string();
    }

    let mut out = String::with_capacity(input.len());
    let mut cursor = 0usize;
    while cursor < input.len() {
        let rem_lower = &lower_haystack[cursor..];
        let Some(rel_idx) = rem_lower.find(&lower_needle) else {
            out.push_str(&input[cursor..]);
            break;
        };
        let abs_idx = cursor + rel_idx;
        if input.is_char_boundary(abs_idx) && input.is_char_boundary(abs_idx + needle.len()) {
            out.push_str(&input[cursor..abs_idx]);
            out.push_str(replacement);
            cursor = abs_idx + needle.len();
        } else {
            out.push_str(&input[cursor..]);
            break;
        }
    }
    out
}

fn redact_users_path_segments(input: &str, marker: &str, sep: char) -> String {
    let lower = input.to_ascii_lowercase();
    if !lower.contains(marker) {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len());
    let mut cursor = 0usize;
    while cursor < input.len() {
        let rem_lower = &lower[cursor..];
        let Some(rel_idx) = rem_lower.find(marker) else {
            out.push_str(&input[cursor..]);
            break;
        };
        let marker_end = cursor + rel_idx + marker.len();
        if !input.is_char_boundary(marker_end) {
            out.push_str(&input[cursor..]);
            break;
        }
        out.push_str(&input[cursor..marker_end]);
        let tail = &input[marker_end..];
        let seg_len = tail
            .find([sep, '\\', '/', '"', '\'', '\r', '\n', '`'])
            .unwrap_or(tail.len());
        if seg_len > 0 {
            out.push_str("<USER>");
            cursor = marker_end + seg_len;
        } else {
            cursor = marker_end;
        }
    }
    out
}

fn is_boundary_char(c: char) -> bool {
    matches!(
        c,
        '\\' | '/'
            | '"'
            | '\''
            | ' '
            | '\t'
            | '\r'
            | '\n'
            | ':'
            | '@'
            | '('
            | ')'
            | ','
            | ';'
            | '`'
    )
}

fn redact_bounded_username(input: &str, username: &str) -> String {
    let trimmed = username.trim();
    if trimmed.len() < 3 {
        return input.to_string();
    }
    let lower_user = trimmed.to_ascii_lowercase();
    if matches!(
        lower_user.as_str(),
        "amd" | "gpu" | "bin" | "win" | "tmp" | "usr" | "user" | "root" | "admin" | "test"
    ) {
        return input.to_string();
    }

    let lower_haystack = input.to_ascii_lowercase();
    if !lower_haystack.contains(&lower_user) {
        return input.to_string();
    }

    let mut out = String::with_capacity(input.len());
    let mut cursor = 0usize;
    while cursor < input.len() {
        let rem_lower = &lower_haystack[cursor..];
        let Some(rel_idx) = rem_lower.find(&lower_user) else {
            out.push_str(&input[cursor..]);
            break;
        };
        let abs_idx = cursor + rel_idx;
        let end_idx = abs_idx + lower_user.len();
        if !input.is_char_boundary(abs_idx) || !input.is_char_boundary(end_idx) {
            out.push_str(&input[cursor..]);
            break;
        }
        let prev_ok = input[..abs_idx]
            .chars()
            .next_back()
            .is_none_or(is_boundary_char);
        let next_ok = input[end_idx..].chars().next().is_none_or(is_boundary_char);
        if prev_ok && next_ok {
            out.push_str(&input[cursor..abs_idx]);
            out.push_str("<USER>");
        } else {
            out.push_str(&input[cursor..end_idx]);
        }
        cursor = end_idx;
    }
    out
}

/// Strips Windows `%USERPROFILE%`, `\Users\<name>\`, `/Users/<name>/`, and bounded `%USERNAME%`
/// strings from diagnostic text so hardware reports contain zero personal user paths.
pub fn sanitize_report_text_with_env(
    input: &str,
    userprofile: Option<&str>,
    username: Option<&str>,
) -> String {
    let mut text = input.to_string();
    if let Some(up) = userprofile.map(str::trim).filter(|s| !s.is_empty()) {
        text = replace_case_insensitive_literal(&text, up, r"C:\Users\<USER>");
        let forward = up.replace('\\', "/");
        text = replace_case_insensitive_literal(&text, &forward, "C:/Users/<USER>");
        let escaped = up.replace('\\', "\\\\");
        text = replace_case_insensitive_literal(&text, &escaped, r"C:\\Users\\<USER>");
    }
    text = redact_users_path_segments(&text, r":\\users\\", '\\');
    text = redact_users_path_segments(&text, r":\users\", '\\');
    text = redact_users_path_segments(&text, ":/users/", '/');
    if let Some(un) = username {
        text = redact_bounded_username(&text, un);
    }
    text
}

/// Strips Windows local username and home-directory paths from generated report text.
pub fn sanitize_report_text(input: &str) -> String {
    let userprofile = std::env::var("USERPROFILE").ok();
    let username = std::env::var("USERNAME").ok();
    sanitize_report_text_with_env(input, userprofile.as_deref(), username.as_deref())
}

/// Generates an anonymized GitHub hardware compatibility report matching
/// `.github/ISSUE_TEMPLATE/hardware_report.md` (with optional live GPU kernel verification).
pub fn generate_github_hardware_report(verify: bool) -> ForgeResult<HardwareReportOutput> {
    let snapshot = discover_all();
    let (backends, missing_dependencies) = probe_all_backends(&snapshot, None);
    let amf_caps = probe_amf_capabilities().ok();

    let mut capabilities = vec![CapabilityResult {
        name: "Device discovery".to_string(),
        status: if snapshot.gpus.is_empty() {
            SupportStatus::Unsupported
        } else {
            SupportStatus::Verified
        },
        detail: Some(format!("{} GPU(s) enumerated", snapshot.gpus.len())),
    }];
    if verify {
        capabilities.extend(run_live_capability_checks(&snapshot.gpus));
    } else {
        capabilities.extend(default_untested_capabilities());
    }

    let report = DiagnosticReport {
        system: snapshot.system,
        gpus: snapshot.gpus,
        backends,
        capabilities,
        missing_dependencies,
        windows_audit: Some(snapshot.windows_audit),
    };

    let primary = report.gpus.first();
    let gfx_str = primary
        .and_then(|g| g.gfx_target.as_ref().map(|t| t.base.as_str()))
        .unwrap_or("gfxXXXX");
    let gpu_name = primary
        .map(|g| g.name.as_str())
        .unwrap_or("Unknown AMD GPU");
    let arch_str = primary
        .map(|g| g.architecture.to_string())
        .unwrap_or_else(|| "Unknown".to_string());
    let matched_str = primary
        .and_then(match_profile_for_device)
        .map(|m| format!("{} (`{}`)", m.profile.device.name, m.id))
        .unwrap_or_else(|| "Generic AMD (unmatched)".to_string());
    let vram_gib = primary
        .map(|g| {
            format!(
                "{:.2} GiB",
                g.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            )
        })
        .unwrap_or_else(|| "n/a".to_string());
    let rebar_str = primary
        .and_then(|g| g.rebar_enabled)
        .map(|b| {
            if b {
                "ENABLED"
            } else {
                "DISABLED (256 MiB BAR)"
            }
        })
        .unwrap_or("Unknown");
    let driver_str = primary
        .map(
            |g| match (g.adrenalin_version.as_deref(), g.driver_version.as_deref()) {
                (Some(a), Some(d)) => format!("Adrenalin {a} ({d})"),
                (Some(a), None) => format!("Adrenalin {a}"),
                (None, Some(d)) => d.to_string(),
                (None, None) => "Unknown".to_string(),
            },
        )
        .unwrap_or_else(|| "Unknown".to_string());

    let runtime_summary = report
        .backends
        .iter()
        .map(|b| {
            let ver = b.version.as_deref().unwrap_or("--");
            format!("{}={} ({})", b.kind, b.status, ver)
        })
        .collect::<Vec<_>>()
        .join(", ");

    let mut md = String::new();
    md.push_str("## Hardware & Software\n\n");
    md.push_str(&format!("- **GPU Model**: {gpu_name}\n"));
    md.push_str(&format!("- **Matched Built-in Profile**: {matched_str}\n"));
    md.push_str(&format!(
        "- **Architecture & `gfx` Target**: {arch_str} / `{gfx_str}`\n"
    ));
    md.push_str(&format!("- **VRAM**: {vram_gib}\n"));
    if let Some(cus) = primary.and_then(|g| g.compute_units) {
        md.push_str(&format!(
            "- **Compute Units**: {cus} CUs ({} WGPs)\n",
            primary.and_then(|g| g.wgp_count).unwrap_or(cus / 2)
        ));
    }
    md.push_str(&format!("- **Resizable BAR Status**: {rebar_str}\n"));
    md.push_str(&format!(
        "- **Windows Build**: {} ({}, Build {})\n",
        report.system.os_name, report.system.os_version, report.system.os_build
    ));
    md.push_str(&format!(
        "- **CPU / System RAM**: {} ({} threads) / {:.2} GiB\n",
        report.system.cpu_name,
        report.system.logical_cores,
        report.system.total_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    ));
    md.push_str(&format!("- **Adrenalin Driver Version**: {driver_str}\n"));
    md.push_str(&format!("- **Runtimes**: {runtime_summary}\n\n"));

    if let Some(ref audit) = report.windows_audit {
        md.push_str("## Windows GPU Health Audit\n\n");
        md.push_str(&format!(
            "- **HAGS (`HwSchMode`)**: {}\n",
            match audit.hags_enabled {
                Some(true) => format!("ENABLED (`HwSchMode={}`)", audit.hags_mode.unwrap_or(2)),
                Some(false) => format!("DISABLED (`HwSchMode={}`)", audit.hags_mode.unwrap_or(1)),
                None => "Windows default (unset)".to_string(),
            }
        ));
        md.push_str(&format!(
            "- **WDDM `TdrDelay`**: {}s{}\n",
            audit.effective_tdr_delay_secs,
            if audit.tdr_delay_secs.is_some() {
                " (registry override)"
            } else {
                " (Windows default)"
            }
        ));
        md.push_str(&format!(
            "- **`LongPathsEnabled`**: {}\n",
            if audit.long_paths_enabled == Some(true) {
                "ENABLED"
            } else {
                "DISABLED"
            }
        ));
        if !audit.advisories.is_empty() {
            md.push_str("\nAdvisories:\n");
            for adv in &audit.advisories {
                md.push_str(&format!("- {adv}\n"));
            }
        }
        md.push('\n');
    }

    if let Some(ref amf) = amf_caps {
        let hw_enc: Vec<&str> = amf
            .codecs
            .iter()
            .filter(|c| {
                c.direction == AmfCodecDirection::Encode
                    && c.acceleration == AmfAcceleration::Hardware
            })
            .map(|c| c.codec.as_str())
            .collect();
        let hw_dec: Vec<&str> = amf
            .codecs
            .iter()
            .filter(|c| {
                c.direction == AmfCodecDirection::Decode
                    && c.acceleration == AmfAcceleration::Hardware
            })
            .map(|c| c.codec.as_str())
            .collect();
        md.push_str("## AMF Hardware Video Capabilities\n\n");
        md.push_str(&format!("- **AMF Runtime**: `{}`\n", amf.runtime_version));
        md.push_str(&format!(
            "- **Hardware Encoders**: {}\n",
            if hw_enc.is_empty() {
                "None".to_string()
            } else {
                hw_enc.join(", ")
            }
        ));
        md.push_str(&format!(
            "- **Hardware Decoders**: {}\n\n",
            if hw_dec.is_empty() {
                "None".to_string()
            } else {
                hw_dec.join(", ")
            }
        ));
    }

    md.push_str("## Verification Commands Run\n\n```powershell\n");
    if verify {
        md.push_str("forge report --verify\n");
    } else {
        md.push_str("forge report\n");
    }
    md.push_str("```\n\n## Observed Results\n\n");
    md.push_str("| Capability Check | Status | Detail |\n| :--- | :--- | :--- |\n");
    for cap in &report.capabilities {
        let detail = cap.detail.as_deref().unwrap_or("--").replace('|', "\\|");
        md.push_str(&format!("| {} | {} | {} |\n", cap.name, cap.status, detail));
    }
    md.push('\n');

    let diag_json = serde_json::to_string_pretty(&report)
        .map_err(|e| ForgeError::ParseError(format!("failed to serialize report JSON: {e}")))?;
    md.push_str("<details>\n<summary>Sanitized DiagnosticReport JSON</summary>\n\n```json\n");
    md.push_str(&diag_json);
    md.push_str("\n```\n\n</details>\n");

    let raw_title = format!("compat({gfx_str}): {gpu_name} hardware verification report");
    Ok(HardwareReportOutput {
        issue_title: sanitize_report_text(&raw_title),
        markdown: sanitize_report_text(&md),
        sanitized: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizer_strips_windows_usernames_across_path_variants() {
        let sample = concat!(
            "Model: C:\\Users\\Alice Example\\models\\qwen.gguf\n",
            "Forward: d:/users/Alice Example/cache/llama\n",
            "Json: C:\\\\Users\\\\Alice Example\\\\.lmstudio\\\\cfg.json\n",
            "ShortUser: Hello from Bob_Dev on C:\\Users\\Bob_Dev\\Desktop and \\Users\\Bob_Dev\\!\n",
        );
        let sanitized =
            sanitize_report_text_with_env(sample, Some(r"C:\Users\Alice Example"), Some("Bob_Dev"));

        assert!(!sanitized.contains("Alice Example"));
        assert!(!sanitized.contains("Bob_Dev"));
        assert!(sanitized.contains(r"C:\Users\<USER>\models\qwen.gguf"));
        assert!(sanitized.contains("d:/users/<USER>/cache/llama"));
        assert!(sanitized.contains(r"C:\\Users\\<USER>\\.lmstudio\\cfg.json"));
        assert!(sanitized.contains("Hello from <USER>"));

        // Ensure short/generic tokens inside words (like "amd" or "user") are never mangled.
        let benign = "AMD Radeon RX 6900 XT vulkan_benchmark user_space";
        assert_eq!(
            sanitize_report_text_with_env(benign, None, Some("amd")),
            benign
        );
    }
}
