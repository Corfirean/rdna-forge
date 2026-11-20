#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

use commands::{
    apply_env_and_ollama_patch, apply_lm_studio_patch, generate_hardware_report, get_diagnostics,
    get_llm_assets, get_llm_plan, get_llm_recommendations, get_telemetry_snapshot,
    get_top_vram_processes, get_tuning_report, get_video_caps, reset_tuning, rollback_env_patch,
    rollback_lm_studio_patch, run_live_verification, run_llm_autotune, run_video_benchmark,
    run_vulkan_benchmark, set_power_limit, AppState,
};
use tracing_subscriber::EnvFilter;

fn main() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();

    tauri::Builder::default()
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            get_diagnostics,
            run_live_verification,
            generate_hardware_report,
            get_telemetry_snapshot,
            get_top_vram_processes,
            get_tuning_report,
            set_power_limit,
            reset_tuning,
            get_video_caps,
            run_video_benchmark,
            run_vulkan_benchmark,
            get_llm_assets,
            get_llm_plan,
            get_llm_recommendations,
            run_llm_autotune,
            apply_lm_studio_patch,
            apply_env_and_ollama_patch,
            rollback_lm_studio_patch,
            rollback_env_patch,
        ])
        .run(tauri::generate_context!())
        .expect("error while running RDNA Forge Desktop");
}
