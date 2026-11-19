use thiserror::Error;

/// Canonical error type for all RDNA Forge library crates.
#[derive(Debug, Error)]
pub enum ForgeError {
    #[error("invalid gfx target '{0}': expected format like 'gfx1030'")]
    InvalidGfxTarget(String),

    #[error("invalid hardware profile: {0}")]
    InvalidProfile(String),

    #[error("no suitable GPU device found: {0}")]
    DeviceNotFound(String),

    #[error("unsupported GPU capability: {0}")]
    UnsupportedCapability(String),

    #[error("invalid tensor or dispatch dimension: {0}")]
    InvalidDimension(String),

    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("requested buffer size ({requested_bytes} bytes) exceeds device maxBufferSize ({max_bytes} bytes)")]
    BufferTooLarge {
        requested_bytes: u64,
        max_bytes: u64,
    },

    #[error(
        "insufficient VRAM: requested {requested_bytes} bytes, available {available_bytes} bytes"
    )]
    InsufficientVram {
        requested_bytes: u64,
        available_bytes: u64,
    },

    #[error("required runtime '{runtime}' is unavailable: {reason}")]
    RuntimeUnavailable { runtime: String, reason: String },

    #[error("GPU execution failed: {0}")]
    ExecutionFailed(String),

    #[error("failed to parse external output: {0}")]
    ParseError(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type ForgeResult<T> = Result<T, ForgeError>;
