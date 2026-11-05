# Stage 1: Build static / optimized binaries
FROM rust:1.80-slim-bookworm AS builder

WORKDIR /build

# Install compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace sources
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build release CLI binary
RUN cargo build --release --bin llmzip

# Stage 2: Minimal runtime container
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy compiled binary from builder stage
COPY --from=builder /build/target/release/llmzip /usr/local/bin/llmzip

# Non-root user execution
RUN useradd -u 1000 -m llmzipuser
USER llmzipuser

ENTRYPOINT ["llmzip"]
CMD ["--help"]
