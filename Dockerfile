# LlamaForge-ROCm — Docker image with ROCm/HIP llama.cpp + the LlamaForge backend.
#
# Builds llama.cpp with GGML_HIP=ON for a configurable set of AMD GPU targets
# (build arg AMDGPU_TARGETS), then installs the pure-stdlib LlamaForge backend.
# CUDA/CPU/Metal remain supported by the upstream project; this image is the
# ROCm variant for AMD GPU hosts (Unraid, or any Docker host with amdgpu).
#
# Build examples:
#   docker build -t llamaforge-rocm .                                  # broad default targets
#   docker build --build-arg AMDGPU_TARGETS=gfx1030 -t llamaforge-rocm .  # one GPU, faster build
#   docker build --build-arg LLAMACPP_REF=master -t llamaforge-rocm .    # pin a llama.cpp ref

# ---- stage 1: build llama.cpp with HIP ----
FROM rocm/dev-ubuntu-22.04:6.3.3 AS build

ARG LLAMACPP_REF=master
# Broad-but-sane default; narrow this to your GPU(s) for a much faster build.
ARG AMDGPU_TARGETS="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1010;gfx1030;gfx1100;gfx1101;gfx1102"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMACPP_REF}" https://github.com/ggml-org/llama.cpp.git .

# HIP build. AMDGPU_TARGETS is the configurable multi-arch list; GGML_HIPBLAS
# is the modern backend name (GGML_HIP is accepted as an alias on older tags).
RUN cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIP=ON \
        -DGGML_HIPBLAS=ON \
        -DAMDGPU_TARGETS="${AMDGPU_TARGETS}" \
        -DGGML_NATIVE=ON \
    && cmake --build build --config Release --parallel "$(nproc)"

# ---- stage 2: runtime ----
FROM rocm/dev-ubuntu-22.04:6.3.3 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 lsof \
    && rm -rf /var/lib/apt/lists/*

# llama-server binary + its shared libs from the build stage.
COPY --from=build /src/build/bin/llama-server /usr/local/bin/llama-server

# LlamaForge backend (pure stdlib) + web UI.
WORKDIR /app
COPY backend ./backend
COPY web ./web
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh

# Two volumes: config (config.json, models.ini, logs) and models (GGUF files).
VOLUME ["/app/config", "/app/models"]

# llama.cpp router + LlamaForge dashboard.
EXPOSE 8080 8090

ENTRYPOINT ["/app/docker-entrypoint.sh"]
