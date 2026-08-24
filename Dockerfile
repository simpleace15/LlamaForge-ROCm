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
# -complete carries the full ROCm toolchain (hipcc/hipconfig) needed to compile;
# the plain tag is runtime-only and cannot build. Matches llama.cpp's own
# .devops/rocm.Dockerfile (24.04 + ROCm 7.2.1).
FROM rocm/dev-ubuntu-24.04:7.2.1-complete AS build

ARG LLAMACPP_REF=master
# Broad-but-sane default, matching llama.cpp's own ROCM_DOCKER_ARCH. Narrow this
# to your GPU(s) for a much faster build. (Vega gfx900/gfx906 are omitted —
# ROCm 7.x dropped them; use a 6.x base if you need those.)
ARG AMDGPU_TARGETS="gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMACPP_REF}" https://github.com/ggml-org/llama.cpp.git .

# HIP build. AMDGPU_TARGETS is the configurable multi-arch list. HIPCXX/HIP_PATH
# are set explicitly (matching llama.cpp's own .devops/rocm.Dockerfile) so the
# build uses the base image's HIP clang rather than relying on PATH defaults.
# GGML_BACKEND_DL=ON builds the backends as dynamically-loaded .so files, which
# the runtime stage copies alongside the binary.
RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS="${AMDGPU_TARGETS}" \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_NATIVE=ON \
    && cmake --build build --config Release --parallel "$(nproc)"

# Collect the shared libs (HIP backend .so + any transitive deps) so the
# runtime stage can copy them in one pass.
RUN mkdir -p /src/lib \
    && find build -name "*.so*" -exec cp -P {} /src/lib \;

# ---- stage 2: runtime ----
FROM rocm/dev-ubuntu-24.04:7.2.1 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 lsof libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# llama-server binary + its shared libs from the build stage. With
# GGML_BACKEND_DL=ON the HIP backend is a .so that llama-server loads at
# runtime from its own directory, so the .so files must sit next to the binary
# (the official rocm.Dockerfile does the same — binary and .so in one dir).
COPY --from=build /src/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /src/lib/ /usr/local/bin/

# The .so files (libllama-server-impl.so, libggml-hip.so, ...) live in
# /usr/local/bin next to the binary, but the dynamic linker does not search
# there by default. Without this, llama-server fails at startup with
# "libllama-server-impl.so: cannot open shared object file".
ENV LD_LIBRARY_PATH=/usr/local/bin

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
