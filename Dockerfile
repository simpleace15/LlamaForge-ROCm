# LlamaForge-ROCm — Docker image with ROCm/HIP + Vulkan llama.cpp + the backend.
#
# Builds llama.cpp with BOTH GGML_HIP=ON and GGML_VULKAN=ON into a single
# llama-server binary (default), so one instance can switch between ROCm and
# Vulkan at runtime via `--device HIP0,...` / `--device Vulkan0,...` — no
# rebuild, no second container. The AMD_BACKEND build arg narrows to a single
# backend for a smaller/faster build when you only ever want one.
#
# Build examples:
#   docker build -t llamaforge-rocm .                                  # dual-backend (HIP + Vulkan)
#   docker build --build-arg AMDGPU_TARGETS=gfx1030 -t llamaforge-rocm .  # narrow HIP targets, faster build
#   docker build --build-arg AMD_BACKEND=rocm -t llamaforge-rocm .       # HIP only
#   docker build --build-arg AMD_BACKEND=vulkan -t llamaforge-vulkan .   # Vulkan only
#   docker build --build-arg LLAMACPP_REF=master -t llamaforge-rocm .    # pin a llama.cpp ref

# ---- stage 1: build llama.cpp ----
# The ROCm toolchain image (-complete) carries hipcc/hipconfig needed to
# compile the HIP backend, and matches llama.cpp's own .devops/rocm.Dockerfile
# (24.04 + ROCm 7.2.1). The Vulkan backend needs only the Vulkan headers +
# loader, which are installed on top. Building both backends into one binary
# is what enables the runtime ROCm/Vulkan toggle.
FROM rocm/dev-ubuntu-24.04:7.2.1-complete AS build

# "both" (default) = HIP + Vulkan in one binary; "rocm" = HIP only; "vulkan" =
# Vulkan only. Re-declared after FROM so the RUN steps can branch on it.
ARG AMD_BACKEND=both
ARG LLAMACPP_REF=master
# Broad-but-sane default, matching llama.cpp's own ROCM_DOCKER_ARCH. Narrow this
# to your GPU(s) for a much faster build. (Vega gfx900/gfx906 are omitted —
# ROCm 7.x dropped them; use a 6.x base if you need those.) HIP-only: the
# Vulkan backend ignores AMDGPU_TARGETS.
ARG AMDGPU_TARGETS="gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build python3 \
    && rm -rf /var/lib/apt/lists/*

# Vulkan headers + loader + shader compiler, needed to compile/link the Vulkan
# backend. glslc is the Vulkan shader compiler (its own package, NOT in
# libvulkan-dev); spirv-headers is required by FindVulkan.cmake
# (SPIRV-Headers_DIR). Installed whenever Vulkan is in the build (both/vulkan).
RUN if [ "${AMD_BACKEND}" != "rocm" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            libvulkan-dev vulkan-tools mesa-vulkan-drivers glslc spirv-headers \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMACPP_REF}" https://github.com/ggml-org/llama.cpp.git .

# Assemble the backend flags. HIPCXX/HIP_PATH are set explicitly (matching
# llama.cpp's own .devops/rocm.Dockerfile) so the HIP build uses the base
# image's HIP clang rather than relying on PATH defaults. GGML_BACKEND_DL=ON
# builds the backends as dynamically-loaded .so files, which the runtime stage
# copies alongside the binary. The RADV Vulkan driver is installed in the
# runtime stage (mesa-vulkan-drivers); /dev/dri passthrough only exposes the
# device nodes, not the driver itself.
RUN BACKEND_FLAGS="" \
    && if [ "${AMD_BACKEND}" != "vulkan" ]; then \
         BACKEND_FLAGS="-DGGML_HIP=ON -DAMDGPU_TARGETS=${AMDGPU_TARGETS}"; \
       fi \
    && if [ "${AMD_BACKEND}" != "rocm" ]; then \
         BACKEND_FLAGS="${BACKEND_FLAGS} -DGGML_VULKAN=ON"; \
       fi \
    && HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
       cmake -B build -S . \
           -DCMAKE_BUILD_TYPE=Release \
           ${BACKEND_FLAGS} \
           -DGGML_BACKEND_DL=ON \
           -DGGML_CPU_ALL_VARIANTS=ON \
           -DGGML_NATIVE=ON \
    && cmake --build build --config Release --parallel "$(nproc)"

# Collect the shared libs (backend .so + any transitive deps) so the
# runtime stage can copy them in one pass.
RUN mkdir -p /src/lib \
    && find build -name "*.so*" -exec cp -P {} /src/lib \;

# ---- stage 2: runtime ----
# The ROCm toolchain image is kept for the runtime so rocm-smi is present
# (perf-level pinning + telemetry); the Vulkan loader + RADV driver are added
# on top so the dual-backend binary can also use the Vulkan backend.
FROM rocm/dev-ubuntu-24.04:7.2.1-complete AS runtime

ARG AMD_BACKEND=both

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 lsof libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Vulkan loader (libvulkan1) + the RADV driver so the Vulkan backend can reach
# the GPU. mesa-vulkan-drivers ships the RADV ICD (radeon_icd.x86_64.json) and
# libvulkan_radeon.so — these must be INSIDE the container; /dev/dri passthrough
# does NOT provide the driver. Installed whenever Vulkan is in the build.
RUN if [ "${AMD_BACKEND}" != "rocm" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            libvulkan1 vulkan-tools mesa-vulkan-drivers \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# llama-server binary + its shared libs from the build stage. With
# GGML_BACKEND_DL=ON the backends are .so files that llama-server loads at
# runtime from its own directory, so the .so files must sit next to the binary
# (the official rocm.Dockerfile does the same — binary and .so in one dir).
COPY --from=build /src/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /src/lib/ /usr/local/bin/

# The .so files (libllama-server-impl.so, libggml-hip.so, libggml-vulkan.so, ...)
# live in /usr/local/bin next to the binary, but the dynamic linker does not
# search there by default. Without this, llama-server fails at startup with
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
