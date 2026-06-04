# ==========================================================
# Stage 1: Build Java 25 & GraalVM AOT Shared Library (.so)
# ==========================================================
FROM ubuntu:24.04 AS builder

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install core build tools, zlib development headers and Maven
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    zlib1g-dev \
    maven \
    && rm -rf /var/lib/apt/lists/*

# Detect host architecture and download Oracle GraalVM JDK 25
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        GRAAL_ARCH="x64"; \
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        GRAAL_ARCH="aarch64"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    echo "Downloading GraalVM JDK 25 for Linux $GRAAL_ARCH..." && \
    curl -LfsSo /tmp/graalvm.tar.gz "https://download.oracle.com/graalvm/25/latest/graalvm-jdk-25_linux-${GRAAL_ARCH}_bin.tar.gz" && \
    mkdir -p /opt/graalvm && \
    tar -xzf /tmp/graalvm.tar.gz -C /opt/graalvm --strip-components=1 && \
    rm /tmp/graalvm.tar.gz

# Set up GraalVM JDK 25 environment variables
ENV JAVA_HOME=/opt/graalvm
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /build

# Copy Maven Project file
COPY pom.xml .

# Pre-resolve dependencies to cache them in Docker layer
RUN mvn dependency:go-offline -B || true

# Copy source code
COPY src/ src/

# Run Maven compile & package (which triggers native-image compilation)
# JUnit tests will run during "test" phase using SIMD Vector API on JVM
RUN mvn clean package

# ==========================================================
# Stage 2: C Client Testing Environment
# ==========================================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies for testing the C application
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled shared library and the exported C headers from builder
COPY --from=builder /build/target/lunar_core.so ./liblunar_core.so
COPY --from=builder /build/target/lunar_core.h .
COPY --from=builder /build/target/graal_isolate.h .
COPY test_client.c .

# Compile test client, linking it to our newly built liblunar_core.so
RUN gcc -o test_client test_client.c -I. -L. -llunar_core -Wl,-rpath,.

# Execute the test client by default
CMD ["./test_client"]
