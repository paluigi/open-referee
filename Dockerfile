# Open Referee — serve the PWA from a container.
# The image installs the package from the source tree (same artifact as PyPI).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY static ./static
COPY templates ./templates
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python .

FROM python:3.12-slim-bookworm
RUN useradd -m -u 1000 referee
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" OPEN_REFEREE_HOME=/data
USER referee
VOLUME /data
EXPOSE 8410
CMD ["open-referee", "serve", "--host", "0.0.0.0", "--port", "8410"]
