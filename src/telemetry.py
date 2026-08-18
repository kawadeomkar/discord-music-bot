import logging
import os
import sys
from typing import Optional, TYPE_CHECKING
from collections.abc import Sequence

if TYPE_CHECKING:
    from multiprocessing.queues import Queue

import structlog
from structlog.typing import EventDict, WrappedLogger
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_ON,
    Decision,
    Sampler,
    SamplingResult,
)
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import Link, SpanKind
from opentelemetry.trace.span import TraceState
from opentelemetry.util.types import Attributes

from src.config import ENVIRONMENT

if TYPE_CHECKING:
    # The SDK providers are imported lazily in _setup_traces/_setup_logs so the
    # exporter stack loads only when telemetry is enabled; these type-only imports let
    # the globals below keep their real types.
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk.trace import TracerProvider

_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "discord-music-bot")
_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

_tracer_provider: Optional["TracerProvider"] = None
_log_provider: Optional["LoggerProvider"] = None

# discord.py makes these startup HTTP calls with no user-visible parent; suppress
# them so Tempo isn't cluttered with orphaned root spans. Confirmed from live traces.
_DISCORD_INTERNAL_URL_PATTERNS = (
    "gateway.discord.gg",  # WebSocket gateway hostname
    "/api/v10/gateway/bot",  # HTTP pre-flight to resolve gateway URL
    "/oauth2/applications/@me",  # older token validation path
    "/api/v10/users/@me",  # current token validation path
    "discord.gg/?v=",  # WebSocket connection URL fragment
)


class _DiscordGatewayFilter(Sampler):
    """Drop discord.py-internal aiohttp spans that have no user-visible parent."""

    def __init__(self, inner: Sampler = ALWAYS_ON) -> None:
        self._inner = inner

    def should_sample(
        self,
        parent_context: Optional[Context],
        trace_id: int,
        name: str,
        kind: Optional[SpanKind] = None,
        attributes: Attributes = None,
        links: Optional[Sequence[Link]] = None,
        trace_state: Optional[TraceState] = None,
    ) -> SamplingResult:
        url = str((attributes or {}).get("http.url", ""))
        if any(p in url for p in _DISCORD_INTERNAL_URL_PATTERNS):
            return SamplingResult(Decision.DROP)
        return self._inner.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return "_DiscordGatewayFilter"


def setup_telemetry() -> None:
    """Initialize OTel SDK and structlog. No-op when OTEL_SDK_DISABLED=true. A second
    call is a no-op too: it would add a second root LoggingHandler (duplicate Loki
    records) and orphan the first TracerProvider's exporter.
    """
    global _tracer_provider
    if _tracer_provider is not None:
        return
    _configure_structlog()
    if os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true":
        return
    _setup_traces()
    _setup_logs()
    _setup_auto_instrumentation()


def shutdown_telemetry() -> None:
    """Flush and shut down OTel exporters. Called from MusicBotApp.close() via executor."""
    if _tracer_provider is not None:
        _tracer_provider.force_flush()
        _tracer_provider.shutdown()
    if _log_provider is not None:
        _log_provider.force_flush()
        _log_provider.shutdown()


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def configure_worker_logging(log_queue: Optional["Queue"] = None) -> None:
    """Configure structured logging inside a yt-dlp pool worker (called via
    ytdlp_pool._worker_init, which wraps this so a failure can't break the pool). With a
    log_queue (production; the thread-pool test seam passes none), the stdout
    StreamHandler is REPLACED by a QueueHandler — the parent re-emits every record it
    drains, so keeping it would double-print — and worker_id is bound as a contextvar.
    yt-dlp's warnings thus reach Loki through the parent's LoggingHandler with worker_id
    and, via run()'s context propagation, the originating trace_id."""
    _configure_structlog()
    if log_queue is None:
        return
    import logging.handlers
    import multiprocessing

    root = logging.root
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.INFO)
    structlog.contextvars.bind_contextvars(
        worker_id=multiprocessing.current_process().name
    )


# ── Internal setup ────────────────────────────────────────────────────────────


def _add_otel_context(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """Structlog processor: inject trace_id and span_id into every log event."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def _add_environment(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """Structlog processor: stamp every log event with the current environment."""
    event_dict["environment"] = ENVIRONMENT
    return event_dict


def setup_cli_logging() -> None:
    """Structured logging for the one-shot operator CLIs, without the OTel SDK.
    `src.util.get_logger` hands back a structlog proxy that is inert until something
    configures structlog, and only this module does; unconfigured it falls back to
    PrintLoggerFactory — readable text on stdout that never enters the logging module,
    so `2> errors.log` captures nothing. Deliberately not setup_telemetry(): a hand-run
    one-shot needs no TracerProvider or OTLP flush, and the SDK's LoggingHandler is
    deprecated upstream, so importing it under `filterwarnings = error` fails the suite.
    """
    _configure_structlog()


def _configure_structlog() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # guild_id, command, etc.
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            _add_environment,  # environment: production | staging | development
            _add_otel_context,  # trace_id, span_id
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    # Always add a stdout StreamHandler so logs reach `docker logs` and local dev even
    # with the SDK off — under OTEL_SDK_DISABLED=true nothing else adds a handler, so
    # without this all output is silently dropped.
    if not any(isinstance(h, logging.StreamHandler) for h in logging.root.handlers):
        logging.root.addHandler(logging.StreamHandler(sys.stdout))
    logging.root.setLevel(logging.INFO)


def _setup_traces() -> None:
    global _tracer_provider
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    resource = Resource.create(
        {
            SERVICE_NAME: _SERVICE_NAME,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: ENVIRONMENT,
        }
    )
    exporter = OTLPSpanExporter(endpoint=_OTLP_ENDPOINT, insecure=True)
    provider = TracerProvider(resource=resource, sampler=_DiscordGatewayFilter())
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer_provider = provider


def _setup_logs() -> None:
    global _log_provider
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

    resource = Resource.create(
        {
            SERVICE_NAME: _SERVICE_NAME,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: ENVIRONMENT,
        }
    )
    exporter = OTLPLogExporter(endpoint=_OTLP_ENDPOINT, insecure=True)
    provider = LoggerProvider(resource=resource)
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    _log_provider = provider

    # Bridge: stdlib root logger → OTel log records → Loki.
    # structlog routes through stdlib via LoggerFactory, so this captures everything.
    handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
    logging.root.addHandler(handler)


def _setup_auto_instrumentation() -> None:
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor

    RedisInstrumentor().instrument()
    AioHttpClientInstrumentor().instrument()
