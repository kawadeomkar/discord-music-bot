import functools
import inspect
import logging
import os
import sys
from typing import Optional

import structlog
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_ON,
    Decision,
    Sampler,
    SamplingResult,
)

_SDK_DISABLED = os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true"
_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "discord-music-bot")
_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

_tracer_provider: Optional[object] = None
_log_provider: Optional[object] = None

# discord.py makes these HTTP calls during startup with no user-visible parent.
# Suppress them to avoid cluttering Tempo with orphaned root spans.
# Patterns confirmed from live traces: gateway URL resolution and token validation.
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
        parent_context,
        trace_id,
        name,
        kind,
        attributes=None,
        links=None,
        trace_state=None,
    ):
        url = str((attributes or {}).get("http.url", ""))
        if any(p in url for p in _DISCORD_INTERNAL_URL_PATTERNS):
            return SamplingResult(Decision.DROP)
        return self._inner.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return "_DiscordGatewayFilter"


def setup_telemetry() -> None:
    """Initialize OTel SDK and structlog. No-op when OTEL_SDK_DISABLED=true.

    Guarded against double-call: a second call is a no-op. This matters because
    calling setup_telemetry() twice would add a second LoggingHandler to the root
    logger (duplicate Loki records) and orphan the first TracerProvider's exporter.
    """
    global _tracer_provider
    if _tracer_provider is not None:
        return
    _configure_structlog()
    if _SDK_DISABLED:
        return
    _setup_traces()
    _setup_logs()
    _setup_auto_instrumentation()


def shutdown_telemetry() -> None:
    """Flush and shut down OTel exporters. Called from MusicBotApp.close() via executor."""
    if _tracer_provider is not None:
        _tracer_provider.force_flush()  # type: ignore[union-attr]
        _tracer_provider.shutdown()  # type: ignore[union-attr]
    if _log_provider is not None:
        _log_provider.force_flush()  # type: ignore[union-attr]
        _log_provider.shutdown()  # type: ignore[union-attr]


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def traced(
    func=None,
    *,
    name=None,
    attributes=None,
    record_exceptions=True,
    set_status_on_exception=True,
):
    """Wrap a sync or async function in an OTel span.

    For functions that swallow exceptions (catch without re-raising), call
    trace.get_current_span().record_exception() / .set_status() manually inside
    the except block — the decorator only records exceptions that propagate out.
    """

    def decorator(fn):
        span_name = name or f"{fn.__module__}.{fn.__qualname__}"
        static_attrs = attributes or {}
        tracer = get_tracer(fn.__module__)

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name, attributes=static_attrs):
                    try:
                        return await fn(*args, **kwargs)
                    except Exception as e:
                        if record_exceptions:
                            trace.get_current_span().record_exception(e)
                        if set_status_on_exception:
                            trace.get_current_span().set_status(
                                StatusCode.ERROR, f"{type(e).__name__}: {e}"
                            )
                        raise

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name, attributes=static_attrs):
                    try:
                        return fn(*args, **kwargs)
                    except Exception as e:
                        if record_exceptions:
                            trace.get_current_span().record_exception(e)
                        if set_status_on_exception:
                            trace.get_current_span().set_status(
                                StatusCode.ERROR, f"{type(e).__name__}: {e}"
                            )
                        raise

            return sync_wrapper

    return decorator(func) if func is not None else decorator


# ── Internal setup ────────────────────────────────────────────────────────────


def _add_otel_context(logger, method, event_dict):
    """Structlog processor: inject trace_id and span_id into every log event."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def _configure_structlog() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # guild_id, command, etc.
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
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
    # Always add a stdout StreamHandler so logs are visible in `docker logs` and
    # local dev regardless of whether the OTel SDK is enabled. Without this,
    # log output is silently dropped when OTEL_SDK_DISABLED=true (no LoggingHandler
    # added either in that path).
    if not any(isinstance(h, logging.StreamHandler) for h in logging.root.handlers):
        logging.root.addHandler(logging.StreamHandler(sys.stdout))
    logging.root.setLevel(logging.INFO)


def _setup_traces() -> None:
    global _tracer_provider
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    resource = Resource.create({SERVICE_NAME: _SERVICE_NAME})
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

    resource = Resource.create({SERVICE_NAME: _SERVICE_NAME})
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
