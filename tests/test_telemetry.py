"""Tests for src/telemetry.py — OTel setup, the gateway sampler, and structlog wiring.

The setup path runs only at process start and fails only in production, so it is
exercised against real SDK objects and an in-memory exporter rather than mocks —
the assertions reach an actually-emitted span. Everything here mutates
process-global state (root logger handlers, the module's provider globals,
structlog's configuration), so every test restores it.
"""

import logging
import logging.handlers
import multiprocessing
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import structlog
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ALWAYS_ON, Decision

import src.telemetry as telemetry
from src import config


@pytest.fixture(autouse=True)
def restore_global_state() -> Iterator[None]:
    """Save and restore everything setup_telemetry() reaches into.

    structlog included: _configure_structlog() reconfigures it PROCESS-wide, and
    conftest's configure_structlog_for_tests is session-scoped, so without this the
    production JSON chain (and cache_logger_on_first_use) stands for every test that
    runs after this file.
    """
    handlers = list(logging.root.handlers)
    level = logging.root.level
    structlog_config = structlog.get_config()
    saved = (
        telemetry._tracer_provider,
        telemetry._log_provider,
        telemetry._setup_done,
    )
    yield
    logging.root.handlers = handlers
    logging.root.setLevel(level)
    structlog.configure(**structlog_config)
    (
        telemetry._tracer_provider,
        telemetry._log_provider,
        telemetry._setup_done,
    ) = saved


@pytest.fixture
def clean_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present setup_telemetry() with a fresh process."""
    monkeypatch.setattr(telemetry, "_setup_done", False)
    monkeypatch.setattr(telemetry, "_tracer_provider", None)
    monkeypatch.setattr(telemetry, "_log_provider", None)


class TestDiscordGatewayFilter:
    """discord.py makes these HTTP calls at startup with no user-visible parent;
    unsampled they clutter Tempo with orphaned root spans."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://gateway.discord.gg",
            "https://discord.com/api/v10/gateway/bot",
            "https://discord.com/api/v10/users/@me",
            "https://discord.com/oauth2/applications/@me",
            "wss://gateway.discord.gg/?v=10",
        ],
    )
    def test_drops_discord_internal_spans(self, url: str) -> None:
        result = telemetry._DiscordGatewayFilter().should_sample(
            None, 1, "GET", attributes={"http.url": url}
        )
        assert result.decision is Decision.DROP

    def test_keeps_ordinary_spans(self) -> None:
        result = telemetry._DiscordGatewayFilter().should_sample(
            None, 1, "GET", attributes={"http.url": "https://api.spotify.com/v1/tracks"}
        )
        assert result.decision is not Decision.DROP

    def test_keeps_spans_with_no_url_attribute(self) -> None:
        """Every internal span this bot creates is attribute-free — dropping
        those would silence the whole application trace."""
        result = telemetry._DiscordGatewayFilter().should_sample(None, 1, "player.loop")
        assert result.decision is not Decision.DROP

    def test_delegates_to_the_inner_sampler(self) -> None:
        """The filter subtracts spans; it must not add any the inner sampler
        would have dropped."""
        result = telemetry._DiscordGatewayFilter(inner=ALWAYS_OFF).should_sample(
            None, 1, "anything"
        )
        assert result.decision is Decision.DROP

    def test_description_is_stable(self) -> None:
        assert telemetry._DiscordGatewayFilter(ALWAYS_ON).get_description() == (
            "_DiscordGatewayFilter"
        )


class TestSetupTelemetry:
    def test_disabled_sdk_configures_logging_but_no_providers(
        self, clean_setup: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
        telemetry.setup_telemetry()

        assert telemetry._tracer_provider is None
        assert telemetry._log_provider is None
        # Logs must still reach stdout — otherwise disabling tracing silently
        # disables `docker logs`.
        assert any(isinstance(h, logging.StreamHandler) for h in logging.root.handlers)

    def test_second_call_is_a_noop_when_sdk_is_disabled(
        self, clean_setup: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Disabled path: no provider is built, so the guard cannot key off one."""
        monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
        telemetry.setup_telemetry()

        with patch.object(telemetry, "_configure_structlog") as reconfigure:
            telemetry.setup_telemetry()

        reconfigure.assert_not_called()

    def test_second_call_is_a_noop_when_sdk_is_enabled(
        self, clean_setup: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second full setup would add a second LoggingHandler (duplicate Loki
        records) and orphan the first TracerProvider's exporter."""
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        with (
            patch.object(telemetry, "_setup_traces") as traces,
            patch.object(telemetry, "_setup_logs") as logs,
            patch.object(telemetry, "_setup_auto_instrumentation") as instr,
        ):
            telemetry.setup_telemetry()
            telemetry.setup_telemetry()

        traces.assert_called_once()
        logs.assert_called_once()
        instr.assert_called_once()

    def test_enabled_sdk_runs_the_full_setup(
        self, clean_setup: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        with (
            patch.object(telemetry, "_setup_traces") as traces,
            patch.object(telemetry, "_setup_logs") as logs,
            patch.object(telemetry, "_setup_auto_instrumentation") as instr,
        ):
            telemetry.setup_telemetry()

        traces.assert_called_once()
        logs.assert_called_once()
        instr.assert_called_once()

    @pytest.mark.parametrize("value", ["TRUE", "True", "true"])
    def test_disabled_flag_is_case_insensitive(
        self, clean_setup: None, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("OTEL_SDK_DISABLED", value)
        with patch.object(telemetry, "_setup_traces") as traces:
            telemetry.setup_telemetry()
        traces.assert_not_called()


class TestSetupTraces:
    def test_builds_a_provider_that_exports_spans(
        self, clean_setup: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real provider with an in-memory exporter, not a mock: the thing
        worth checking is that a span actually comes out the other end."""
        exporter = InMemorySpanExporter()

        def _fake_provider(**kwargs: Any) -> TracerProvider:
            provider = TracerProvider(**kwargs)
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            return provider

        with (
            patch("opentelemetry.sdk.trace.TracerProvider", side_effect=_fake_provider),
            patch(
                "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
            ),
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor"),
            patch.object(trace, "set_tracer_provider"),
        ):
            telemetry._setup_traces()

        provider = telemetry._tracer_provider
        assert provider is not None
        provider.get_tracer(__name__).start_span("smoke").end()
        assert [s.name for s in exporter.get_finished_spans()] == ["smoke"]

    def test_provider_carries_service_and_environment(self, clean_setup: None) -> None:
        with (
            patch(
                "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
            ),
            patch.object(trace, "set_tracer_provider"),
        ):
            telemetry._setup_traces()

        provider = telemetry._tracer_provider
        assert provider is not None
        attrs = provider.resource.attributes
        assert attrs["service.name"] == telemetry._SERVICE_NAME
        assert attrs["deployment.environment"] == config.ENVIRONMENT

    def test_gateway_filter_is_installed_as_the_sampler(
        self, clean_setup: None
    ) -> None:
        """Without this the orphaned discord.py spans reach Tempo."""
        with (
            patch(
                "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
            ),
            patch.object(trace, "set_tracer_provider"),
        ):
            telemetry._setup_traces()

        provider = telemetry._tracer_provider
        assert provider is not None
        assert isinstance(provider.sampler, telemetry._DiscordGatewayFilter)


class TestSetupLogs:
    # opentelemetry-sdk deprecated LoggingHandler in favour of the one in
    # opentelemetry-instrumentation-logging, and `filterwarnings = ["error"]` turns
    # the resulting DeprecationWarning into a failure. Production emits it too;
    # delete this marker once the handler migration lands.
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_bridges_the_root_logger_to_otel(self, clean_setup: None) -> None:
        with patch(
            "opentelemetry.exporter.otlp.proto.grpc._log_exporter.OTLPLogExporter"
        ):
            telemetry._setup_logs()

        from opentelemetry.sdk._logs import LoggingHandler

        assert telemetry._log_provider is not None
        assert any(isinstance(h, LoggingHandler) for h in logging.root.handlers), (
            "structlog routes through stdlib, so this handler is what reaches Loki"
        )


class TestShutdownTelemetry:
    def test_flushes_and_shuts_down_both_providers(self) -> None:
        tracer, logs = MagicMock(), MagicMock()
        with (
            patch.object(telemetry, "_tracer_provider", tracer),
            patch.object(telemetry, "_log_provider", logs),
        ):
            telemetry.shutdown_telemetry()

        # Flush before shutdown, or buffered spans are dropped on exit.
        tracer.force_flush.assert_called_once()
        tracer.shutdown.assert_called_once()
        logs.force_flush.assert_called_once()
        logs.shutdown.assert_called_once()

    def test_is_a_noop_when_telemetry_never_started(self) -> None:
        with (
            patch.object(telemetry, "_tracer_provider", None),
            patch.object(telemetry, "_log_provider", None),
        ):
            telemetry.shutdown_telemetry()  # must not raise


class TestStructlogProcessors:
    def test_environment_is_stamped_on_every_event(self) -> None:
        event = telemetry._add_environment(None, "info", {"event": "hello"})
        assert event["environment"] == config.ENVIRONMENT

    def test_trace_context_is_injected_inside_a_span(self) -> None:
        provider = TracerProvider()
        with provider.get_tracer(__name__).start_as_current_span("s"):
            event = telemetry._add_otel_context(None, "info", {"event": "x"})

        assert len(event["trace_id"]) == 32
        assert len(event["span_id"]) == 16

    def test_trace_context_is_absent_outside_a_span(self) -> None:
        """An invalid context must not stamp all-zero ids, which would look
        like a real trace in Loki and correlate to nothing."""
        with patch.object(trace, "get_current_span", return_value=trace.INVALID_SPAN):
            event = telemetry._add_otel_context(None, "info", {"event": "x"})

        assert "trace_id" not in event
        assert "span_id" not in event


class TestConfigureStructlog:
    def test_adds_a_stdout_handler(self, clean_setup: None) -> None:
        logging.root.handlers = []
        telemetry._configure_structlog()
        assert any(isinstance(h, logging.StreamHandler) for h in logging.root.handlers)

    def test_does_not_stack_duplicate_stdout_handlers(self, clean_setup: None) -> None:
        """Double-printing every line is the failure this guard prevents."""
        logging.root.handlers = []
        telemetry._configure_structlog()
        telemetry._configure_structlog()
        assert (
            sum(isinstance(h, logging.StreamHandler) for h in logging.root.handlers)
            == 1
        )


class TestConfigureWorkerLogging:
    def test_without_a_queue_only_configures_structlog(self) -> None:
        """The thread-pool test seam passes no queue — the worker shares the
        parent's handlers and must not have them ripped out."""
        logging.root.handlers = [logging.StreamHandler()]
        telemetry.configure_worker_logging(None)
        assert not any(
            isinstance(h, logging.handlers.QueueHandler) for h in logging.root.handlers
        )

    def test_with_a_queue_replaces_handlers_rather_than_adding(self) -> None:
        """Replaced, not added: the parent re-emits every record it drains, so
        keeping the worker's stdout handler would double-print."""
        queue: Any = multiprocessing.Queue()
        try:
            telemetry.configure_worker_logging(queue)
            assert [type(h) for h in logging.root.handlers] == [
                logging.handlers.QueueHandler
            ]
        finally:
            queue.close()
            structlog.contextvars.clear_contextvars()

    def test_binds_worker_id_for_every_line(self) -> None:
        queue: Any = multiprocessing.Queue()
        try:
            telemetry.configure_worker_logging(queue)
            bound = structlog.contextvars.get_contextvars()
            assert bound["worker_id"] == multiprocessing.current_process().name
        finally:
            queue.close()
            structlog.contextvars.clear_contextvars()


class TestGetTracer:
    def test_returns_a_tracer(self) -> None:
        assert telemetry.get_tracer("some.module") is not None
