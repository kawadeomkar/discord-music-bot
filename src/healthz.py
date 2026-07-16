"""Liveness-probe endpoint for the Kubernetes pipeline.

Exposes ``GET /healthz`` on ``0.0.0.0:$HEALTHZ_PORT``. When ``HEALTHZ_PORT``
is unset the server never starts — the Docker Compose pipeline sets no port
and is byte-for-byte unaffected.

The probe is deliberately dumb: it returns 200 whenever the event loop is
alive and serving, regardless of gateway state. Returning non-200 on
``not is_ready()`` would make the kubelet kill the pod during a Discord
outage or a long reconnect — a restart loop that makes an external problem
worse. Gateway health is observability's job; the body fields feed it.
Design: docs/K8S_DEPLOYMENT_PLAN.md §3.4.
"""

import math
import os
from typing import Optional

import discord
from aiohttp import web

from src.util import get_logger

log = get_logger(__name__)


async def start_healthz(bot: discord.Client) -> Optional[web.AppRunner]:
    port = os.environ.get("HEALTHZ_PORT")
    if not port:
        return None  # compose pipeline: env unset → no server, zero behavior change

    async def healthz(_req: web.Request) -> web.Response:
        # bot.latency is NaN until the first heartbeat ACK, and json.dumps
        # emits bare `NaN` — invalid JSON. Probes only read the status code,
        # but the body feeds observability, so keep it parseable.
        latency = bot.latency
        return web.json_response(
            {
                "ready": bot.is_ready(),
                "latency_s": latency if math.isfinite(latency) else None,
                "guilds": len(bot.guilds),
            }
        )

    app = web.Application()
    app.router.add_get("/healthz", healthz)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(port)).start()
    log.info(f"healthz endpoint listening on 0.0.0.0:{port}")
    return runner
