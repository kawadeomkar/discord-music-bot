"""Whether a URL's host is on the public internet.

Every `-play` names a host the bot then fetches from three places (yt-dlp, the
stream probe, ffmpeg), so a link that resolves to loopback, a private range or
a link-local address is refused before any of them opens a connection. The
policy is the pure `is_public_address`; `refusal_reason` resolves a URL's host
through the event loop, bounded. See docs/ARCHITECTURE.md#fetch-host-policy.
"""

import asyncio
import ipaddress
import socket
from typing import Final, Optional, Union
from urllib.parse import urlsplit

from src.util import get_logger

log = get_logger(__name__)

IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]

# Bound on one host resolution. A stalled resolver must not park a command.
RESOLVE_TIMEOUT_SECS: Final[float] = 5.0

# NAT64's well-known prefix embeds an IPv4 in the low 32 bits.
_NAT64: Final = ipaddress.IPv6Network("64:ff9b::/96")


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> Optional[ipaddress.IPv4Address]:
    """The IPv4 an IPv6 address stands for (mapped or NAT64), else None."""
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip in _NAT64:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


def is_public_address(ip: IPAddress) -> bool:
    """True only for a globally routable unicast address. `is_global` already
    refuses loopback, unspecified, link-local, RFC 1918, ULA, CGNAT and the
    reserved blocks; multicast is refused separately (it is not "private" to
    ipaddress), and an IPv6 address embedding an IPv4 is judged as that IPv4."""
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = _embedded_ipv4(ip)
        if embedded is not None:
            ip = embedded
    return ip.is_global and not ip.is_multicast


def url_host(url: str) -> str:
    """The hostname of `url`, lowercased, "" when it has none. A scheme-less
    link is read as https so the host is not mistaken for a path."""
    if "://" not in url:
        url = f"https://{url}"
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


async def resolve_addresses(host: str) -> list[IPAddress]:
    """Every address `host` resolves to, through the loop's resolver. Raises
    what getaddrinfo raises; a literal address resolves to itself."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    addresses: list[IPAddress] = []
    for *_, sockaddr in infos:
        try:
            addresses.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    return addresses


def refusal_for(host: str, addresses: list[IPAddress]) -> Optional[str]:
    """Why `host` is refused given what it resolved to, or None when every
    address is public. ANY non-public answer refuses the host: a name that
    resolves to both cannot be pinned to the public one at fetch time."""
    if not addresses:
        return f"`{host}` did not resolve to any address."
    for ip in addresses:
        if not is_public_address(ip):
            return f"`{host}` points at a private or local network address."
    return None


async def refusal_reason(url: str) -> Optional[str]:
    """None when `url` names a host on the public internet; otherwise one line
    saying why it is refused. Resolution that fails or does not finish within
    RESOLVE_TIMEOUT_SECS refuses too: a host that cannot be resolved cannot be
    fetched, and the caller's own fetch would only report it later and worse."""
    host = url_host(url)
    if not host:
        return "That link has no host to connect to."
    try:
        async with asyncio.timeout(RESOLVE_TIMEOUT_SECS):
            addresses = await resolve_addresses(host)
    # TimeoutError first: it is an OSError, and the clause below would take it.
    except TimeoutError:
        log.warning(f"resolving {host!r} did not finish within {RESOLVE_TIMEOUT_SECS}s")
        return f"`{host}` could not be resolved in time."
    except (socket.gaierror, UnicodeError, OSError) as e:
        log.info(f"host {host!r} did not resolve: {e}")
        return f"`{host}` could not be resolved."
    return refusal_for(host, addresses)
