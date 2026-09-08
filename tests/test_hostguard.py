"""Tests for src/hostguard.py — the public-internet host policy every fetch
site consults before connecting."""

import asyncio
import ipaddress
import socket
from unittest.mock import AsyncMock

import pytest

from src import hostguard
from src.hostguard import (
    is_public_address,
    refusal_for,
    refusal_reason,
    resolve_addresses,
    url_host,
)


class TestIsPublicAddress:
    @pytest.mark.parametrize(
        "raw",
        [
            "127.0.0.1",  # loopback
            "127.255.255.254",
            "0.0.0.0",  # unspecified
            "10.0.0.1",  # RFC 1918
            "172.16.5.5",
            "172.31.255.255",
            "192.168.1.1",
            "169.254.169.254",  # link-local (cloud metadata)
            "100.64.0.1",  # CGNAT
            "192.0.0.1",  # IETF protocol assignments
            "192.0.2.1",  # TEST-NET
            "198.18.0.1",  # benchmarking
            "224.0.0.1",  # multicast
            "239.255.255.250",
            "240.0.0.1",  # reserved
            "255.255.255.255",  # broadcast
            "::1",
            "::",
            "fc00::1",  # ULA
            "fdff::1",
            "fe80::1",  # link-local
            "ff02::1",  # multicast
            "::ffff:127.0.0.1",  # IPv4-mapped loopback
            "::ffff:10.1.2.3",
            "::ffff:169.254.169.254",
            "64:ff9b::7f00:1",  # NAT64 of 127.0.0.1
            "64:ff9b::a9fe:a9fe",  # NAT64 of 169.254.169.254
            "2001::1",  # Teredo
            "2001:db8::1",  # documentation
            "2002:7f00:1::",  # 6to4 of 127.0.0.1
        ],
    )
    def test_refused(self, raw: str) -> None:
        assert is_public_address(ipaddress.ip_address(raw)) is False

    @pytest.mark.parametrize(
        "raw",
        [
            "8.8.8.8",
            "93.184.216.34",
            "172.32.0.1",  # just past 172.16/12
            "172.15.255.255",  # just before it
            "11.0.0.1",
            "2606:4700::1111",
            "2a00:1450:4001:80b::200e",
            "::ffff:8.8.8.8",  # IPv4-mapped public
            "64:ff9b::808:808",  # NAT64 of 8.8.8.8
        ],
    )
    def test_allowed(self, raw: str) -> None:
        assert is_public_address(ipaddress.ip_address(raw)) is True


class TestUrlHost:
    @pytest.mark.parametrize(
        "url,host",
        [
            ("https://www.YouTube.com/watch?v=x", "www.youtube.com"),
            ("http://user:pw@example.com:8080/p", "example.com"),
            ("youtu.be/x?t=1", "youtu.be"),
            ("http://[::1]/x", "::1"),
            ("http://127.0.0.1:80/x", "127.0.0.1"),
            ("http:///nohost", ""),
            ("", ""),
            ("http://[bad/", ""),  # urlsplit raises ValueError
        ],
    )
    def test_host(self, url: str, host: str) -> None:
        assert url_host(url) == host


class TestRefusalFor:
    def test_all_public_is_allowed(self) -> None:
        addresses = [
            ipaddress.ip_address("8.8.8.8"),
            ipaddress.ip_address("2606:4700::1111"),
        ]
        assert refusal_for("dns.example", addresses) is None

    def test_one_private_answer_refuses_the_host(self) -> None:
        """A name resolving to a public AND a private address cannot be pinned to
        the public one when aiohttp or ffmpeg resolve it again, so it is refused."""
        addresses = [ipaddress.ip_address("8.8.8.8"), ipaddress.ip_address("10.0.0.1")]
        reason = refusal_for("dual.example", addresses)
        assert reason is not None and "private or local" in reason

    def test_no_answers_refuse(self) -> None:
        reason = refusal_for("nowhere.example", [])
        assert reason is not None and "did not resolve" in reason


class TestRefusalReason:
    async def test_public_host_passes(self, public_hosts: AsyncMock) -> None:
        assert await refusal_reason("https://example.com/x") is None
        public_hosts.assert_awaited_once_with("example.com")

    async def test_private_host_is_refused(self, public_hosts: AsyncMock) -> None:
        public_hosts.return_value = [ipaddress.ip_address("192.168.0.9")]
        reason = await refusal_reason("http://nas.local/song.mp3")
        assert reason is not None and "nas.local" in reason

    async def test_literal_private_address_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A literal IP goes through the real resolver, which answers it as
        itself — no DNS is involved."""
        monkeypatch.setattr(hostguard, "resolve_addresses", resolve_addresses)
        reason = await refusal_reason("http://127.0.0.1:8080/x")
        assert reason is not None and "private or local" in reason

    async def test_hostless_url_is_refused(self, public_hosts: AsyncMock) -> None:
        assert await refusal_reason("https:///nohost") is not None
        public_hosts.assert_not_awaited()

    async def test_resolution_failure_is_refused(self, public_hosts: AsyncMock) -> None:
        public_hosts.side_effect = socket.gaierror(-2, "Name or service not known")
        reason = await refusal_reason("https://nx.example/x")
        assert reason is not None and "could not be resolved" in reason

    async def test_slow_resolution_is_refused_within_the_bound(
        self, public_hosts: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def stall(host: str) -> list[hostguard.IPAddress]:
            await asyncio.sleep(5)
            return []

        public_hosts.side_effect = stall
        monkeypatch.setattr(hostguard, "RESOLVE_TIMEOUT_SECS", 0.05)
        async with asyncio.timeout(2):
            reason = await refusal_reason("https://slow.example/x")
        assert reason is not None and "in time" in reason


class TestResolveAddresses:
    async def test_literal_addresses_resolve_to_themselves(self) -> None:
        assert await resolve_addresses("127.0.0.1") == [
            ipaddress.ip_address("127.0.0.1")
        ]
        assert await resolve_addresses("::1") == [ipaddress.ip_address("::1")]

    async def test_localhost_resolves_to_loopback(self) -> None:
        """The one name every host file answers; proves the loop resolver path
        without touching DNS."""
        addresses = await resolve_addresses("localhost")
        assert addresses and all(a.is_loopback for a in addresses)
