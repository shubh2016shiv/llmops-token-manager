"""Derive rate-limit buckets from trusted network identity."""

from __future__ import annotations

from ipaddress import ip_address, ip_network
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from fastapi import Request


def get_client_ip(request: Request) -> str:
    """Honor forwarding headers only from configured, trusted TCP peers."""
    peer = request.client.host if request.client else "unknown"
    hops = settings.rate_limit_trusted_proxy_hops
    forwarded = request.headers.get("X-Forwarded-For")
    networks = settings.rate_limit_trusted_proxy_networks
    if not (hops and forwarded and networks):
        return peer

    try:
        peer_address = ip_address(peer)
        if not any(
            peer_address in ip_network(network, strict=False) for network in networks
        ):
            return peer
        addresses = [part.strip() for part in forwarded.split(",")]
        if len(addresses) < hops:
            return peer
        return str(ip_address(addresses[-hops]))
    except ValueError:
        return peer


async def ip_only_key(request: Request) -> str:
    """Use the verified client address as the rate-limit bucket."""
    return get_client_ip(request)
