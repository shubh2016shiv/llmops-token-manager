"""
Deriving the rate-limit bucket key from an incoming request.

----
What is a "bucket key" and why does it matter?
-----------------------------------------------
The rate limiter doesn't count ALL requests together — that would be useless.
Instead, it counts requests per "bucket." A bucket is just a named group of
requests that share the same limit.

The bucket key is the label on the bucket. It answers: "which group does this
request belong to?" For example:

    Key: "192.168.1.5"
    → All requests from this IP share one 10/minute budget.

    Key: "192.168.1.5:alice"
    → Alice's login attempts from this IP share one 5/minute budget.
      Bob's login attempts from the same IP get a DIFFERENT budget
      (key: "192.168.1.5:bob").

    Key: "ms-gateway:10.0.0.1"
    → The "ms-gateway" microservice from this IP gets its own budget,
      separate from "ms-pipeline" from the same IP.

Choosing the right key for each endpoint is a design decision:
- Too broad (just IP) → one noisy user behind a NAT blocks everyone.
- Too narrow (IP + username + user-agent + timestamp) → attacker just
  rotates one field to get unlimited budgets.
- Just right → depends on what you're protecting.

----
Client IP extraction: the X-Forwarded-For header
-------------------------------------------------
When your app runs behind a reverse proxy (Nginx, Cloudflare, AWS ALB),
the proxy receives the real user's request first, then forwards it to your
app. From your app's perspective, EVERY request appears to come from the
proxy's IP, not the real user.

    Real user (1.2.3.4) → Nginx (10.0.0.1) → Your app
                            Your app sees THIS IP ☝

To fix this, proxies add an HTTP header called X-Forwarded-For that
preserves the chain of IPs:

    X-Forwarded-For: 1.2.3.4, 10.0.0.1

Each proxy APPENDS the IP it received the request from to the RIGHT. So the
rightmost entries are the ones added by YOUR infrastructure (trusted); the
leftmost entries are client-supplied (untrusted — they can be spoofed!).

Our `get_client_ip` reads from a TRUSTED position in this chain: N hops
from the right, where N = settings.rate_limit_trusted_proxy_hops (the number
of proxies in front of your app). With 1 proxy, we read the second-from-right
entry (the real client). Entries to the left are ignored — they're client-
supplied and malicious actors can set them to anything.

If there's no proxy (hops=0), we fall back to the raw TCP connection address.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from fastapi import Request


def get_client_ip(request: Request) -> str:
    """
    Extract the real client IP, handling the case where the app is behind a proxy.

    The algorithm:
    1. If a trusted number of proxy hops is configured AND X-Forwarded-For is
       present, read the IP at position `-trusted_proxy_hops` from the right.
    2. Otherwise, use the raw TCP peer address (request.client.host).
    3. If even that is unavailable, return "unknown" (shouldn't happen, but
       better than crashing).

    Why read from the right instead of the left?
    ............................................
    The LEFTmost entry in X-Forwarded-For is whatever the CLIENT put there
    (or the first proxy in an honest chain). A malicious client can set it
    to anything — including rotating it on every request to get unlimited
    rate-limit buckets. The RIGHTmost entries are the ones YOUR proxy added,
    and they can't be spoofed (assuming your proxy strips incoming X-Forwarded-For
    and sets its own — which is the standard configuration).
    """
    trusted_proxy_hops = settings.rate_limit_trusted_proxy_hops
    forwarded_for = request.headers.get("X-Forwarded-For")
    if trusted_proxy_hops >= 1 and forwarded_for:
        forwarded_ips = [ip.strip() for ip in forwarded_for.split(",") if ip.strip()]
        if len(forwarded_ips) >= trusted_proxy_hops:
            # Read the IP N positions from the right. With 1 proxy hop, this
            # gives you the rightmost entry — the IP your proxy forwarded from
            # (i.e., the actual client, assuming your proxy strips/stamps the
            # header correctly and there are no intermediate proxies you don't
            # know about).
            return forwarded_ips[-trusted_proxy_hops]
    return request.client.host if request.client else "unknown"


async def ip_only_key(request: Request) -> str:
    """
    Build a bucket key from client IP only.

    All requests from the same IP share one rate-limit budget. This is the
    simplest key and works well for endpoints where you don't have (or don't
    trust) user identity — like token generation and token refresh, where the
    user hasn't authenticated yet.
    """
    return get_client_ip(request)


async def service_id_key(request: Request) -> str:
    """
    Build a bucket key from service ID + client IP: "service_id:client_ip".

    This is used for internal service-to-service calls (token acquisition).
    Each upstream microservice identifies itself with an X-Service-Id header,
    and gets its OWN rate-limit budget separate from other services.

    Example keys for a token acquire endpoint:
        "ms-llm-gateway:10.0.1.5"    ← gateway service from that IP
        "ms-content-pipeline:10.0.1.5" ← pipeline service (same IP, different budget!)
        "unknown:10.0.1.5"            ← missing X-Service-Id header

    Without service bucketing, one misbehaving microservice could exhaust the
    shared rate-limit budget and starve all other services. With it, each
    service is independently limited.

    Falls back to "unknown" when X-Service-Id is missing — the request is
    still rate-limited, but in a separate bucket from identified services,
    so an unlabeled caller can't interfere with known-good services.
    """
    service_id = request.headers.get("X-Service-Id", "unknown").strip() or "unknown"
    client_ip = get_client_ip(request)
    return f"{service_id}:{client_ip}"
