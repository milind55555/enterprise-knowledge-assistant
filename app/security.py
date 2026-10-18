"""API-key authentication with role-based access control (RBAC) and an audit trail."""
from __future__ import annotations

import hmac
import json
import time
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from app import config


@dataclass(frozen=True)
class User:
    role: str
    key_id: str  # a short, non-secret fingerprint used in the audit log


def _fingerprint(key: str) -> str:
    return key[:3] + "***" + key[-2:] if len(key) > 6 else "***"


def authenticate(x_api_key: str | None = Header(default=None)) -> User:
    if not x_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-API-Key header")
    role = None
    for known, r in config.load_api_keys().items():
        # constant-time comparison; loop over every key so timing does not reveal matches
        if hmac.compare_digest(known.encode(), x_api_key.encode()):
            role = r
    if role is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
    return User(role=role, key_id=_fingerprint(x_api_key))


def require_admin(user: User = Depends(authenticate)) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required")
    return user


def audit(user: User, event: str, **details) -> None:
    config.AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), "role": user.role, "key": user.key_id, "event": event, **details}
    with config.AUDIT_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
