"""JWT verification across the legacy/asymmetric migration.

The interesting test here is `test_public_key_cannot_be_used_as_an_hmac_secret`. A JWKS is
public, so during a migration an attacker can read the signing public key and try to sign a
forged token with HS256 using that public key as the shared secret. Any verifier that hands
the caller-supplied `alg` and a single key to `jwt.decode` accepts it. This one must not.
"""

import datetime as _dt

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from api import auth

SEAT = "11111111-1111-1111-1111-111111111111"
LEGACY_SECRET = "legacy-shared-secret-value"
PROJECT = "https://uikdknfxcqklldmfshug.supabase.co"


def _claims(**over):
    now = _dt.datetime.now(_dt.timezone.utc)
    base = {
        "sub": SEAT,
        "aud": auth.JWT_AUDIENCE,
        "role": "authenticated",
        "email": "andy@example.com",
        "exp": now + _dt.timedelta(hours=1),
        "iat": now,
    }
    base.update(over)
    return base


class _Req:
    def __init__(self, token=None):
        self.headers = {"authorization": f"Bearer {token}"} if token else {}


def _forge_hs256(claims, secret):
    """Build an HS256 JWT with an arbitrary secret, bypassing PyJWT's key guardrails.

    Needed only for the algorithm-confusion test: PyJWT will not encode with a PEM as an
    HMAC secret, so the attack cannot be expressed through its API.
    """
    import base64, hashlib, hmac, json

    def seg(d):
        raw = json.dumps(d, separators=(",", ":"), default=str).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=")

    signing_input = seg({"alg": "HS256", "typ": "JWT"}) + b"." + seg(claims)
    sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return (signing_input + b"." + base64.urlsafe_b64encode(sig).rstrip(b"=")).decode()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    auth.reset_jwks_cache()
    yield
    auth.reset_jwks_cache()


@pytest.fixture(scope="module")
def rsa_pair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()).decode()
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return priv, pub


def _install_jwks(monkeypatch, public_pem):
    """Point the asymmetric branch at a known public key without network access."""
    class _Key:
        key = public_pem

    class _Client:
        def get_signing_key_from_jwt(self, token):
            return _Key()

    monkeypatch.setattr(auth, "_jwks_client", lambda: _Client())


# ------------------------------------------------------------------ configuration

def test_auth_unconfigured_treats_everyone_as_anonymous():
    assert auth.auth_configured() is False
    assert auth.get_current_user(_Req("anything")) is None


@pytest.mark.parametrize("env,expected", [
    ({"SUPABASE_JWT_SECRET": LEGACY_SECRET}, True),          # legacy only
    ({"SUPABASE_URL": PROJECT}, True),                       # migrated only
    ({"SUPABASE_JWT_SECRET": LEGACY_SECRET,
      "SUPABASE_URL": PROJECT}, True),                       # mid-migration
])
def test_either_path_alone_counts_as_configured(monkeypatch, env, expected):
    """Requiring BOTH would refuse a correctly configured server at each end of the move."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert auth.auth_configured() is expected


def test_jwks_url_derives_from_project_url(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", PROJECT + "/")
    assert auth.jwks_url() == f"{PROJECT}{auth.JWKS_PATH}"


# ------------------------------------------------------------------ legacy HS256

def test_legacy_hs256_token_verifies(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    token = jwt.encode(_claims(), LEGACY_SECRET, algorithm="HS256")
    assert auth.get_current_user(_Req(token))["sub"] == SEAT


def test_hs256_refused_once_the_legacy_secret_is_revoked(monkeypatch):
    """Post-revoke state: URL configured, secret gone. HS256 is no longer ours."""
    monkeypatch.setenv("SUPABASE_URL", PROJECT)
    token = jwt.encode(_claims(), LEGACY_SECRET, algorithm="HS256")
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        auth.get_current_user(_Req(token))
    assert e.value.status_code == 401


def test_wrong_secret_is_refused(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    token = jwt.encode(_claims(), "some-other-secret", algorithm="HS256")
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        auth.get_current_user(_Req(token))


# ------------------------------------------------------------------ asymmetric

def test_rs256_token_verifies_via_jwks(monkeypatch, rsa_pair):
    priv, pub = rsa_pair
    monkeypatch.setenv("SUPABASE_URL", PROJECT)
    _install_jwks(monkeypatch, pub)
    token = jwt.encode(_claims(), priv, algorithm="RS256")
    assert auth.get_current_user(_Req(token))["sub"] == SEAT


def test_both_kinds_verify_during_the_migration(monkeypatch, rsa_pair):
    """The whole point: already-issued HS256 tokens stay valid while new ones are RS256."""
    priv, pub = rsa_pair
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    monkeypatch.setenv("SUPABASE_URL", PROJECT)
    _install_jwks(monkeypatch, pub)

    old = jwt.encode(_claims(), LEGACY_SECRET, algorithm="HS256")
    new = jwt.encode(_claims(), priv, algorithm="RS256")
    assert auth.get_current_user(_Req(old))["sub"] == SEAT
    assert auth.get_current_user(_Req(new))["sub"] == SEAT


def test_asymmetric_token_without_project_url_is_refused(monkeypatch, rsa_pair):
    priv, _ = rsa_pair
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    token = jwt.encode(_claims(), priv, algorithm="RS256")
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        auth.get_current_user(_Req(token))


# ------------------------------------------------------------------ the attack

def test_public_key_cannot_be_used_as_an_hmac_secret(monkeypatch, rsa_pair):
    """Algorithm confusion. A JWKS is PUBLIC -- this key is not a secret.

    Forge a token by signing HS256 with the published public key as the HMAC secret. A
    verifier that trusts the header's `alg` against whichever key it has accepts this and
    hands the attacker any `sub` they like. Ours must not, because the HS256 branch checks
    the legacy secret and only the legacy secret.
    """
    _priv, pub = rsa_pair
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    monkeypatch.setenv("SUPABASE_URL", PROJECT)
    _install_jwks(monkeypatch, pub)

    # Hand-rolled, because jwt.encode() REFUSES to use a PEM as an HMAC secret. That
    # guardrail protects us from writing the bug; it does not protect us from receiving
    # the attack, so the test has to build the token the way an attacker would.
    forged = _forge_hs256(_claims(sub="attacker"), pub)

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        auth.get_current_user(_Req(forged))
    assert e.value.status_code == 401


def test_unsupported_algorithm_is_refused(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    token = jwt.encode(_claims(), LEGACY_SECRET, algorithm="HS512")
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        auth.get_current_user(_Req(token))


def test_none_algorithm_is_refused(monkeypatch):
    """`alg: none` is the oldest JWT attack there is. It must not reach a branch."""
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    token = jwt.encode(_claims(), key="", algorithm="none")
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        auth.get_current_user(_Req(token))


# ------------------------------------------------------------------ expiry + deps

def test_expired_token_reads_as_anonymous_not_an_error(monkeypatch):
    """Deliberate: lets the client prompt a re-auth instead of the endpoint refusing."""
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    past = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=2)
    token = jwt.encode(_claims(exp=past, iat=past), LEGACY_SECRET, algorithm="HS256")
    assert auth.get_current_user(_Req(token)) is None


def test_no_header_is_anonymous(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    assert auth.get_current_user(_Req()) is None


def test_require_user_503s_when_unconfigured():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        auth.require_user(None)
    assert e.value.status_code == 503


def test_require_user_401s_when_anonymous(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", PROJECT)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        auth.require_user(None)
    assert e.value.status_code == 401


def test_user_id_or_none():
    assert auth.user_id_or_none({"sub": SEAT}) == SEAT
    assert auth.user_id_or_none(None) is None
