from sessionrouter import pii
from sessionrouter.pii import Sanitizer, classify_tier, detect
from sessionrouter.types import Message, PrivacyTier


def test_detect_email_phone():
    fs = detect("Email me at jane.doe@corp.io or call 415-555-0132.")
    kinds = {f.kind for f in fs}
    assert "EMAIL" in kinds and "PHONE" in kinds


def test_luhn_credit_card():
    assert any(f.kind == "CREDIT_CARD"
               for f in detect("card: 4242 4242 4242 4242"))
    assert not any(f.kind == "CREDIT_CARD"
                   for f in detect("order 1234 5678 9012 3456"))  # fails Luhn


def test_api_key_and_ssn():
    fs = detect("key sk-ant-api03-AAAABBBBCCCCDDDD1234 ssn 078-05-1120")
    kinds = {f.kind for f in fs}
    assert "API_KEY" in kinds or "SSN" in kinds


def test_tier_classification():
    assert classify_tier([]) == PrivacyTier.OPEN
    assert classify_tier(detect("email me at a@b.com")) == PrivacyTier.REDACT
    assert classify_tier(detect("ssn 078-05-1120")) == PrivacyTier.STRICT
    assert classify_tier(detect("card 4242424242424242")) == PrivacyTier.STRICT


def test_sanitize_stable_placeholders_and_restore():
    s = Sanitizer()
    t1, f1 = s.sanitize("sess", "Contact jane@corp.io about this.")
    assert "jane@corp.io" not in t1 and "[EMAIL_1]" in t1
    t2, _ = s.sanitize("sess", "again: jane@corp.io")
    assert "[EMAIL_1]" in t2  # same placeholder within session
    restored = s.de_sanitize("sess", "Replied to [EMAIL_1] with details.")
    assert "jane@corp.io" in restored


def test_restore_map_isolation_between_sessions():
    s = Sanitizer()
    s.sanitize("a", "x@y.com")
    s.sanitize("b", "p@q.com")
    assert s.restore_map("a") != s.restore_map("b")
    assert s.de_sanitize("b", "hi [EMAIL_1]") == "hi p@q.com"
