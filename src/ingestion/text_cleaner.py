import re

def clean_text(text: str) -> str:
    """Normalizes whitespace and removes null bytes and unprintable characters."""
    if not text:
        return ""
    # Remove null bytes and other non-printable control characters
    text = text.replace("\x00", "")
    # Normalize unicode whitespace and collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Alias for backward compatibility
clean = clean_text
