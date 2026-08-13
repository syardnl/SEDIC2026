import httpx, os

SYSTEM = """You are a maritime domain awareness analyst.
Given a list of vessel detections for a single frame, write a concise
2-sentence threat summary for a duty officer. Be factual and direct.
Flag HIGH PRIORITY detections immediately."""

def summarise(detections: list[dict]) -> str:
    """Returns a plain-text threat narrative, or empty string on failure."""
    if not detections:
        return "No vessels detected in this frame."
    prompt = "Detections:\n" + "\n".join(
        f"- {d['class_name']} ({d['threat_level']}, conf {d['confidence']:.0%})"
        for d in detections
    )
    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 200,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=8.0,
        )
        return resp.json()["content"][0]["text"]
    except Exception:
        return ""   # Silent fallback — GUI continues normally