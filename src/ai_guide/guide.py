def make_guide_text(poi: dict) -> str:
    c = poi["card"]
    lines = []
    lines.append(f"📍 {poi['name']} ({poi['city']}, {poi['country']})")
    lines.append("")
    lines.append(c["summary"])
    lines.append(f"Era: {c.get('era', 'Unknown')}")
    lines.append(f"Style: {c.get('style', 'Unknown')}")
    lines.append("")

    if c.get("facts"):
        lines.append("Interesting facts:")
        for f in c["facts"]:
            lines.append(f" - {f}")
        lines.append("")

    if c.get("tips"):
        lines.append("Visitor tips:")
        for t in c["tips"]:
            lines.append(f" - {t}")

    return "\n".join(lines)
