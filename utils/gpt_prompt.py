def generate_prompt(county: str, baseline: dict, goal_type: str) -> str:
    """Generate a context-aware planning prompt for GPT."""
    return f"""
    Given the following data for {county} County:
    - Absenteeism: {baseline['Absenteeism']}%
    - Literacy: {baseline['Literacy']}%
    - Youth Programs: {baseline['Youth_Programs']}
    - Internet Access: {baseline['Internet_Access']}%
    Suggest 3 specific, actionable steps that could help achieve the goal: "{goal_type}".
    Include cost considerations, potential partners, and expected impact.
    """
