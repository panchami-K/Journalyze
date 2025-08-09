# src/insights/period_feedback.py

CBT_PROMPT_LIBRARY = {
    "catastrophizing": (
        "Try to list the evidence for and against your feared outcome. "
        "What is a more realistic possibility?"
    ),
    "overgeneralization": (
        "Can you think of times when this hasn't been true? "
        "What are the exceptions to the rule?"
    ),
    "should statements": (
        "What would happen if you changed 'should' to 'prefer'? "
        "Are your standards flexible or absolute?"
    ),
    "mind reading": (
        "Do you have evidence for what others think, or is it a guess? "
        "Could you ask instead of assuming?"
    ),
    "fortune telling": (
        "What predictions are you making that feel inevitable? "
        "What's the likelihood they actually happen?"
    ),
    "personalization": (
        "Are there other factors at play beyond you? "
        "Are you holding yourself responsible for things outside your control?"
    ),
    "none detected": (
        "You are showing flexible thinking patterns. "
        "Try to reinforce these strengths through self-awareness and gratitude."
    )
}

EMOTION_CONTEXT_FEEDBACK = {
    "shame": "This period appears to include self-critical thinking. Reflect on moments where you showed courage or growth.",
    "guilt": "You may be holding yourself to very high standards. Consider how self-forgiveness might support your well-being.",
    "anxiety": "This period reflects high worry. Journaling about what you can control may reduce unnecessary mental loops.",
    "fear": "Take time to write out what you fear and where that fear comes from. Name fears to reduce their power.",
    "sadness": "Consider exploring small moments of hope, joy or gratitude through your journaling practice.",
    "hopelessness": "If things feel stuck, write about one small thing that moved forward this week.",
    "joy": "This is a motivating emotional period—try to reflect on what enabled this mood and how you can sustain it.",
    "hope": "Use hope as a foundation to set new intentions. Reflect on your strengths and what you’ve overcome.",
    "gratitude": "It’s a good time to reflect on what nourishes you. Consider writing appreciation letters or gratitude entries.",
    "relief": "Notice what changed that made you feel relief. Can you create more space for that?",
    "confusion": "Try summarizing what you do know when things feel foggy. Self-clarity builds confidence.",
    "connection": "Relationships nourish our sense of identity. Reflect on conversations or moments that built closeness."
}


def feedback_for_period_row(row):
    feedback = []
    # Emotion-aware reflective prompts (always present)
    top_emotion = row.get("top_emotion", "").lower()
    for emotion_key, emotion_advice in EMOTION_CONTEXT_FEEDBACK.items():
        if emotion_key in top_emotion:
            feedback.append(f"Emotion Insight: {emotion_advice}")
            break
    else:
        feedback.append("Reflect on how your emotions changed this period. Use journaling to deepen your understanding.")

    # Cognitive distortion feedback only for meaningful distortions
    if row.get("common_distortions"):
        # Limit to the top 2 non-'none detected' distortions
        n = 0
        for distortion, freq in list(row["common_distortions"].items()):
            if distortion == "none detected":
                continue
            prompt = CBT_PROMPT_LIBRARY.get(distortion, "")
            if prompt:
                feedback.append(f"For '{distortion}' ({freq}): {prompt}")
                n += 1
            if n == 2:
                break

    return " ".join(feedback)



def attach_period_feedback(df_period_summary):
    """Attach automated feedback/prompts to the period summary DataFrame."""
    feedback_series = df_period_summary.apply(feedback_for_period_row, axis=1)
    df_period_summary["cbt_feedback"] = feedback_series
    return df_period_summary


def print_period_feedback(df_period_feedback):
    print("\n=== PERIODIC FEEDBACK & PROMPTS ===")
    for idx, row in df_period_feedback.iterrows():
        print(f"Period: {row['period']}\nTop Emotion: {row['top_emotion']}\nFeedback:\n{row['cbt_feedback']}\n")
