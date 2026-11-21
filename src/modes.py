"""Student and researcher settings. Mode-specific behaviour lives here."""
from __future__ import annotations

from src.tools import REGISTRY

# google-genai SDK docs (current) use gemini-3.8-flash as the fast model.
FLASH_MODEL = "gemini-3.8-flash"
PRO_MODEL = "gemini-3.8-pro"

STUDENT_TOOLS = [
    "profile",
    "summarize",
    "group_aggregate",
    "plot",
    "correlate",
    "quick_model",
]

STUDENT_PROMPT = (
    "You are a data analyst helping a student. Use only the provided tools. "
    "Plan briefly, call a tool, then answer. Short sentences. At most 120 words. "
    "Give 2 or 3 key points and 1 next step. The first time you use a technical term, "
    "define it in brackets. No code, no jargon, no formulas. "
    "Never invent numbers or column names. If the tools did not measure it, say you could not determine it. "
    "If the sample is small, say so. When you mention a model, repeat the plain-language verdict from quick_model."
)

RESEARCHER_PROMPT = (
    "You are a data analyst assistant. You can only answer using the provided tools. "
    "Plan briefly, call tools, then answer. Be precise. Include effect sizes and uncertainty. "
    "Give 3-6 key points. Cite the tool result that supports each claim. "
    "Then give 2-4 next analyses. Say what you could NOT determine. "
    "Never invent numbers or columns. Warn about small samples, class imbalance "
    "(accuracy versus the majority baseline), data leakage, and multiple comparisons when relevant. "
    "When training, report recall and PR-AUC next to accuracy and the majority baseline."
)

MODES = {
    "student": {
        "label": "🎓 Student",
        "caption": "Short answers, six basic tools, and a lower token cap.",
        "model": FLASH_MODEL,
        "pro_model": FLASH_MODEL,
        "tools": STUDENT_TOOLS,
        "max_tool_calls": 3,
        "max_output_tokens": 400,
        "history_turns": 2,
        "system": STUDENT_PROMPT,
        "questions": [
            "What is in this data?",
            "Which columns matter most?",
            "Is there anything wrong with this data?",
        ],
    },
    "researcher": {
        "label": "🔬 Researcher",
        "caption": "Every tool, longer answers, optional deeper model.",
        "model": FLASH_MODEL,
        "pro_model": PRO_MODEL,
        "tools": list(REGISTRY),
        "max_tool_calls": 8,
        "max_output_tokens": 1200,
        "history_turns": 4,
        "system": RESEARCHER_PROMPT,
        "questions": [],
    },
}

PRIVACY_NOTE = (
    "Sent to Gemini: one line per column (name, dtype, missing %, unique count, "
    "min/max or top values). Five sample rows are included only if you turn that on. "
    "The full dataset stays in this session. The model cannot run code. Python computes every number."
)
