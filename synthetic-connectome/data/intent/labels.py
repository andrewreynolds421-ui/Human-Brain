"""
Intent taxonomy for the on-device router.

The router sits between Porcupine's wake-word detection and the network. Its
job is one decision: can this utterance be handled on the phone, and if not,
which submind should it go to?

Design constraints that shaped these labels
-------------------------------------------
* They must be predictable from the utterance ALONE, with no conversation
  history — the router runs on a single wake-word phrase.
* They must map to something the app can act on. A label the app cannot use
  is a label not worth predicting.
* The set must be small. Every extra class needs its own training examples,
  and you are bootstrapping from one person's usage.

The two-tier split is the point: LOCAL_* means the phone acts alone and no
API call happens at all. Everything else picks a destination for a call that
was going to happen anyway, and saves tool-schema tokens rather than the
whole round trip.
"""

# Handled entirely on-device. No network, no API spend, no latency.
LOCAL_LABELS = [
    "LOCAL_LAUNCH_APP",   # "open spotify", "launch maps"
    "LOCAL_TRACK",        # "log 80 kilos", "record blood pressure 120 over 80"
    "LOCAL_STATUS",       # "status report", "what are the subminds doing"
    "LOCAL_CANCEL",       # "never mind", "cancel", "stop"
]

# Needs the API. The label picks the submind, which narrows the tool schemas.
REMOTE_LABELS = [
    "REMOTE_HEALTH",      # -> ASCLEPIUS
    "REMOTE_FINANCE",     # -> PLUTUS
    "REMOTE_GEO",         # -> regional subminds
    "REMOTE_DEVICE",      # -> RASPUTIN, device tools (workspace, run_code, browse)
    "REMOTE_DASHBOARD",   # -> RASPUTIN, manage_ui / customize_dashboard
    "REMOTE_CHAT",        # -> RASPUTIN, general conversation and web search
]

LABELS = LOCAL_LABELS + REMOTE_LABELS
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

# Where a remote label sends the request.
LABEL_TO_TARGET = {
    "REMOTE_HEALTH": "ASCLEPIUS",
    "REMOTE_FINANCE": "PLUTUS",
    "REMOTE_GEO": "RASPUTIN",     # RASPUTIN consults the right region
    "REMOTE_DEVICE": "RASPUTIN",
    "REMOTE_DASHBOARD": "RASPUTIN",
    "REMOTE_CHAT": "RASPUTIN",
}

# Which backend tool names imply which label, for auto-labelling from the
# tool_calls table. Only tools whose selection is genuinely determined by the
# utterance appear here.
TOOL_TO_LABEL = {
    "launch_app": "LOCAL_LAUNCH_APP",
    "track": "LOCAL_TRACK",
    "update_my_status": "LOCAL_STATUS",

    "fitbit": "REMOTE_HEALTH",

    "market": "REMOTE_FINANCE",
    "market_outlook": "REMOTE_FINANCE",
    "congress_trades": "REMOTE_FINANCE",

    "show_on_map": "REMOTE_GEO",
    "draw_on_map": "REMOTE_GEO",
    "find_route": "REMOTE_GEO",
    "find_places": "REMOTE_GEO",
    "route_search": "REMOTE_GEO",
    "open_navigation": "REMOTE_GEO",
    "get_location": "REMOTE_GEO",

    "workspace": "REMOTE_DEVICE",
    "run_code": "REMOTE_DEVICE",
    "browse": "REMOTE_DEVICE",

    "manage_ui": "REMOTE_DASHBOARD",
    "customize_dashboard": "REMOTE_DASHBOARD",
    "set_directive": "REMOTE_DASHBOARD",

    "web_search": "REMOTE_CHAT",
    "create_project": "REMOTE_CHAT",
    "update_project": "REMOTE_CHAT",

    # A turn that needed no tool at all. Logged by run_tool_use_loop under
    # this sentinel; without it the gold set had no REMOTE_CHAT examples and
    # a router trained on it had never seen "just answer me".
    "__none__": "REMOTE_CHAT",
}

# Deliberately absent from TOOL_TO_LABEL:
#   create_graph  — appears across every domain; predicts nothing.
#   open_link     — usually a follow-up action, not an opening utterance.
#   consult_submind — handled separately: the CONSULTED submind is the label,
#                     which is a far better signal than the tool name.

CONSULT_TO_LABEL = {
    "ASCLEPIUS": "REMOTE_HEALTH",
    "PLUTUS": "REMOTE_FINANCE",
    # Regional subminds all imply geographic work.
    "VERDE": "REMOTE_GEO", "KHEPRI": "REMOTE_GEO", "BOREAS": "REMOTE_GEO",
    "TAIGA": "REMOTE_GEO", "KURNAI": "REMOTE_GEO", "CHARLEMAGNE": "REMOTE_GEO",
}

# Below this, route to RASPUTIN rather than acting on the prediction.
#
# Asymmetric on purpose. A wrong REMOTE_* label costs some wasted tool
# schemas and a slightly worse answer. A wrong LOCAL_* label means the phone
# acts without ever asking the model — logging a weight that was never
# stated, or opening an app the user did not name. So local actions need to
# clear a much higher bar, and destructive-ish ones should confirm out loud
# before acting regardless of score.
CONFIDENCE_THRESHOLD = 0.60

# Per-label thresholds for local action, set by REVERSIBILITY rather than by
# one global number.
#
# Measured on the benchmark at a uniform 0.85 threshold, over 10 seeds:
#
#   LOCAL_LAUNCH_APP   60 actions   0 wrong   0.0%
#   LOCAL_STATUS       23 actions   0 wrong   0.0%
#   LOCAL_CANCEL       11 actions   0 wrong   0.0%
#   LOCAL_TRACK        32 actions   4 wrong  12.5%   <-- and it is the only
#                                                        one that WRITES
#
# A wrong launch_app opens the wrong app; you close it. A wrong track writes
# a health measurement you never stated into a record you will later read as
# fact and may show a clinician. Same confidence, entirely different cost,
# so they should not share a bar.
LOCAL_THRESHOLDS = {
    "LOCAL_LAUNCH_APP": 0.85,   # reversible: close it
    "LOCAL_STATUS": 0.85,       # read-only
    "LOCAL_CANCEL": 0.85,       # a no-op if wrong
    "LOCAL_TRACK": 0.95,        # writes durable health data
}

# Labels that must read back what they understood and wait, however
# confident. Confidence is calibrated against the classifier's own view of
# the transcript — it cannot detect that ASR heard "eighty" as "eighteen".
# Confirmation catches the failure the score structurally cannot see.
CONFIRM_BEFORE_ACTING = {"LOCAL_TRACK"}

# Kept for compatibility; LOCAL_THRESHOLDS takes precedence.
LOCAL_ACTION_THRESHOLD = 0.85


def policy() -> dict:
    """
    The threshold policy as data, for export into router.json.

    Kotlin reads this block at load time rather than carrying its own copy
    of the numbers. One source of truth; the model file and the policy that
    interprets its scores travel together.
    """
    return {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "local_thresholds": dict(LOCAL_THRESHOLDS),
        "confirm_before_acting": sorted(CONFIRM_BEFORE_ACTING),
        "label_to_target": dict(LABEL_TO_TARGET),
        "default_local_threshold": LOCAL_ACTION_THRESHOLD,
    }


def route(label: str, confidence: float) -> dict:
    """
    Turn a prediction into an action.

    Threshold policy lives here, in one place, so Kotlin never reimplements
    it and the two can never drift apart.
    """
    if label in LOCAL_LABELS:
        threshold = LOCAL_THRESHOLDS.get(label, LOCAL_ACTION_THRESHOLD)
        if confidence >= threshold:
            return {
                "action": "local",
                "label": label,
                "confidence": confidence,
                "confirm": label in CONFIRM_BEFORE_ACTING,
            }
        # Confident enough to guess, not confident enough to act unasked.
        return {"action": "remote", "target": "RASPUTIN", "label": label,
                "confidence": confidence, "reason": "below local threshold"}

    if confidence < CONFIDENCE_THRESHOLD:
        return {"action": "remote", "target": "RASPUTIN", "label": "REMOTE_CHAT",
                "confidence": confidence, "reason": "below threshold"}

    return {"action": "remote", "target": LABEL_TO_TARGET[label],
            "label": label, "confidence": confidence}
