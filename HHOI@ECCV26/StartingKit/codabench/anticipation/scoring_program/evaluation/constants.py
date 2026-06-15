"""Constants for UDIVA-HHOI Event Anticipation (Tracks 3 & 4)."""

ROOT_KEY = "anticipation"
PARTICIPANTS = ("participant_a", "participant_b")
MAX_HYPOTHESES = 5

VERBAL_EVENT_LENGTH = 2
NONVERBAL_EVENT_LENGTH = 3

INSERTION_COST = 1.0
DELETION_COST = 1.0
TRANSPOSITION_COST = 0.5

VERBAL_WEIGHTS = (0.8, 0.2)         # utterance_type, target
NONVERBAL_WEIGHTS = (0.4, 0.4, 0.2) # highlevel_action, lowlevel_action, target

VERBAL_ATTRIBUTE_NAMES = ("utterance_type", "target")
NONVERBAL_ATTRIBUTE_NAMES = ("highlevel_action", "lowlevel_action", "target")

# The following are verbal and non-verbal wildcard values for different attributes. These are special annotation values
# that can match any valid value for those attributes.
VERBAL_WILDCARDS = {
    "utterance_type": ["unintelligible"],
    "target": ["unclear"]
}
NONVERBAL_WILDCARDS = {
    "highlevel_action": ["unintentional", "unclear"],
    "lowlevel_action": ["unclear", "none"],
    "target": ["unclear"]
}

SUBTASKS = (
    "next_action",
    "verbal_2s",
    "nonverbal_2s",
    "verbal_nonverbal_2s",
)