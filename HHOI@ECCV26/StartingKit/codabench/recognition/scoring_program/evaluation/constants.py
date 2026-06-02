UTTERANCE_TYPES = [
    # 1. Instruction
    "instruct", "explain", "clarify", "suggest", "command", "draw_attention",
    # 2. Decision-Making
    "agree", "disagree", "doubt", "resolve_conflict", "confirm_selection", "reject_selection", "discuss",
    # 3. Support
    "encourage", "reassure", "praise", "criticize", "assist",
    # 4. Questioning
    "seek_help", "seek_confirmation", "seek_clarification", "seek_information", "check_progress", "request",
    # 5. Declare
    "declare_step", "declare_selection", "express_concern", "express_intent", "express_observation", "express_other",
    # 6. Acknowledgement
    "positive_acknowledgement", "negative_acknowledgement", "other_acknowledgement",
    # 6. Other (present in the groundtruth, but not included as valid categories in the mAP)
    # "unintelligible"
]

HIGHLEVEL_ACTIONS = [
    # 1. Communicative
    "imitate", "request", "demonstrate", "positive_acknowledgement", "negative_acknowledgement", "other_acknowledgement",
    # 2. Manipulative
    "open", "close", "assemble", "disassemble", "relocate", "select", "discard", "give", "receive", "correct", "take", "show", "play", "make_room", "organize", "prepare", "keep", "withdraw",
    # 3. Cognitive
    "inspect_check", "draw_attention", "pay_attention", "verify",
    # 4. Other
    "search", "wait", "assist",
    # 4. Other (present in the groundtruth, but not included as valid categories in the mAP)
    # "unintentional", "unclear"
]

# The following are wildcard values for different attributes. These are special annotation values
# that can match any valid value for those attributes.
WILDCARDS = {
    "utterance_type": ["unintelligible"],
    "highlevel_action": ["unintentional", "unclear"],
    "lowlevel_action": ["unclear", "none"],
    "target": ["unclear"]
}