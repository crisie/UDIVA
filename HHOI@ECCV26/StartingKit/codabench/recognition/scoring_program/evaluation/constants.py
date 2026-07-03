# List of utterance_types used for computing mAP (verbal) and the partial metric mAP (utterance_type)
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
    # 7. Other (present in the groundtruth, but not included as valid categories in the mAP)
    # "unintelligible"
]

# List of high_level actions used for computing mAP (non-verbal) and the partial metric mAP (highlevel_action)
HIGHLEVEL_ACTIONS = [
    # 1. Communicative
    "imitate", "request", "demonstrate", "positive_acknowledgement", "negative_acknowledgement", "other_acknowledgement",
    # 2. Manipulative
    "open", "close", "assemble", "disassemble", "relocate", "select", "discard", "give", "receive", "correct", "take", "show", "play", "make_room", "organize", "prepare", "keep", "withdraw",
    # 3. Cognitive
    "inspect_check", "draw_attention", "pay_attention", "verify",
    # 4. Other
    "search", "wait", "assist",
    # 5. Other (present in the groundtruth, but not included as valid categories in the mAP)
    # "unintentional", "unclear"
]

# List of low_level actions used for computing the partial metric mAP (lowlevel-action) only.
LOWLEVEL_ACTIONS = [
    "look_at", "hold", "pick_up", "attach", "place", "move", "pointing_at",
    "spread", "rotate", "flip", "pass", "move_head", "release", "grasp",
    "move_body", "hand_gesture", "drop", "detach", "press", "nod", "lift",
    "face_gesture", "tap", "throw"
    # Present in the groundtruth, but not included as valid categories in the mAP)
    # "unclear", "none"
]

# The following are wildcard values for the attribute matching. They can match any valid value for those attributes.
WILDCARDS = {
    "utterance_type": ["unintelligible"],
    "highlevel_action": ["unintentional", "unclear"],
    "lowlevel_action": ["unclear", "none"],
    "target": ["unclear"]
}