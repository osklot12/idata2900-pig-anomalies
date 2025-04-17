from src.utils.norsvin_behavior_class import NorsvinBehaviorClass

RESIZE_SHAPE = (640, 640)

NORMALIZE_RANGE = (0, 1)

CLASS_COUNTS = {
    NorsvinBehaviorClass.BELLY_NOSING: 1885,
    NorsvinBehaviorClass.TAIL_BITING: 1073,
    NorsvinBehaviorClass.EAR_BITING: 1008,
    NorsvinBehaviorClass.TAIL_DOWN: 1107
}