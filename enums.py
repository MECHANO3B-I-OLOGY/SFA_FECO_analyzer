from enum import Enum, auto

class FileMode(Enum):
    UNSELECTED = auto()
    APPEND = auto()
    OVERWRITE = auto()

class TimeUnits(Enum):
    UNSELECTED = auto()
    SECONDS = 's'
    MINUTES = 'min'
    HOURS = 'hr'

class LineMethod(Enum):
    CREST_TO_CREST = auto()
    AVERAGE = auto()

    