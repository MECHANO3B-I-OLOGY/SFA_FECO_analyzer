from enum import Enum, auto

class CalibrationValues(Enum):
    HG_GREEN = 546.075
    HG_YELLOW_1 = 576.959 
    HG_YELLOW_2 = 579.065

class FileMode(Enum):
    UNSELECTED = auto()
    APPEND = auto()
    OVERWRITE = auto()

class TimeUnits(Enum):
    UNSELECTED = auto()
    SECONDS = 's'
    MINUTES = 'min'
    HOURS = 'hr'