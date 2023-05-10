from enum import Enum


# Classes
class Normalizations(Enum):
    """Defines Normalization types for audio processing.
    """
    NONE = 0
    PEAK = 1
    RMS = 2
    Z_SCORE = 3
    MIN_MAX = 4


class Instruments(Enum):
    """Defines instrument types for audio processing.
    """
    DEFAULT = 1
    VOICE = 2
    GUITAR = 3
    PIANO = 4
    STRINGS = 5
    DRUMS = 6

    def get_string(self):
        """Returns the name of the instrument as a string.
        """
        return self.name
    
    @staticmethod
    def from_string(s: str):
        """Returns the enum value of a given string.

        :return: :py:class:`Instruments` of the given string.

        :raises Exception: If the string does not correspond to a known instrument.
        """
        if s == "DEFAULT":
            return Instruments.DEFAULT
        if s == "VOICE":
            return Instruments.VOICE
        if s == "GUITAR":
            return Instruments.GUITAR
        if s == "PIANO":
            return Instruments.PIANO
        if s == "STRINGS":
            return Instruments.STRINGS
        if s == "DRUMS":
            return Instruments.DRUMS
        raise Exception("Couldn't parse string into instrument")
    
    @staticmethod
    def from_index(index: int):
        """Returns the enum value of a given index.

        :return: :py:class:`Instruments` of the given index.

        :raises Exception: If the index does not correspond to a known instrument.
        """
        if index == 1:
            return Instruments.DEFAULT
        if index == 2:
            return Instruments.VOICE
        if index == 3:
            return Instruments.GUITAR
        if index == 4:
            return Instruments.PIANO
        if index == 5:
            return Instruments.STRINGS
        if index == 6:
            return Instruments.DRUMS
        raise Exception("Couldn't parse string into instrument")
    
    def get_fundamental_frequency_range(self) -> list:
        """Returns the fundamental frequency range for a given instrument.

        :returns: A :py:type:`list` containing the lower and upper frequency range limits.
        """
        if self == Instruments.DEFAULT:
            return [20.0, 8000.0]
        if self == Instruments.VOICE:
            return [80.0, 4000.0]
        if self == Instruments.GUITAR:
            return [20.0, 5000.0]
        if self == Instruments.PIANO:
            return [20.0, 4500.0]
        if self == Instruments.STRINGS:
            return [20.0, 3500.0]
        if self == Instruments.DRUMS:
            return [20.0, 10000.0]


class BColors:
    """Colors used to print and debug
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


__DEBUGGER_ACTIVE = True
__PRINTING_ACTIVE = True
__PRINTING_DATA_ACTIVE = True


def print_success(string, flush=True):
    if __PRINTING_ACTIVE:
        print(BColors.OKGREEN + "[OK] " + str(string) + BColors.ENDC, flush=flush)


def print_info(string, flush=True):
    if __PRINTING_ACTIVE:
        print(BColors.OKBLUE + "[INFO] " + BColors.UNDERLINE + str(string) + BColors.ENDC, flush=flush)


def print_data(channel, data, flush=True):
    if __PRINTING_DATA_ACTIVE:
        print(BColors.OKCYAN + "[DATA - Channel " + str(channel) + "] ", data, BColors.ENDC, flush=flush)


def print_warning(string, flush=True):
    if __PRINTING_ACTIVE:
        print(BColors.WARNING + BColors.BOLD + "[WARNING] " + str(string) + BColors.ENDC, flush=flush)


def print_error(string, flush=True):
    if __PRINTING_ACTIVE:
        print(BColors.FAIL + BColors.BOLD + "[ERROR] " + str(string) + BColors.ENDC, flush=flush)


def print_dbg(string, flush=True):
    if __PRINTING_ACTIVE and __DEBUGGER_ACTIVE:
        print(BColors.OKGREEN + "[DBG] " + str(string) + BColors.ENDC, flush=flush)
