from typing import NamedTuple

__all__ = ["MetacalRecord"]


class MetacalRecord(NamedTuple):
    index: int

    e1: float
    e2: float
    e1err: float
    e2err: float
    snr: float

    R11_p: float
    R11_m: float
    R11: float

    R22_p: float
    R22_m: float
    R22: float

    R12_p: float
    R12_m: float
    R12: float

    R21_p: float
    R21_m: float
    R21: float

    flag_noshear: int
    flag_1p: int
    flag_1m: int
    flag_2p: int
    flag_2m: int

    @classmethod
    def dtypes(cls) -> list:
        return [
            ("index", "i4"),
            ("e1", "f4"),
            ("e2", "f4"),
            ("e1err", "f4"),
            ("e2err", "f4"),
            ("snr", "f4"),
            ("R11_p", "f4"),
            ("R11_m", "f4"),
            ("R11", "f4"),
            ("R22_p", "f4"),
            ("R22_m", "f4"),
            ("R22", "f4"),
            ("R12_p", "f4"),
            ("R12_m", "f4"),
            ("R12", "f4"),
            ("R21_p", "f4"),
            ("R21_m", "f4"),
            ("R21", "f4"),
            ("flag_noshear", "i4"),
            ("flag_1p", "i4"),
            ("flag_1m", "i4"),
            ("flag_2p", "i4"),
            ("flag_2m", "i4"),
        ]
