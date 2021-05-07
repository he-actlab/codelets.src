from typing import Dict
from dataclasses import dataclass

@dataclass(frozen=True)
class Datatype:
    type: str
    bitwidth: int

    def __str__(self):
        return f"{self.type}{self.bitwidth}"

    def to_json(self):
        blob = {}
        blob['type'] = self.type
        blob['bitwidth'] = self.bitwidth
        return blob

    @staticmethod
    def from_json(dt_obj: Dict):
        return Datatype(type=dt_obj['type'], bitwidth=dt_obj['bitwidth'])

    @staticmethod
    def from_str(dt_str: str):
        idx = 0

        while not dt_str[idx].isdigit() and idx < len(dt_str):
            idx += 1
        type_part = dt_str[:idx].upper()
        bit_part = int(dt_str[idx:])
        return Datatype(type=type_part, bitwidth=bit_part)

    def bytes(self):
        return self.bitwidth // 8

    def bits(self):
        return self.bitwidth