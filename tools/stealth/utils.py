from collections import defaultdict


class UniqueNameGenerator:
    _name_id: dict[str, int]

    def __init__(self) -> None:
        self._name_id = defaultdict(lambda: 0)

    def get_unique_name(self, name: str) -> str:
        name_id = self._name_id[name]
        self._name_id[name] += 1
        return f"{name}_{name_id}"


def int_to_name(i: int) -> str:
    D_TO_NAME_MAP: dict[int, str] = {
        0: "ZERO",
        1: "ONE",
        2: "TWO",
        3: "THREE",
        4: "FOUR",
        5: "FIVE",
        6: "SIX",
        7: "SEVEN",
        8: "EIGHT",
        9: "NINE"
    }

    i_str = str(i)
    ret = []
    for c in i_str:
        ret.append(D_TO_NAME_MAP[int(c)])
    return "_".join(ret)
