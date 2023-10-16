from collections import defaultdict


class UniqueNameGenerator:
    _NAME_ID: dict[str, int] = defaultdict(lambda: 0)

    @staticmethod
    def get_unique_name(name: str) -> str:
        name_id = UniqueNameGenerator._NAME_ID[name]
        UniqueNameGenerator._NAME_ID[name] += 1
        return f"{name}_{name_id}"
    
    @staticmethod
    def reset() -> None:
        UniqueNameGenerator._NAME_ID = defaultdict(lambda: 0)
 