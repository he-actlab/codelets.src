from .utils import sequence_to_string


def generate_return(outputs: tuple[str]) -> str:
    return f"return {sequence_to_string(outputs)}"
