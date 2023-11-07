import numpy as np
from stealth.stealth_codelet import interpret, StealthCodelet
from .result import UnitTestResult


class UnitTest:
    _inputs: tuple[np.ndarray, ...]
    _expected_outputs: tuple[np.ndarray, ...]

    def __init__(self, inputs: tuple[np.ndarray, ...], expected_outputs: tuple[np.ndarray, ...]) -> None:
        self._inputs = inputs
        self._expected_outputs = expected_outputs
    
    @property
    def inputs(self) -> tuple[np.ndarray, ...]:
        return self._inputs
    
    def run(self, codelet: StealthCodelet) -> UnitTestResult:
        actual_outputs: tuple[np.ndarray, ...] = interpret(codelet, self.inputs)
        for actual_output, expected_output in zip(actual_outputs, self._expected_outputs):
            if not np.array_equal(actual_output, expected_output):
                return UnitTestResult(False, "Actual output does not match expected output.")
        return UnitTestResult(True)
