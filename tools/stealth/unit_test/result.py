from typing import Optional


class UnitTestResult:
    _is_correct: bool
    _error_message: Optional[str]

    def __init__(self, is_correct: bool, error_message: Optional[str] = None) -> None:
        self._is_correct = is_correct
        self._error_message = error_message
    
    @property
    def is_correct(self) -> bool:
        return self._is_correct
    
    @property
    def error_message(self) -> Optional[str]:
        return self._error_message
