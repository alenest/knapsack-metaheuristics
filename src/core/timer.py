"""
Утилиты для замера времени выполнения алгоритмов.
"""

import time
import functools
from typing import Callable, Any, Dict


def timed(func: Callable) -> Callable:
    """
    Декоратор для замера времени выполнения функции.
    Добавляет поле 'time_sec' в возвращаемый словарь.
    
    Использование:
        @timed
        def solve(instance, params):
            return {'profit': 100, 'assignment': [...]}
    
    Возвращает:
        Словарь с результатами + поле 'time_sec'
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Dict[str, Any]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        result['time_sec'] = end - start
        return result
    return wrapper


def measure_time(func: Callable, *args, **kwargs) -> tuple:
    """
    Измеряет время выполнения произвольной функции.
    
    Returns:
        (результат, время_в_секундах)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


class Timer:
    """
    Контекстный менеджер для замера времени блока кода.
    
    Использование:
        with Timer() as timer:
            # код
        print(f"Время: {timer.elapsed:.4f} сек")
    """
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start