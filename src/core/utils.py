"""
Вспомогательные функции для работы с решениями задачи о нескольких рюкзаках.
"""

from typing import List, Tuple


def calculate_total_weight(assignment: List[int], weights: List[float]) -> float:
    """
    Вычисляет суммарный вес всех взятых предметов.
    """
    total = 0.0
    for i, knap in enumerate(assignment):
        if knap != 0:  # 0 = предмет не взят
            total += weights[i]
    return total


def calculate_profit(assignment: List[int], profits: List[float]) -> float:
    """
    Вычисляет суммарную ценность всех взятых предметов.
    """
    total = 0.0
    for i, knap in enumerate(assignment):
        if knap != 0:
            total += profits[i]
    return total


def calculate_weights_per_knapsack(assignment: List[int], weights: List[float], 
                                   m: int) -> List[float]:
    """
    Вычисляет суммарный вес в каждом рюкзаке.
    """
    result = [0.0] * m
    for i, knap in enumerate(assignment):
        if knap != 0:
            result[knap - 1] += weights[i]
    return result


def is_valid_solution(assignment: List[int], weights: List[float], 
                      capacities: List[float]) -> Tuple[bool, List[float]]:
    """
    Проверяет допустимость решения.
    
    Args:
        assignment: Список длины n, где значение — номер рюкзака (0 = не взят)
        weights: Веса предметов
        capacities: Вместимости рюкзаков (длина m)
        
    Returns:
        (допустимо_ли, веса_по_рюкзакам)
    """
    m = len(capacities)
    weights_per_knap = calculate_weights_per_knapsack(assignment, weights, m)
    
    for i in range(m):
        if weights_per_knap[i] > capacities[i] + 1e-9:  # небольшой допуск на погрешность
            return False, weights_per_knap
    
    return True, weights_per_knap


def encode_assignment(assignment: List[int]) -> str:
    """
    Кодирует распределение в строку для сохранения в CSV.
    """
    return ','.join(str(x) for x in assignment)


def decode_assignment(encoded: str) -> List[int]:
    """
    Декодирует строку обратно в список.
    """
    return [int(x) for x in encoded.split(',')]