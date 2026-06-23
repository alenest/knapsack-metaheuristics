"""
Алгоритм полного перебора для задачи о нескольких рюкзаках.
Гарантированно находит оптимальное решение.
Сложность: O((M+1)^N) — применим только для малых N.
"""

from pathlib import Path
import sys
from typing import Dict, Any, List

# Добавляем путь для импорта core
sys.path.append(str(Path(__file__).parent.parent))

from core.timer import timed
from core.utils import is_valid_solution, calculate_profit, calculate_weights_per_knapsack


@timed
def solve(instance: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Полный перебор для задачи о нескольких рюкзаках.
    
    Args:
        instance: Словарь с данными экземпляра
        params: Параметры алгоритма (пока не используются)
    
    Returns:
        Словарь с результатами:
            profit: float
            assignment: List[int]
            weights_per_knapsack: List[float]
            iterations: int (0 для единообразия)
            params_used: dict
            is_optimal: bool (всегда True)
    """
    weights = instance['weights']
    profits = instance['profits']
    capacities = instance['capacities']
    n = len(weights)
    m = len(capacities)
    
    best_profit = -1.0
    best_assignment = [0] * n
    
    # Рекурсивный перебор
    def backtrack(idx: int, current_assignment: List[int], 
                  current_weight_per_knap: List[float], 
                  current_profit: float) -> None:
        nonlocal best_profit, best_assignment
        
        # Если все предметы рассмотрены
        if idx == n:
            if current_profit > best_profit:
                best_profit = current_profit
                best_assignment = current_assignment.copy()
            return
        
        # Опции: 0 (не брать) или положить в один из рюкзаков
        options = list(range(m + 1))
        
        for knap in options:
            new_weight_per_knap = current_weight_per_knap.copy()
            new_profit = current_profit
            
            if knap == 0:
                backtrack(idx + 1, current_assignment + [0], 
                          new_weight_per_knap, new_profit)
            else:
                new_weight = current_weight_per_knap[knap - 1] + weights[idx]
                if new_weight <= capacities[knap - 1] + 1e-9:
                    new_weight_per_knap[knap - 1] = new_weight
                    new_profit += profits[idx]
                    backtrack(idx + 1, current_assignment + [knap], 
                              new_weight_per_knap, new_profit)
    
    initial_weights = [0.0] * m
    backtrack(0, [], initial_weights, 0.0)
    
    valid, weights_per_knap = is_valid_solution(best_assignment, weights, capacities)
    
    return {
        'profit': best_profit,
        'assignment': best_assignment,
        'weights_per_knapsack': weights_per_knap,
        'iterations': 0,
        'params_used': params,
        'is_optimal': True,
    }