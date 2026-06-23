"""
Алгоритм имитации отжига (Simulated Annealing) для задачи о нескольких рюкзаках.
Поддерживает настройку гиперпараметров.
"""

import random
import math
from typing import Dict, Any, List, Optional

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.timer import timed
from core.utils import (
    is_valid_solution,
    calculate_profit,
    calculate_weights_per_knapsack
)


def generate_initial_solution(weights: List[float], capacities: List[float],
                              rng: random.Random) -> List[int]:
    """
    Генерирует начальное решение жадным способом с перемешиванием:
    предметы обрабатываются в случайном порядке, каждый помещается
    в рюкзак с наименьшей загрузкой, если влезает.
    """
    n = len(weights)
    m = len(capacities)
    assignment = [0] * n
    current_weights = [0.0] * m
    
    indices = list(range(n))
    rng.shuffle(indices)
    
    for i in indices:
        w = weights[i]
        best_knap = -1
        best_load = float('inf')
        for k in range(m):
            if current_weights[k] + w <= capacities[k] + 1e-9:
                if current_weights[k] < best_load:
                    best_load = current_weights[k]
                    best_knap = k
        if best_knap != -1:
            assignment[i] = best_knap + 1
            current_weights[best_knap] += w
        else:
            assignment[i] = 0
    return assignment


def generate_neighbor(assignment: List[int], weights: List[float],
                      capacities: List[float], rng: random.Random,
                      swap_probability: float = 0.3,
                      max_attempts: int = 100) -> Optional[List[int]]:
    """
    Генерирует соседнее решение путём случайного изменения:
    - swap_probability: вероятность выполнить обмен вместо перемещения.
    - С вероятностью 0.3 удаляем предмет, с вероятностью 0.7 перемещаем.
    - Если выбран swap: меняем местами два предмета из разных рюкзаков.
    """
    n = len(assignment)
    m = len(capacities)
    
    for _ in range(max_attempts):
        new_assignment = assignment.copy()
        
        # Решаем, какую операцию выполнить
        if rng.random() < swap_probability and n >= 2:
            # === SWAP: обмен двумя предметами между рюкзаками ===
            items = [i for i, k in enumerate(assignment) if k != 0]
            if len(items) >= 2:
                idx1, idx2 = rng.sample(items, 2)
                knap1, knap2 = assignment[idx1], assignment[idx2]
                if knap1 != knap2:
                    new_assignment[idx1] = knap2
                    new_assignment[idx2] = knap1
                    valid, _ = is_valid_solution(new_assignment, weights, capacities)
                    if valid:
                        return new_assignment
        else:
            # === MOVE: перемещение одного предмета ===
            idx = rng.randrange(n)
            current_knap = assignment[idx]
            
            # С вероятностью 0.3 пытаемся удалить, иначе переместить в другой рюкзак
            if rng.random() < 0.3:
                new_knap = 0
            else:
                options = [k for k in range(m + 1) if k != current_knap]
                if not options:
                    continue
                new_knap = rng.choice(options)
            
            if new_knap == current_knap:
                continue
            
            new_assignment[idx] = new_knap
            valid, _ = is_valid_solution(new_assignment, weights, capacities)
            if valid:
                return new_assignment
    
    return None


@timed
def solve(instance: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Запускает алгоритм имитации отжига.
    
    Параметры (оптимальные для n=10 по результатам тюнинга):
        - temperature: начальная температура (1000.0)
        - cooling_rate: коэффициент охлаждения (0.95)
        - iterations_per_temp: итераций на одну температуру (50)
        - min_temperature: минимальная температура (0.001)
        - swap_probability: вероятность swap-операции (0.2)
        - seed: зерно для случайности
    """
    weights = instance['weights']
    profits = instance['profits']
    capacities = instance['capacities']
    n = len(weights)
    m = len(capacities)
    
    # Извлекаем параметры с оптимальными значениями по умолчанию
    temperature = params.get('temperature', 1000.0)
    cooling_rate = params.get('cooling_rate', 0.95)
    iterations_per_temp = params.get('iterations_per_temp', 50)
    min_temperature = params.get('min_temperature', 0.001)
    swap_probability = params.get('swap_probability', 0.2)
    seed = params.get('seed', None)
    
    rng = random.Random(seed) if seed is not None else random.Random()
    
    # Генерируем начальное решение
    current_assignment = generate_initial_solution(weights, capacities, rng)
    current_profit = calculate_profit(current_assignment, profits)
    best_assignment = current_assignment.copy()
    best_profit = current_profit
    
    # Основной цикл
    iteration_count = 0
    while temperature > min_temperature:
        for _ in range(iterations_per_temp):
            iteration_count += 1
            neighbor = generate_neighbor(
                current_assignment, weights, capacities, rng,
                swap_probability=swap_probability,
                max_attempts=100
            )
            if neighbor is None:
                continue
            neighbor_profit = calculate_profit(neighbor, profits)
            delta = neighbor_profit - current_profit
            
            if delta > 0:
                current_assignment = neighbor
                current_profit = neighbor_profit
                if current_profit > best_profit:
                    best_assignment = current_assignment.copy()
                    best_profit = current_profit
            else:
                if rng.random() < math.exp(delta / temperature):
                    current_assignment = neighbor
                    current_profit = neighbor_profit
        
        temperature *= cooling_rate
    
    valid, weights_per_knap = is_valid_solution(best_assignment, weights, capacities)
    
    return {
        'profit': best_profit,
        'assignment': best_assignment,
        'weights_per_knapsack': weights_per_knap,
        'iterations': iteration_count,
        'params_used': params,
        'is_optimal': False
    }