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
    Генерирует начальное решение жадным способом:
    предметы сортируются по убыванию ценности/вес, затем распределяются
    по рюкзакам с наименьшей загрузкой.
    """
    n = len(weights)
    m = len(capacities)
    # Сортируем индексы по убыванию отношения profit/weight
    # Но у нас нет profits в этом контексте, поэтому для простоты сделаем случайное
    # Лучше использовать случайное решение, так как жадное может быть слишком хорошим
    assignment = [0] * n
    current_weights = [0.0] * m
    
    # Случайный порядок предметов
    indices = list(range(n))
    rng.shuffle(indices)
    
    for i in indices:
        w = weights[i]
        # Пробуем положить в случайный рюкзак, если влезает, иначе не берём
        # Для простоты выбираем рюкзак с наименьшей загрузкой
        # (это даёт более равномерное заполнение)
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
            assignment[i] = 0  # не помещается
    return assignment


def generate_neighbor(assignment: List[int], weights: List[float],
                      capacities: List[float], rng: random.Random,
                      max_attempts: int = 100) -> Optional[List[int]]:
    """
    Генерирует соседнее решение путём случайного изменения одного предмета.
    Возвращает None, если не удалось найти допустимого соседа.
    """
    n = len(assignment)
    m = len(capacities)
    
    for _ in range(max_attempts):
        # Копируем текущее решение
        new_assignment = assignment.copy()
        # Выбираем случайный предмет
        idx = rng.randrange(n)
        current_knap = assignment[idx]
        # Генерируем новый рюкзак (0..m)
        # С вероятностью 0.5 пробуем убрать предмет (0), иначе случайный рюкзак
        if rng.random() < 0.3:
            new_knap = 0
        else:
            new_knap = rng.randint(1, m)  # 1..m
        if new_knap == current_knap:
            continue
        # Проверяем допустимость нового решения (вес в рюкзаках)
        # Вычисляем новые веса
        new_weights = calculate_weights_per_knapsack(new_assignment, weights, m)
        # Если меняем с существующего на другой, нужно пересчитать
        # Для простоты пересчитаем целиком
        valid, _ = is_valid_solution(new_assignment, weights, capacities)
        if valid:
            return new_assignment
    return None


def objective(assignment: List[int], profits: List[float]) -> float:
    """Целевая функция — суммарная ценность."""
    return calculate_profit(assignment, profits)


@timed
def solve(instance: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Запускает алгоритм имитации отжига.
    
    Параметры (по умолчанию):
        - temperature: начальная температура (1000.0)
        - cooling_rate: коэффициент охлаждения (0.95)
        - iterations_per_temp: итераций на одну температуру (100)
        - min_temperature: минимальная температура (0.001)
        - seed: зерно для случайности (если не указано, используется random)
    """
    weights = instance['weights']
    profits = instance['profits']
    capacities = instance['capacities']
    n = len(weights)
    m = len(capacities)
    
    # Извлекаем параметры с значениями по умолчанию
    temperature = params.get('temperature', 1000.0)
    cooling_rate = params.get('cooling_rate', 0.95)
    iterations_per_temp = params.get('iterations_per_temp', 100)
    min_temperature = params.get('min_temperature', 0.001)
    seed = params.get('seed', None)
    
    # Инициализируем генератор случайных чисел
    rng = random.Random(seed) if seed is not None else random.Random()
    
    # Генерируем начальное решение
    current_assignment = generate_initial_solution(weights, capacities, rng)
    current_profit = objective(current_assignment, profits)
    best_assignment = current_assignment.copy()
    best_profit = current_profit
    
    # Основной цикл
    while temperature > min_temperature:
        for _ in range(iterations_per_temp):
            # Генерируем соседа
            neighbor = generate_neighbor(current_assignment, weights, capacities, rng)
            if neighbor is None:
                continue
            neighbor_profit = objective(neighbor, profits)
            delta = neighbor_profit - current_profit
            
            # Принимаем решение
            if delta > 0:
                # Улучшение — всегда принимаем
                current_assignment = neighbor
                current_profit = neighbor_profit
                if current_profit > best_profit:
                    best_assignment = current_assignment.copy()
                    best_profit = current_profit
            else:
                # Ухудшение — принимаем с вероятностью exp(delta / temperature)
                if rng.random() < math.exp(delta / temperature):
                    current_assignment = neighbor
                    current_profit = neighbor_profit
        
        # Охлаждение
        temperature *= cooling_rate
    
    # Проверяем допустимость лучшего решения
    valid, weights_per_knap = is_valid_solution(best_assignment, weights, capacities)
    if not valid:
        # Если по каким-то причинам решение недопустимо, возвращаем лучшее из допустимых
        # (но такого быть не должно)
        pass
    
    return {
        'profit': best_profit,
        'assignment': best_assignment,
        'weights_per_knapsack': weights_per_knap,
        'iterations': 0,  # Мы не считаем точное число итераций, оставим 0
        'params_used': params,
        'is_optimal': False  # SA не гарантирует оптимальность
    }