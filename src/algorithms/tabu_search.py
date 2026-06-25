"""
Алгоритм табу-поиска (Tabu Search) для задачи о нескольких рюкзаках.
Поддерживает перемещения и обмены предметов, табу-список, аспирацию и перезапуск.
"""

import random
from typing import Dict, Any, List, Tuple, Optional

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
    """Жадное начальное решение со случайным порядком предметов."""
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
    return assignment


@timed
def solve(instance: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Параметры (рекомендуемые после тюнинга):
        tabu_tenure = 10
        max_iterations = 1000
        neighborhood_size = 50
        swap_probability = 0.3
    """
    weights = instance['weights']
    profits = instance['profits']
    capacities = instance['capacities']
    n = len(weights)
    m = len(capacities)

    # Параметры
    tabu_tenure = params.get('tabu_tenure', 10)
    max_iterations = params.get('max_iterations', 1000)
    neighborhood_size = params.get('neighborhood_size', 50)
    swap_probability = params.get('swap_probability', 0.3)
    seed = params.get('seed', None)
    rng = random.Random(seed) if seed is not None else random.Random()

    # Начальное решение
    current_assignment = generate_initial_solution(weights, capacities, rng)
    current_profit = calculate_profit(current_assignment, profits)
    best_assignment = current_assignment.copy()
    best_profit = current_profit

    # Табу-список (храним как список и множество для быстрой проверки)
    tabu_list = []
    tabu_set = set()

    def add_tabu(move: Tuple) -> None:
        """Добавляет ход в табу-список, удаляя самый старый при переполнении."""
        tabu_list.append(move)
        key = _move_key(move)
        tabu_set.add(key)
        if len(tabu_list) > tabu_tenure:
            old = tabu_list.pop(0)
            old_key = _move_key(old)
            tabu_set.discard(old_key)

    def is_tabu(move: Tuple) -> bool:
        return _move_key(move) in tabu_set

    def _move_key(move: Tuple) -> str:
        """Уникальный строковый ключ для хода."""
        if move[0] == 'move':
            return f"move_{move[1]}_{move[2]}"
        else:  # swap
            return f"swap_{move[1]}_{move[2]}"

    iteration = 0
    no_improve_count = 0

    while iteration < max_iterations:
        iteration += 1
        neighbors = []
        attempts = 0
        max_attempts = neighborhood_size * 20

        # Генерация соседей
        while len(neighbors) < neighborhood_size and attempts < max_attempts:
            attempts += 1
            if rng.random() < swap_probability:
                # Обмен предметами из двух разных рюкзаков
                items = [i for i, k in enumerate(current_assignment) if k != 0]
                if len(items) >= 2:
                    idx1, idx2 = rng.sample(items, 2)
                    knap1, knap2 = current_assignment[idx1], current_assignment[idx2]
                    if knap1 != knap2:
                        move = ('swap', idx1, idx2)
                        if is_tabu(move):
                            continue
                        new_assignment = current_assignment.copy()
                        new_assignment[idx1] = knap2
                        new_assignment[idx2] = knap1
                        valid, _ = is_valid_solution(new_assignment, weights, capacities)
                        if valid:
                            neighbors.append((new_assignment, move))
            else:
                # Перемещение предмета в другой рюкзак (включая 0)
                idx = rng.randrange(n)
                current_knap = current_assignment[idx]
                options = [k for k in range(m + 1) if k != current_knap]
                if not options:
                    continue
                new_knap = rng.choice(options)
                move = ('move', idx, new_knap)
                if is_tabu(move):
                    continue
                new_assignment = current_assignment.copy()
                new_assignment[idx] = new_knap
                valid, _ = is_valid_solution(new_assignment, weights, capacities)
                if valid:
                    neighbors.append((new_assignment, move))

        if not neighbors:
            # Нет допустимых соседей – перезапуск
            current_assignment = generate_initial_solution(weights, capacities, rng)
            current_profit = calculate_profit(current_assignment, profits)
            if current_profit > best_profit:
                best_assignment = current_assignment.copy()
                best_profit = current_profit
            tabu_list.clear()
            tabu_set.clear()
            no_improve_count = 0
            continue

        # Выбор лучшего соседа (с аспирацией)
        best_neighbor = None
        best_neighbor_profit = -1
        best_move = None
        for neighbor, move in neighbors:
            profit = calculate_profit(neighbor, profits)
            if is_tabu(move) and profit <= best_profit:
                continue  # табу-ход не лучше текущего лучшего
            if profit > best_neighbor_profit:
                best_neighbor_profit = profit
                best_neighbor = neighbor
                best_move = move

        if best_neighbor is None:
            # Все соседи – табу и не лучше лучшего – перезапуск
            current_assignment = generate_initial_solution(weights, capacities, rng)
            current_profit = calculate_profit(current_assignment, profits)
            if current_profit > best_profit:
                best_assignment = current_assignment.copy()
                best_profit = current_profit
            tabu_list.clear()
            tabu_set.clear()
            no_improve_count = 0
            continue

        # Применяем ход
        current_assignment = best_neighbor
        current_profit = best_neighbor_profit
        add_tabu(best_move)

        if current_profit > best_profit:
            best_profit = current_profit
            best_assignment = current_assignment.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Перезапуск при застревании (10% от max_iterations без улучшения)
        if no_improve_count > max_iterations // 10:
            current_assignment = generate_initial_solution(weights, capacities, rng)
            current_profit = calculate_profit(current_assignment, profits)
            tabu_list.clear()
            tabu_set.clear()
            no_improve_count = 0

    valid, weights_per_knap = is_valid_solution(best_assignment, weights, capacities)
    return {
        'profit': best_profit,
        'assignment': best_assignment,
        'weights_per_knapsack': weights_per_knap,
        'iterations': iteration,
        'params_used': params,
        'is_optimal': False
    }