"""
Сохранение результатов экспериментов в CSV.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime


def save_results(results: List[Dict[str, Any]], 
                 output_path: Path,
                 append: bool = True) -> None:
    """Сохраняет список результатов в CSV-файл."""
    if not results:
        return
    
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    
    priority_keys = [
        'timestamp', 'instance_id', 'n', 'm', 'algorithm', 'run_id',
        'profit', 'total_weight', 'time_sec', 'gap_percent', 'is_optimal',
        'iterations', 'params', 'seed', 'capacities'
    ]
    other_keys = sorted([k for k in all_keys if k not in priority_keys])
    fieldnames = [k for k in priority_keys if k in all_keys] + other_keys
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if append and output_path.exists() else 'w'
    
    with open(output_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        for row in results:
            processed_row = {}
            for key, value in row.items():
                if key in ['params', 'metadata'] and isinstance(value, dict):
                    processed_row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                elif isinstance(value, list):
                    processed_row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    processed_row[key] = value
            writer.writerow(processed_row)


def append_result(result: Dict[str, Any], output_path: Path) -> None:
    """Добавляет один результат в CSV-файл (прогрессивная запись)."""
    save_results([result], output_path, append=True)


def load_optimal_profits(results_path: Path) -> Dict[tuple, float]:
    """
    Загружает оптимальные значения прибыли из файла с результатами брутфорса.
    Возвращает словарь {(instance_id, n): optimal_profit}.
    """
    optimal = {}
    if not results_path.exists():
        return optimal
    
    with open(results_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('algorithm') == 'brute' and row.get('is_optimal') == 'True':
                try:
                    instance_id = int(row['instance_id'])
                    n = int(row['n'])
                    profit = float(row['profit'])
                    key = (instance_id, n)
                    if key not in optimal or profit > optimal[key]:
                        optimal[key] = profit
                except (ValueError, KeyError):
                    continue
    return optimal


def load_existing_keys(results_path: Path) -> Set[tuple]:
    """
    Загружает множество уже существующих ключей (instance_id, n, algorithm, params_str)
    из CSV-файла результатов, чтобы избежать повторных вычислений.
    """
    keys = set()
    if not results_path.exists():
        return keys
    
    with open(results_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                instance_id = int(row['instance_id'])
                n = int(row['n'])
                algorithm = row['algorithm']
                params_str = row.get('params', '{}')
                # Для корректного сравнения нормализуем params_str (JSON-строка)
                # При сохранении мы используем sort_keys=True, поэтому сравнение строк корректно
                keys.add((instance_id, n, algorithm, params_str))
            except (ValueError, KeyError):
                continue
    return keys


def format_result(instance: Dict[str, Any],
                  algorithm_name: str,
                  run_id: int,
                  profit: float,
                  assignment: List[int],
                  weights_per_knapsack: List[float],
                  time_sec: float,
                  iterations: int,
                  params: Dict[str, Any],
                  optimal_profit: Optional[float] = None,
                  is_optimal: Optional[bool] = None) -> Dict[str, Any]:
    """
    Форматирует результат одного запуска в единый словарь для CSV.
    """
    total_weight = sum(weights_per_knapsack)
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'instance_id': instance['instance_id'],
        'n': instance['n'],
        'm': instance['m'],
        'algorithm': algorithm_name,
        'run_id': run_id,
        'profit': profit,
        'total_weight': total_weight,
        'time_sec': time_sec,
        'iterations': iterations,
        'params': params,
        'capacities': instance['capacities'],
        'assignment': assignment,
        'weights_per_knapsack': weights_per_knapsack,
    }
    
    if 'seed' in instance['metadata']:
        result['seed'] = instance['metadata']['seed']
    
    if optimal_profit is not None and optimal_profit > 0:
        gap = (optimal_profit - profit) / optimal_profit * 100
        result['gap_percent'] = round(gap, 4)
    else:
        result['gap_percent'] = None
    
    if is_optimal is not None:
        result['is_optimal'] = is_optimal
    
    return result