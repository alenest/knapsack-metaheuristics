#!/usr/bin/env python
"""
Скрипт для автоматического перебора параметров алгоритма имитации отжига.
Запускается на одном экземпляре (по умолчанию n=10, instance_id=1)
и проверяет все комбинации параметров, записывая результаты в CSV.

Использование:
    python src/tune_sa.py
    python src/tune_sa.py --instance data/n_10/instance_0001
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.loader import load_instance
from core.result_saver import append_result, load_optimal_profits, format_result
from core.utils import is_valid_solution
from algorithms import simulated_annealing


def tune_sa(instance_path: str, output_file: str = 'results/tuning_results.csv'):
    """
    Перебирает все комбинации параметров SA на указанном экземпляре.
    """
    instance = load_instance(Path(instance_path))
    print(f"Тестируем на: n={instance['n']}, instance_id={instance['instance_id']}")
    
    optimal_profits = load_optimal_profits(Path('results/full_results.csv'))
    opt_profit = optimal_profits.get((instance['instance_id'], instance['n']), None)
    if opt_profit:
        print(f"Оптимальная прибыль: {opt_profit}")
    else:
        print("Оптимальная прибыль не найдена, gap не будет вычислен")
    
    # Сокращённые сетки для быстрого тюнинга
    temperatures = [500, 1000, 2000]
    cooling_rates = [0.95, 0.98, 0.99]
    iterations_per_temp = [50, 100, 200]
    swap_probabilities = [0.2, 0.3, 0.4]
    
    total_combinations = (len(temperatures) * len(cooling_rates) *
                          len(iterations_per_temp) * len(swap_probabilities))
    print(f"Всего комбинаций: {total_combinations}")
    
    counter = 0
    for temp in temperatures:
        for cooling in cooling_rates:
            for iterations in iterations_per_temp:
                for swap_prob in swap_probabilities:
                    counter += 1
                    params = {
                        'temperature': temp,
                        'cooling_rate': cooling,
                        'iterations_per_temp': iterations,
                        'swap_probability': swap_prob,
                        'seed': 12345
                    }
                    
                    print(f"[{counter}/{total_combinations}] "
                          f"T={temp}, cooling={cooling}, iter={iterations}, swap={swap_prob}")
                    
                    result = simulated_annealing.solve(instance, params)
                    valid, weights_per_knap = is_valid_solution(
                        result['assignment'],
                        instance['weights'],
                        instance['capacities']
                    )
                    
                    record = format_result(
                        instance=instance,
                        algorithm_name='sa_tune',
                        run_id=1,
                        profit=result['profit'],
                        assignment=result['assignment'],
                        weights_per_knapsack=weights_per_knap,
                        time_sec=result['time_sec'],
                        iterations=result['iterations'],
                        params=params,
                        optimal_profit=opt_profit,
                        is_optimal=False
                    )
                    
                    # Добавляем поля для удобства
                    record['temperature'] = temp
                    record['cooling_rate'] = cooling
                    record['iterations_per_temp'] = iterations
                    record['swap_probability'] = swap_prob
                    
                    append_result(record, Path(output_file))
    
    print(f"\nГотово! Результаты сохранены в {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Автоматический перебор параметров для SA')
    parser.add_argument('--instance', type=str, default='data/n_10/instance_0001',
                        help='Путь к экземпляру')
    parser.add_argument('--output', type=str, default='results/tuning_results.csv',
                        help='Выходной CSV-файл')
    args = parser.parse_args()
    tune_sa(args.instance, args.output)


if __name__ == '__main__':
    main()