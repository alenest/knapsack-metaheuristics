"""
Тюнинг параметров табу-поиска (Tabu Search).
Аналог tune_sa.py – перебирает сетку параметров на одном экземпляре.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from core.loader import load_instance
from core.result_saver import append_result, load_optimal_profits, format_result
from core.utils import is_valid_solution
from algorithms import tabu_search


def tune_ts(instance_path: str, output_file: str = 'results/tuning_results_ts.csv'):
    instance = load_instance(Path(instance_path))
    print(f"Тестируем на: n={instance['n']}, instance_id={instance['instance_id']}")

    optimal_profits = load_optimal_profits(Path('results/full_results.csv'))
    opt_profit = optimal_profits.get((instance['instance_id'], instance['n']), None)
    if opt_profit:
        print(f"Оптимальная прибыль: {opt_profit}")
    else:
        print("Оптимальная прибыль не найдена, gap не будет вычислен")

    # Сетка параметров
    tabu_tenures = [5, 10, 15, 20]
    max_iterations_list = [500, 1000, 2000]
    neighborhood_sizes = [20, 50, 100]
    swap_probabilities = [0.2, 0.3, 0.4]

    total = (len(tabu_tenures) * len(max_iterations_list) *
             len(neighborhood_sizes) * len(swap_probabilities))
    print(f"Всего комбинаций: {total}")

    counter = 0
    for tenure in tabu_tenures:
        for max_iter in max_iterations_list:
            for neigh_size in neighborhood_sizes:
                for swap_prob in swap_probabilities:
                    counter += 1
                    params = {
                        'tabu_tenure': tenure,
                        'max_iterations': max_iter,
                        'neighborhood_size': neigh_size,
                        'swap_probability': swap_prob,
                        'seed': 12345  # фиксированный для воспроизводимости
                    }
                    print(f"[{counter}/{total}] "
                          f"tenure={tenure}, iter={max_iter}, neigh={neigh_size}, swap={swap_prob}")

                    result = tabu_search.solve(instance, params)
                    valid, weights_per_knap = is_valid_solution(
                        result['assignment'],
                        instance['weights'],
                        instance['capacities']
                    )

                    record = format_result(
                        instance=instance,
                        algorithm_name='ts_tune',
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
                    # Добавляем параметры как отдельные колонки для удобства анализа
                    record['tabu_tenure'] = tenure
                    record['max_iterations'] = max_iter
                    record['neighborhood_size'] = neigh_size
                    record['swap_probability'] = swap_prob

                    append_result(record, Path(output_file))

    print(f"\nГотово! Результаты сохранены в {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str,
                        default='data/n_10/instance_0001',
                        help='Путь к экземпляру для тюнинга')
    parser.add_argument('--output', type=str,
                        default='results/tuning_results_ts.csv',
                        help='Путь для сохранения результатов')
    args = parser.parse_args()
    tune_ts(args.instance, args.output)


if __name__ == '__main__':
    main()