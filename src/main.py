#!/usr/bin/env python
"""
Главный запускатель для экспериментов с задачами о нескольких рюкзаках.

Поддерживаемые алгоритмы:
    - brute    : полный перебор (эталон, ТОЛЬКО ДЛЯ МАЛЫХ n)
    - sa       : имитация отжига (будет реализован)
    - ts       : табу-поиск (будет реализован)
    - ga       : генетический алгоритм (будет реализован)

Примеры использования:
    # Запустить брутфорс на одном экземпляре
    python main.py --algorithm brute --instance data/n_10/instance_0001

    # Запустить брутфорс на всех экземплярах для n=10
    python main.py --algorithm brute --n 10

    # Запустить брутфорс на всех экземплярах для n от 10 до 20
    python main.py --algorithm brute --n-min 10 --n-max 20

    # Запустить все алгоритмы на всех экземплярах (по 30 повторов)
    python main.py --all-algorithms --all --runs 30
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

# Добавляем пути для импорта модулей
sys.path.append(str(Path(__file__).parent))

from core.loader import (
    load_instance,
    load_instances_by_n,
    load_instances_by_n_range,
    load_all_instances,
    get_available_n
)
from core.result_saver import (
    append_result,
    load_optimal_profits,
    load_existing_keys,
    format_result
)
from core.utils import is_valid_solution
from core.timer import timed

# Импортируем алгоритмы
from algorithms import brute_force

# Регистрация алгоритмов
ALGORITHMS = {
    'brute': brute_force.solve,
    # 'sa': simulated_annealing.solve,      # будет добавлен позже
    # 'ts': tabu_search.solve,
    # 'ga': genetic_algorithm.solve,
}

# Максимальное n для брутфорса (защита от дурака)
BRUTE_MAX_N = 25


def run_algorithm_on_instance(algorithm_func, instance: Dict[str, Any],
                              params: Dict[str, Any], runs: int = 1,
                              algorithm_name: str = None,
                              optimal_profit: Optional[float] = None,
                              result_writer=None,
                              existing_keys: Set[tuple] = None) -> None:
    """
    Запускает алгоритм на одном экземпляре заданное число раз.
    Результаты сразу записываются через result_writer (прогрессивно).
    Если уже есть результаты с такими же параметрами, пропускает.
    """
    if algorithm_name is None:
        algorithm_name = algorithm_func.__name__

    # Преобразуем params в каноническую JSON-строку для сравнения
    params_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
    
    # Проверяем, есть ли уже результаты для этого (instance_id, n, algorithm, params)
    instance_id = instance['instance_id']
    n = instance['n']
    key = (instance_id, n, algorithm_name, params_str)
    
    if existing_keys is not None and key in existing_keys:
        print(f"    Пропускаем: уже есть результаты для алгоритма {algorithm_name} на этом экземпляре с такими параметрами")
        return

    for run_id in range(1, runs + 1):
        # Запускаем алгоритм
        result = algorithm_func(instance, params)

        # Проверяем допустимость решения
        valid, weights_per_knap = is_valid_solution(
            result['assignment'],
            instance['weights'],
            instance['capacities']
        )

        # Формируем запись для CSV
        record = format_result(
            instance=instance,
            algorithm_name=algorithm_name,
            run_id=run_id,
            profit=result['profit'],
            assignment=result['assignment'],
            weights_per_knapsack=weights_per_knap,
            time_sec=result['time_sec'],
            iterations=result.get('iterations', 0),
            params=params,
            optimal_profit=optimal_profit,
            is_optimal=result.get('is_optimal', None)
        )
        # Прогрессивно записываем
        result_writer(record)
    
    # Добавляем ключ в множество существующих (чтобы не дублировать в рамках текущего запуска)
    if existing_keys is not None:
        existing_keys.add(key)


def main():
    parser = argparse.ArgumentParser(
        description='Запуск метаэвристик для задачи о нескольких рюкзаках',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Примеры:
  python main.py --algorithm brute --instance data/n_10/instance_0001
  python main.py --algorithm brute --n 10
  python main.py --algorithm brute --n-min 10 --n-max 20
  python main.py --all-algorithms --all --runs 30
        """
    )

    # Выбор алгоритмов
    parser.add_argument('--algorithm', type=str, choices=list(ALGORITHMS.keys()),
                        help='Название алгоритма для запуска')
    parser.add_argument('--all-algorithms', action='store_true',
                        help='Запустить все доступные алгоритмы')

    # Выбор данных
    parser.add_argument('--instance', type=str,
                        help='Путь к конкретному экземпляру (папка instance_XXXX)')
    parser.add_argument('--n', type=int,
                        help='Запустить на всех экземплярах для данного n')
    parser.add_argument('--n-min', type=int,
                        help='Минимальное n для диапазона (используется с --n-max)')
    parser.add_argument('--n-max', type=int,
                        help='Максимальное n для диапазона (используется с --n-min)')
    parser.add_argument('--all', action='store_true',
                        help='Запустить на всех экземплярах (все n)')

    # Параметры запуска
    parser.add_argument('--runs', type=int, default=1,
                        help='Количество запусков для усреднения (по умолчанию: 1)')
    parser.add_argument('--params', type=str, default='{}',
                        help='JSON-строка с гиперпараметрами алгоритма')
    parser.add_argument('--params-file', type=str, default=None,
                        help='Путь к JSON-файлу с гиперпараметрами')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Папка для сохранения результатов (по умолчанию: results)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Папка с данными (по умолчанию: data)')

    args = parser.parse_args()

    # Проверка корректности комбинации аргументов выбора данных
    has_instance = args.instance is not None
    has_n = args.n is not None
    has_range = (args.n_min is not None) or (args.n_max is not None)
    has_all = args.all

    if (args.n_min is not None) != (args.n_max is not None):
        parser.error("--n-min и --n-max должны указываться вместе")

    ways = [has_instance, has_n, has_range, has_all]
    if sum(ways) != 1:
        parser.error("Необходимо указать ровно один способ выбора данных: --instance, --n, --n-min/--n-max или --all")

    # Загружаем параметры алгоритма
    if args.params_file:
        with open(args.params_file, 'r', encoding='utf-8') as f:
            params = json.load(f)
    else:
        params = json.loads(args.params)

    # Определяем, какие алгоритмы запускать
    algorithms_to_run = {}
    if args.all_algorithms:
        algorithms_to_run = ALGORITHMS.copy()
    elif args.algorithm:
        algorithms_to_run[args.algorithm] = ALGORITHMS[args.algorithm]
    else:
        print("Ошибка: укажите --algorithm или --all-algorithms")
        sys.exit(1)

    # Загружаем экземпляры
    instances = []
    data_dir = Path(args.data_dir)

    if args.instance:
        instance_path = Path(args.instance)
        if not instance_path.exists():
            print(f"Ошибка: экземпляр {instance_path} не найден")
            sys.exit(1)
        instances = [load_instance(instance_path)]
        print(f"Загружен 1 экземпляр: {instance_path}")
    elif args.n is not None:
        instances = load_instances_by_n(args.n, data_dir)
        if not instances:
            print(f"Ошибка: не найдены экземпляры для n={args.n} в папке {data_dir}")
            sys.exit(1)
        print(f"Загружено {len(instances)} экземпляров для n={args.n}")
    elif args.n_min is not None and args.n_max is not None:
        instances = load_instances_by_n_range(args.n_min, args.n_max, data_dir)
        if not instances:
            print(f"Ошибка: не найдены экземпляры для n в диапазоне [{args.n_min}, {args.n_max}]")
            sys.exit(1)
        print(f"Загружено {len(instances)} экземпляров для n от {args.n_min} до {args.n_max}")
    elif args.all:
        all_data = load_all_instances(data_dir)
        for n, inst_list in all_data.items():
            instances.extend(inst_list)
        if not instances:
            print(f"Ошибка: не найдены экземпляры в папке {data_dir}")
            sys.exit(1)
        print(f"Загружено {len(instances)} экземпляров для всех n")

    instances.sort(key=lambda x: (x['n'], x['instance_id']))

    # Загружаем оптимальные значения
    results_file = Path(args.output_dir) / 'full_results.csv'
    optimal_profits = load_optimal_profits(results_file)
    if optimal_profits:
        print(f"Загружено {len(optimal_profits)} оптимальных значений из {results_file}")

    # Загружаем существующие ключи, чтобы избежать повторных вычислений
    existing_keys = load_existing_keys(results_file)
    if existing_keys:
        print(f"Загружено {len(existing_keys)} существующих записей из {results_file}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Функция для прогрессивной записи
    def writer(result_record):
        append_result(result_record, results_file)

    # Для каждого алгоритма
    for algo_name, algo_func in algorithms_to_run.items():
        print(f"\nЗапуск алгоритма: {algo_name}")
        total = len(instances)

        for idx, instance in enumerate(instances, start=1):
            n = instance['n']
            instance_id = instance['instance_id']

            if algo_name == 'brute' and n > BRUTE_MAX_N:
                print(f"  [{idx}/{total}] Пропускаем n={n}, id={instance_id} (слишком большой для брутфорса, > {BRUTE_MAX_N})")
                continue

            print(f"  [{idx}/{total}] n={n}, id={instance_id}")

            opt_profit = optimal_profits.get((instance_id, n), None)

            run_algorithm_on_instance(
                algorithm_func=algo_func,
                instance=instance,
                params=params,
                runs=args.runs,
                algorithm_name=algo_name,
                optimal_profit=opt_profit,
                result_writer=writer,
                existing_keys=existing_keys
            )

    print(f"\nВсе результаты сохранены в {results_file}")


if __name__ == '__main__':
    main()