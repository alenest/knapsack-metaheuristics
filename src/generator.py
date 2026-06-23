"""
Генератор экземпляров задачи о нескольких рюкзаках (Multiple Knapsack Problem).
- Предметы: веса и ценности генерируются независимо (без корреляции).
- Поддерживаются целые или вещественные числа для весов и ценностей.
- Вместимости рюкзаков по умолчанию случайные (неравные).
- Сохраняет данные в структуру data/n_{n}/instance_{id}/
"""

import argparse
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


def get_next_instance_id(n_dir: Path) -> int:
    """
    Находит наименьший свободный номер для нового экземпляра в папке n_dir.
    Если есть пропуски, заполняет их; если все заняты — добавляет новый.
    """
    existing = list(n_dir.glob('instance_*'))
    if not existing:
        return 1

    numbers = []
    for p in existing:
        try:
            num = int(p.name.split('_')[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue

    numbers.sort()
    expected = 1
    for num in numbers:
        if num == expected:
            expected += 1
        else:
            return expected
    return expected


def generate_weights(n: int, weight_min: float, weight_max: float,
                     integer: bool, rng: random.Random) -> List[float]:
    """
    Генерирует веса предметов.
    - integer=True: целые числа в диапазоне [weight_min, weight_max]
    - integer=False: вещественные числа в диапазоне [weight_min, weight_max]
    """
    if integer:
        return [float(rng.randint(int(weight_min), int(weight_max))) for _ in range(n)]
    else:
        return [rng.uniform(weight_min, weight_max) for _ in range(n)]


def generate_profits(n: int, profit_min: float, profit_max: float,
                     integer: bool, rng: random.Random) -> List[float]:
    """
    Генерирует ценности предметов (независимо от весов).
    - integer=True: целые числа в диапазоне [profit_min, profit_max]
    - integer=False: вещественные числа в диапазоне [profit_min, profit_max]
    """
    if integer:
        return [float(rng.randint(int(profit_min), int(profit_max))) for _ in range(n)]
    else:
        return [rng.uniform(profit_min, profit_max) for _ in range(n)]


def generate_capacities(m: int, total_weight: float, capacity_ratio: float,
                        mode: str, rng: random.Random) -> List[float]:
    """
    Генерирует вместимости рюкзаков.
    - mode='equal': все рюкзаки имеют одинаковую вместимость.
    - mode='random': вместимости распределены случайно (доли от общего бюджета).
    Возвращает список длиной m.
    """
    total_capacity = capacity_ratio * total_weight

    if mode == 'equal':
        cap = total_capacity / m
        return [cap] * m

    elif mode == 'random':
        # Генерируем m случайных коэффициентов (например, от 0.5 до 1.5)
        coeffs = [rng.uniform(0.5, 1.5) for _ in range(m)]
        sum_coeffs = sum(coeffs)
        # Нормализуем так, чтобы сумма = total_capacity
        return [total_capacity * c / sum_coeffs for c in coeffs]

    else:
        raise ValueError(f"Unknown capacities mode: {mode}")


def generate_instance(params: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """
    Генерирует один экземпляр задачи по переданным параметрам.
    Возвращает словарь с данными.
    """
    n = params['n']
    m = params['m']
    weight_min = params['weight_min']
    weight_max = params['weight_max']
    profit_min = params['profit_min']
    profit_max = params['profit_max']
    capacity_ratio = params['capacity_ratio']
    capacities_mode = params['capacities_mode']
    weights_integer = params['weights_integer']
    profits_integer = params['profits_integer']

    weights = generate_weights(n, weight_min, weight_max, weights_integer, rng)
    profits = generate_profits(n, profit_min, profit_max, profits_integer, rng)
    total_weight = sum(weights)
    capacities = generate_capacities(m, total_weight, capacity_ratio, capacities_mode, rng)

    return {
        'weights': weights,
        'profits': profits,
        'capacities': capacities,
        'metadata': {
            'n': n,
            'm': m,
            'seed': params.get('seed'),
            'capacity_ratio': capacity_ratio,
            'capacities_mode': capacities_mode,
            'weight_min': weight_min,
            'weight_max': weight_max,
            'profit_min': profit_min,
            'profit_max': profit_max,
            'weights_integer': weights_integer,
            'profits_integer': profits_integer,
            'generated_at': datetime.now().isoformat()
        }
    }


def save_instance(instance: Dict[str, Any], instance_dir: Path) -> None:
    """Сохраняет экземпляр в папку instance_dir: meta.json и data.npz."""
    instance_dir.mkdir(parents=True, exist_ok=False)

    # Сохраняем метаданные (все параметры, кроме больших массивов)
    meta = instance['metadata'].copy()
    meta['capacities'] = instance['capacities']
    with open(instance_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Сохраняем большие массивы в сжатый бинарный формат
    np.savez_compressed(
        instance_dir / 'data.npz',
        weights=np.array(instance['weights'], dtype=np.float64),
        profits=np.array(instance['profits'], dtype=np.float64)
    )


def main():
    parser = argparse.ArgumentParser(
        description='Генератор задачи о нескольких рюкзаках',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python src/generator.py --n 50 --num 10
  python src/generator.py --n 100 --num 5 --seed 42 --m 8
  python src/generator.py --n 200 --num 3 --capacity-ratio 0.7 --capacities-mode equal
        """
    )

    # Обязательные параметры
    parser.add_argument('--n', type=int, required=True,
                        help='Количество предметов')

    # Количество экземпляров
    parser.add_argument('--num', type=int, default=1,
                        help='Количество экземпляров для генерации (по умолчанию: 1)')

    # Зерно
    parser.add_argument('--seed', type=int, default=None,
                        help='Базовое зерно (если не указано, выбирается случайно)')

    # Веса
    parser.add_argument('--weight-min', type=float, default=1.0,
                        help='Минимальный вес (по умолчанию: 1.0)')
    parser.add_argument('--weight-max', type=float, default=100.0,
                        help='Максимальный вес (по умолчанию: 100.0)')
    parser.add_argument('--int-weights', action='store_true', default=True,
                        help='Генерировать целые веса (по умолчанию: включено)')
    parser.add_argument('--float-weights', action='store_false', dest='int_weights',
                        help='Генерировать вещественные веса')

    # Ценности
    parser.add_argument('--profit-min', type=float, default=1.0,
                        help='Минимальная ценность (по умолчанию: 1.0)')
    parser.add_argument('--profit-max', type=float, default=100.0,
                        help='Максимальная ценность (по умолчанию: 100.0)')
    parser.add_argument('--int-profits', action='store_true', default=True,
                        help='Генерировать целые ценности (по умолчанию: включено)')
    parser.add_argument('--float-profits', action='store_false', dest='int_profits',
                        help='Генерировать вещественные ценности')

    # Рюкзаки
    parser.add_argument('--m', type=int, default=None,
                        help='Количество рюкзаков (если не указано, выбирается случайно от 2 до max(2, n//10))')
    parser.add_argument('--capacity-ratio', type=float, default=0.5,
                        help='Доля от суммы весов, распределяемая между рюкзаками (по умолчанию: 0.5)')
    parser.add_argument('--capacities-mode', choices=['equal', 'random'], default='random',
                        help='Способ генерации вместимостей: equal (все равны) или random (случайные доли) (по умолчанию: random)')

    # Выходная папка
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Корневая папка для данных (по умолчанию: data)')

    args = parser.parse_args()

    # Базовый seed
    base_seed = args.seed if args.seed is not None else random.randint(1, 1000000)
    print(f"Базовый seed: {base_seed}")

    # Определяем количество рюкзаков m
    if args.m is not None:
        m = args.m
    else:
        # Случайно от 2 до max(2, n//10)
        max_m = max(2, args.n // 10)
        rng_m = random.Random(base_seed)
        m = rng_m.randint(2, max_m)
    print(f"Количество рюкзаков: {m}")

    # Папка для данного n
    n_dir = Path(args.output_dir) / f'n_{args.n}'
    n_dir.mkdir(parents=True, exist_ok=True)

    # Параметры, общие для всех экземпляров серии
    base_params = {
        'n': args.n,
        'm': m,
        'weight_min': args.weight_min,
        'weight_max': args.weight_max,
        'profit_min': args.profit_min,
        'profit_max': args.profit_max,
        'capacity_ratio': args.capacity_ratio,
        'capacities_mode': args.capacities_mode,
        'weights_integer': args.int_weights,
        'profits_integer': args.int_profits,
    }

    # Генерируем num экземпляров
    generated = 0
    for i in range(args.num):
        seed = base_seed + i
        rng = random.Random(seed)
        params = base_params.copy()
        params['seed'] = seed

        instance = generate_instance(params, rng)

        instance_id = get_next_instance_id(n_dir)
        instance_dir = n_dir / f'instance_{instance_id:04d}'

        save_instance(instance, instance_dir)
        generated += 1
        print(f"  Создан экземпляр {instance_id:04d} (seed={seed})")

    print(f"Всего сгенерировано {generated} экземпляров для n={args.n} в папке {n_dir}")


if __name__ == '__main__':
    main()