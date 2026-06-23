"""
Загрузчик экземпляров задачи из папки data/.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_instance(instance_path: Path) -> Dict[str, Any]:
    """Загружает один экземпляр задачи из папки instance_XXXX."""
    meta_path = instance_path / 'meta.json'
    data_path = instance_path / 'data.npz'
    
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {instance_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data.npz not found in {instance_path}")
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    data = np.load(data_path)
    weights = data['weights'].tolist()
    profits = data['profits'].tolist()
    
    instance_id = int(instance_path.name.split('_')[1])
    
    return {
        'weights': weights,
        'profits': profits,
        'capacities': metadata['capacities'],
        'metadata': metadata,
        'instance_id': instance_id,
        'n': metadata['n'],
        'm': metadata['m']
    }


def load_instances_by_n(n: int, data_dir: Path = Path('data')) -> List[Dict[str, Any]]:
    """Загружает все экземпляры для заданного n."""
    n_dir = data_dir / f'n_{n}'
    if not n_dir.exists():
        return []
    
    instances = []
    for instance_dir in sorted(n_dir.glob('instance_*')):
        try:
            instances.append(load_instance(instance_dir))
        except Exception as e:
            print(f"  Предупреждение: не удалось загрузить {instance_dir.name}: {e}")
    
    return instances


def load_instances_by_n_range(n_min: int, n_max: int, data_dir: Path = Path('data')) -> List[Dict[str, Any]]:
    """Загружает все экземпляры для n в диапазоне [n_min, n_max]."""
    instances = []
    for n_dir in data_dir.glob('n_*'):
        try:
            n = int(n_dir.name.split('_')[1])
            if n_min <= n <= n_max:
                instances.extend(load_instances_by_n(n, data_dir))
        except (IndexError, ValueError):
            continue
    return instances


def load_all_instances(data_dir: Path = Path('data')) -> Dict[int, List[Dict[str, Any]]]:
    """Загружает все экземпляры для всех n."""
    result = {}
    for n_dir in sorted(data_dir.glob('n_*')):
        try:
            n = int(n_dir.name.split('_')[1])
            result[n] = load_instances_by_n(n, data_dir)
        except (IndexError, ValueError):
            continue
    return result


def get_available_n(data_dir: Path = Path('data')) -> List[int]:
    """Возвращает список доступных n в папке data."""
    n_list = []
    for n_dir in data_dir.glob('n_*'):
        try:
            n = int(n_dir.name.split('_')[1])
            n_list.append(n)
        except (IndexError, ValueError):
            continue
    return sorted(n_list)