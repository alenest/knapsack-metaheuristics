#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для анализа результатов экспериментов.
Генерирует как общие графики, так и отдельные для каждой размерности.
Автоматически подхватывает новые алгоритмы.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Настройки
# ----------------------------------------------------------------------
INPUT_CSV = Path("results/full_results.csv")
OUTPUT_DIR = Path("analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Отключаем предупреждения (для чистоты вывода)
import warnings
warnings.filterwarnings("ignore")

# Стиль графиков
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["legend.fontsize"] = 11

# ----------------------------------------------------------------------
# 2. Загрузка и предобработка
# ----------------------------------------------------------------------
print("Загрузка данных...")
df = pd.read_csv(INPUT_CSV)

# Удаляем записи тюнинга
df = df[~df["algorithm"].str.contains("_tune", na=False)]

# Приводим типы
df["instance_id"] = df["instance_id"].astype(int)
df["n"] = df["n"].astype(int)
df["profit"] = df["profit"].astype(float)
df["time_sec"] = df["time_sec"].astype(float)
df["iterations"] = df["iterations"].astype(float)

print(f"Загружено записей: {len(df)}")
print(f"Уникальные алгоритмы: {df['algorithm'].unique().tolist()}")

# ----------------------------------------------------------------------
# 3. Оптимальные значения (из брутфорса)
# ----------------------------------------------------------------------
brute = df[df["algorithm"] == "brute"].copy()
if not brute.empty:
    optimal = brute.groupby(["instance_id", "n"])["profit"].max().reset_index()
    optimal.rename(columns={"profit": "optimal_profit"}, inplace=True)
    print(f"Найдено оптимальных значений: {len(optimal)}")
else:
    optimal = pd.DataFrame(columns=["instance_id", "n", "optimal_profit"])
    print("Внимание: брутфорс не найден, gap не будет вычислен.")

# ----------------------------------------------------------------------
# 4. Агрегация по запускам
# ----------------------------------------------------------------------
agg = df.groupby(["instance_id", "n", "algorithm"]).agg(
    best_profit=("profit", "max"),
    median_time=("time_sec", "median"),
    median_iterations=("iterations", "median"),
    std_profit=("profit", "std")
).reset_index()
agg["std_profit"] = agg["std_profit"].fillna(0)

# Добавляем gap
if not optimal.empty:
    agg = agg.merge(optimal, on=["instance_id", "n"], how="left")
    agg["gap_percent"] = np.where(
        agg["optimal_profit"].notna() & (agg["optimal_profit"] > 0),
        (agg["optimal_profit"] - agg["best_profit"]) / agg["optimal_profit"] * 100,
        np.nan
    )
else:
    agg["gap_percent"] = np.nan

# Исключаем brute из сравнения
agg = agg[agg["algorithm"] != "brute"]
print(f"Агрегировано записей (уникальных экземпляр-алгоритм): {len(agg)}")

# ----------------------------------------------------------------------
# 5. Функция для построения и сохранения графиков по размерностям
# ----------------------------------------------------------------------
def save_plot(fig, filename, subdir=""):
    """Сохраняет график в указанную подпапку."""
    path = OUTPUT_DIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  - {filename} сохранён в {path}")

# ----------------------------------------------------------------------
# 6. Построение графиков
# ----------------------------------------------------------------------
print("\nПостроение графиков...")

# --- 6.1. Общий график времени (scatter) ---
fig, ax = plt.subplots()
sns.scatterplot(data=agg, x="n", y="median_time", hue="algorithm", alpha=0.7, ax=ax)
ax.set_title("Время выполнения (медиана по запускам)")
ax.set_xlabel("Количество предметов (n)")
ax.set_ylabel("Время, секунды")
ax.set_yscale("log")
ax.legend(title="Алгоритм")
save_plot(fig, "time_scatter_all.png")

# --- 6.2. Отдельные графики для каждой размерности ---
# Для каждого уникального n строим три боксплота: gap (если есть), время, прибыль
for n_val in sorted(agg["n"].unique()):
    sub_data = agg[agg["n"] == n_val].copy()
    if sub_data.empty:
        continue
    
    subdir = f"n_{n_val}"
    
    # 6.2.1. Gap (только если есть данные)
    gap_data = sub_data[sub_data["gap_percent"].notna()]
    if not gap_data.empty:
        fig, ax = plt.subplots()
        sns.boxplot(data=gap_data, x="algorithm", y="gap_percent", ax=ax)
        ax.set_title(f"Отклонение от оптимума (gap, %) для n={n_val}")
        ax.set_xlabel("Алгоритм")
        ax.set_ylabel("Gap, %")
        ax.grid(True, linestyle="--", alpha=0.6)
        save_plot(fig, "gap_boxplot.png", subdir)
    
    # 6.2.2. Время выполнения
    fig, ax = plt.subplots()
    sns.boxplot(data=sub_data, x="algorithm", y="median_time", ax=ax)
    ax.set_title(f"Время выполнения для n={n_val}")
    ax.set_xlabel("Алгоритм")
    ax.set_ylabel("Время, секунды")
    ax.grid(True, linestyle="--", alpha=0.6)
    save_plot(fig, "time_boxplot.png", subdir)
    
    # 6.2.3. Лучшая прибыль
    fig, ax = plt.subplots()
    sns.boxplot(data=sub_data, x="algorithm", y="best_profit", ax=ax)
    ax.set_title(f"Лучшая найденная прибыль для n={n_val}")
    ax.set_xlabel("Алгоритм")
    ax.set_ylabel("Прибыль")
    ax.grid(True, linestyle="--", alpha=0.6)
    save_plot(fig, "profit_boxplot.png", subdir)
    
    # 6.2.4. Стабильность (если есть std>0)
    stable = sub_data[sub_data["std_profit"] > 0]
    if not stable.empty:
        fig, ax = plt.subplots()
        sns.boxplot(data=stable, x="algorithm", y="std_profit", ax=ax)
        ax.set_title(f"Разброс прибыли по запускам для n={n_val}")
        ax.set_xlabel("Алгоритм")
        ax.set_ylabel("Стандартное отклонение прибыли")
        ax.grid(True, linestyle="--", alpha=0.6)
        save_plot(fig, "stability_boxplot.png", subdir)

# --- 6.3. Дополнительно: общий график прибыли для больших n (n>=50) ---
large = agg[agg["n"] >= 50].copy()
if not large.empty:
    fig, ax = plt.subplots()
    sns.boxplot(data=large, x="n", y="best_profit", hue="algorithm", ax=ax)
    ax.set_title("Лучшая прибыль на больших размерностях")
    ax.set_xlabel("Количество предметов (n)")
    ax.set_ylabel("Прибыль")
    ax.legend(title="Алгоритм")
    save_plot(fig, "profit_large_all.png")

# ----------------------------------------------------------------------
# 7. Сводная таблица с добавлением средней прибыли
# ----------------------------------------------------------------------
print("\n" + "="*70)
print("СВОДНАЯ ТАБЛИЦА ПО РАЗМЕРНОСТЯМ И АЛГОРИТМАМ")
print("="*70)

summary = agg.groupby(["n", "algorithm"]).agg(
    count=("instance_id", "count"),
    mean_gap=("gap_percent", "mean"),
    median_gap=("gap_percent", "median"),
    pct_optimal=("gap_percent", lambda x: (x == 0).mean() * 100 if len(x) > 0 else np.nan),
    mean_profit=("best_profit", "mean"),
    median_profit=("best_profit", "median"),
    mean_time=("median_time", "mean"),
    median_time=("median_time", "median"),
    mean_std_profit=("std_profit", "mean")
).reset_index()

# Округление
for col in ["mean_gap", "median_gap", "pct_optimal", "mean_profit", "median_profit",
            "mean_time", "median_time", "mean_std_profit"]:
    if col in summary.columns:
        summary[col] = summary[col].round(4)

# Вывод по n
for n_val in sorted(summary["n"].unique()):
    print(f"\n--- Размерность n = {n_val} ---")
    sub = summary[summary["n"] == n_val].drop(columns=["n"])
    print(sub.to_string(index=False))

# Сохраняем таблицу
summary.to_csv(OUTPUT_DIR.parent / "summary_stats.csv", index=False)
print(f"\nСводная таблица сохранена в {OUTPUT_DIR.parent / 'summary_stats.csv'}")

print("\nАнализ завершён. Все графики в папке:", OUTPUT_DIR)