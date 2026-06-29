#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для первичного анализа результатов экспериментов.
Читает full_results.csv, строит графики и выводит сводные таблицы.
Автоматически адаптируется под любые алгоритмы, добавленные в CSV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Настройки
# ----------------------------------------------------------------------
INPUT_CSV = Path("results/full_results.csv")   # путь к исходным данным
OUTPUT_DIR = Path("analysis/figures")          # папка для сохранения графиков
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Стиль графиков
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

# ----------------------------------------------------------------------
# 2. Загрузка и предобработка данных
# ----------------------------------------------------------------------
print("Загрузка данных...")
df = pd.read_csv(INPUT_CSV)

# Удаляем записи тюнинга (они не нужны для основного анализа)
df = df[~df["algorithm"].str.contains("_tune", na=False)]

# Преобразуем типы
df["instance_id"] = df["instance_id"].astype(int)
df["n"] = df["n"].astype(int)
df["profit"] = df["profit"].astype(float)
df["time_sec"] = df["time_sec"].astype(float)
df["iterations"] = df["iterations"].astype(float)

print(f"Загружено записей: {len(df)}")
print(f"Уникальные алгоритмы: {df['algorithm'].unique().tolist()}")

# ----------------------------------------------------------------------
# 3. Вычисление оптимальной прибыли (из брутфорса)
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
# 4. Агрегация по запускам (для каждого экземпляра и алгоритма)
# ----------------------------------------------------------------------
# Группируем по (instance_id, n, algorithm) и вычисляем:
#   - лучшую прибыль (max)
#   - медианное время (устойчиво к выбросам)
#   - медианное количество итераций
#   - стандартное отклонение прибыли (стабильность)
agg = df.groupby(["instance_id", "n", "algorithm"]).agg(
    best_profit=("profit", "max"),
    median_time=("time_sec", "median"),
    median_iterations=("iterations", "median"),
    std_profit=("profit", "std")
).reset_index()

# Заполняем NaN в std_profit (если только один запуск) значением 0
agg["std_profit"] = agg["std_profit"].fillna(0)

print(f"Агрегировано записей (уникальных экземпляр-алгоритм): {len(agg)}")

# ----------------------------------------------------------------------
# 5. Добавляем колонку с gap (относительное отклонение от оптимума)
# ----------------------------------------------------------------------
if not optimal.empty:
    agg = agg.merge(optimal, on=["instance_id", "n"], how="left")
    # Вычисляем gap только если есть оптимальная прибыль и она > 0
    agg["gap_percent"] = np.where(
        agg["optimal_profit"].notna() & (agg["optimal_profit"] > 0),
        (agg["optimal_profit"] - agg["best_profit"]) / agg["optimal_profit"] * 100,
        np.nan
    )
else:
    agg["gap_percent"] = np.nan

# Удаляем алгоритм brute из агрегированного датасета (он не нужен для сравнения)
agg = agg[agg["algorithm"] != "brute"]

# ----------------------------------------------------------------------
# 6. Построение графиков
# ----------------------------------------------------------------------
print("Построение графиков...")

# ----------------------------------------------------------------------
# График 1: Boxplot gap (только для n=10,20)
# ----------------------------------------------------------------------
gap_data = agg[agg["n"].isin([10, 20]) & agg["gap_percent"].notna()]
if not gap_data.empty:
    plt.figure()
    sns.boxplot(data=gap_data, x="n", y="gap_percent", hue="algorithm")
    plt.title("Отклонение от оптимума (gap, %) на малых размерностях")
    plt.xlabel("Количество предметов (n)")
    plt.ylabel("Gap, %")
    plt.legend(title="Алгоритм")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gap_boxplot.png", dpi=150)
    plt.close()
    print("  - gap_boxplot.png сохранён")
else:
    print("  - пропущен (нет данных для gap)")

# ----------------------------------------------------------------------
# График 2: Scatter времени выполнения (медианное время на экземпляр)
# ----------------------------------------------------------------------
if not agg.empty:
    plt.figure()
    # Используем логарифмическую шкалу по времени, если есть большой разброс
    sns.scatterplot(data=agg, x="n", y="median_time", hue="algorithm", alpha=0.7)
    plt.title("Время выполнения (медиана по запускам)")
    plt.xlabel("Количество предметов (n)")
    plt.ylabel("Время, секунды")
    plt.yscale("log")  # логарифмическая шкала для лучшей визуализации
    plt.legend(title="Алгоритм")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_scatter.png", dpi=150)
    plt.close()
    print("  - time_scatter.png сохранён")

# ----------------------------------------------------------------------
# График 3: Boxplot времени по размерностям и алгоритмам
# ----------------------------------------------------------------------
if not agg.empty:
    plt.figure()
    # Создаём столбец для группировки (n + алгоритм)
    agg["n_alg"] = agg["n"].astype(str) + " - " + agg["algorithm"]
    sns.boxplot(data=agg, x="n_alg", y="median_time")
    plt.title("Распределение времени выполнения (медиана по запускам)")
    plt.xlabel("Размерность - Алгоритм")
    plt.ylabel("Время, секунды")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_boxplot.png", dpi=150)
    plt.close()
    print("  - time_boxplot.png сохранён")

# ----------------------------------------------------------------------
# График 4: Стабильность (boxplot стандартных отклонений прибыли)
# ----------------------------------------------------------------------
# Используем только те алгоритмы, где есть более одного запуска (std>0)
stability_data = agg[agg["std_profit"] > 0]
if not stability_data.empty:
    plt.figure()
    # Добавим небольшой jitter для наглядности
    sns.boxplot(data=stability_data, x="n", y="std_profit", hue="algorithm")
    plt.title("Стабильность алгоритмов (разброс прибыли по запускам)")
    plt.xlabel("Количество предметов (n)")
    plt.ylabel("Стандартное отклонение прибыли")
    plt.legend(title="Алгоритм")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stability_boxplot.png", dpi=150)
    plt.close()
    print("  - stability_boxplot.png сохранён")
else:
    print("  - пропущен (нет данных с несколькими запусками)")

# ----------------------------------------------------------------------
# График 5: Сравнение лучшей прибыли на больших n (без эталона)
# ----------------------------------------------------------------------
large_data = agg[agg["n"] >= 50].copy()
if not large_data.empty:
    plt.figure()
    sns.boxplot(data=large_data, x="n", y="best_profit", hue="algorithm")
    plt.title("Лучшая найденная прибыль на больших размерностях")
    plt.xlabel("Количество предметов (n)")
    plt.ylabel("Прибыль")
    plt.legend(title="Алгоритм")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "profit_large_boxplot.png", dpi=150)
    plt.close()
    print("  - profit_large_boxplot.png сохранён")

# ----------------------------------------------------------------------
# 7. Сводные таблицы (вывод в консоль)
# ----------------------------------------------------------------------
print("\n" + "="*70)
print("СВОДНЫЕ ТАБЛИЦЫ")
print("="*70)

# Таблица 1: средние показатели по группам (n, algorithm)
summary = agg.groupby(["n", "algorithm"]).agg(
    count=("instance_id", "count"),
    mean_gap=("gap_percent", "mean"),
    median_gap=("gap_percent", "median"),
    pct_optimal=("gap_percent", lambda x: (x == 0).mean() * 100 if len(x) > 0 else np.nan),
    mean_time=("median_time", "mean"),
    median_time=("median_time", "median"),
    mean_std_profit=("std_profit", "mean")
).reset_index()

# Округлим для удобства
for col in ["mean_gap", "median_gap", "pct_optimal", "mean_time", "median_time", "mean_std_profit"]:
    if col in summary.columns:
        summary[col] = summary[col].round(4)

# Выводим таблицу, группируя по n
for n_val in sorted(summary["n"].unique()):
    print(f"\n--- Размерность n = {n_val} ---")
    sub = summary[summary["n"] == n_val].drop(columns=["n"])
    print(sub.to_string(index=False))

# ----------------------------------------------------------------------
# 8. Сохранение сводной таблицы в CSV (опционально)
# ----------------------------------------------------------------------
summary.to_csv(OUTPUT_DIR.parent / "summary_stats.csv", index=False)
print(f"\nСводная таблица сохранена в {OUTPUT_DIR.parent / 'summary_stats.csv'}")

# ----------------------------------------------------------------------
# 9. Дополнительно: матрица корреляции (очень кратко)
# ----------------------------------------------------------------------
# Можно вывести корреляцию между размерностью, временем и прибылью (по лучшим результатам)
# Это необязательно, но добавим для полноты.
if not agg.empty:
    corr_data = agg[["n", "best_profit", "median_time", "median_iterations"]].copy()
    # Удалим алгоритм-зависимые выбросы? Лучше сгруппировать по n и алгоритму? 
    # Для простоты возьмём средние по алгоритмам.
    corr_mean = agg.groupby("n")[["best_profit", "median_time"]].mean().reset_index()
    if len(corr_mean) > 1:
        corr = corr_mean[["best_profit", "median_time"]].corr()
        print("\nКорреляция между средней прибылью и временем (по размерностям):")
        print(corr)

print("\nАнализ завершён. Графики сохранены в папке:", OUTPUT_DIR)