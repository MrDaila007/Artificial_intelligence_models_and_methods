#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования веб-приложения
"""

import sys
import os
import numpy as np

def test_lab1():
    """Тестирование Lab1"""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ LAB1: Кластеризация")
    print("=" * 60)
    
    # Настройка путей
    lab1_path = os.path.join(os.path.dirname(__file__), 'Lab1')
    sys.path.insert(0, lab1_path)
    os.chdir(lab1_path)
    
    # Очистка кэша
    modules_to_remove = [k for k in sys.modules.keys() if 'solution' in k]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    try:
        from solution import HierarchicalClustering, compute_mismatch_measure
        
        # Тестовые данные
        np.random.seed(42)
        class1 = np.random.randn(20, 2) + np.array([0, 0])
        class2 = np.random.randn(20, 2) + np.array([3, 3])
        class3 = np.random.randn(20, 2) + np.array([0, 3])
        X = np.vstack([class1, class2, class3])
        true_labels = np.array([1]*20 + [2]*20 + [3]*20)
        
        print(f"✓ Данные созданы: {X.shape[0]} объектов, {X.shape[1]} признаков")
        
        # Тест кластеризации
        clusterer = HierarchicalClustering(metric='euclidean')
        clusterer.fit(X, n_clusters=3)
        print("✓ Кластеризация выполнена")
        
        # Тест меры несоответствия
        mu = compute_mismatch_measure(true_labels, clusterer.labels_)
        print(f"✓ Мера несоответствия: μ = {mu:.4f}")
        
        # Проверка результатов
        unique_clusters = len(np.unique(clusterer.labels_))
        print(f"✓ Число кластеров: {unique_clusters}")
        
        print("\n✅ LAB1: Все тесты пройдены!")
        return True
        
    except Exception as e:
        print(f"\n❌ LAB1: Ошибка - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lab2():
    """Тестирование Lab2"""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ LAB2: Классификация")
    print("=" * 60)
    
    # Настройка путей
    lab2_path = os.path.join(os.path.dirname(__file__), 'Lab2')
    sys.path.insert(0, lab2_path)
    os.chdir(lab2_path)
    
    # Очистка кэша
    modules_to_remove = [k for k in sys.modules.keys() if 'solution' in k]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    try:
        from solution import PatternRecognitionClassifier, run_experiment
        from sklearn.model_selection import train_test_split
        
        # Тестовые данные
        np.random.seed(42)
        class1 = np.random.randn(30, 2) + np.array([0, 0])
        class2 = np.random.randn(30, 2) + np.array([3, 3])
        class3 = np.random.randn(30, 2) + np.array([0, 3])
        X = np.vstack([class1, class2, class3])
        y = np.array([1]*30 + [2]*30 + [3]*30)
        
        print(f"✓ Данные созданы: {X.shape[0]} объектов, {X.shape[1]} признаков")
        
        # Тест разбиения
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✓ Разбиение: train={len(X_train)}, test={len(X_test)}")
        
        # Тест классификации
        clf = PatternRecognitionClassifier(
            distance_metric='euclidean',
            comparison_func='mean'
        )
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(f"✓ Классификация: Φ^A = {score:.4f}")
        
        # Тест полного эксперимента
        results_df = run_experiment(X, y, test_size=0.2, random_state=42)
        print(f"✓ Полный эксперимент: {len(results_df)} комбинаций")
        print(f"  Лучший результат: {results_df['Φ^A'].max():.4f}")
        
        print("\n✅ LAB2: Все тесты пройдены!")
        return True
        
    except Exception as e:
        print(f"\n❌ LAB2: Ошибка - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Тестирование импортов для веб-приложения"""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ИМПОРТОВ ДЛЯ ВЕБ-ПРИЛОЖЕНИЯ")
    print("=" * 60)
    
    try:
        import streamlit as st
        print("✓ streamlit импортирован")
        
        import pandas as pd
        print("✓ pandas импортирован")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib импортирован")
        
        print("\n✅ Все зависимости доступны!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Ошибка импорта: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ВЕБ-ПРИЛОЖЕНИЯ")
    print("=" * 60 + "\n")
    
    results = []
    
    # Тест зависимостей
    results.append(("Зависимости", test_imports()))
    
    # Тест Lab1
    results.append(("Lab1", test_lab1()))
    
    # Тест Lab2
    results.append(("Lab2", test_lab2()))
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("Веб-приложение готово к использованию.")
        print("\nЗапустите: streamlit run app.py")
    else:
        print("❌ НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ")
        print("Проверьте ошибки выше.")
    print("=" * 60 + "\n")
    
    sys.exit(0 if all_passed else 1)

