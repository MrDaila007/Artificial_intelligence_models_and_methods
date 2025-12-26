# -*- coding: utf-8 -*-
"""
Веб-интерфейс для лабораторной работы №1
Распознавание образов без обучения (иерархическая кластеризация)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Добавляем путь к модулю solution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from solution import HierarchicalClustering, compute_mismatch_measure

st.set_page_config(
    page_title="Лаб. работа №1: Кластеризация",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Лабораторная работа №1")
st.markdown("### Распознавание образов без обучения (иерархическая кластеризация)")

# Боковая панель для загрузки данных
st.sidebar.header("📁 Загрузка данных")
uploaded_file = st.sidebar.file_uploader(
    "Выберите CSV файл",
    type=['csv'],
    help="Загрузите файл с данными в формате CSV"
)

# Параметры кластеризации
st.sidebar.header("⚙️ Параметры")
metric = st.sidebar.selectbox(
    "Метрика расстояния",
    ["euclidean", "minkowski", "hamming"],
    help="Выберите метрику для вычисления расстояний между объектами"
)

n_clusters = st.sidebar.number_input(
    "Число кластеров (l)",
    min_value=2,
    max_value=20,
    value=3,
    step=1,
    help="Целевое число кластеров для кластеризации"
)

# Основная область
if uploaded_file is not None:
    try:
        # Загрузка данных
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Файл загружен: {uploaded_file.name}")
        
        # Выбор столбцов
        st.header("🔧 Настройка данных")
        
        col1, col2 = st.columns(2)
        
        with col1:
            label_col = st.selectbox(
                "Столбец с метками классов",
                df.columns.tolist(),
                index=len(df.columns) - 1 if len(df.columns) > 0 else 0,
                help="Выберите столбец, содержащий метки классов"
            )
        
        with col2:
            # Автоматически определяем числовые столбцы
            numeric_cols = [col for col in df.columns 
                          if col != label_col 
                          and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
            
            feature_cols = st.multiselect(
                "Столбцы признаков",
                numeric_cols,
                default=numeric_cols,
                help="Выберите столбцы, которые будут использоваться как признаки"
            )
        
        if len(feature_cols) == 0:
            st.warning("⚠️ Выберите хотя бы один столбец признаков!")
        else:
            # Подготовка данных
            data = df[feature_cols].values.astype(float)
            labels = df[label_col].values
            
            # Преобразование меток в числа
            if labels.dtype == 'object':
                unique_labels = np.unique(labels)
                label_map = {label: i+1 for i, label in enumerate(unique_labels)}
                labels = np.array([label_map[l] for l in labels])
            
            # Информация о данных
            st.info(f"📊 Загружено {len(data)} объектов, {len(feature_cols)} признаков, {len(np.unique(labels))} классов")
            
            # Предпросмотр данных
            with st.expander("👀 Предпросмотр данных"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Кнопка запуска кластеризации
            if st.button("🚀 Запустить кластеризацию", type="primary", use_container_width=True):
                with st.spinner("Выполняется кластеризация..."):
                    # Кластеризация
                    clusterer = HierarchicalClustering(metric=metric)
                    clusterer.fit(data, n_clusters)
                    
                    # Вычисление меры несоответствия
                    mu = compute_mismatch_measure(labels, clusterer.labels_)
                    
                    # Результаты
                    st.header("📈 Результаты кластеризации")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Метрика", metric)
                    with col2:
                        st.metric("Число кластеров", n_clusters)
                    with col3:
                        st.metric("μ(T₀,T₁)", f"{mu:.4f}")
                    
                    # Разбиение на кластеры
                    st.subheader("📊 Разбиение X' = (X'₁, ..., X'ₗ)")
                    cluster_counts = {}
                    for i in range(1, n_clusters + 1):
                        count = np.sum(clusterer.labels_ == i)
                        cluster_counts[f"X'_{i}"] = count
                    
                    cluster_df = pd.DataFrame({
                        "Кластер": list(cluster_counts.keys()),
                        "Число объектов": list(cluster_counts.values())
                    })
                    st.dataframe(cluster_df, use_container_width=True, hide_index=True)
                    
                    # Визуализация
                    st.subheader("📉 Визуализация")
                    
                    if data.shape[1] >= 2:
                        # График 1: Исходные данные с истинными метками
                        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                        
                        scatter1 = axes[0].scatter(data[:, 0], data[:, 1], 
                                                   c=labels, cmap='tab10', alpha=0.7, s=50)
                        axes[0].set_xlabel('Признак 1')
                        axes[0].set_ylabel('Признак 2')
                        axes[0].set_title('Исходные данные (истинные метки классов)')
                        plt.colorbar(scatter1, ax=axes[0], label='Класс')
                        
                        # График 2: Результаты кластеризации
                        scatter2 = axes[1].scatter(data[:, 0], data[:, 1], 
                                                   c=clusterer.labels_, cmap='viridis', alpha=0.7, s=50)
                        axes[1].set_xlabel('Признак 1')
                        axes[1].set_ylabel('Признак 2')
                        axes[1].set_title('Результаты кластеризации')
                        plt.colorbar(scatter2, ax=axes[1], label='Кластер')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Дендрограмма
                        if clusterer.linkage_matrix is not None:
                            st.subheader("🌳 Дендрограмма")
                            fig2, ax = plt.subplots(figsize=(12, 6))
                            from scipy.cluster.hierarchy import dendrogram
                            dendrogram(clusterer.linkage_matrix, ax=ax)
                            ax.set_title('Дендрограмма иерархической кластеризации')
                            ax.set_xlabel('Индекс объекта')
                            ax.set_ylabel('Расстояние')
                            plt.tight_layout()
                            st.pyplot(fig2)
                    else:
                        st.info("Для визуализации нужно минимум 2 признака")
                    
                    # Сохранение результатов в сессии
                    st.session_state['clusterer'] = clusterer
                    st.session_state['mu'] = mu
                    st.session_state['data'] = data
                    st.session_state['labels'] = labels
                    
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        st.exception(e)
else:
    st.info("👈 Загрузите CSV файл в боковой панели, чтобы начать работу")
    
    # Пример использования
    with st.expander("📖 Инструкция"):
        st.markdown("""
        ### Как использовать:
        1. **Загрузите CSV файл** через боковую панель
        2. **Выберите столбец с метками классов** (обычно последний)
        3. **Выберите столбцы признаков** (числовые столбцы)
        4. **Настройте параметры**:
           - Метрика расстояния (euclidean, minkowski, hamming)
           - Число кластеров (l)
        5. **Нажмите "Запустить кластеризацию"**
        
        ### Формат данных:
        - CSV файл с числовыми признаками
        - Один столбец с метками классов (может быть текстовым)
        - Первая строка - заголовки столбцов
        """)

