# -*- coding: utf-8 -*-
"""
Веб-интерфейс для лабораторной работы №2
Распознавание образов с обучением
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

from solution import PatternRecognitionClassifier, run_experiment
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Лаб. работа №2: Классификация",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Лабораторная работа №2")
st.markdown("### Распознавание образов с обучением")

# Боковая панель для загрузки данных
st.sidebar.header("📁 Загрузка данных")
uploaded_file = st.sidebar.file_uploader(
    "Выберите CSV файл",
    type=['csv'],
    help="Загрузите файл с данными в формате CSV"
)

# Параметры классификации
st.sidebar.header("⚙️ Параметры")
metric = st.sidebar.selectbox(
    "Метрика расстояния",
    ["euclidean", "minkowski", "hamming"],
    help="Выберите метрику для вычисления расстояний"
)

comparison_func = st.sidebar.selectbox(
    "Функция сравнения",
    ["mean", "knn", "min"],
    help="Функция сравнения объекта с классом"
)

k = st.sidebar.number_input(
    "Параметр k (для knn)",
    min_value=1,
    max_value=20,
    value=3,
    step=1,
    help="Число ближайших соседей для метода knn"
)

test_size = st.sidebar.slider(
    "Размер тестовой выборки",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05,
    help="Доля данных для контрольной выборки"
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
            
            # Разбиение на обучающую и контрольную выборки
            if st.button("✂️ Разбить выборку", use_container_width=True):
                X_train, X_test, y_train, y_test = train_test_split(
                    data, labels, test_size=test_size, stratify=labels, random_state=42
                )
                
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                
                st.success("✅ Выборка разбита!")
                
                # Информация о разбиении
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Обучающая выборка", f"{len(X_train)} объектов")
                with col2:
                    st.metric("Контрольная выборка", f"{len(X_test)} объектов")
                
                # Проверка ограничения t_i / m_i >= 0.2
                st.subheader("📋 Проверка ограничения t_i / m_i >= 0.2")
                constraint_df = []
                all_ok = True
                for cls in np.unique(labels):
                    m_i = np.sum(y_train == cls)
                    t_i = np.sum(y_test == cls)
                    ratio = t_i / m_i if m_i > 0 else 0
                    status = "✅" if ratio >= 0.2 else "❌"
                    if ratio < 0.2:
                        all_ok = False
                    constraint_df.append({
                        "Класс": cls,
                        "m_i (обучение)": m_i,
                        "t_i (контроль)": t_i,
                        "t_i/m_i": f"{ratio:.2f}",
                        "Статус": status
                    })
                
                constraint_df = pd.DataFrame(constraint_df)
                st.dataframe(constraint_df, use_container_width=True, hide_index=True)
                
                if not all_ok:
                    st.warning("⚠️ Некоторые классы не удовлетворяют ограничению t_i/m_i >= 0.2")
            
            # Обучение и тестирование
            if 'X_train' in st.session_state:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🎓 Обучить и протестировать", type="primary", use_container_width=True):
                        X_train = st.session_state['X_train']
                        X_test = st.session_state['X_test']
                        y_train = st.session_state['y_train']
                        y_test = st.session_state['y_test']
                        
                        with st.spinner("Обучение классификатора..."):
                            classifier = PatternRecognitionClassifier(
                                distance_metric=metric,
                                comparison_func=comparison_func,
                                k=k
                            )
                            classifier.fit(X_train, y_train)
                            score = classifier.score(X_test, y_test)
                            
                            st.session_state['classifier'] = classifier
                            st.session_state['score'] = score
                            
                            st.success("✅ Классификация завершена!")
                            
                            # Результаты
                            st.header("📈 Результаты классификации")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Метрика", metric)
                            with col2:
                                st.metric("Функция", comparison_func)
                            with col3:
                                st.metric("Параметр k", k if comparison_func == 'knn' else "—")
                            with col4:
                                st.metric("Φ^A", f"{score:.4f}")
                            
                            # Визуализация
                            if data.shape[1] >= 2:
                                st.subheader("📉 Визуализация")
                                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                                
                                # Левый график: исходные данные
                                scatter1 = axes[0].scatter(data[:, 0], data[:, 1], 
                                                           c=labels, cmap='tab10', alpha=0.7, s=50)
                                axes[0].set_xlabel('Признак 1')
                                axes[0].set_ylabel('Признак 2')
                                axes[0].set_title('Исходные данные (все)')
                                plt.colorbar(scatter1, ax=axes[0], label='Класс')
                                
                                # Правый график: разбиение на train/test
                                axes[1].scatter(X_train[:, 0], X_train[:, 1], 
                                               c=y_train, cmap='tab10', alpha=0.5, 
                                               marker='o', label='Обучающая', s=50)
                                axes[1].scatter(X_test[:, 0], X_test[:, 1], 
                                               c=y_test, cmap='tab10', alpha=1.0, 
                                               marker='*', label='Контрольная', s=150, edgecolors='black')
                                axes[1].legend()
                                axes[1].set_xlabel('Признак 1')
                                axes[1].set_ylabel('Признак 2')
                                axes[1].set_title('Разбиение выборки')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                
                with col2:
                    if st.button("🔬 Полный эксперимент", use_container_width=True):
                        with st.spinner("Выполняется полный эксперимент..."):
                            results_df = run_experiment(data, labels, test_size=test_size, random_state=42)
                            
                            st.header("📊 Результаты полного эксперимента")
                            st.dataframe(results_df, use_container_width=True, hide_index=True)
                            
                            # Визуализация результатов
                            st.subheader("📈 Сравнение комбинаций")
                            
                            # Находим лучшую комбинацию
                            best_idx = results_df['Φ^A'].idxmax()
                            best_row = results_df.loc[best_idx]
                            
                            st.success(f"🏆 Лучшая комбинация: {best_row['Метрика']} + {best_row['Функция сравнения']} (Φ^A = {best_row['Φ^A']:.4f})")
                            
                            # График сравнения
                            fig, ax = plt.subplots(figsize=(12, 6))
                            x_labels = [f"{row['Метрика']}\n{row['Функция сравнения']}" 
                                       for _, row in results_df.iterrows()]
                            ax.bar(range(len(results_df)), results_df['Φ^A'])
                            ax.set_xticks(range(len(results_df)))
                            ax.set_xticklabels(x_labels, rotation=45, ha='right')
                            ax.set_ylabel('Φ^A')
                            ax.set_title('Сравнение функционала качества для разных комбинаций')
                            ax.grid(axis='y', alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
            else:
                st.info("👆 Сначала разбейте выборку на обучающую и контрольную")
                    
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
           - Функция сравнения (mean, knn, min)
           - Параметр k (для knn)
           - Размер тестовой выборки
        5. **Нажмите "Разбить выборку"**
        6. **Нажмите "Обучить и протестировать"** или **"Полный эксперимент"**
        
        ### Формат данных:
        - CSV файл с числовыми признаками
        - Один столбец с метками классов (может быть текстовым)
        - Первая строка - заголовки столбцов
        """)

