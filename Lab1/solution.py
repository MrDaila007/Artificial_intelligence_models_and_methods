# -*- coding: utf-8 -*-
"""
Лабораторная работа №1
Задача распознавания образов без обучения
Алгоритм иерархической кластеризации

Автор: Eliseev D. I.
Дата: 10.10.2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Условный импорт tkinter (для работы без GUI)
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Предупреждение: tkinter не установлен. GUI недоступен.")

# ============================================================================
# МЕТРИКИ РАССТОЯНИЯ
# ============================================================================

def euclidean_distance(x1, x2):
    """Метрика Евклида: s(x1, x2) = sqrt(sum((x1i - x2i)^2))"""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def minkowski_distance(x1, x2, p=2):
    """Метрика Минковского: s(x1, x2) = (sum(|x1i - x2i|^p))^(1/p)"""
    return np.power(np.sum(np.abs(x1 - x2) ** p), 1/p)


def hamming_distance(x1, x2):
    """Метрика Хэмминга: s(x1, x2) = sum(|x1i - x2i|)"""
    return np.sum(np.abs(x1 - x2))


# ============================================================================
# АЛГОРИТМ ИЕРАРХИЧЕСКОЙ КЛАСТЕРИЗАЦИИ
# ============================================================================

class HierarchicalClustering:
    """
    Алгоритм иерархической кластеризации для задачи T1
    (распознавание образов без обучения)
    """
    
    def __init__(self, metric='euclidean'):
        """
        Инициализация алгоритма
        
        Parameters:
        -----------
        metric : str
            Метрика расстояния: 'euclidean', 'minkowski', 'hamming'
        """
        self.metric = metric
        self.clusters = None
        self.linkage_matrix = None
        self.labels_ = None
        
    def fit(self, X, n_clusters):
        """
        Выполнение кластеризации
        
        Parameters:
        -----------
        X : np.ndarray
            Матрица объектов (n_samples, n_features)
        n_clusters : int
            Целевое число кластеров (l)
            
        Returns:
        --------
        self
        """
        # Шаг 0: Каждый объект — отдельный кластер
        # Шаги 1-4: Иерархическая агломеративная кластеризация
        
        self.linkage_matrix = linkage(X, method='average', metric=self.metric)
        self.labels_ = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        
        # Формируем кластеры
        self.clusters = {}
        for i in range(1, n_clusters + 1):
            self.clusters[i] = X[self.labels_ == i]
            
        return self
    
    def get_clustering_vector(self, x, n_clusters):
        """
        Получить кластеризационный вектор C(x) для объекта x
        
        Parameters:
        -----------
        x : np.ndarray
            Объект
        n_clusters : int
            Число кластеров
            
        Returns:
        --------
        np.ndarray
            Кластеризационный вектор C(x) = (C1(x), ..., Cl(x))
        """
        C = np.zeros(n_clusters)
        idx = np.where((self.clusters == x).all(axis=1))[0]
        if len(idx) > 0:
            cluster_id = self.labels_[idx[0]]
            C[cluster_id - 1] = 1
        return C
    
    def plot_dendrogram(self, title='Дендрограмма иерархической кластеризации'):
        """Построить дендрограмму"""
        plt.figure(figsize=(12, 6))
        dendrogram(self.linkage_matrix)
        plt.title(title)
        plt.xlabel('Индекс объекта')
        plt.ylabel('Расстояние')
        plt.tight_layout()
        plt.show()


# ============================================================================
# ВЫЧИСЛЕНИЕ МЕРЫ НЕСООТВЕТСТВИЯ
# ============================================================================

def compute_mismatch_measure(true_labels, predicted_labels):
    """
    Вычисление меры несоответствия информации μ(T0, T1)
    
    Parameters:
    -----------
    true_labels : np.ndarray
        Истинные метки классов (информационный вектор P)
    predicted_labels : np.ndarray
        Предсказанные метки кластеров (кластеризационный вектор C)
        
    Returns:
    --------
    float
        Мера несоответствия μ ∈ [0, 1]
    """
    n_samples = len(true_labels)
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(predicted_labels)
    
    l = len(unique_true)
    
    # Шаг 0: Формируем наборы A1, ..., Al
    # Шаг 1: Для каждого Ai находим максимальное число вхождений
    
    # Матрица соответствия
    contingency = np.zeros((l, len(unique_pred)))
    for i, true_class in enumerate(unique_true):
        mask = true_labels == true_class
        for j, pred_class in enumerate(unique_pred):
            contingency[i, j] = np.sum(predicted_labels[mask] == pred_class)
    
    # Жадный алгоритм поиска лучшего соответствия
    used_pred = set()
    total_matches = 0
    
    for _ in range(l):
        best_match = 0
        best_i, best_j = -1, -1
        for i in range(l):
            for j in range(len(unique_pred)):
                if j not in used_pred and contingency[i, j] > best_match:
                    best_match = contingency[i, j]
                    best_i, best_j = i, j
        if best_j != -1:
            used_pred.add(best_j)
            total_matches += best_match
            contingency[best_i, :] = -1  # Исключаем строку
    
    # Мера несоответствия
    mu = (n_samples - total_matches) / n_samples
    
    return mu


# ============================================================================
# ГРАФИЧЕСКИЙ ИНТЕРФЕЙС
# ============================================================================

class ClusteringApp:
    """Графический интерфейс для лабораторной работы №1"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Лаб. работа №1: Распознавание образов без обучения")
        self.root.geometry("800x600")
        
        self.data = None
        self.labels = None
        self.clusterer = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Фрейм для управления
        control_frame = ttk.LabelFrame(self.root, text="Управление")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Кнопка загрузки данных
        ttk.Button(control_frame, text="Загрузить данные", 
                   command=self.load_data).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Выбор метрики
        ttk.Label(control_frame, text="Метрика:").pack(side=tk.LEFT, padx=5)
        self.metric_var = tk.StringVar(value="euclidean")
        metric_combo = ttk.Combobox(control_frame, textvariable=self.metric_var,
                                    values=["euclidean", "minkowski", "hamming"])
        metric_combo.pack(side=tk.LEFT, padx=5)
        
        # Число кластеров
        ttk.Label(control_frame, text="Число кластеров (l):").pack(side=tk.LEFT, padx=5)
        self.n_clusters_var = tk.StringVar(value="3")
        ttk.Entry(control_frame, textvariable=self.n_clusters_var, 
                  width=5).pack(side=tk.LEFT, padx=5)
        
        # Кнопка запуска
        ttk.Button(control_frame, text="Кластеризация", 
                   command=self.run_clustering).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Фрейм для результатов
        result_frame = ttk.LabelFrame(self.root, text="Результаты")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Текстовое поле для вывода
        self.result_text = tk.Text(result_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, 
                                   command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Кнопки визуализации
        vis_frame = ttk.Frame(self.root)
        vis_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(vis_frame, text="Показать дендрограмму", 
                   command=self.show_dendrogram).pack(side=tk.LEFT, padx=5)
        ttk.Button(vis_frame, text="Показать кластеры", 
                   command=self.show_clusters).pack(side=tk.LEFT, padx=5)
        ttk.Button(vis_frame, text="Показать исходные данные", 
                   command=self.show_raw_data).pack(side=tk.LEFT, padx=5)
        
    def load_data(self):
        """Загрузка данных из файла"""
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            try:
                df = pd.read_csv(filepath)
                self.result_text.insert(tk.END, f"Загружен файл: {filepath}\n")
                self.result_text.insert(tk.END, f"Размер данных: {df.shape}\n")
                self.result_text.insert(tk.END, f"Столбцы: {list(df.columns)}\n\n")
                
                # Диалог выбора столбца с метками классов
                self._show_column_selector(df)
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")
    
    def _show_column_selector(self, df):
        """Диалог выбора столбцов для признаков и меток"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Выбор столбцов")
        dialog.geometry("500x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Заголовок
        ttk.Label(dialog, text="Настройка данных", 
                  font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Фрейм для выбора метки
        label_frame = ttk.LabelFrame(dialog, text="Столбец с метками классов")
        label_frame.pack(fill=tk.X, padx=10, pady=5)
        
        label_var = tk.StringVar()
        label_combo = ttk.Combobox(label_frame, textvariable=label_var, 
                                   values=list(df.columns), state="readonly", width=40)
        label_combo.pack(pady=10, padx=10)
        if len(df.columns) > 0:
            label_combo.current(len(df.columns) - 1)
        
        # Фрейм для выбора признаков с чекбоксами
        feature_frame = ttk.LabelFrame(dialog, text="Столбцы признаков (выберите нужные)")
        feature_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Canvas со скроллом для чекбоксов
        canvas = tk.Canvas(feature_frame)
        scrollbar = ttk.Scrollbar(feature_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Чекбоксы для каждого числового столбца
        feature_vars = {}
        numeric_cols = [col for col in df.columns 
                       if df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        for col in df.columns:
            var = tk.BooleanVar(value=(col in numeric_cols))
            cb = ttk.Checkbutton(scrollable_frame, text=f"{col} ({df[col].dtype})", 
                                 variable=var)
            cb.pack(anchor='w', padx=5, pady=2)
            feature_vars[col] = var
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Кнопки "Выбрать все" и "Снять все"
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        def select_all():
            for col, var in feature_vars.items():
                if col != label_var.get():
                    var.set(True)
        
        def deselect_all():
            for var in feature_vars.values():
                var.set(False)
        
        ttk.Button(btn_frame, text="Выбрать все", command=select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Снять все", command=deselect_all).pack(side=tk.LEFT, padx=5)
        
        # Предпросмотр данных
        preview_frame = ttk.LabelFrame(dialog, text="Предпросмотр (первые 5 строк)")
        preview_frame.pack(fill=tk.X, padx=10, pady=5)
        
        preview_text = tk.Text(preview_frame, height=4, wrap=tk.NONE)
        preview_scroll_x = ttk.Scrollbar(preview_frame, orient="horizontal", command=preview_text.xview)
        preview_text.configure(xscrollcommand=preview_scroll_x.set)
        preview_text.insert(tk.END, df.head().to_string())
        preview_text.config(state=tk.DISABLED)
        preview_text.pack(fill=tk.X, padx=5, pady=5)
        preview_scroll_x.pack(fill=tk.X)
        
        def on_ok():
            label_col = label_var.get()
            if not label_col:
                messagebox.showwarning("Внимание", "Выберите столбец с метками!")
                return
            
            # Собираем выбранные признаки
            feature_cols = [col for col, var in feature_vars.items() 
                           if var.get() and col != label_col]
            
            if len(feature_cols) == 0:
                messagebox.showerror("Ошибка", "Выберите хотя бы один столбец признаков!")
                return
            
            # Проверяем, что все выбранные столбцы числовые
            non_numeric = [col for col in feature_cols 
                          if df[col].dtype not in ['int64', 'float64', 'int32', 'float32']]
            if non_numeric:
                messagebox.showerror("Ошибка", 
                    f"Столбцы {non_numeric} не являются числовыми!")
                return
            
            self.data = df[feature_cols].values.astype(float)
            self.labels = df[label_col].values
            
            # Преобразуем метки в числа если нужно
            if self.labels.dtype == 'object':
                unique_labels = np.unique(self.labels)
                label_map = {label: i+1 for i, label in enumerate(unique_labels)}
                self.labels = np.array([label_map[l] for l in self.labels])
            
            self.result_text.insert(tk.END, f"Признаки: {feature_cols}\n")
            self.result_text.insert(tk.END, f"Метки: {label_col}\n")
            self.result_text.insert(tk.END, f"Загружено {len(self.data)} объектов, {len(feature_cols)} признаков\n")
            self.result_text.insert(tk.END, f"Число классов: {len(np.unique(self.labels))}\n\n")
            
            dialog.destroy()
        
        # Кнопки OK/Отмена
        ok_frame = ttk.Frame(dialog)
        ok_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(ok_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(ok_frame, text="Отмена", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
                
    def run_clustering(self):
        """Запуск кластеризации"""
        if self.data is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные!")
            return
            
        try:
            n_clusters = int(self.n_clusters_var.get())
            metric = self.metric_var.get()
            
            # Создаём и запускаем кластеризатор
            self.clusterer = HierarchicalClustering(metric=metric)
            self.clusterer.fit(self.data, n_clusters)
            
            # Вычисляем меру несоответствия
            if self.labels is not None:
                mu = compute_mismatch_measure(self.labels, self.clusterer.labels_)
                self.result_text.insert(tk.END, f"\n{'='*50}\n")
                self.result_text.insert(tk.END, f"РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ\n")
                self.result_text.insert(tk.END, f"{'='*50}\n")
                self.result_text.insert(tk.END, f"Метрика: {metric}\n")
                self.result_text.insert(tk.END, f"Число кластеров: {n_clusters}\n")
                self.result_text.insert(tk.END, f"Мера несоответствия μ(T0, T1): {mu:.4f}\n")
                
                # Вывод разбиения
                self.result_text.insert(tk.END, f"\nРазбиение X' = (X'1, ..., X'l):\n")
                for i in range(1, n_clusters + 1):
                    count = np.sum(self.clusterer.labels_ == i)
                    self.result_text.insert(tk.END, f"  X'{i}: {count} объектов\n")
                    
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при кластеризации: {e}")
            
    def show_dendrogram(self):
        """Показать дендрограмму"""
        if self.clusterer is not None:
            self.clusterer.plot_dendrogram()
        else:
            messagebox.showwarning("Внимание", "Сначала выполните кластеризацию!")
            
    def show_clusters(self):
        """Показать визуализацию кластеров"""
        if self.clusterer is None or self.data is None:
            messagebox.showwarning("Внимание", "Сначала выполните кластеризацию!")
            return
            
        if self.data.shape[1] >= 2:
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(self.data[:, 0], self.data[:, 1], 
                                  c=self.clusterer.labels_, cmap='viridis')
            plt.colorbar(scatter, label='Кластер')
            plt.xlabel('Признак 1')
            plt.ylabel('Признак 2')
            plt.title('Визуализация кластеров')
            plt.tight_layout()
            plt.show()
            
    def show_raw_data(self):
        """Показать исходные данные с истинными метками"""
        if self.data is None or self.labels is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные!")
            return
            
        if self.data.shape[1] >= 2:
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(self.data[:, 0], self.data[:, 1], 
                                  c=self.labels, cmap='tab10', alpha=0.7)
            plt.colorbar(scatter, label='Класс')
            plt.xlabel('Признак 1')
            plt.ylabel('Признак 2')
            plt.title('Исходные данные (истинные метки классов)')
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showinfo("Информация", "Для визуализации нужно минимум 2 признака")


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    # Пример использования без GUI
    print("Лабораторная работа №1: Распознавание образов без обучения")
    print("=" * 60)
    
    # Генерация тестовых данных
    np.random.seed(42)
    
    # 3 класса по 30 объектов
    class1 = np.random.randn(30, 2) + np.array([0, 0])
    class2 = np.random.randn(30, 2) + np.array([5, 5])
    class3 = np.random.randn(30, 2) + np.array([0, 5])
    
    X = np.vstack([class1, class2, class3])
    true_labels = np.array([1]*30 + [2]*30 + [3]*30)
    
    print(f"Размер выборки X⁰: {X.shape[0]} объектов")
    print(f"Число признаков: {X.shape[1]}")
    print(f"Число классов l: {len(np.unique(true_labels))}")
    
    # Кластеризация
    clusterer = HierarchicalClustering(metric='euclidean')
    clusterer.fit(X, n_clusters=3)
    
    # Мера несоответствия
    mu = compute_mismatch_measure(true_labels, clusterer.labels_)
    
    print(f"\nРезультаты:")
    print(f"  Мера несоответствия μ(T0, T1): {mu:.4f}")
    
    for i in range(1, 4):
        count = np.sum(clusterer.labels_ == i)
        print(f"  Кластер X'{i}: {count} объектов")
    
    # Запуск GUI (если tkinter доступен)
    if TKINTER_AVAILABLE:
        print("\n" + "=" * 60)
        print("Запуск графического интерфейса...")
        root = tk.Tk()
        app = ClusteringApp(root)
        root.mainloop()
    else:
        print("\nGUI недоступен (tkinter не установлен)")
        print("Установите: sudo apt-get install python3-tk")
