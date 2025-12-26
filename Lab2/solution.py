# -*- coding: utf-8 -*-
"""
Лабораторная работа №2
Задача распознавания образов с обучением

Автор: Eliseev D. I.
Дата: 26.09.2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Условный импорт tkinter (для работы без GUI)
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Предупреждение: tkinter не установлен. GUI недоступен.")


# ============================================================================
# МЕТРИКИ РАССТОЯНИЯ (Шаг 1 алгоритма)
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
# ФУНКЦИИ СРАВНЕНИЯ С КЛАССАМИ (Шаг 2 алгоритма)
# ============================================================================

def mean_distance_to_class(x, X_class, distance_func):
    """
    Среднее расстояние до класса Xi
    f_i(x) = (m_i)^(-1) * sum(s(x, x_j)) для x_j в X_i(о)
    """
    if len(X_class) == 0:
        return np.inf
    distances = [distance_func(x, x_j) for x_j in X_class]
    return np.mean(distances)


def knn_distance_to_class(x, X_class, distance_func, k=3):
    """
    k ближайших соседей
    Среднее расстояние до k ближайших объектов класса
    """
    if len(X_class) == 0:
        return np.inf
    distances = [distance_func(x, x_j) for x_j in X_class]
    distances_sorted = sorted(distances)
    k_actual = min(k, len(distances_sorted))
    return np.mean(distances_sorted[:k_actual])


def min_distance_to_class(x, X_class, distance_func):
    """
    Минимальное расстояние до объектов класса Xi
    f_i(x) = min(s(x, x_j)) для x_j в X_i(о)
    """
    if len(X_class) == 0:
        return np.inf
    distances = [distance_func(x, x_j) for x_j in X_class]
    return np.min(distances)


# ============================================================================
# РЕШАЮЩЕЕ ПРАВИЛО (Шаг 3 алгоритма)
# ============================================================================

def decision_rule(f_values):
    """
    Решающее правило по минимуму оценки до класса
    P_i^A(x) = 1 если f_i(x) = min{f_1(x), ..., f_l(x)}
    
    Parameters:
    -----------
    f_values : list
        Значения функций сравнения [f_1(x), ..., f_l(x)]
        
    Returns:
    --------
    np.ndarray
        Классификационный вектор P^A(x)
    """
    l = len(f_values)
    P = np.zeros(l)
    min_idx = np.argmin(f_values)
    P[min_idx] = 1
    return P


def get_predicted_class(f_values):
    """Получить номер предсказанного класса (1-indexed)"""
    return np.argmin(f_values) + 1


# ============================================================================
# КЛАССИФИКАТОР
# ============================================================================

class PatternRecognitionClassifier:
    """
    Классификатор для задачи распознавания образов с обучением
    """
    
    def __init__(self, distance_metric='euclidean', comparison_func='mean', k=3):
        """
        Parameters:
        -----------
        distance_metric : str
            Метрика расстояния: 'euclidean', 'minkowski', 'hamming'
        comparison_func : str
            Функция сравнения с классами: 'mean', 'knn', 'min'
        k : int
            Параметр k для метода k ближайших соседей
        """
        self.distance_metric = distance_metric
        self.comparison_func = comparison_func
        self.k = k
        
        # Словарь метрик
        self.distance_functions = {
            'euclidean': euclidean_distance,
            'minkowski': lambda x1, x2: minkowski_distance(x1, x2, p=2),
            'hamming': hamming_distance
        }
        
        # Данные обучающей выборки по классам
        self.classes = {}
        self.n_classes = 0
        self.class_labels = None
        
    def fit(self, X_train, y_train):
        """
        Обучение классификатора (запоминание обучающей выборки)
        
        Parameters:
        -----------
        X_train : np.ndarray
            Обучающая выборка X^0_обуч
        y_train : np.ndarray
            Метки классов
        """
        self.class_labels = np.unique(y_train)
        self.n_classes = len(self.class_labels)
        
        # Группируем объекты по классам: X_i(о)
        self.classes = {}
        for label in self.class_labels:
            self.classes[label] = X_train[y_train == label]
            
        return self
    
    def _compute_class_distances(self, x):
        """
        Вычислить расстояния до всех классов для объекта x
        
        Returns:
        --------
        list
            Значения f_i(x) для всех классов
        """
        dist_func = self.distance_functions[self.distance_metric]
        f_values = []
        
        for label in self.class_labels:
            X_class = self.classes[label]
            
            if self.comparison_func == 'mean':
                f = mean_distance_to_class(x, X_class, dist_func)
            elif self.comparison_func == 'knn':
                f = knn_distance_to_class(x, X_class, dist_func, self.k)
            elif self.comparison_func == 'min':
                f = min_distance_to_class(x, X_class, dist_func)
            else:
                raise ValueError(f"Unknown comparison function: {self.comparison_func}")
                
            f_values.append(f)
            
        return f_values
    
    def predict(self, X_test):
        """
        Классификация объектов контрольной выборки
        
        Parameters:
        -----------
        X_test : np.ndarray
            Контрольная выборка X^0_контр
            
        Returns:
        --------
        np.ndarray
            Предсказанные метки классов
        """
        predictions = []
        
        for x in X_test:
            f_values = self._compute_class_distances(x)
            pred_idx = np.argmin(f_values)
            predictions.append(self.class_labels[pred_idx])
            
        return np.array(predictions)
    
    def get_classification_vectors(self, X_test):
        """
        Получить классификационные векторы P^A(x) для всех объектов
        """
        vectors = []
        for x in X_test:
            f_values = self._compute_class_distances(x)
            P = decision_rule(f_values)
            vectors.append(P)
        return np.array(vectors)
    
    def score(self, X_test, y_test):
        """
        Вычислить функционал качества Φ^A(X^0_контр)
        
        Φ^A = t^0 / t, где t^0 — число правильно классифицированных объектов
        """
        predictions = self.predict(X_test)
        t0 = np.sum(predictions == y_test)
        t = len(y_test)
        return t0 / t


# ============================================================================
# ЭКСПЕРИМЕНТ
# ============================================================================

def run_experiment(X, y, test_size=0.2, random_state=42):
    """
    Провести эксперимент с различными комбинациями функций
    
    Returns:
    --------
    pd.DataFrame
        Таблица результатов
    """
    # Разбиение на обучающую и контрольную выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Проверка ограничения: t_i / m_i >= 0.2
    unique_classes = np.unique(y)
    for cls in unique_classes:
        m_i = np.sum(y_train == cls)
        t_i = np.sum(y_test == cls)
        ratio = t_i / m_i if m_i > 0 else 0
        print(f"Класс {cls}: m_i={m_i}, t_i={t_i}, t_i/m_i={ratio:.2f}")
    
    results = []
    
    # Перебор комбинаций
    distance_metrics = ['euclidean', 'minkowski', 'hamming']
    comparison_funcs = ['mean', 'knn', 'min']
    
    for metric in distance_metrics:
        for comp_func in comparison_funcs:
            clf = PatternRecognitionClassifier(
                distance_metric=metric,
                comparison_func=comp_func,
                k=3
            )
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            
            results.append({
                'Метрика': metric,
                'Функция сравнения': comp_func,
                'Φ^A': score
            })
    
    return pd.DataFrame(results)


# ============================================================================
# ГРАФИЧЕСКИЙ ИНТЕРФЕЙС
# ============================================================================

class RecognitionApp:
    """Графический интерфейс для лабораторной работы №2"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Лаб. работа №2: Распознавание образов с обучением")
        self.root.geometry("900x700")
        
        self.data = None
        self.labels = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classifier = None
        
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
        
        # Выбор функции сравнения
        ttk.Label(control_frame, text="Функция:").pack(side=tk.LEFT, padx=5)
        self.comp_var = tk.StringVar(value="mean")
        comp_combo = ttk.Combobox(control_frame, textvariable=self.comp_var,
                                  values=["mean", "knn", "min"])
        comp_combo.pack(side=tk.LEFT, padx=5)
        
        # Параметр k
        ttk.Label(control_frame, text="k:").pack(side=tk.LEFT, padx=5)
        self.k_var = tk.StringVar(value="3")
        ttk.Entry(control_frame, textvariable=self.k_var, 
                  width=5).pack(side=tk.LEFT, padx=5)
        
        # Размер тестовой выборки
        ttk.Label(control_frame, text="test_size:").pack(side=tk.LEFT, padx=5)
        self.test_size_var = tk.StringVar(value="0.2")
        ttk.Entry(control_frame, textvariable=self.test_size_var, 
                  width=5).pack(side=tk.LEFT, padx=5)
        
        # Кнопки действий
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(action_frame, text="Разбить выборку", 
                   command=self.split_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Обучить и тестировать", 
                   command=self.train_and_test).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Полный эксперимент", 
                   command=self.full_experiment).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Визуализация", 
                   command=self.show_visualization).pack(side=tk.LEFT, padx=5)
        
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
                
    def split_data(self):
        """Разбиение данных на обучающую и контрольную выборки"""
        if self.data is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные!")
            return
            
        test_size = float(self.test_size_var.get())
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, test_size=test_size, stratify=self.labels
        )
        
        self.result_text.insert(tk.END, f"\nРазбиение выборки:\n")
        self.result_text.insert(tk.END, f"  X_обуч: {len(self.X_train)} объектов\n")
        self.result_text.insert(tk.END, f"  X_контр: {len(self.X_test)} объектов\n")
        
        # Проверка ограничения
        for cls in np.unique(self.labels):
            m_i = np.sum(self.y_train == cls)
            t_i = np.sum(self.y_test == cls)
            ratio = t_i / m_i if m_i > 0 else 0
            self.result_text.insert(tk.END, f"  Класс {cls}: t_i/m_i = {ratio:.2f}\n")
            
    def train_and_test(self):
        """Обучение и тестирование классификатора"""
        if self.X_train is None:
            messagebox.showwarning("Внимание", "Сначала разбейте выборку!")
            return
            
        metric = self.metric_var.get()
        comp_func = self.comp_var.get()
        k = int(self.k_var.get())
        
        self.classifier = PatternRecognitionClassifier(
            distance_metric=metric,
            comparison_func=comp_func,
            k=k
        )
        self.classifier.fit(self.X_train, self.y_train)
        score = self.classifier.score(self.X_test, self.y_test)
        
        self.result_text.insert(tk.END, f"\n{'='*50}\n")
        self.result_text.insert(tk.END, f"РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ\n")
        self.result_text.insert(tk.END, f"{'='*50}\n")
        self.result_text.insert(tk.END, f"Метрика: {metric}\n")
        self.result_text.insert(tk.END, f"Функция сравнения: {comp_func}\n")
        self.result_text.insert(tk.END, f"Параметр k: {k}\n")
        self.result_text.insert(tk.END, f"Функционал качества Φ^A: {score:.4f}\n")
        
    def full_experiment(self):
        """Полный эксперимент со всеми комбинациями"""
        if self.data is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные!")
            return
            
        test_size = float(self.test_size_var.get())
        results_df = run_experiment(self.data, self.labels, test_size=test_size)
        
        self.result_text.insert(tk.END, f"\n{'='*60}\n")
        self.result_text.insert(tk.END, f"РЕЗУЛЬТАТЫ ПОЛНОГО ЭКСПЕРИМЕНТА\n")
        self.result_text.insert(tk.END, f"{'='*60}\n")
        self.result_text.insert(tk.END, results_df.to_string(index=False))
        self.result_text.insert(tk.END, "\n")
        
    def show_visualization(self):
        """Показать визуализацию данных"""
        if self.data is None or self.labels is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные!")
            return
            
        if self.data.shape[1] < 2:
            messagebox.showinfo("Информация", "Для визуализации нужно минимум 2 признака")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Левый график: исходные данные
        scatter1 = axes[0].scatter(self.data[:, 0], self.data[:, 1], 
                                   c=self.labels, cmap='tab10', alpha=0.7)
        axes[0].set_xlabel('Признак 1')
        axes[0].set_ylabel('Признак 2')
        axes[0].set_title('Исходные данные (все)')
        plt.colorbar(scatter1, ax=axes[0], label='Класс')
        
        # Правый график: обучающая и тестовая выборки
        if self.X_train is not None and self.X_test is not None:
            axes[1].scatter(self.X_train[:, 0], self.X_train[:, 1], 
                           c=self.y_train, cmap='tab10', alpha=0.5, 
                           marker='o', label='Обучающая', s=50)
            axes[1].scatter(self.X_test[:, 0], self.X_test[:, 1], 
                           c=self.y_test, cmap='tab10', alpha=1.0, 
                           marker='*', label='Контрольная', s=150, edgecolors='black')
            axes[1].legend()
            axes[1].set_title('Разбиение выборки')
        else:
            axes[1].text(0.5, 0.5, 'Выполните разбиение выборки', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Разбиение выборки (не выполнено)')
        
        axes[1].set_xlabel('Признак 1')
        axes[1].set_ylabel('Признак 2')
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    print("Лабораторная работа №2: Распознавание образов с обучением")
    print("=" * 60)
    
    # Генерация тестовых данных
    np.random.seed(42)
    
    # 3 класса по 50 объектов
    class1 = np.random.randn(50, 4) + np.array([0, 0, 0, 0])
    class2 = np.random.randn(50, 4) + np.array([3, 3, 3, 3])
    class3 = np.random.randn(50, 4) + np.array([0, 3, 0, 3])
    
    X = np.vstack([class1, class2, class3])
    y = np.array([1]*50 + [2]*50 + [3]*50)
    
    print(f"Размер выборки X⁰: {X.shape[0]} объектов")
    print(f"Число признаков: {X.shape[1]}")
    print(f"Число классов l: {len(np.unique(y))}")
    
    # Разбиение на обучающую и контрольную выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nРазбиение выборки:")
    print(f"  X_обуч: {len(X_train)} объектов")
    print(f"  X_контр: {len(X_test)} объектов")
    
    # Проверка ограничения t_i / m_i >= 0.2
    print("\nПроверка ограничения t_i / m_i >= 0.2:")
    for cls in np.unique(y):
        m_i = np.sum(y_train == cls)
        t_i = np.sum(y_test == cls)
        ratio = t_i / m_i
        status = "✓" if ratio >= 0.2 else "✗"
        print(f"  Класс {cls}: m_i={m_i}, t_i={t_i}, t_i/m_i={ratio:.2f} {status}")
    
    # Тестирование классификатора
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ КЛАССИФИКАТОРОВ")
    print("=" * 60)
    
    configs = [
        ('euclidean', 'mean'),
        ('euclidean', 'knn'),
        ('euclidean', 'min'),
        ('minkowski', 'mean'),
        ('hamming', 'mean'),
    ]
    
    for metric, comp_func in configs:
        clf = PatternRecognitionClassifier(
            distance_metric=metric,
            comparison_func=comp_func,
            k=3
        )
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(f"  {metric:12s} + {comp_func:4s}: Φ^A = {score:.4f}")
    
    # Запуск GUI (если tkinter доступен)
    if TKINTER_AVAILABLE:
        print("\n" + "=" * 60)
        print("Запуск графического интерфейса...")
        root = tk.Tk()
        app = RecognitionApp(root)
        root.mainloop()
    else:
        print("\nGUI недоступен (tkinter не установлен)")
        print("Установите: sudo apt-get install python3-tk")
