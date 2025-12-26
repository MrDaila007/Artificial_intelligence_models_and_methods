# -*- coding: utf-8 -*-
"""
Единый веб-интерфейс для лабораторных работ
"""

import streamlit as st

st.set_page_config(
    page_title="Лабораторные работы по ИИ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Навигация
st.sidebar.title("🤖 Лабораторные работы")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Выберите лабораторную работу:",
    ["📊 Lab 1: Кластеризация", "🎯 Lab 2: Классификация"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Информация")
st.sidebar.info("""
**Лабораторная работа №1:**
Распознавание образов без обучения
(Иерархическая кластеризация)

**Лабораторная работа №2:**
Распознавание образов с обучением
(Классификация)
""")

# Загрузка соответствующих страниц
if page == "📊 Lab 1: Кластеризация":
    import sys
    import os
    
    # Сохраняем текущее состояние
    original_cwd = os.getcwd()
    original_path = sys.path.copy()
    
    try:
        # Очищаем кэш модулей solution
        modules_to_remove = [k for k in sys.modules.keys() if 'solution' in k]
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        # Переходим в директорию Lab1
        lab1_path = os.path.join(os.path.dirname(__file__), 'Lab1')
        os.chdir(lab1_path)
        # Очищаем пути от других Lab директорий и добавляем текущую
        sys.path = [lab1_path] + [p for p in original_path if 'Lab' not in p]
        
        # Импортируем и выполняем код из Lab1/app.py
        with open('app.py', 'r', encoding='utf-8') as f:
            code = f.read()
            exec(code, {'__file__': os.path.join(lab1_path, 'app.py'), '__name__': '__main__'})
    finally:
        # Восстанавливаем исходное состояние
        os.chdir(original_cwd)
        sys.path[:] = original_path
        
elif page == "🎯 Lab 2: Классификация":
    import sys
    import os
    
    # Сохраняем текущее состояние
    original_cwd = os.getcwd()
    original_path = sys.path.copy()
    
    try:
        # Очищаем кэш модулей solution
        modules_to_remove = [k for k in sys.modules.keys() if 'solution' in k]
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        # Переходим в директорию Lab2
        lab2_path = os.path.join(os.path.dirname(__file__), 'Lab2')
        os.chdir(lab2_path)
        # Очищаем пути от других Lab директорий и добавляем текущую
        sys.path = [lab2_path] + [p for p in original_path if 'Lab' not in p]
        
        # Импортируем и выполняем код из Lab2/app.py
        with open('app.py', 'r', encoding='utf-8') as f:
            code = f.read()
            exec(code, {'__file__': os.path.join(lab2_path, 'app.py'), '__name__': '__main__'})
    finally:
        # Восстанавливаем исходное состояние
        os.chdir(original_cwd)
        sys.path[:] = original_path

