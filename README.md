# Artificial Intelligence Models and Methods

Laboratory works on pattern recognition algorithms: unsupervised and supervised learning approaches.

## 📋 Overview

This repository contains two laboratory works implementing pattern recognition algorithms:

- **Lab 1**: Pattern Recognition without Learning (Hierarchical Clustering)
- **Lab 2**: Pattern Recognition with Learning (Classification)

Both labs include:
- Python implementations with GUI (tkinter) and web interface (Streamlit)
- LaTeX reports with theoretical background and results
- Test datasets and comprehensive documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- LaTeX distribution (for report compilation)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Artificial_intelligence_models_and_methods
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install tkinter for desktop GUI:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS
brew install python-tk
```

### Running the Applications

#### Web Interface (Recommended)

Run the unified web application:
```bash
streamlit run app.py
```

Or run individual lab applications:
```bash
# Lab 1
streamlit run Lab1/app.py

# Lab 2
streamlit run Lab2/app.py
```

The application will open in your browser at `http://localhost:8501`

#### Desktop GUI

Run the Python scripts directly:
```bash
# Lab 1
python3 Lab1/solution.py

# Lab 2
python3 Lab2/solution.py
```

#### Console Mode

The scripts also work in console mode (without GUI) if tkinter is not available.

## 📁 Project Structure

```
Artificial_intelligence_models_and_methods/
├── Lab1/                    # Laboratory work #1: Clustering
│   ├── solution.py          # Main implementation
│   ├── app.py               # Streamlit web interface
│   ├── report.tex           # LaTeX report
│   ├── report.pdf           # Compiled report
│   └── README.md            # Lab 1 documentation
│
├── Lab2/                    # Laboratory work #2: Classification
│   ├── solution.py          # Main implementation
│   ├── app.py               # Streamlit web interface
│   ├── report.tex           # LaTeX report
│   ├── report.pdf           # Compiled report
│   └── README.md            # Lab 2 documentation
│
├── app.py                   # Unified web application
├── data/                    # Test datasets
│   └── iris.csv             # Sample dataset
├── Doc/                     # Documentation
│   └── Методическая-разработка-к-лабораторным-для-магистрантов.txt
├── requirements.txt         # Python dependencies
├── test_web.py             # Web application tests
├── WEB_README.md           # Web interface documentation
└── README.md               # This file
```

## 🔬 Laboratory Works

### Lab 1: Hierarchical Clustering (Unsupervised Learning)

**Topic**: Pattern recognition without learning using hierarchical clustering algorithm.

**Features**:
- Implementation of hierarchical agglomerative clustering
- Distance metrics: Euclidean, Minkowski, Hamming
- Mismatch measure computation: μ(T₀, T₁)
- Dendrogram visualization
- Interactive web and desktop interfaces

**Key Algorithm**:
1. Initialize: each object is a separate cluster
2. Compute distance matrix
3. Merge closest clusters iteratively
4. Form final clusters based on target number

**Usage**:
```python
from Lab1.solution import HierarchicalClustering, compute_mismatch_measure

clusterer = HierarchicalClustering(metric='euclidean')
clusterer.fit(X, n_clusters=3)
mu = compute_mismatch_measure(true_labels, clusterer.labels_)
```

See [Lab1/README.md](Lab1/README.md) for detailed documentation.

### Lab 2: Pattern Recognition with Learning (Supervised Learning)

**Topic**: Pattern recognition with learning using training and test sets.

**Features**:
- Train/test split with validation
- Comparison functions: mean distance, k-NN, minimum distance
- Decision rule based on minimum evaluation
- Quality functional: Φ^A (accuracy)
- Full experiment with all metric/function combinations
- Interactive web and desktop interfaces

**Key Algorithm**:
1. Split dataset into training and test sets
2. For each test object, compute distances to all classes
3. Apply comparison function (mean/knn/min)
4. Assign to class with minimum distance
5. Compute accuracy (Φ^A)

**Usage**:
```python
from Lab2.solution import PatternRecognitionClassifier

clf = PatternRecognitionClassifier(
    distance_metric='euclidean',
    comparison_func='mean',
    k=3
)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)  # Returns Φ^A
```

See [Lab2/README.md](Lab2/README.md) for detailed documentation.

## 📊 Data Format

The applications accept CSV files with the following structure:

- **Features**: Numerical columns (int64, float64, int32, float32)
- **Labels**: One column with class labels (can be text or numeric)
- **Header**: First row contains column names

Example:
```csv
feature1,feature2,feature3,label
1.5,2.3,0.5,ClassA
2.1,1.8,0.7,ClassB
...
```

## 🧪 Testing

Run the test suite:
```bash
python3 test_web.py
```

This will test:
- All dependencies are installed
- Lab1 clustering functionality
- Lab2 classification functionality
- Import statements for web applications

## 📝 Reports

LaTeX reports are available for both labs:
- `Lab1/report.tex` - Hierarchical clustering report
- `Lab2/report.tex` - Classification report

To compile:
```bash
cd Lab1
xelatex report.tex
xelatex report.tex  # Run twice for references

cd ../Lab2
xelatex report.tex
xelatex report.tex
```

**Note**: Use XeLaTeX for proper Cyrillic character support.

## 🛠️ Technologies

- **Python 3.8+**: Core implementation
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **SciPy**: Clustering algorithms
- **scikit-learn**: Machine learning utilities
- **Streamlit**: Web interface framework
- **tkinter**: Desktop GUI (optional)
- **LaTeX/XeLaTeX**: Report generation

## 🌐 Deployment

The web application can be deployed to **Streamlit Cloud** (free hosting):

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" and select your repository
5. Set main file to `app.py`
6. Click "Deploy"

Your app will be live at `https://your-app-name.streamlit.app`

See [DEPLOY.md](DEPLOY.md) for detailed deployment instructions.

**Note**: GitHub Pages only supports static sites. For Python web apps, use Streamlit Cloud or other PaaS providers (Heroku, Railway, Render).

## 📚 Documentation

- [Lab1 README](Lab1/README.md) - Detailed Lab 1 documentation
- [Lab2 README](Lab2/README.md) - Detailed Lab 2 documentation
- [Web Interface Guide](WEB_README.md) - Streamlit app usage
- [Deployment Guide](DEPLOY.md) - How to deploy to Streamlit Cloud

## 🎯 Features

### Web Interface
- ✅ Interactive file upload
- ✅ Column selection with checkboxes
- ✅ Data preview
- ✅ Real-time parameter adjustment
- ✅ Visualization with matplotlib
- ✅ Results export
- ✅ Navigation between labs

### Desktop GUI
- ✅ File browser for CSV loading
- ✅ Column selection dialog
- ✅ Parameter configuration
- ✅ Results display
- ✅ Visualization windows

### Console Mode
- ✅ Command-line execution
- ✅ Batch processing
- ✅ Script integration

## 📄 License

See [LICENSE](LICENSE) file for details.

## 👤 Author

Eliseev D. I.

## 📅 Date

2025

## 🤝 Contributing

This is an academic project. For questions or improvements, please open an issue.

## 📖 References

- Methodological guide: `Doc/Методическая-разработка-к-лабораторным-для-магистрантов.txt`
- Scikit-learn documentation: https://scikit-learn.org/
- Streamlit documentation: https://docs.streamlit.io/

---

**Note**: This repository is part of a master's degree program in Artificial Intelligence Models and Methods.
