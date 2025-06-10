# CV Filtering and Matching System

An advanced CV filtering and matching system that utilizes Natural Language Processing (NLP) and machine learning to help recruiters analyze, rank, and classify candidate CVs against job descriptions.

## ✨ Features

- **Smart CV Analysis** - Extract key information from CVs and job descriptions automatically
- **Similarity Scoring** - Calculate detailed similarity scores between job requirements and candidate profiles
- **Candidate Ranking** - Rank candidates based on customizable criteria and weights
- **ML Classification** - Classify CVs as matches or non-matches using machine learning
- **Interactive Dashboard** - Visualize results with charts and detailed analysis views
- **Dual Interface** - Use either the web interface or command line for batch processing

## 📋 Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for larger datasets)
- 2GB free disk space

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Lebjawi-Tech/Resume-Analyzer.git
cd cv-filtering

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -e .

# Download required NLTK and spaCy data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_md
```

### Running the Web Interface

```bash
cv_filter web --port 5000
```

Then open your browser and navigate to `http://localhost:5000`

### Using Docker

```bash
# Build the Docker image
docker build -t cv-filtering .

# Run the container
docker run -p 5000:5000 cv-filtering
```

## 💻 Command Line Usage

### Ranking CVs

```bash
cv_filter rank --job path/to/job_description.txt --cv-dir path/to/cv_directory --output rankings.csv
```

### Classifying CVs

```bash
cv_filter classify --job path/to/job_description.txt --cv-dir path/to/cv_directory --model path/to/model.joblib --output classifications.csv
```

### Training a Classifier

```bash
cv_filter train --dataset path/to/training_data.csv --model-type random_forest --output models/my_classifier.joblib
```

## 📊 Web Interface Guide

The web interface provides an intuitive dashboard with the following tabs:

1. **Upload** - Upload job descriptions and candidate CVs
2. **Settings** - Configure weights and parameters for analysis
3. **Similarity Ranking** - View ranked candidates with similarity scores 
4. **Classification** - ML-based classification of candidates
5. **Detailed Analysis** - In-depth view of individual candidate profiles


## 📁 Supported File Formats

### Job Descriptions
- Plain text (.txt)

### CVs/Resumes
- Plain text (.txt)
- PDF documents (.pdf)
- Microsoft Word documents (.docx)

### Training Dataset
For training a classifier, the dataset should be a CSV file with the following columns:
- `job_text`: The text of the job description
- `cv_text`: The text of the CV
- `is_match`: 1 if the CV is a match for the job, 0 otherwise

## 🧠 How It Works

The system uses a combination of techniques:

1. **Text Extraction** - Extracts text from various document formats
2. **Preprocessing** - Cleans and normalizes text data
3. **Feature Extraction** - Identifies key skills, education, experience requirements
4. **Embedding** - Creates vector representations using transformers or TF-IDF
5. **Similarity Calculation** - Computes similarity scores between job and CV sections
6. **Machine Learning** - Classifies candidates using trained models

## 📊 Project Structure

```
cv-filtering/
├── data/
│   ├── raw/                # Raw job descriptions and CVs
│   └── processed/          # Processed data files
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for experimentation
├── src/
│   ├── data/               # Data loading and preprocessing
│   ├── features/           # Feature extraction and embedding
│   ├── models/             # Similarity and classification models
│   └── web/                # Web interface
├── tests/                  # Unit tests
├── README.md
├── requirements.txt
├── setup.py
└── Dockerfile
```

## 🔧 Advanced Configuration

The system can be configured through various parameters:

### Embedding Methods
- `--embedding tfidf` - Use TF-IDF for document embedding (faster)
- `--embedding transformer` - Use transformer models (more accurate)

### Classification Models
- `random_forest` - Balanced performance (default)
- `gradient_boosting` - Higher accuracy but slower
- `svm` - Good for smaller datasets
- `logistic` - Fastest but less accurate

### Hyperparameter Optimization
Add `--optimize` when training to perform hyperparameter tuning:

```bash
cv_filter train --dataset dataset.csv --model-type random_forest --optimize
```

## 📝 License


