# Sentiment Analysis of ChatGPT Reviews using Deep Learning Models

## Overview
This project focuses on performing sentiment analysis on user reviews of ChatGPT. The goal is to classify reviews into three sentiment categories: Negative (1-2 stars), Neutral (3 stars), and Positive (4-5 stars). The project employs various deep learning models, including Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU) with attention mechanisms, and BERT (Bidirectional Encoder Representations from Transformers).

## Dataset
- **Source**: `ChatGPT_Reviews.csv`
- **Description**: The dataset contains user reviews and ratings for ChatGPT. It includes columns for reviews and ratings.
- **Preprocessing**: Text cleaning (lowercasing, removing punctuation, numbers, stopwords, lemmatization), handling missing values, and encoding sentiments.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn, WordCloud
  - NLP: NLTK (Natural Language Toolkit)
  - Deep Learning: TensorFlow, Keras
  - Transformers: Hugging Face Transformers (for BERT)
  - Machine Learning: Scikit-learn

## Installation
1. Clone or download the project repository.
2. Ensure Python 3.7+ is installed.
3. Install required libraries using pip:
   ```
   pip install numpy pandas tensorflow scikit-learn matplotlib seaborn wordcloud nltk transformers torch
   ```
4. Download NLTK resources (if not already done):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('punkt_tab')
   ```

## Usage
1. Place the `ChatGPT_Reviews.csv` file in the project directory.
2. Open the Jupyter notebook `projet2_v2 (2).ipynb` in Jupyter Notebook or JupyterLab.
3. Run the cells sequentially to:
   - Load and preprocess the data
   - Perform exploratory data analysis (EDA)
   - Train and evaluate the models (RNN, LSTM, GRU with attention, BERT)
   - View classification reports and confusion matrices

## Models and Evaluation
- **RNN with Attention**: Simple RNN with attention mechanism for sequence modeling.
- **LSTM with Attention**: LSTM network enhanced with attention to focus on relevant parts of the input.
- **GRU with Attention**: GRU model with attention for efficient sequence processing.
- **BERT**: Pre-trained BERT model fine-tuned for sentiment classification.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

## Results
The models are trained on a subset of 7000 reviews and evaluated on a test set. BERT typically achieves the highest accuracy due to its contextual understanding of language.

## Contributing
Feel free to fork the repository and submit pull requests for improvements or additional features.

## License
This project is for educational purposes. Please check the license of the dataset and libraries used.
