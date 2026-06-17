# ML, Deep Learning & NLP Applications Capstone Report

## 1. Introduction
I chose the Fake News dataset because it is a real text-classification problem and it let me practice the full pipeline from cleaning text to comparing machine-learning and neural-network models.

## 2. Dataset Description
The dataset contains two classes: **fake** and **true** news articles. After combining the two CSV files, the project uses **44,898 total rows**. The data includes `title`, `text`, `subject`, and `date`. The class distribution is:

- Fake: 23,481
- True: 21,417

The dataset is close to balanced, so accuracy and F1-score are both useful.

## 3. Models Used
I trained three traditional machine-learning models on the cleaned text and TF-IDF features:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM (SGDClassifier)

Best Step 1 model: **Linear SVM (SGDClassifier)** with F1-score **0.986**.

## 4. Neural Network
I also trained a small feed-forward neural network with dense layers and dropout on a balanced sample of the same cleaned data. The network used two hidden layers, ReLU activations, dropout regularization, and 20 epochs. The final validation curve showed learning over time, but the model was still weaker than the best full-data classical model.

Neural-network test results on the balanced sample:
- Accuracy: 0.950
- Precision: 0.959
- Recall: 0.940
- F1-score: 0.949

## 5. Results & Comparison
The best overall model in this project was **Linear SVM (SGDClassifier)**. It achieved the highest F1-score and also the strongest accuracy, precision, and recall on the test set.

The neural network was useful for learning and comparison, but it did not beat the strongest classical model. For this dataset, the TF-IDF + linear model approach was more efficient and gave better performance.

## 6. What I Learned
I learned that text preprocessing matters a lot, and that simpler models can outperform more complex ones on TF-IDF features. I also learned how to avoid data leakage by fitting TF-IDF only on the training set. Finally, I saw how to compare models fairly using the same metrics and confusion matrices.
