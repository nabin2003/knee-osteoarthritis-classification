# Knee Osteoarthritis Classification Using Deep Learning

## Project Overview

This project focuses on classifying knee osteoarthritis severity from X-ray images using deep learning. The model predicts five severity levels, from KL Grade 0 to KL Grade 4.

The project is based on the practical learning materials provided during the course. I used the existing project structure to train and evaluate the model and understand how deep learning can be applied to medical image classification.

## Dataset

The project uses knee X-ray images organized into five classes according to the Kellgren–Lawrence (KL) grading system.

- Grade 0: No osteoarthritis
- Grade 1: Doubtful osteoarthritis
- Grade 2: Mild osteoarthritis
- Grade 3: Moderate osteoarthritis
- Grade 4: Severe osteoarthritis

## Methodology

The project uses a ResNet18-based deep learning model implemented with PyTorch.

The dataset was prepared for training and validation using five-fold stratified cross-validation. The model was trained separately for each fold, and the saved models were evaluated using classification metrics.

## Evaluation Results

The five-fold evaluation produced the following average results:

| Metric | Result |
|---|---|
| Accuracy | 41.49% |
| Balanced Accuracy | 40.85% |
| Macro F1-score | 32.09% |

The results show that the model can recognize some osteoarthritis severity levels, but its performance varies between classes and validation folds.

The model particularly struggles with KL Grade 1. One validation fold also produced significantly lower accuracy than the others.

## What I Learned

Through this project, I learned how to prepare image datasets, work with a pretrained neural network architecture, train a classification model, and evaluate its performance.

I also learned that medical image classification can be challenging because some severity levels look similar and the dataset may contain unequal numbers of images for different classes.

## How to Run

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Prepare the dataset in the directory structure expected by the project.

Run dataset preparation:

```bash
python data_preparation.py
```

Train the model:

```bash
python main.py
```

Evaluate the saved models:

```bash
python evaluate.py
```

The evaluation reports and confusion matrices are saved in `outputs/evaluation/`.

## Limitations

The model's accuracy is relatively low, and its predictions are not reliable enough for clinical diagnosis. Further improvements could include more balanced training data, additional image augmentation, and hyperparameter tuning.

## AI Assistance

I used ChatGPT for help with debugging, understanding errors, and preparing parts of the evaluation code and documentation.