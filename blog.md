---
title: "Text Classification with TF-IDF and Logistic Regression"
slug: "/vaishnavm/class-competition"
date: 2026-05-06
author: vaishnavm
description: "Classifying movie reviews using logistic regression"
tags:
  - class competition
---

## Task

We had to classify text into three buckets — not a review (0), positive review (1), or negative review (2). The dataset had around 70k training samples and 17.5k test samples. Scoring was done using macro F1.

## Data

Label 0 had about 32k samples, label 1 had 19k, and label 2 had 18k. So not-review texts made up almost half the data. Some texts were really short, others were super long. I had to clean out HTML tags and URLs from the text since a lot of it was scraped from the web.

## Approach

I went with logistic regression and TF-IDF, which we learned about in Unit 3. For TF-IDF I used word n-grams (1 to 3) and character n-grams (2 to 6), 200k features each. Character n-grams helped with catching word parts and typos. I set class_weight to balanced since the classes weren't evenly split.

## Results

I ran 5-fold cross validation and got a mean macro F1 of about 0.921. On the Kaggle leaderboard I scored around 0.926 and placed 20th out of 42 teams. The random baseline was 0.326 so that's a decent improvement.

Submission name: Vaishnav Mandlik

## Error Analysis

The model mostly struggles with non-review texts that sound like reviews (product reviews for games or books). It also has trouble with reviews that are mixed or sarcastic. Short texts are harder too since there's less to work with.

## How to Run

```
git clone https://github.com/uazhlt-ms-program/grad-level-term-project-kaggle-competition-vaishnav-mandlik-1.git
cd grad-level-term-project-kaggle-competition-vaishnav-mandlik-1
pip install -r requirements.txt
# put train.csv and test.csv in data/
python classify.py
```

## What I'd Do Differently

Try naive bayes and combine it with logistic regression. Maybe add stemming. Do a grid search over hyperparameters instead of just guessing. Could also try word embeddings from Unit 4.

## Code

https://github.com/uazhlt-ms-program/grad-level-term-project-kaggle-competition-vaishnav-mandlik-1
