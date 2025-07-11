import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 模拟数据集（真实使用时可读取CSV文件）
data = {
    'text': [
        'Congratulations, you have won a free lottery ticket',
        'Call now to claim your free prize',
        'Hey, are we still on for lunch tomorrow?',
        'You are selected for a free credit card',
        'Please review the meeting notes and send feedback',
        'Win a free vacation trip to Bahamas',
        'Let’s catch up sometime next week',
        'Limited offer! Buy one get one free now'
    ],
    'label': [1, 1, 0, 1, 0, 1, 0, 1]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# 文本转为词频向量
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# 标签
y = df['label']

# 拆分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=78)

# 模型训练
model = MultinomialNB()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 看一下预测例子
example = ["Win a free iPhone now", "Can we have a meeting tomorrow?"]
example_vec = vectorizer.transform(example)
pred = model.predict(example_vec)

for text, label in zip(example, pred):
    print(f"'{text}' --> {'Spam' if label == 1 else 'Not Spam'}")
