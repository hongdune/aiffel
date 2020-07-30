```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

digits = load_digits() #데이터 준비
digits_data = digits.data #Feature Data 지정하기
digits_label = digits.target #Label Data 지정하기
digits_name = digits.target_names

print(digits_name) #Target Names 출력해보기

print(digits.DESCR) #데이터 Describe 해 보기

#train, test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(digits_data, 
                                                    digits_label, 
                                                    test_size=0.2, 
                                                    random_state=15)
```

    [0 1 2 3 4 5 6 7 8 9]
    .. _digits_dataset:
    
    Optical recognition of handwritten digits dataset
    --------------------------------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 5620
        :Number of Attributes: 64
        :Attribute Information: 8x8 image of integer pixels in the range 0..16.
        :Missing Attribute Values: None
        :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
        :Date: July; 1998
    
    This is a copy of the test set of the UCI ML hand-written digits datasets
    https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
    
    The data set contains images of hand-written digits: 10 classes where
    each class refers to a digit.
    
    Preprocessing programs made available by NIST were used to extract
    normalized bitmaps of handwritten digits from a preprinted form. From a
    total of 43 people, 30 contributed to the training set and different 13
    to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of
    4x4 and the number of on pixels are counted in each block. This generates
    an input matrix of 8x8 where each element is an integer in the range
    0..16. This reduces dimensionality and gives invariance to small
    distortions.
    
    For info on NIST preprocessing routines, see M. D. Garris, J. L. Blue, G.
    T. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C.
    L. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,
    1994.
    
    .. topic:: References
    
      - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their
        Applications to Handwritten Digit Recognition, MSc Thesis, Institute of
        Graduate Studies in Science and Engineering, Bogazici University.
      - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.
      - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.
        Linear dimensionalityreduction using relevance weighted LDA. School of
        Electrical and Electronic Engineering Nanyang Technological University.
        2005.
      - Claudio Gentile. A New Approximate Maximal Margin Classification
        Algorithm. NIPS. 2000.



```python
from sklearn.tree import DecisionTreeClassifier #DecisionTree
decision_tree = DecisionTreeClassifier(random_state=15)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.90      0.95        31
               1       0.84      0.82      0.83        38
               2       0.73      0.87      0.80        38
               3       0.71      0.74      0.73        27
               4       0.94      0.80      0.87        41
               5       0.87      0.94      0.90        35
               6       0.87      0.89      0.88        38
               7       0.91      0.94      0.93        34
               8       0.79      0.77      0.78        35
               9       0.80      0.77      0.79        43
    
        accuracy                           0.84       360
       macro avg       0.85      0.85      0.84       360
    weighted avg       0.85      0.84      0.85       360
    



```python
from sklearn.ensemble import RandomForestClassifier #Random Forest

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.97      0.94      0.95        31
               1       0.95      0.97      0.96        38
               2       1.00      1.00      1.00        38
               3       1.00      0.96      0.98        27
               4       0.95      1.00      0.98        41
               5       0.97      1.00      0.99        35
               6       1.00      0.95      0.97        38
               7       1.00      1.00      1.00        34
               8       0.94      0.97      0.96        35
               9       1.00      0.98      0.99        43
    
        accuracy                           0.98       360
       macro avg       0.98      0.98      0.98       360
    weighted avg       0.98      0.98      0.98       360
    



```python
from sklearn import svm #SVM
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.97      0.98        31
               1       0.95      1.00      0.97        38
               2       1.00      1.00      1.00        38
               3       0.96      0.96      0.96        27
               4       0.98      0.98      0.98        41
               5       1.00      1.00      1.00        35
               6       1.00      1.00      1.00        38
               7       1.00      1.00      1.00        34
               8       0.97      0.94      0.96        35
               9       0.98      0.98      0.98        43
    
        accuracy                           0.98       360
       macro avg       0.98      0.98      0.98       360
    weighted avg       0.98      0.98      0.98       360
    



```python
from sklearn.linear_model import SGDClassifier #SGD
sgd_model = SGDClassifier()
sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        31
               1       0.84      0.97      0.90        38
               2       1.00      0.95      0.97        38
               3       0.96      0.85      0.90        27
               4       1.00      0.98      0.99        41
               5       0.97      0.97      0.97        35
               6       0.97      1.00      0.99        38
               7       0.92      1.00      0.96        34
               8       0.86      0.86      0.86        35
               9       0.97      0.88      0.93        43
    
        accuracy                           0.95       360
       macro avg       0.95      0.95      0.95       360
    weighted avg       0.95      0.95      0.95       360
    



```python
from sklearn.linear_model import LogisticRegression #Logistic Regression
logistic_model = LogisticRegression(max_iter = 10000)

logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.97      0.98        31
               1       0.95      0.97      0.96        38
               2       1.00      0.97      0.99        38
               3       0.96      0.93      0.94        27
               4       0.93      1.00      0.96        41
               5       0.94      0.97      0.96        35
               6       1.00      0.97      0.99        38
               7       1.00      1.00      1.00        34
               8       0.94      0.94      0.94        35
               9       0.98      0.95      0.96        43
    
        accuracy                           0.97       360
       macro avg       0.97      0.97      0.97       360
    weighted avg       0.97      0.97      0.97       360
    


sklearn.metrics 에서 제공하는 평가지표 중 가장 적절한 것은 Random Forest와 SVM 두가지이다. 평가 항목 중 precision과 recall 두가지 영역에서 거의 동일한 점수를 얻었으며 정확도 역시 98%로 높았기 때문이다. Logistic 회귀 역시 회귀 횟수를 증가시키면 높아질 수 있지만 다른 두 방법에 비해서 비효율적이란 결과를 얻었다.
