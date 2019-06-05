import tensorflow as tf
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

mnist = fetch_mldata("MNIST original")
X_train, X_test, y_train, y_test = train_test_split(mnist["data"], mnist["target"], 
                                                    random_state=42, test_size=0.33)

scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                         feature_columns=feature_cols)
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)

y_pred = dnn_clf.predict(X_test)
print(accuracy_score(y_test, y_pred["classes"]))