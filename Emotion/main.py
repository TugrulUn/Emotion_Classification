import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv("veriseti.csv")

data.head(10)
data.isna().any()
data

Y = data['sh'].copy()
X = data.drop('sh', axis = 1).copy()

print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 10)

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)
decision_pred = clf_gini.predict(X_test)

cm = confusion_matrix(y_test, decision_pred)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")
plt.show()

print("Decision Tree Confusion Matrix:")
print(classification_report(y_test, decision_pred))



# import numpy as np
# import pandas as pd
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# import tensorflow as tf
# from sklearn.decomposition import PCA
#
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# data = pd.read_csv('emotions.csv')
#
#
# sns.countplot(data['label'], color='lightblue')
#
# ###################################################
#
# # sample = data.loc[0, 'fft_0_b':'fft_749_b']
# # plt.figure(figsize=(16, 10))
# # plt.plot(range(len(sample)), sample)
# # plt.title("Features fft_0_b through fft_749_b")
# # plt.show()
#
# ###################################################
#
# pca = PCA(10).fit(data.drop('label', axis=1))
# explained_variance = pca.explained_variance_ratio_
# plt.plot(np.cumsum(explained_variance))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()
#
# first_c = pca.components_[0]
# second_c = pca.components_[1]
# print(first_c)
# print(second_c)
#
# ##################################################
#
# print(data['label'].value_counts())
# label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
#
#
#
# def preprocess_inputs(df):
#     df = df.copy()
#
#     df['label'] = df['label'].replace(label_mapping)
#     print(df.shape)
#     y = df['label'].copy()
#     X = df.drop('label', axis=1).copy()
#
#     scaler = StandardScaler()
#     scaled_df = scaler.fit_transform(X)
#     pca = PCA(n_components=30)
#     pca_vectors = pca.fit_transform(scaled_df)
#     for index, var in enumerate(pca.explained_variance_ratio_):
#         print("Explained Variance ratio by Principal Component ", (index + 1), " : ", var)
#
#     plt.figure()
#     plt.plot(pca.explained_variance_ratio_)
#     plt.xticks(rotation='horizontal')
#
#     plt.figure(figsize=(25, 8))
#     sns.scatterplot(x=pca_vectors[:, 0], y=pca_vectors[:, 1], hue=y)
#     plt.title('Principal Components vs Class distribution', fontsize=16)
#     plt.ylabel('Principal Component 2', fontsize=16)
#     plt.xlabel('Principal Component 1', fontsize=16)
#     plt.xticks(rotation='vertical');
#
#     print(X.shape)
#     print(y.shape)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
#
#     return X_train, X_test, y_train, y_test
#
# X_train, X_test, y_train, y_test = preprocess_inputs(data)



# print(X_train)
# print(y_train)




# inputs = tf.keras.Input(shape=(X_train.shape[1],))
#
# expand_dims = tf.expand_dims(inputs, axis=2)
#
# gru = tf.keras.layers.GRU(256, return_sequences=True)(expand_dims)
#
# flatten = tf.keras.layers.Flatten()(gru)
#
# outputs = tf.keras.layers.Dense(3, activation='softmax')(flatten)
#
#
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
# print(model.summary())
#
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# history = model.fit(
#     X_train,
#     y_train,
#     validation_split=0.2,
#     batch_size=32,
#     epochs=10,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=5,
#             restore_best_weights=True
#         )
#     ]
# )
#
# model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
# print("Test Accuracy: {:.3f}%".format(model_acc * 100))
#
# keras_pred = np.array(list(map(lambda x: np.argmax(x), model.predict(X_test))))
#
# cm = confusion_matrix(y_test, keras_pred)
#
# plt.figure(figsize=(8, 8))
# sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
# plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
# plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Keras Confusion Matrix")
# plt.show()
# # df_cm = pd.DataFrame(cm)
# # heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Greens")
# print("Keras Confusion Matrix:")
# print(classification_report(y_test, keras_pred))

#########################################################################

# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors = 10)
# knn.fit(X_train, y_train)
# prediction = knn.predict(X_test)
# cm = confusion_matrix(y_test, prediction)
#
#
# plt.figure(figsize=(8, 8))
# sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
# plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
# plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("KNN Confusion Matrix")
# plt.show()
# # df_cm = pd.DataFrame(cm)
# # heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Greens")
# print("KNN Confusion Matrix:")
# print(classification_report(y_test, prediction))

################################################################################

# from sklearn.naive_bayes import MultinomialNB
#
# naive_bayes = MultinomialNB(alpha=0.01)
# naive_bayes.fit(X_train, y_train)
# naive_pred = naive_bayes.predict(X_test)
# cm = confusion_matrix(y_test, naive_pred)

# from sklearn.naive_bayes import GaussianNB
#
# gnb = GaussianNB(priors=None, var_smoothing=1e-09)
# gnb.fit(X_train, y_train)
# naive_pred =gnb.predict((X_test))
# cm = confusion_matrix(y_test, naive_pred)
#
# plt.figure(figsize=(8, 8))
# sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
# plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
# plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Gaussian Naive Bayes Matrix")
# plt.show()
# # df_cm = pd.DataFrame(cm)
# # heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Greens")
# print("Gaussian Naive Bayes Confusion Matrix:")
# print(classification_report(y_test, naive_pred))
#
# ####################################################################################
#
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=10, random_state=0)
# classifier.fit(X_train, y_train)
# forest_pred = classifier.predict(X_test)
#
# cm = confusion_matrix(y_test, forest_pred)
#
# plt.figure(figsize=(8, 8))
# sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
# plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
# plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Random Forest Confusion Matrix")
# plt.show()
#
# print("Random Forest Confusion Matrix:")
# print(classification_report(y_test, forest_pred))
#
# ########################################################################################

# from sklearn.tree import DecisionTreeClassifier
#
# clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
# clf_gini.fit(X_train, y_train)
# decision_pred = clf_gini.predict(X_test)
#
# cm = confusion_matrix(y_test, decision_pred)
#
# plt.figure(figsize=(8, 8))
# sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
# plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
# plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Decision Tree Confusion Matrix")
# plt.show()
#
# print("Decision Tree Confusion Matrix:")
# print(classification_report(y_test, decision_pred))

#######################################################################################

# from xgboost import XGBRegressor
# my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# my_model.fit(X_train, y_train,  early_stopping_rounds=5,
#              eval_set=[(X_test, y_test)], verbose=False)
#
# boosted_pred = my_model.predict(X_test)
#
# cm = confusion_matrix(y_test, boosted_pred)
#
# plt.figure(figsize=(8, 8))
# sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
# plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
# plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Boosted Trees Confusion Matrix")
# plt.show()
#
# print("Boosted Trees Confusion Matrix:")
# print(classification_report(y_test, boosted_pred))



