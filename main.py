import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras.optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU,ReLU
from tensorflow.keras.layers import Dropout
from sklearn.metrics import confusion_matrix,accuracy_score

data = pd.read_csv("Churn_Modelling.csv")

y = data["Exited"]
data.drop(["Exited","RowNumber","CustomerId","Surname"],inplace=True,axis=1)
X = data
##Feature engineering

geograpgy = pd.get_dummies(X["Geography"],drop_first=True)
gender = pd.get_dummies(X["Gender"],drop_first=True)

##Concat with df

X.drop(["Geography","Gender"],inplace=True,axis=1)

X = pd.concat([X,geograpgy,gender],axis=1)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=420)


##Feature Scaling

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

##ANN
Classifier = Sequential()

##Input layer

Classifier.add(Dense(units=11,activation="relu"))
#Classifier.add(Dropout(0.3))
##First Hidden layer

Classifier.add(Dense(units=6,activation="relu"))
#Classifier.add(Dropout(0.3))
##Second hidden layer
Classifier.add(Dense(units=7,activation="relu"))
#Classifier.add(Dropout(0.3))
##output layer
Classifier.add(Dense(units=1,activation="sigmoid"))
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)

#Classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
Classifier.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)


model_history = Classifier.fit(X_train_scaled,y_train,validation_split=0.33,batch_size=10,epochs=1000,callbacks=early_stopping)


print(model_history.history.keys())


plt.plot(model_history.history["accuracy"])
plt.plot(model_history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model accuracy")
plt.legend(["train","test"],loc = "upper left")
plt.show()

plt.plot(model_history.history["loss"])
plt.plot(model_history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model loss")
plt.legend(["train","test"],loc = "upper left")
plt.show()


y_pred = Classifier.predict(X_test)
y_pred = (y_pred >= 0.5)

##confusion matrix

cm = confusion_matrix(y_test,y_pred)
score = accuracy_score(y_test,y_pred)

print(cm)
print(score)

model_history_data = pd.DataFrame({"loss":model_history.history["loss"],
                                   "val_loss":model_history.history["val_loss"],
                                   "acc":model_history.history["accuracy"],
                                   "val_acc":model_history.history["val_accuracy"]})
model_history_data.to_csv("model_history.csv")


file_name = "churn_model.h5"
Classifier.save(file_name)

filename_scaler = 'sc_model.pickle'
pickle.dump(sc, open(filename_scaler, 'wb'))

