import tensorflow as tf
import pickle
def predict_cnn(data):
    model = tf.keras.models.load_model("F:\Project\churn_model\churn_model_file.h5")
    scalar_model = pickle.load(open("F:\Project\churn_model\sc_model.pickle", 'rb'))
    data_scaled = scalar_model.transform(data)
    y_pred = model.predict(data_scaled)
    y_pred = (y_pred >= 0.5)
    return y_pred