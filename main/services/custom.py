import tensorflow as tf
from tensorflow.keras import backend as K

'''
Custom cce, plate_acc and acc for plate recognition using CNN
'''

# Custom Metrics


def cat_acc(y_true, y_pred):
    # Reorganiza los tensores de verdadero y predicho en una forma adecuada.
    y_true = K.reshape(y_true, shape=(-1, 7, 37))
    y_pred = K.reshape(y_pred, shape=(-1, 7, 37))
    # Calcula la precisión categórica utilizando las métricas incorporadas de Keras.
    return K.mean(tf.keras.metrics.categorical_accuracy(y_true, y_pred))


def plate_acc(y_true, y_pred):
    # Reorganiza los tensores de verdadero y predicho en una forma adecuada.
    y_true = K.reshape(y_true, shape=(-1, 7, 37))
    y_pred = K.reshape(y_pred, shape=(-1, 7, 37))
    # Compara las predicciones con los valores verdaderos y calcula la precisión de la placa.
    et = K.equal(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(
        K.cast(K.all(et, axis=-1, keepdims=False), dtype='float32')
    )


def top_3_k(y_true, y_pred):
    # Reorganiza los tensores de verdadero y predicho en una forma adecuada.
    y_true = K.reshape(y_true, (-1, 37))
    y_pred = K.reshape(y_pred, (-1, 37))
    # Calcula la precisión top-3 utilizando la métrica incorporada de Keras.
    return K.mean(
        tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
    )


# Custom loss
def cce(y_true, y_pred):
    # Reorganiza los tensores de verdadero y predicho en una forma adecuada.
    y_true = K.reshape(y_true, shape=(-1, 37))
    y_pred = K.reshape(y_pred, shape=(-1, 37))
    # Calcula la pérdida de entropía cruzada categórica utilizando la función de pérdida incorporada de Keras.
    return K.mean(
        tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=False, label_smoothing=0.2
        )
    )