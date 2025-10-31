#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, regularizers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import h5py

file_path = "/eos/user/s/ssaha/SWAN_projects/AD_Trigger_1/ntuples/lam_output/lam_test/EB_482596.h5"

with h5py.File(file_path, "r") as f:
    datasets = {k: np.array(f[k]) for k in f}
    Ofl_data = datasets["Offline_data"]

class NormalizepT(tf.keras.layers.Layer): #base layer class
    def __init__(self, scale_factor, **kwargs): #constructor, set up initial state 
        super().__init__(**kwargs) #calls the above constructor to initialze the arguments properly 
        self.scale_factor = scale_factor
        
    def call(self, data): #(N, 15, 3) for offline data
        data_scaled = tf.concat([tf.expand_dims(data[:,:,0]* self.scale_factor, axis = -1), data[:,:,1:]],axis = -1)
        #print("data_scaled",data_scaled)
        return data_scaled

Ofl_tensor = tf.convert_to_tensor(Ofl_data, dtype=tf.float32)
norm_layer = NormalizepT(scale_factor=0.005)
Ofl_norm = norm_layer(Ofl_tensor)

Ofl_pTnorm = tf.reshape(Ofl_norm, (-1,45))
Ofl_pTnorm = Ofl_pTnorm.numpy()
#print("ofl_flat", Ofl_pTnorm, Ofl_pTnorm.shape)
#print("ofl_data", Ofl_data, Ofl_data.shape)
print(np.max(Ofl_pTnorm))
print(np.max(Ofl_data))


# In[4]:


def safe_log1p_clip(t, min_val=0.0, max_val=1e20):
    t = tf.clip_by_value(t, min_val, max_val)
    return tf.math.log1p(t)

def create_large_AE(input_dim, h_dim_1, h_dim_2, h_dim_3, h_dim_4, latent_dim, 
                    l2_reg=0.01, dropout_rate=0.0):
    
    encoder_inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(h_dim_1, kernel_regularizer=regularizers.l2(l2_reg))(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_2, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_3, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_4, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    z = layers.Dense(latent_dim, kernel_regularizer=regularizers.l2(l2_reg))(x)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)

    #clipping values?
    z = layers.Lambda(lambda t: tf.clip_by_value(t, -1e3, 1e3))(z)

    encoder = Model(inputs=encoder_inputs, outputs=z, name="encoder")

    # ---------------- Decoder ----------------
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(h_dim_4, kernel_regularizer=regularizers.l2(l2_reg))(decoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)    

    x = layers.Dense(h_dim_3, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_2, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_1, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(input_dim, kernel_regularizer=regularizers.l2(l2_reg))(x)

    decoder = Model(inputs=decoder_inputs, outputs=outputs, name="decoder")

    # ---------------- Autoencoder ----------------
    ae_inputs = layers.Input(shape=(input_dim,))
    
    # Forward pass with debug Lambda layers
    encoded = encoder(ae_inputs)
    decoded = decoder(encoded)
    
    ae = Model(ae_inputs, outputs=decoded, name="autoencoder")

    return ae, encoder, decoder
    
INPUT_DIM = 45 #No MET in Offline data yet
H_DIM_1 = 100
H_DIM_2 = 100
H_DIM_3 = 64
H_DIM_4 = 32
LATENT_DIM = 4
L2_reg_coupling = 0.0001
dropout_p = 0.1

ae_mse, encoder_mse, decoder_mse = create_large_AE(INPUT_DIM, H_DIM_1, H_DIM_2, H_DIM_3, H_DIM_4,
                                       LATENT_DIM, l2_reg=L2_reg_coupling, dropout_rate=dropout_p)

X = Ofl_pTnorm 

X_train, X_val = train_test_split(
    X, test_size=0.15, random_state=0
)
# Pack X and HLT into y_train, y_val, give HLT score info for disco calc, dependent on the batch size
y_train = X_train.astype(np.float32)
y_val   = X_val.astype(np.float32)

print("Done!")


optimizer_mse = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm = 1.0)

ae_mse.compile(
    optimizer=optimizer_mse,
    loss = "mse",
    metrics=[],run_eagerly=True
)
history_mse = ae_mse.fit(
    X_train[:], X_train[:],
    validation_data=(X_val[:], X_val[:]),
    epochs=100,
    batch_size=512,
    verbose=2
)
print("Training done")

ae_mse.save("trained_model_mse_offline.keras")

y_pred_mse = ae_mse.predict(X_val, batch_size=512)

#Not masking nay invalid values, just plain MSE 
score_mse = np.mean((X_val - y_pred_mse)**2, axis=1)

with h5py.File("mse_scores_offline_test.h5", "w") as f:
    f.create_dataset("offline_AD_score", data=score_mse)