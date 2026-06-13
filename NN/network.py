# Aggiungi le importazioni per la CNN e per le Functional API
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Flatten, Reshape
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import L2

# Custom
import sys
sys.path.insert(0, '../utils')
import constants

class Network():
    def __init__(self, model_type, hp):
        self.model_type = model_type
        self.hp = hp
        
        # Sicurezza: Se hp è vuoto, garantisce un batch_size di default
        if not self.hp or 'batch_size' not in self.hp:
            self.hp['batch_size'] = 100
            
        self.model = None # Usiamo 'None' inizialmente perché useremo le API Funzionali
        self.callbacks = []

    def add_checkpoint_callback(self, model_path):
        self.callbacks.append(
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=False
            )
        )

    def build_model(self, input_dim):
        
        # ==========================================
        # 0. INGRESSO E GESTIONE DEL PLAINTEXT (PTX)
        # ==========================================
        main_input = Input(shape=(input_dim,), name='main_input')

        # Capiamo se l'utente sta usando il ptx guardando il nome del modello
        has_ptx = ('ptx_binary' in self.model_type) or ('ptx_scalar' in self.model_type)
        
        if has_ptx:
            # Calcoliamo quanti bit/valori occupano il PTX (8 per binario, 1 per scalare)
            ptx_dim = 8 if 'ptx_binary' in self.model_type else 1
            trace_dim = input_dim - ptx_dim
            
            # TRUCCO MAGICO: Tagliamo l'input direttamente dentro TensorFlow!
            # Così non devi cambiare nulla nel training.py
            trace_features = Lambda(lambda x: x[:, :trace_dim], name='split_trace')(main_input)
            ptx_features = Lambda(lambda x: x[:, trace_dim:], name='split_ptx')(main_input)
        else:
            trace_dim = input_dim
            trace_features = main_input

        # ==========================================
        # 1. COSTRUZIONE ARCHITETTURE
        # ==========================================
        
        if 'MLP' in self.model_type:
            # --- ARCHITETTURA MLP_BEST DI ASCAD ---
            # All'MLP diamo in pasto tutto l'input intero (traccia + ptx se c'è)
            # per mantenere esatta compatibilità con i tuoi test precedenti.
            x = Dense(200, activation='relu', name='fc1')(main_input)
            x = Dense(200, activation='relu', name='fc2')(x)
            x = Dense(200, activation='relu', name='fc3')(x)
            x = Dense(200, activation='relu', name='fc4')(x)
            x = Dense(200, activation='relu', name='fc5')(x)
            final_features = x

        elif 'CNN' in self.model_type:
            # --- ARCHITETTURA CNN_BEST2 DI ASCAD ---
            # La CNN DEVE guardare SOLO i campioni elettrici puri (trace_features)
            x = Reshape((trace_dim, 1), name='Reshape_for_Conv1D')(trace_features)
            
            x = Conv1D(64, 11, strides=2, activation='relu', padding='same', name='block1_conv1')(x)
            x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
            
            x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
            x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
            
            x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
            x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
            
            x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
            x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
            
            x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
            x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
            
            x = Flatten(name='flatten')(x)
            
            # --- INTEGRAZIONE PTX DOPO L'ESTRAZIONE DELLE FEATURE ---
            if has_ptx:
                x = Concatenate(name='concat_ptx')([x, ptx_features])
                
            x = Dense(4096, activation='relu', name='fc_cnn_1')(x)
            x = Dense(4096, activation='relu', name='fc_cnn_2')(x)
            final_features = x
        
        elif 'ZAID' in self.model_type:
            # --- ARCHITETTURA ZAID ET AL. (CHES 2020) PER ASCAD ---
            # Parametri estratti dalle tabelle ufficiali del paper
            
            x = Reshape((trace_dim, 1), name='Reshape_for_Conv1D')(trace_features)
            
            # Un solo blocco convoluzionale piccolissimo
            x = Conv1D(filters=4, kernel_size=1, strides=1, activation='selu', padding='same', name='block1_conv1')(x)
            x = AveragePooling1D(pool_size=2, strides=2, name='block1_pool')(x)
            
            x = Flatten(name='flatten')(x)
            
            # --- INTEGRAZIONE PTX (Se presente, anche se si consiglia 'none') ---
            if has_ptx:
                x = Concatenate(name='concat_ptx')([x, ptx_features])
                
            # Livelli densi compatti (solo 10 neuroni!)
            x = Dense(10, activation='selu', kernel_initializer='he_uniform', name='fc_cnn_1')(x)
            x = Dense(10, activation='selu', kernel_initializer='he_uniform', name='fc_cnn_2')(x)
            
            final_features = x

        else:
            raise RuntimeError(f'"{self.model_type}" is not a valid model')

        # ==========================================
        # 2. GESTIONE DINAMICA DELL'OUTPUT E DELLA LOSS
        # ==========================================
        if 'out_binary' in self.model_type:
            predictions = Dense(8, activation='sigmoid', name='predictions')(final_features)
            loss_func = 'binary_crossentropy'
        elif 'HW_SO' in self.model_type:
            predictions = Dense(9, activation='softmax', name='predictions')(final_features)
            loss_func = 'categorical_crossentropy'
        else:
            predictions = Dense(256, activation='softmax', name='predictions')(final_features)
            loss_func = 'categorical_crossentropy'

        # Assembliamo il modello finale con la Functional API
        self.model = Model(inputs=main_input, outputs=predictions)

        # ==========================================
        # 3. COMPILATION E OTTIMIZZATORE SOTA
        # ==========================================
        
        # Scegli l'ottimizzatore in base all'architettura
        if 'ZAID' in self.model_type:
            from tensorflow.keras.optimizers import Adam
            opt = Adam(learning_rate=0.001)
        else:
            from tensorflow.keras.optimizers import RMSprop
            opt = RMSprop(learning_rate=0.00001)

        self.model.compile(
            optimizer=opt,
            loss=loss_func,
            metrics=['accuracy']
        )