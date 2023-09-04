import numpy as np
from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping, CSVLogger
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold
import timeit




class ANN():
    def __init__(self, n_input, n_neurons, n_output):
        self.model = Sequential(
            [
                layers.Dense(n_neurons, activation='relu', input_dim=n_input),
                layers.Dense(n_neurons, activation='relu'),
                layers.Dense(n_output, activation='softmax')
            ]
        )

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        
    def fit(self, X, y, max_epochs = 300, patience = 5):

        early_stop = EarlyStopping(monitor="val_loss", mode="min", restore_best_weights=True, patience=patience, verbose=0)
        y_cat = to_categorical(y, len(np.unique(y)))

        X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.3, random_state=0, stratify=y)

        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=max_epochs, callbacks=[early_stop], verbose=1)


    def cv_fit(self, X, y, max_epochs = 300, patience = 5, cv = 5):
        # Fix so its also one hot encoded

        models = []
        scores = []

        splitter = StratifiedKFold(n_splits=cv)
        splits = splitter.split(X, y)

        early_stop = EarlyStopping(monitor="val_loss", mode="min", restore_best_weights=True, patience=patience, verbose=0)
        
        y_cat = to_categorical(y, len(np.unique(y)))

        print(f'Training model with cv = {cv} splits')
        for i, (train_index, val_index) in enumerate(splits):
            
            model_cv = self.model
            log = CSVLogger(f'.\model_{i}_training_log.csv', append = False)

            X_train = X[train_index]
            y_train = y_cat[train_index]
            X_val = X[val_index]
            y_val = y_cat[val_index]
            
            # Fit and evaluate the model
            model_cv.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=max_epochs, callbacks=early_stop, verbose=1)
            val_loss, val_accuracy = model_cv.evaluate(X_val, y_val)

            # Append the results
            models.append(model_cv)
            scores.append(val_accuracy)
        
        # 'Refit' the best model
        self.model = models[np.argmax(scores)]
        self.val_score_ = np.amax(scores)

    def predict(self, X):        

        pred_probability = self.model.predict(X, verbose = 0)

        # Single predictions
        if len(np.shape(pred_probability)) == 1:
            return np.argmax(pred_probability)
        
        # Batch predictions
        else:
            return np.argmax(pred_probability, axis = 1) # value from 0 to n_classes

    @staticmethod
    def accuracy_score(y_true, y_pred):
        pass



#TODO Add a gpu implemented version