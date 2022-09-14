"""
@author: supervampire
@email: wangxutong@iie.ac.cn
@version: Created in 2022 03/25.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import  LinearSVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Dropout
from tensorflow.keras.models import Model
import pandas as pd
import shap
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 0.001):
            print("\n\nReached 0.001 loss value so stop training!\n\n")
            self.model.stop_training = True

class Deep_NN(object):
    def __init__(self, n_features):
        self.n_features = n_features
        self.normal = StandardScaler()
        self.model = self.build_model()
        self.trainingStopCallback = haltCallback()

        lr = 0.1
        momentum = 0.9
        decay = 0.000001
        opt = tf.keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay)

        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def fit(self, X, y):
        self.normal.fit(X)
        self.model.fit(self.normal.transform(X), y, batch_size=512, epochs=50, callbacks=[self.trainingStopCallback])

    def predict(self, X):
        return self.model.predict(self.normal.transform(X), batch_size=512)

    def build_model(self):
        model = None
        # with tf.device('/cpu:0'):
        input1 = Input(shape=(self.n_features,))
        dense1 = Dense(128, activation='relu')(input1)
        norm1 = BatchNormalization()(dense1)
        drop1 = Dropout(0.5)(norm1)
        dense2 = Dense(64, activation='relu')(drop1)
        norm2 = BatchNormalization()(dense2)
        drop2 = Dropout(0.5)(norm2)
        dense3 = Dense(32, activation='relu')(drop2)
        norm3 = BatchNormalization()(dense3)
        drop3 = Dropout(0.5)(norm3)
        dense4 = Dense(1)(drop3)
        out = Activation('sigmoid')(dense4)
        model = Model(inputs=[input1], outputs=[out])
        return model

    def explain(self, X_back, X_exp, n_samples=100):
        # self.exp = shap.GradientExplainer(self.model, self.normal.transform(X_back))
        self.exp = shap.DeepExplainer(self.model, self.normal.transform(X_back))
        # return self.exp.shap_values(self.normal.transform(X_exp), nsamples=n_samples)
        return self.exp.shap_values(self.normal.transform(X_exp))


class Transfer:
    def __init__(self,x_test_pos,x_train_pos,y_train_pos,new_x_pos,new_y_pos,poison_num,watermark_size,id):
        self.x_test = np.load(x_test_pos)
        self.x_train = np.load(x_train_pos)
        self.y_train = np.load(y_train_pos)
        self.new_x = np.load(new_x_pos)
        self.new_y = np.load(new_y_pos)
        self.id = id
        self.poison_num = poison_num
        self.watermark_size = watermark_size
        self.normal1 = StandardScaler().fit(self.x_train)
        self.normal2 = StandardScaler().fit(self.new_x)
        self.record = []


    def Linear_SVC(self):
        print(f'-----------------------------------------------------')
        model = LinearSVC(C=1,random_state=4)
        model.fit(self.normal1.transform(self.x_train), self.y_train)
        bst = model.predict(self.normal1.transform(self.x_test))
        pred = [1 if i>0.5 else 0 for i in bst]
        
        ori = [1 if i>0.5 else 0 for i in model.predict(self.normal1.transform(self.x_train))]
        print(f'[SVM][Acc] Ori Acc on whole dataset :{accuracy_score(ori,self.y_train)}')

        detection_rate = accuracy_score(pred,np.array([1]*len(self.x_test)))

        new_model = LinearSVC(C=1,random_state=4)
        new_model.fit(self.normal2.transform(self.new_x),self.new_y)
        new_bst = new_model.predict(self.normal2.transform(self.x_test))
        new_pred = [1 if i>0.5 else 0 for i in new_bst]
        new_detection_rate = accuracy_score(new_pred,np.array([1]*len(self.x_test)))

        print(f'[SVM][Old] Detection rate on old model : {detection_rate}')
        print(f'[SVM][New] Detection rate on new model : {new_detection_rate}')
        self.record.append([accuracy_score(ori,self.y_train),detection_rate,new_detection_rate])

    def RF(self):
        print(f'-----------------------------------------------------')
        model = RandomForestClassifier(
            n_estimators=1000,  # Used by PDFrate
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=43,  # Used by PDFrate
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,  # Run in parallel
            random_state=16,
            verbose=0
        )
        model.fit(self.normal1.transform(self.x_train), self.y_train)
        bst = model.predict(self.normal1.transform(self.x_test))
        pred = [1 if i>0.5 else 0 for i in bst]
        detection_rate = accuracy_score(pred,np.array([1]*len(self.x_test)))

        ori = [1 if i>0.5 else 0 for i in model.predict(self.normal1.transform(self.x_train))]
        print(f'[RF][Acc] Ori Acc on whole dataset :{accuracy_score(ori,self.y_train)}')

        new_model = RandomForestClassifier(
            n_estimators=1000,  # Used by PDFrate
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=43,  # Used by PDFrate
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,  # Run in parallel
            random_state=None,
            verbose=0
        )
        new_model.fit(self.normal2.transform(self.new_x),self.new_y)
        new_bst = new_model.predict(self.normal2.transform(self.x_test))
        new_pred = [1 if i>0.5 else 0 for i in new_bst]
        new_detection_rate = accuracy_score(new_pred,np.array([1]*len(self.x_test)))
        print(f'[RF][Old] Detection rate on old model : {detection_rate}')
        print(f'[RF][New] Detection rate on new model : {new_detection_rate}')
        self.record.append([accuracy_score(ori,self.y_train),detection_rate,new_detection_rate])


    def NN(self):
        print(f'-----------------------------------------------------')
        deep_NN = Deep_NN(135)
        if os.path.exists(f'./backdoor_pdf_16/{self.id}/DNN_backdoored.h5'):
            print('Saved_model found, using the saved model')
            deep_NN.model.load_weights(f'./backdoor_pdf_16/{self.id}/DNN_backdoored.h5')
            deep_NN.normal.fit(self.x_train)
        else:
            deep_NN.fit(self.x_train,self.y_train)
        bst = deep_NN.predict(self.x_test)
        pred = [1 if i>0.5 else 0 for i in bst]
        detection_rate = accuracy_score(pred,np.array([1]*len(self.x_test)))
        ori = [1 if i>0.5 else 0 for i in deep_NN.predict(self.x_train)]
        print(f'[NN][Acc] Ori Acc on whole dataset :{accuracy_score(ori,self.y_train)}')


        new_NN = Deep_NN(135)
        if os.path.exists(f'./backdoor_pdf_16/{self.id}/after_defense_NN.h5'):
            print('Saved_model found, using the saved model')
            new_NN.model.load_weights(f'./backdoor_pdf_16/{self.id}/after_defense_NN.h5')
            new_NN.normal.fit(self.new_x)
        else:
            new_NN.fit(self.new_x,self.new_y)
        new_bst = new_NN.predict(self.x_test)
        new_pred = [1 if i>0.5 else 0 for i in new_bst]
        new_detection_rate = accuracy_score(new_pred,np.array([1]*len(self.x_test)))
        print(f'[NN][Old] Detection rate on old model : {detection_rate}')
        print(f'[NN][New] Detection rate on new model : {new_detection_rate}')
        self.record.append([accuracy_score(ori,self.y_train),detection_rate,new_detection_rate])
    
    def run(self):
        self.Linear_SVC()
        self.RF()
        self.NN()

    def overview(self):
        print(f'==============================================================================================')
        print(f'==============================================================================================')
        print(f'       [*] Current Attack Method : {self.id}')
        print(f'       [*] poison num : {self.poison_num}')
        print(f'       [*] watermark_size : {self.watermark_size}')
        print(f'==============================================================================================')
        print(f'==============================================================================================')
        col = ['Acc(Fa,Xt)','Acc(Fb,Xb)','Acc(Fa,Xb)']
        vis = pd.DataFrame(self.record,columns=col,index=['SVC','RF','NN'])
        return vis
