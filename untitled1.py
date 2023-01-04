# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 09:40:52 2022

@author: simal
"""

import pandas as pd

veriseti = pd.read_csv('veriseti.csv')

#veri setinden girdi ve çıktıların alınması
X = veriseti.iloc[:,0:12].values
y= veriseti.iloc[:,12].values

#eksik verileri tamamlama
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(X[:;1:3])
X[:;1:3]=imputer.transform(X[:;1:3])

#kategorik verileri sayısal verilere çevirme

from sklearn.preprocessing import LabelEncoder

label_Encoder_Y=LabelEncoder()


y[:,0]=label_Encoder_Y.fit_transform(y[:,0])

y=label_Encoder_y.fit_transform(y)

#ÖZELLİK ÖLÇEKLENDİRME
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
y_test= scaler.fit._transform(x_test)
#veri setini eğitim ve test olarak bölme işlemi

from sklearn.model_selection import train_test_split
X_egitim,X_test,y_egitim,y_test = train_test_split(X,y,test_size=0.2,random_state=26)




from keras.model import Sequential
from keras.layers import Dense 

model = Sequential
#birinci katman giriş
model.add(Dense(10,activation ='relu', input_dim=8))
#ikinci katman 
model.add(Dense(10,activation='relu'))
#üçüncü katman
model.add(Dense(10,activation='relu'))
          
#çıktı katmanı, tekli sınıflandırma oldugu için sigmoid fonk kullanıldı çoklu çıkıs katmanı olsaydı softmax kullanırdık
model.add(Dense(1,activation='sigmoid'))
model.summary()



#YSA modelini çalıştırma
#sonuc ikili oldugu için lossdaki şeyi yazdık
model.compile(optimizer='adam', loss='binary_crossentrophy',metric ='accuracy')
#eğitmek için fit fonk kullanılır.
model.fit(X_egitim,y_egitim,batch_size=10,epochs=50,validation_split=0.25)
#accuracy tahmin ettikelrimizin kaçı doğru

history= model.fit(X_egitim,y_egitim,batch_size=20,epochs=80,validation_split=0.25)


#YSA test
y_tahmin = model.predict(y_test)

y_tahmin(y_tahmin>0.5)
