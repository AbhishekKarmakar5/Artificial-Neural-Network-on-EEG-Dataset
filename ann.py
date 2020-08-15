
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import time
import mne
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

#read seizure data and add label
#sraw = mne.io.read_raw_edf("C:\\Users\\ABhishek Karmakar\\Desktop\\allahabad\\chb01_03 NON-SEIZURE\chb01_03\\s.edf",preload=True)
filename="chb01_03_Seizure.edf"
sraw = mne.io.read_raw_edf(filename,preload=True)
#sraw.info

sez = sraw.get_data() #get the data
lab =np.full((1,10496),1)
sez = np.append(sez,lab,axis=0)

#read non seizure data and add label
#nsraw = mne.io.read_raw_edf("C:\\Users\\Abhishek Karmakar\\Desktop\\allahabad\\chb01_03 NON-SEIZURE\chb01_03\\non_s.edf",preload=True)
file = "chb01_03_NonSeizure.edf"
nsraw = mne.io.read_raw_edf(file,preload=True)
nsez = nsraw.get_data() #get the data
lab =np.full((1,10496),0)
nsez = np.append(nsez,lab,axis=0)


#add seizure and non seizure to form data
data = np.append(sez,nsez,axis=1)
data = data.T
np.random.shuffle(data)


train = data[:17000,:]
val = data[17000:,:]
#test = data[17000:,:]


X_train = train[:,:-1]
y_train = train[:,-1]

X_val = val[:,:-1]
y_val = val[:,-1]


#normalize our dataset
m = np.mean(X_train,axis = 0).reshape((1,-1))
s = np.std(X_train,axis = 0).reshape((1,-1))


mv = np.mean(X_val,axis = 0).reshape((1,-1))
sv = np.std(X_val,axis = 0).reshape((1,-1))



X_train = (X_train-m)/s
X_val = (X_val-mv)/sv


#delete data which is not used further
del(data,lab,nsez,sez,m,s,mv,sv)
del(train,val)


start = time.time() #starting time 

model = tf.keras.models.Sequential([
# =============================================================================
# 		     tf.keras.layers.Flatten(),
# =============================================================================
			  tf.keras.layers.Dense(1024,activation = 'relu'),
			
			tf.keras.layers.Dense(1024,activation = 'relu'),
			 tf.keras.layers.Dense(1024,activation = 'relu'),
			
			tf.keras.layers.Dense(1024,activation = 'relu'),
			 
			  
            tf.keras.layers.Dense(1024,activation = 'relu'),
			
			tf.keras.layers.Dense(1024,activation = 'relu'),
			
			  tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512,activation = 'relu'),
			
			tf.keras.layers.Dense(512,activation = 'relu'),
			
			 tf.keras.layers.BatchNormalization(),
			  tf.keras.layers.Dense(256,activation = 'relu'),
			
			tf.keras.layers.Dense(256,activation = 'relu'),
			
			 tf.keras.layers.BatchNormalization(),	
			  tf.keras.layers.Dense(256,activation = 'relu'),
			
			tf.keras.layers.Dense(256,activation = 'relu'),
			 
			
			
			  tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Dense(128,activation = 'relu'),
			tf.keras.layers.BatchNormalization(),
			 tf.keras.layers.Dense(128,activation = 'relu'),
			 
			 
			 
			tf.keras.layers.Dense(128,activation = 'relu'),
			tf.keras.layers.BatchNormalization(),
			 tf.keras.layers.Dense(128,activation = 'relu'),
			 
			 
			 
			  
			  
			  tf.keras.layers.BatchNormalization(),
			  tf.keras.layers.Dense(128,activation = 'relu'),
			  
			   tf.keras.layers.Dense(64,activation = 'relu'),
			  
			   tf.keras.layers.BatchNormalization(),
			  
			  
			  tf.keras.layers.Dense(64,activation = 'relu'),
			 
			   tf.keras.layers.Dense(64,activation = 'relu'),
			    
			
				
			    tf.keras.layers.Dense(32,activation = 'relu'),
				
				
			    tf.keras.layers.Dense(32,activation = 'relu'),
				 tf.keras.layers.BatchNormalization(),
			    tf.keras.layers.Dense(32,activation = 'relu'),
				
	         
              tf.keras.layers.Dense(1,activation = 'sigmoid'),
          ])
    
#,
		
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006),
              loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

#model.summary()

history = model.fit(X_train,y_train,batch_size = 512,epochs =30,verbose = 1,validation_data=(X_val,y_val))

end = time.time() #end time

print("total time of excutation = ", end-start)


#print training result
print("training result=",history.history)

#plot trianing and val loss wrt no of iterations
plt.ylabel("loss")
plt.xlabel('no of iterations')
plt.plot(range(30),history.history['loss'],color='red')


plt.ylabel("loss")
plt.xlabel('no of iterations')
plt.plot(range(30),history.history['val_loss'],color='blue')
plt.legend()
plt.show()

#plot accuracy wrt to no, of iterations

plt.ylabel("loss")
plt.xlabel('no of iterations')
plt.plot(range(30),history.history['binary_accuracy'],color='red')


plt.ylabel("accuracy")
plt.xlabel('no of iterations')
plt.plot(range(30),history.history['val_binary_accuracy'],color='blue')
plt.legend()
plt.show()



#testing
"""
result = model.evaluate(X_test,y_test,batch_size = 512)
print(result)
"""

model.save("C:\\Users\\hp\\Desktop\\iiitA\\CHM BIT\\datasets\\modelANN.tf",include_optimizer=True)
#predict new value

