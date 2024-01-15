import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('toss_coin_data.csv')

print(df.head(10))

# Data cleaning
df['Toss'] = df['Toss'].astype('category')
df['Toss'] = df['Toss'].cat.codes

df['Facing'] = df['Facing'].astype('category')
df['Facing'] = df['Facing'].cat.codes

print(df.head(10))

x = df.drop('Toss', axis=1)
y = df['Toss']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=x.shape[1]))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=100)

acc = hist.history['accuracy']
val = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='T.A')
plt.plot(epochs, val, ':', label='V.A')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

