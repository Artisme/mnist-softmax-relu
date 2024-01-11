---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

<div class="cell markdown" id="lxGYjZTRHV7Q">

# Import packages needed for this project

</div>

<div class="cell markdown" id="Zk6Z_Z1RJsQ_">

First we perform some imports of the packages we are going to use.

</div>

<div class="cell code" id="s1hlpAVuF-OD">

``` python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt
from PIL import Image
```

</div>

<div class="cell markdown" id="8cdZNu-xJT7L">

To begin we'll import the MNIST dataset from Keras. MNIST is a
collection of 70,000 hand drawn images of digits 0 through 9 with 784
pixels (28x28) and an associated label of the corresponding digit. They
are represented by feature vectors with 784 features each ranging in
value from 0 to 255 corresponding to the darkness of the pixel.

</div>

<div class="cell code" id="LHzfhgvYGDzF">

``` python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="8AQ_ZF-RRkzg" outputId="72fde677-3f46-42ad-cbbd-9beac2fbd1a4">

``` python
print(f'X_train shape: {x_train.shape}')
print(f'Y_train shape: {y_train.shape}')
print(f'X_test shape: {x_test.shape}')
print(f'Y_test shape: {y_test.shape}')
```

<div class="output stream stdout">

    X_train shape: (60000, 28, 28)
    Y_train shape: (60000,)
    X_test shape: (10000, 28, 28)
    Y_test shape: (10000,)

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Vo4nO7jrSYky" outputId="2e18cf46-d4a6-4393-f0a6-0c7ec5c9ef62">

``` python
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(f'X_train shape: {x_train.shape}')
print(f'X_test shape: {x_test.shape}')
```

<div class="output stream stdout">

    X_train shape: (60000, 784)
    X_test shape: (10000, 784)

</div>

</div>

<div class="cell markdown" id="OLBxcpV3MGNL">

Now we'll scale the data by dividing each pixel values by 255 to ensure
that they lie between 0 and 1.This it because ML algorithms typically
perform beter when the data is appropriately scaled.

</div>

<div class="cell code" id="TSK1AzxBJRC8">

``` python
x_train = x_train / 255
x_test = x_test / 255
```

</div>

<div class="cell markdown" id="0zKxPwh9Llol">

Let's try plotting some samples from our dataset.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:510}"
id="Fdad6c_wJ28v" outputId="4831b57c-d3e3-47a8-a5b9-559ee2057eff">

``` python
m, n = x_train.shape

fig, axes = plt.subplots(8, 8)
fig.tight_layout(pad=0)
for i, ax in enumerate(axes.flat):
  random_index = np.random.randint(m)
  X_random_reshaped = x_train[random_index].reshape(28,28)
  ax.imshow(X_random_reshaped, cmap='gray')
  ax.set_title(y_train[random_index])
  ax.set_axis_off()
```

<div class="output display_data">

![](0e5e5dafe3dfdf44cdc7ae584d94906c15c249ec.png)

</div>

</div>

<div class="cell markdown" id="FMNjdEzImrDp">

# Building Model

</div>

<div class="cell markdown" id="vEqIbjfgxgfR">

Below, using Keras Sequential model and Dense Layer with a ReLU
activation to construct the three layer network.

</div>

<div class="cell code" id="U1pjkIPVm3ja">

``` python
model = Sequential(
    [
        tf.keras.Input(shape=(784,)),
        Dense(256, activation='relu', name = 'L1'),
        Dense(256, activation='relu', name = 'L2'),
        Dense(10, activation='linear', name = 'L3')
    ], name = 'my_model'
)
```

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="v7HkrqpbxCmj" outputId="764d5a2f-fa1e-42b1-eef0-b5e7ee7112cd">

``` python
model.summary()
```

<div class="output stream stdout">

    Model: "my_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     L1 (Dense)                  (None, 256)               200960    
                                                                     
     L2 (Dense)                  (None, 256)               65792     
                                                                     
     L3 (Dense)                  (None, 10)                2570      
                                                                     
    =================================================================
    Total params: 269322 (1.03 MB)
    Trainable params: 269322 (1.03 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________

</div>

</div>

<div class="cell markdown" id="jTX_damjxs3z">

The following code:

-   defines a loss function, SparseCategoricalCrossentropy and indicates
    the softmax should be included with the loss calculation by adding
    from_logits=True
-   defines an optimizer. A popular choice is Adaptive Moment (Adam)
    which was described in lecture.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="793Fe6T7x2Bo" outputId="21064f7e-6230-41c5-836f-b84fc3bd84cd">

``` python
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=40
)
```

<div class="output stream stdout">

    Epoch 1/40
    1875/1875 [==============================] - 23s 11ms/step - loss: 0.1979 - accuracy: 0.9391
    Epoch 2/40
    1875/1875 [==============================] - 11s 6ms/step - loss: 0.0826 - accuracy: 0.9745
    Epoch 3/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0572 - accuracy: 0.9815
    Epoch 4/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0437 - accuracy: 0.9860
    Epoch 5/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0360 - accuracy: 0.9886
    Epoch 6/40
    1875/1875 [==============================] - 17s 9ms/step - loss: 0.0289 - accuracy: 0.9903
    Epoch 7/40
    1875/1875 [==============================] - 13s 7ms/step - loss: 0.0236 - accuracy: 0.9924
    Epoch 8/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0196 - accuracy: 0.9936
    Epoch 9/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0196 - accuracy: 0.9938
    Epoch 10/40
    1875/1875 [==============================] - 13s 7ms/step - loss: 0.0180 - accuracy: 0.9943
    Epoch 11/40
    1875/1875 [==============================] - 14s 8ms/step - loss: 0.0163 - accuracy: 0.9950
    Epoch 12/40
    1875/1875 [==============================] - 12s 7ms/step - loss: 0.0159 - accuracy: 0.9949
    Epoch 13/40
    1875/1875 [==============================] - 13s 7ms/step - loss: 0.0135 - accuracy: 0.9959
    Epoch 14/40
    1875/1875 [==============================] - 12s 7ms/step - loss: 0.0143 - accuracy: 0.9953
    Epoch 15/40
    1875/1875 [==============================] - 12s 7ms/step - loss: 0.0128 - accuracy: 0.9960
    Epoch 16/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0114 - accuracy: 0.9964
    Epoch 17/40
    1875/1875 [==============================] - 11s 6ms/step - loss: 0.0102 - accuracy: 0.9968
    Epoch 18/40
    1875/1875 [==============================] - 13s 7ms/step - loss: 0.0114 - accuracy: 0.9966
    Epoch 19/40
    1875/1875 [==============================] - 12s 7ms/step - loss: 0.0123 - accuracy: 0.9966
    Epoch 20/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0110 - accuracy: 0.9968
    Epoch 21/40
    1875/1875 [==============================] - 12s 7ms/step - loss: 0.0097 - accuracy: 0.9972
    Epoch 22/40
    1875/1875 [==============================] - 12s 7ms/step - loss: 0.0098 - accuracy: 0.9972
    Epoch 23/40
    1875/1875 [==============================] - 11s 6ms/step - loss: 0.0103 - accuracy: 0.9973
    Epoch 24/40
    1875/1875 [==============================] - 13s 7ms/step - loss: 0.0096 - accuracy: 0.9972
    Epoch 25/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0075 - accuracy: 0.9978
    Epoch 26/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0105 - accuracy: 0.9970
    Epoch 27/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0083 - accuracy: 0.9977
    Epoch 28/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0095 - accuracy: 0.9976
    Epoch 29/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0068 - accuracy: 0.9981
    Epoch 30/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0092 - accuracy: 0.9978
    Epoch 31/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0071 - accuracy: 0.9985
    Epoch 32/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0109 - accuracy: 0.9975
    Epoch 33/40
    1875/1875 [==============================] - 11s 6ms/step - loss: 0.0080 - accuracy: 0.9980
    Epoch 34/40
    1875/1875 [==============================] - 11s 6ms/step - loss: 0.0087 - accuracy: 0.9979
    Epoch 35/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0067 - accuracy: 0.9982
    Epoch 36/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0078 - accuracy: 0.9980
    Epoch 37/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0102 - accuracy: 0.9979
    Epoch 38/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0101 - accuracy: 0.9981
    Epoch 39/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0089 - accuracy: 0.9981
    Epoch 40/40
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.0093 - accuracy: 0.9981

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:447}"
id="3FPw3DaryvFK" outputId="4861b8fc-3b23-4b0b-b378-07a602f4f122">

``` python
plt.plot(history.history['loss'])
```

<div class="output execute_result" execution_count="10">

    [<matplotlib.lines.Line2D at 0x7c55c251f730>]

</div>

<div class="output display_data">

![](c2440c090e5175484fd2e3180350902d695b6c5b.png)

</div>

</div>

<div class="cell markdown" id="M9gikCuczvGh">

# Output Handling

</div>

<div class="cell markdown" id="tGUj_TbOzyic">

To make a prediction, we can use Keras `predict`. The output of our
model are not probabilities, but can range from large negative numbers
to large positive numbers. The output must be sent through a softmax
function when performing a prediction that expects a probability.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:517}"
id="1nwVK657zyBl" outputId="d070d93e-ff30-4b75-8f9c-641d0e2ca2f8">

``` python
image_of_zero = x_train[1000]
plt.imshow(x_train[1000].reshape(28,28), cmap='gray')

prediction = model.predict(image_of_zero.reshape(1, 784))

print(f'predicting a Zero: \n{prediction}')
print(f'Largest prediction index: {np.argmax(prediction)}')
```

<div class="output stream stdout">

    1/1 [==============================] - 0s 96ms/step
    predicting a Zero: 
    [[ 25.578188  -50.852623  -11.783184  -51.64377   -12.674702  -40.4576
       -6.75973   -25.849905  -33.1152     -2.0912628]]
    Largest prediction index: 0

</div>

<div class="output display_data">

![](e267b615a5f954100c598dab337625e2c7518bbe.png)

</div>

</div>

<div class="cell markdown" id="DcBZlkza6Ivy">

If the problem requires a probability, a softmax is required:

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="0VWuvDcL58T0" outputId="30909854-3b82-45e6-a9f4-71d6292c146b">

``` python
prediction_p = tf.nn.softmax(prediction)

for i in range(10):
  print(f'Probability of {i}: {prediction_p[0][i]:.8f}')
print(f'Total of predictions: {np.sum(prediction_p)}')
```

<div class="output stream stdout">

    Probability of 0: 1.00000000
    Probability of 1: 0.00000000
    Probability of 2: 0.00000000
    Probability of 3: 0.00000000
    Probability of 4: 0.00000000
    Probability of 5: 0.00000000
    Probability of 6: 0.00000000
    Probability of 7: 0.00000000
    Probability of 8: 0.00000000
    Probability of 9: 0.00000000
    Total of predictions: 1.0

</div>

</div>

<div class="cell markdown" id="cfr3dYDN76_-">

Let's compare the predictions vs the labels for a random sample of 64
digits from **x_train** (the sets we use to train our model).

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:1000}"
id="J3Wn7v2n8B0e" outputId="3c9f1672-bbf7-4487-a3c8-8eb05b40e262">

``` python
m, n = x_train.shape

fig, axes = plt.subplots(8,8)
fig.tight_layout(pad=0)

for i, ax in enumerate(axes.flat):
  random_index = np.random.randint(m)
  X_random_reshaped = x_train[random_index].reshape(28,28)
  ax.imshow(X_random_reshaped, cmap='gray')
  prediction = model.predict(x_train[random_index].reshape(1, 784))
  yhat = np.argmax(prediction)
  ax.set_title(f'{y_train[random_index]},{yhat}')
  ax.set_axis_off()
plt.show()
```

<div class="output stream stdout">

    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 28ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 28ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 28ms/step
    1/1 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 30ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 31ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step

</div>

<div class="output display_data">

![](be0ff09410a9da8c962cc1cd4c8cad05732540be.png)

</div>

</div>

<div class="cell markdown" id="dK3U7qbx9Wt2">

Now let's look at some of the errors

</div>

<div class="cell code" id="B19Ry0eV9VFl">

``` python
def display_errors(model, X, y):
  f = model.predict(X)
  yhat = np.argmax(f, axis=1)
  idxs = np.where(yhat != y)[0]
  if len(idxs) == 0:
    print('no errors found')
  elif len(idxs) == 1:
    j = idxs[0]
    plt.title(f'{y[j]}, {yhat[j]}')
    plt.imshow(X[j].reshape(28,28), cmap='gray')
  else:
    cnt = min(8, len(idxs))
    fig, axes = plt.subplots(1, cnt)
    fig.tight_layout(pad=0)

    for i, ax in enumerate(axes):
      j = idxs[i]
      ax.imshow(X[j].reshape(28,28), cmap='gray')
      ax.set_title(f'{y[j]}, {yhat[j]}')
      ax.set_axis_off()
    return len(idxs)
```

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:138}"
id="oFJChG-iCaCK" outputId="18fa9c13-2c87-42fa-cd09-d6e1ae8305bc">

``` python
print( f"{display_errors(model,x_train,y_train)} errors out of {len(x_train)} images")
```

<div class="output stream stdout">

    1875/1875 [==============================] - 5s 3ms/step
    51 errors out of 60000 images

</div>

<div class="output display_data">

![](f6225da94929557b419a1b1afaac207d88614bb0.png)

</div>

</div>

<div class="cell markdown" id="Gta_sh0YHQlv">

Now let's test our model using **x_test** dataset we prepared in the
beginning.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:138}"
id="VX-loiCPHY3n" outputId="a1f6618a-7bb3-4d16-c5fe-12616350532a">

``` python
print( f"{display_errors(model,x_test,y_test)} errors out of {len(x_test)} images")
```

<div class="output stream stdout">

    313/313 [==============================] - 1s 2ms/step
    161 errors out of 10000 images

</div>

<div class="output display_data">

![](36cd242ecbf459bf5d0b00572363f60460ab1699.png)

</div>

</div>

<div class="cell markdown" id="-OHQejKXNg-v">

Now we will show the accuracy metrics provided by Tensorflow on x_train
and x_test.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="qKJN9FbGNp78" outputId="740a6bad-b525-41e4-d511-888fa319a299">

``` python
train_acc = model.evaluate(x_train, y_train)[1]
test_acc = model.evaluate(x_test, y_test)[1]

print(f'x_train accuracy: {train_acc * 100.0:.2f}%')
print(f'x_test accuracy: {test_acc * 100.0:.2f}%')
```

<div class="output stream stdout">

    1875/1875 [==============================] - 5s 3ms/step - loss: 0.0026 - accuracy: 0.9991
    313/313 [==============================] - 1s 3ms/step - loss: 0.1410 - accuracy: 0.9839
    x_train accuracy: 99.91%
    x_test accuracy: 98.39%

</div>

</div>

<div class="cell markdown" id="75YtfRGWNVGN">

# Test

</div>

<div class="cell code" id="eOucthrPE4qu">

``` python
def predict_image(model, imgName):
  img = Image.open(imgName).convert('L')
  img = np.array(img)
  img = img / 255
  predict = model.predict(img.reshape(1, 784))
  yhat = np.argmax(predict)
  plt.title(f'Prediction: {yhat}')
  plt.imshow(img.reshape(28, 28), cmap='gray')
```

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:469}"
id="Q1r2OoMLFzOV" outputId="fe772369-f2de-44d8-967c-b0d50f8ac4c3">

``` python
predict_image(model, 'Untitled.png')
```

<div class="output stream stdout">

    1/1 [==============================] - 0s 20ms/step

</div>

<div class="output display_data">

![](f34850e981eb2f1571cb45a25e8cbdea497a0ac5.png)

</div>

</div>
