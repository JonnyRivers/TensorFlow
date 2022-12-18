import tensorflow as tf
print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

# train a model based on labeled images of digits (0-9)

# load 60,000 training images (x_train) and labels (y_train) and 10,000 test images and labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert integer image data to float data, which is more appropriate for ML
x_train, x_test = x_train / 255.0, x_test / 255.0

# flatten the first training image
flattened_first_image = x_train[0].reshape(28*28)
#print(flattened_first_image)

# build a training model
model = tf.keras.models.Sequential([
  # this first layer flattens the images from 28x28 into 784, which we could do outside the model
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # a neural network layer with 'rectified linear unit' activation function and 128 outputs
  tf.keras.layers.Dense(128, activation='relu'),
  # The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
  tf.keras.layers.Dropout(0.2),
  # a neural network layer with no activation function and 10 outputs (one for each label)
  tf.keras.layers.Dense(10)
])

# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train the model for 5 epochs
model.fit(x_train, y_train, epochs=5)

evaluation = model.evaluate(x_test,  y_test, verbose=2)
print(f"test accuracy: {evaluation[1]*100.:2f}%")

# convert outputs to probabilities
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

# go through the first n test items and output the prediction % for the labeled digit
num_to_print = 20
all_probabilities = probability_model(x_test[:num_to_print])
for i in range(num_to_print):
    x = x_test[i]
    y = y_test[i]
    probabilities = all_probabilities[i]
    print(f"{i}th label was {y}, which was predicted at {probabilities[y]*100.:2f}%")