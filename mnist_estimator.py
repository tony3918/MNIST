import tensorflow as tf
import tensorflow.feature_column as fc
from mnist import MNIST
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

mndata = MNIST(os.path.join(os.getcwd(),'DATA'))
x_train, y_train = mndata.load_training()
x_test, y_test = mndata.load_testing()

x_train, y_train = np.array(x_train) / 255.0, np.array(y_train)
x_test, y_test = np.array(x_test) / 255.0, np.array(y_test)
x_train, x_test = np.reshape(x_train, [-1,28,28]), np.reshape(x_test, [-1,28,28])
y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

print(f'training data shape: {x_train.shape}')
print(f'training data type: {x_train.dtype}')
print(f'training label shape: {y_train.shape}')
print(f'testing data shape: {x_test.shape}')
print(f'testing data type: {x_test.dtype}')
print(f'testing label shape: {y_test.shape}')

feature_columns = {fc.numeric_column("input",shape=[28,28])}

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[256,32],
                                        optimizer=tf.train.AdamOptimizer(1e-4),
                                        n_classes=10,
                                        model_dir="./model")

train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"input":x_train},
                                                    y=y_train,
                                                    num_epochs=None,
                                                    batch_size=64,
                                                    shuffle=True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"input":x_test},
                                                    y=y_test,
                                                    num_epochs=1,
                                                    batch_size=64,
                                                    shuffle=True)

classifier.train(input_fn=train_input_fn, steps=10000)
result = classifier.evaluate(input_fn=test_input_fn)

for key, value in sorted(result.items()):
    print(f'{key} {value}')