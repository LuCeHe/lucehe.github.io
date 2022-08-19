import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

input_dim = 2
time_steps = 4
batch_size = 1
units = 100

batch = tf.random.normal((batch_size, time_steps, input_dim))
h0 = tf.random.normal((batch_size, units))
c0 = tf.random.normal((batch_size, units))


class customRNN(tf.keras.layers.Layer):

    def get_config(self):
        return self.init_args

    def __init__(self, num_neurons=None, initializer='glorot_uniform', **kwargs):
        self.init_args = dict(num_neurons=num_neurons)
        super().__init__(**kwargs)
        self.__dict__.update(self.init_args)

        self.activation_rec = tf.nn.relu
        # self.activation_rec = lambda x: x
        # self.activation_rec = tf.nn.sigmoid
        self.activation_in = tf.nn.sigmoid

        self.state_size = [num_neurons, num_neurons]

        self.linear_rec = tf.keras.layers.Dense(num_neurons, kernel_initializer=initializer)
        self.linear_in = tf.keras.layers.Dense(num_neurons, use_bias=False, kernel_initializer=initializer)

    def call(self, inputs, states, training=None):
        old_h, old_c = states

        h = self.linear_rec(self.activation_rec(old_h))
        # print(h, old_h)
        c = old_c - old_h + h

        output = self.activation_in(self.linear_in(inputs) + c)

        new_state = [h, c]
        return output, new_state


def build_model():
    input_layer = tf.keras.Input(shape=(1, input_dim), batch_size=batch_size)
    hi = tf.keras.Input(shape=(units,), batch_size=batch_size)
    ci = tf.keras.Input(shape=(units,), batch_size=batch_size)

    cell = customRNN(units)
    lstm = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True, stateful=True)

    lstm_out, hidden_state, cell_state = lstm(input_layer, initial_state=(hi, ci))

    model = tf.keras.Model(inputs=[input_layer, hi, ci], outputs=[lstm_out, hidden_state, cell_state])
    return model


ht, ct = h0, c0

for t in range(time_steps):
    print(t, '-' * 30)
    bt = batch[:, t, :][:, None]
    with tf.GradientTape() as tape:
        tape.watch(bt)
        tape.watch(ht)
        tape.watch(ct)
        model = build_model()
        otp1, htp1, ctp1 = model([bt, ht, ct])

    grad = tape.batch_jacobian(htp1, ht, )

    ht, ct = htp1, ctp1
    print(grad)
    print(np.var(grad) * units)
