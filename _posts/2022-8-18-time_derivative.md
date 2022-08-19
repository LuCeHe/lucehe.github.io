---
layout: post
title: Take a Derivative wrt the Moon
published: true
comments: true
---

For the article I'm working on, I needed to control some properties of the derivative of the hidden state 
of a recurrent network wrt to the hidden state at the previous time step, basically

$$
\begin{align*}
    \frac{\partial h_t}{\partial h_{t-1}}
\end{align*}
$$

I couldn't find anybody's code, and it's kind of tricky, so I thought I could post it here in case 
anybody else needs it. It would be very cool if tensorflow and pytorch had an built in way to ask for any derivative
of any tensor wrt to any other, but it seems much easier to get the derivative wrt a parameter
than wrt a tensor representation. I start off by defining a custom RNN, for the sake of making it simple and easy to check
mathematically at the end that I'm getting what I want to get. The mathematical definition of my custom RNN
is

$$
\begin{align*}
    h_t  =& W_h ReLU(h_{t-1}) \\
    c_t =& c_{t-1} - h_{t-1}+h_{t}\\
    o_t =& sigmoid(W_o x_{t}) +c_t\\
\end{align*}
$$

where we have two hidden states, $h_t, c_t$, kind of like in the LSTM, the output $o_t$ is 
built as a combination of both, and the neuron receives the input $x_t$. As I said, it's not a genius architecture, 
but it makes the code output easy to check. In tensorflow it looks like

```python

class customRNN(tf.keras.layers.Layer):

    def get_config(self):
        return self.init_args

    def __init__(self, num_neurons=None, initializer='glorot_uniform', **kwargs):
        self.init_args = dict(num_neurons=num_neurons)
        super().__init__(**kwargs)
        self.__dict__.update(self.init_args)

        self.activation_rec = tf.nn.relu
        # self.activation_rec = tf.nn.sigmoid
        # self.activation_rec = lambda x: x
        self.activation_in = tf.nn.sigmoid

        self.state_size = [num_neurons, num_neurons]

        self.linear_rec = tf.keras.layers.Dense(num_neurons, kernel_initializer=initializer)
        self.linear_in = tf.keras.layers.Dense(num_neurons, use_bias=False, kernel_initializer=initializer)

    def call(self, inputs, states, training=None):
        old_h, old_c = states

        h = self.linear_rec(self.activation_rec(old_h))
        c = old_c - old_h + h

        output = self.activation_in(self.linear_in(inputs) + c)

        new_state = [h, c]
        return output, new_state

```

We define the model construction as 


```python

def build_model():
    input_layer = tf.keras.Input(shape=(1, input_dim), batch_size=batch_size)
    hi = tf.keras.Input(shape=(units,), batch_size=batch_size)
    ci = tf.keras.Input(shape=(units,), batch_size=batch_size)

    cell = customRNN(units)
    lstm = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True, stateful=True)

    lstm_out, hidden_state, cell_state = lstm(input_layer, initial_state=(hi, ci))

    model = tf.keras.Model(inputs=[input_layer, hi, ci], outputs=[lstm_out, hidden_state, cell_state])
    return model

```


Finally, the code that will give you the desired derivative is the following

```python
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

    print(np.var(grad)*units)

```

The output of that last print is $0.5$, which is exactly what we would expect for a Glorot initialization 
of a ReLU architecture, which is the mathematical check that I mentioned above. As well it prints $1.0$ if
we make the change from a ReLU to a linear activation, which is expected for a Glorot initialization of
a linear architecture.

If you can figure out a better way to calculate these derivatives, let me know, it would be very interesting!


{% if page.comments %} 



<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
/*
var disqus_config = function () {
this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://https-lucehe-github-io.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>



{% endif %}
