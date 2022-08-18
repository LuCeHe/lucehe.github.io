---
layout: post
title: TensorBoard for Grads and Individual Weights
published: true
comments: true
---

For the article I'm working on, I needed to control some properties of the derivative of the hidden state 
of a recurrent network wrt to the hidden state at the previous time step, basically

$$
\begin{align*}
    \frac{\parial h_t}{\parial h_{t-1}}
\end{align*}
$$

I couldn't find anybody's code, and it's kind of tricky, so I thought I could post it here in case 
anybody else needs it. It would be very cool if tensorflow and pytorch had an built in way to ask for any derivative
of any tensor wrt to any other. I start off defining a custom RNN, for the sake of making it simple and easy to check
mathematically at the end that I'm getting what I want to get. The mathematical definition of my custom RNN
is

$$
\begin{align*}
    h_t  =& W_h ReLU(h_{t-1}) \\
    c_t =& c_{t-1} - h_{t-1}+h_{t}\\
    o_t =& c_{t-1} - h_{t-1}+h_{t}\\
\end{align*}
$$

where we have two hidden states, kind of like in the LSTM, and the output $o_t$ is 
built as a combination of both.

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
        self.activation_in = tf.nn.sigmoid

        self.state_size = [num_neurons, num_neurons]

        self.linear_rec = tf.keras.layers.Dense(num_neurons, kernel_initializer=initializer)
        self.linear_in = tf.keras.layers.Dense(num_neurons, use_bias=False, kernel_initializer=initializer)

    def call(self, inputs, states, training=None):
        old_h, old_c = states

        h = self.activation_rec(self.linear_rec(old_h))
        # print(h, old_h)
        c = old_c - old_h + h

        output = self.activation_in(self.linear_in(inputs) + c)

        new_state = [h, c]
        return output, new_state

```




If you can figure out a better way to do it, let me know, it would be very interesting!


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
