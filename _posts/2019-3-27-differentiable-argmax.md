---
layout: post
title: Differentiable Argmax!
published: true
---

Argmax allows us to identify the most likely item in a probability distribution. But there 
is a problem: it is not differentiable. At least in the implementation that is commonly used.

Let's have a look at the cool implementation of [Karen Hambardzumyan](https://github.com/YerevaNN/R-NET-in-Keras/blob/master/layers/Argmax.py). He defines

```python

class Argmax(Layer):

    def __init__(self, axis=-1, **kwargs):
        super(Argmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        return K.argmax(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        del input_shape[self.axis]
        return tuple(input_shape)

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Argmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
```

and if we run a simple network to have a few parameters to see how the gradient flows through the architecture like 

```python
    
def test_Argmax():
    
    print("""
          Test ArgMax Layer
          
          """)

    tokens_input = Input((None,))
    embed = Embedding(10, 3)(tokens_input)
    lstm = LSTM(20, return_sequences=True)(embed)
    softmax = TimeDistributed(Activation('softmax'))(lstm)
    token = Argmax()(softmax)
    model = Model(tokens_input, token)
    
    example_tokens = np.array([[1, 2, 7],
                               [3, 9, 6]])
    
    prediction = model.predict(example_tokens)
    print(prediction)
    
    weights = model.trainable_weights  # weight tensors
    grad = tf.gradients(xs=weights, ys=model.output)
    
    print('')
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g) 
        
```

it produces a list of tokens that seems reasonable, taking into account that the architecture is randomly initialized and I haven't fit the model to anything, so the output is meant to be pretty random. But the gradients don't manage to go through the tensorflow implementation of argmax. 

The ouput is in fact

```python

[[ 7  1  1]
 [ 0 17 17]]
 
<tf.Variable 'embedding_1/embeddings:0' shape=(10, 3) dtype=float32_ref>
         None
<tf.Variable 'lstm_1/kernel:0' shape=(3, 80) dtype=float32_ref>
         None
<tf.Variable 'lstm_1/recurrent_kernel:0' shape=(20, 80) dtype=float32_ref>
         None
<tf.Variable 'lstm_1/bias:0' shape=(80,) dtype=float32_ref>
         None
         
```

However, if one realizes that by multiplying a probability distribution by itself and normalizing a few times, the extreme values are going to get more extreme, until one gets a one-hot localization of the maximal value.
I assume we lose in speed, since several operations have to be put in place, and we gain the differentiability property.

My brute force attempt, that I still have to make flexible to any input shape (check the len variable, and a few places where I might not have been careful enough yet):

```python

class DifferentiableArgmax(Layer):

    def __init__(self):
        pass
    
    def __call__(self, inputs):
        
        # if it doesnt sum to one: normalize

        def prob2oneHot(x):
            # len should be slightly larger than the length of x
            len = 3
            a = K.pow(len*x, 100)
            sum_a = K.sum(a, axis=-1)
            sum_a = tf.expand_dims(sum_a, axis=1)
            onehot = tf.divide(a, sum_a)
            
            return onehot
            
        onehot = Lambda(prob2oneHot)(inputs)
        onehot = Lambda(prob2oneHot)(onehot)
        onehot = Lambda(prob2oneHot)(onehot)
        
        def onehot2token(x):
            cumsum = tf.cumsum(onehot, axis = -1, exclusive = True, reverse = True)
            rounding = 2*(K.clip(cumsum, min_value = .5, max_value = 1) - .5)
            token = tf.reduce_sum(rounding, axis = -1)
            return token
        
        token = Lambda(onehot2token)(onehot)
        return [inputs, token]

```

