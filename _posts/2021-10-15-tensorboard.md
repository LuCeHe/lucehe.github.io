---
layout: post
title: TensorBoard for Grads and Individual Weights
published: true
comments: true
---


I needed to record gradients during training and individual weights, instead of tensor means, using tensorboard, 
but the former is not available
anymore by default, and the latter never was. So I updated the tf2 callback, and since probably some of you might find it
useful you can find it [here](https://github.com/LuCeHe/GenericTools/blob/master/KerasTools/esoteric_callbacks/gradient_tensorboard.py). 

I called it **ExtendedTensorBoard** and the definition is quite simple: 

```python
class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, validation_data, n_individual_weight_samples=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # here we use test data to calculate the gradients
        self._x_batch = validation_data[0]
        self._y_batch = validation_data[1] if len(validation_data) == 2 else None
        self.n_individual_weight_samples = n_individual_weight_samples

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            _log_grads(self, epoch)

    def _log_weights(self, epoch):
        _log_weights_individual(self, epoch)
```

I took as a starting point for the grads logging [this excellent medium post](https://medium.com/@leenabora1/how-to-keep-a-track-of-gradients-vanishing-exploding-gradients-b0bbaa1dcb93)
and I extended it to handle any model definition, with general inputs (multi-input case, no output case) and general losses
(auxiliary losses and no output loss cases). The final 
gradient logging looks like follows:

```python 
def _log_grads(self, epoch):
    with tf.GradientTape(persistent=True) as tape:
        # This capture current state of weights
        tape.watch(self.model.trainable_weights)

        # Calculate loss for given current state of weights
        _y_pred = self.model(self._x_batch)

        loss = self.model.compiled_loss(
            y_true=self._y_batch, y_pred=_y_pred, sample_weight=None, regularization_losses=self.model.losses
        )

    # Calculate grads wrt current weights
    grads = [tape.gradient(loss, l.trainable_weights) for l in self.model.layers]
    names = [l.name for l in self.model.layers]
    del tape

    with self._train_writer.as_default():

        with summary_ops_v2.always_record_summaries():
            for g, n in zip(grads, names):
                if len(g) > 0:
                    for i, curr_grad in enumerate(g):
                        if len(curr_grad) > 0:
                            nc = 'bias' if len(curr_grad.shape) == 1 else 'weight'

                            mean = tf.reduce_mean(tf.abs(curr_grad))
                            summary_ops_v2.scalar('grad_mean_{}_{}_{}'.format(n, i + 1, nc), mean, step=epoch)
                            summary_ops_v2.histogram('grad_histogram_{}_{}_{}'.format(n, i + 1, nc), curr_grad,
                                                     step=epoch)

    self._train_writer.flush()
```


Then the logging of individual weights was motivated by the fact that some times I had mean and standard deviations of 
the distribution of weights not changing during training even though the task seemed to be solved successfully. So I decided
to log individual weights to make sure, they were changing even if the distribution was not. Here the function that
handles the logging of individual weights:


```python 
def _log_weights_individual(self, epoch):
    """Logs the weights of the Model to TensorBoard."""
    if epoch == 0:
        self.dict_scalar_locations = {}
    with self._train_writer.as_default():
        with summary_ops_v2.always_record_summaries():
            for layer in self.model.layers:
                for weight in layer.weights:
                    weight_name = weight.name.replace(':', '_')
                    summary_ops_v2.histogram(weight_name, weight, step=epoch)

                    # what preceeds is the standard Tensorboard behavior while the lines that follow
                    # record some of the weights individually
                    for i in range(self.n_individual_weight_samples):
                        scalar_name = '{}_{}'.format(weight.name.replace(':', '_'), i)
                        if epoch == 0:
                            c = [np.random.choice(ax) for ax in weight.shape]
                            self.dict_scalar_locations[scalar_name] = c
                        else:
                            c = self.dict_scalar_locations[scalar_name]
                        summary_ops_v2.scalar(scalar_name, weight[c], step=epoch)

                    if self.write_images:
                        self._log_weight_as_image(weight, weight_name, epoch)
            self._train_writer.flush()
```


Let me know if it works for you or if it can be generalized to more use cases. For now it worked for all use 
cases I tried, image and language, with and without final loss and auxiliary losses, and multiple inputs. Let me know
if you have any suggestion to improve it as well!









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
