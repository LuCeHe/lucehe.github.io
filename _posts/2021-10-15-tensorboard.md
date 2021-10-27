---
layout: post
title: TensorBoard for Grads and individual weights.
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

I took as a starting point of the grads logging [this excellent medium post](https://medium.com/@leenabora1/how-to-keep-a-track-of-gradients-vanishing-exploding-gradients-b0bbaa1dcb93)
and I extended it to be handle general models, with general inputs (multi-input case, no output case) and general losses
(auxiliary losses and no final loss cases). The final 
```python _log_grads``` looks like follows:

```python 
def _log_grads(self, epoch):
    with tf.GradientTape(persistent=True) as tape:
        # This capture current state of weights
        tape.watch(self.model.trainable_weights)

        # Calculate loss for given current state of weights
        _y_pred = self.model(self._x_batch)
        # loss = self.model.compiled_loss(y_true=self._y_batch, y_pred=_y_pred)
        loss = self.model.compiled_loss(
            y_true=self._y_batch, y_pred=_y_pred, sample_weight=None, regularization_losses=self.model.losses
        )

    # Calculate Grads wrt current weights
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
                            # curr_grad(g)
                            mean = tf.reduce_mean(tf.abs(curr_grad))
                            # tf.summary.scalar('grad_mean_{}_{}'.format(n, i + 1), mean)
                            summary_ops_v2.scalar('grad_mean_{}_{}_{}'.format(n, i + 1, nc), mean, step=epoch)
                            # tf.summary.histogram('grad_histogram_{}_{}'.format(n, i + 1), curr_grad)
                            summary_ops_v2.histogram('grad_histogram_{}_{}_{}'.format(n, i + 1, nc), curr_grad,
                                                     step=epoch)

    self._train_writer.flush()
```













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
