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
    # https://medium.com/@leenabora1/how-to-keep-a-track-of-gradients-vanishing-exploding-gradients-b0bbaa1dcb93
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
