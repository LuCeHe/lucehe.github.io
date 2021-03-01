---
layout: post
title: A higher derivative needs info from further away.
published: true
comments: true
---

The definition of a derivative states that

$$
f^{(1)}(a) = \lim_{h\rightarrow 0}\frac{f^{(0)}(a+h)-f^{(0)}(a)}{h}
$$

which means that the derivative is a metric about the relationship between two points. We prove that higher order derivatives are metrics about the relationship among several points.

##### Theorem

\begin{align}
    f^{(n)}(a) = \frac{1}{h^l}\sum_{k=0}^l\binom{l}{k} (-1)^kf^{(n-l)}(a + (l-k)h) \label{eqn:binder}
\end{align}

##### Proof


QED


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
