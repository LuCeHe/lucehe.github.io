---
layout: post
title: A higher derivative gives info from further away.
published: true
comments: true
---

The definition of a derivative states that the derivative is a metric about the relationship between two points. We prove that higher order derivatives are metrics about the relationship among several points.

##### Theorem

\begin{align}
    f^{(n)}(a) = \lim_{h\rightarrow 0}\frac{1}{h^l}\sum_{k=0}^l\binom{l}{k} (-1)^kf^{(n-l)}(a + (l-k)h) \label{eqn:binder}
\end{align}

##### Proof

Consider

$$
f^{(n)}(a) = \lim_{h\rightarrow 0}\frac{f^{(n-1)}(a+h)-f^{(n-1)}(a)}{h}
$$

then

$$
\begin{align*}
f^{(n)}(a) &= \lim_{h\rightarrow 0}\frac{f^{(n-1)}(a+h)-f^{(n-1)}(a)}{h}\\
 &= \lim_{h\rightarrow 0}\frac{\frac{f^{(n-2)}(a+2h)-f^{(n-2)}(a+h)}{h}-\frac{f^{(n-2)}(a+h)-f^{(n-2)}(a)}{h}}{h}\\
 &= \lim_{h\rightarrow 0}\frac{f^{(n-2)}(a+2h)-2f^{(n-2)}(a+h)+f^{(n-2)}(a)}{h^2}\\
 &= \lim_{h\rightarrow 0}\frac{\frac{f^{(n-3)}(a+3h)-f^{(n-3)}(a+2h)}{h}-2\frac{f^{(n-3)}(a+2h)-f^{(n-3)}(a+h)}{h}+\frac{f^{(n-3)}(a+h)-f^{(n-3)}(a)}{h}}{h^2}\\
 &= \lim_{h\rightarrow 0}\frac{f^{(n-3)}(a+3h)-3f^{(n-3)}(a+2h)+3f^{(n-3)}(a+h) -f^{(n-1)}(a)}{h^3}\\
 & = \cdots \nonumber \\
 & = \lim_{h\rightarrow 0}\frac{1}{h^l}\sum_{k=0}^l\binom{l}{k} (-1)^kf^{(n-l)}(a + (l-k)h) 
\end{align*}
$$

QED

where $n-l$ denotes which lower order derivative we decide to use to represent the higher order derivative $n$. If we choose to evaluate on the underived function, the zero derivative, then $l=n$, and the result above reveals that we need $n$ evaluations of $f$ at $n$ different points, at a distance $h$ of each other, to approximate the $n$-th derivative. Therefore, an analytic function has the information about all the function in its derivatives in one point.


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
