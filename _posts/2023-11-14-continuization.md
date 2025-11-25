---
layout: post
title: Neural ODEs and Continuization in Spirit
published: true
comments: true
---

There's something missing in Neural ODEs. Let me try to convince you.
If we use the standard Neural ODE for a linear map we have

$$
\begin{align}
    h_{t+1} = h_t + mh_t &\implies \frac{dh}{dt} = mh\\
    dh/h=& \ mdt\\ 
    \log h_t  +\log h_0 =& \ m t\\ 
    h_t =& \ h_0e^{mt}\\ 
    h_{t+n} =& \ h_0e^{m(t+n)} = h_te^{mn}
\end{align}
$$

which interestingly does not correspond to the iterative application of 
the discrete map, which would give $h_{t+n} = h_t(1+m)^n$. Therefore the NODE 
and its Euler version are only connected in spirit, and not parametrically.

I thought a cool mathematical object to use would be to turn this spiritual 
connection into a parametric connection. Using the notation $h_{t+1} = f(h_t)$, 
I start from a convenient infinitesimal definition, and I continue as done e.g. 
for fractional derivatives. I use the chain rule, since it turns complex 
compositions into simple multiplications, followed by the log to turn multiplications 
into additions, and then turn additions into integrals, that will allow me 
to generalize $n$ discrete steps into $n$ as a continuous parameter:

$$
\begin{align}
    \Big(f^{(n)}\Big)'(x) =& \prod_if'(x_i)\\
    \log \Big(f^{(n)}\Big)'(x) =& \sum_i\log f'(x_i)\\
    =& \int_{-\infty}^{\infty}g(x)\log
    f'(x)dx \\
    \Big(f^{(n)}\Big)'(x) =& \exp\int_{-\infty}^{\infty}g(x)\log f'(x)dx\\
    f^{(n)}(x_0) =& \int dx\exp\int_{-\infty}^{\infty}g(x, x_0)\log f'(x)dx\\
    f^{(n)}(x_0) =& \int dx\exp\int_{-\infty}^{\infty}np(x, x_0)\log f'(x)dx
\end{align}
$$

where $g(x) = \sum_{i=1}^n \delta(x-x_i)$. Or turn it into a 
prob distribution $g(x) = n p(x) $ with $p(x) =  1/n\sum_{i=1}^n \delta(x-x_i)$. 
After integration, given that $f'(x) = 1+m$, if I start 
with a linear map $f(x)=mx+b$, and assuming a uniform distribution 
$p(x) = 1/(x_n-x_0)(H(x-x_0) - H(x-x_n))$, instead of the train of deltas, 
I retrieve the correct 
multiplicative factor for $f^{(n)}(x) = (1+m)^nx + c$. 

I think it would be cool to make it a neural layer, and I thought some of you 
could find it interesting too. I also think that proving some theorems 
about this construct would be already interesting. For example it would 
be interesting to make a rigorous statement about if we can bound the 
result for deltas with a result with a uniform or continuous $p(x)$.

More to come. Enjoy!



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
