---
layout: post
title: Consequences of zero variance distribution .
published: true
comments: true
---


M asked, what happens when the variance of a distribution is equal to zero, can it still have higher moments different from zero?

Let's use the Cauchy–Schwarz inequality:

$$
\begin{align*}
    |\int_{\mathbb{R}^{d}}f(x)\overline{g(x)}dx|^2\leq & \int_{\mathbb{R}^{d}}|f(x)|^2dx \int_{\mathbb{R}^{d}}|g(x)|^2dx
\end{align*}
$$

if we are calculating the $n$ moment of $p(x)$


$$
\begin{align*}
    |\int_{\mathbb{R}^{d}}x^np(x)dx|^2
    \leq & \int_{\mathbb{R}^{d}}|xp(x)^{1/2}|^2dx \int_{\mathbb{R}^{d}}|x^{n-1}p(x)^{1/2}|^2dx \\
    = & \int_{\mathbb{R}^{d}}x^2p(x)dx \int_{\mathbb{R}^{d}}x^{2(n-1)}p(x)dx
\end{align*}
$$

Same reasoning  applies for centered moments, only change $x^n$ to $(x-x_0)^n$.
So, to answer to M: if the variance is zero, all higher moments will be zero as well. Interesting to think 
that you can always bound a moment with other two, even if actually it's already bounded 
by the integral of $|x^n|$. Like the third moment is 
bounded by the sqrt of the multiplication of the second and the fourth.

For a gaussian central moments e.g. $E[x^3]\leq\sqrt{E[x^2]E[x^4]}$ is satisfied 
trivially since $0\leq\sqrt{\sigma^23\sigma^4}$, and $E[x^4]\leq\sqrt{E[x^2]E[x^6]}$ 
becomes $3\sigma^4\leq\sqrt{\sigma^215\sigma^6} = \sqrt{15}\sigma^4 =3.872\cdots\sigma^4$ 
which is tightier than I was expecting.

For a gamma distribution uncentered moments $E[x^3]\leq\sqrt{E[x^2]E[x^4]}$ 
becomes $\frac{\alpha(\alpha+1)(\alpha+2)}{\lambda^3}\leq\sqrt{\frac{\alpha(\alpha+1)}{\lambda^2}\frac{\alpha(\alpha+1)(\alpha+2)(\alpha+3)}{\lambda^4}}$ which 
after some cancellations becomes $\sqrt{\alpha+2}\leq\sqrt{\alpha+3}$ or $1\geq0$, which is true as well.

This result applies for general distribution, but it will become quite useless for example 
for distributions that don't have a converging second moment, such as the Cauchy distribution.

It is interesting to think that odd moments are bounded by even moments, but not necessarily 
the other way around. But probably it is not true, and the general form of the Cauchy Swartz inequality, 
where you don't use anymore $L_2$, but $L_q$, could be used to prove as well a bound on even 
moments with odd moments.

A minimal generalization is given by 

$$
\begin{align*}
    |\int_{\mathbb{R}^{d}}x^np(x)dx|^2
    \leq & \int_{\mathbb{R}^{d}}x^{2m}p(x)dx \int_{\mathbb{R}^{d}} x^{2n-2m}p(x)dx \\
    E[x^n]\leq & \sqrt{E[x^{2m}]E[x^{2(n-m)}]}
\end{align*}
$$

and it makes me wonder if it has any value in bounding polinomials by 
substituting $p(x)=\sum_iw_i\delta(x-x_i)$ where $\sum_iw_i=1$.







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
