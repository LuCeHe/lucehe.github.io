---
layout: post
title: Bot with Big Personality
published: true
comments: true
---

I'm working on a bot that has to keep a consistent personality while talking
about anything. What I have now is a bot that can be given a description of an
environment, a description of which personality it should take, and the sentence
that a human is asking to it. For now it is able to provide varied replies.
The language it uses is not yet the sharpest possible but I think it's fun to interact with.

I made it produce as well an action to take and an emotion to show. It's interesting that
when the description of the persona is friendly, it tries to hug everything, while
if the description is not that friendly, it tries to hit everything! Better versions will come!

<script src="https://anvil.works/embed.js" async></script>
<iframe style="width:100%;" data-anvil-embed src="https://XXMJTCRVTPECSYWD.anvil.app/AIVG3CMJWDMTXF7QNOTTEPG3"></iframe>

Things I will eventually do are: use larger model, larger data to train, distill larger models pretrained on gigantic 
datasets, put it in the front end, faster inference, create a task that encourages the model to retrieve info from its 
persona, the past of the convo and the description of the environment.


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
