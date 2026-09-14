---
layout: post
title: Fixing English writing
published: true
comments: true
---

English spelling is a historical accident that we keep teaching as if it were a
language. It is not. It is a pile of etymologies, fashion, and frozen
pronunciations. Kids spend years learning that *one* does not start with *w*,
that *colonel* is *kernel*, and that *ough* is a random number generator.
The honest starting point is not "how it used to be written". It is phonetic
transcription: write the sounds.

IPA already does that, but it asks you to leave the Latin alphabet. Most people
will not. We do not need to. Latin letters plus a few accents are enough.

Hungarian already solved this. Five vowel letters, and marks that mean something
stable. The full Hungarian kit also has two accents, the double acute on *ő* and
*ű*, because *ö* and *ü* can be long. English does not need that axis. It also
does not need a fifth mark, one dot: lowercase *i* already wears a tittle, and
the extra cells are empty anyway.

Keep three types of hat: none, one accent, two dots.

- none: a e i o u
- one accent: á é í ó ú
- two dots: ä ë ï ö ü

Five letters times three hats is fifteen vocalic symbols. English has about
thirteen vowel sounds, give or take the dialect and whether you count diphthongs
as one sound or two. Fifteen is already more than enough. Two cells left over.
Leave them empty, or keep them for a dialect that insists on *lot* versus
*thought*.

Those fifteen are also the boring Latin letters that fonts and keyboards already
have. *ő ű ė ȧ* are how you show off. *ä ö ü á é* is how you write.

A working English subset:

| sound | as in | write |
| --- | --- | --- |
| /ʌ/ | but, blood | a |
| /ɑ/ | father, yacht | á |
| /æ/ | cat | ä |
| /ə/, /ɜ/ | the, about, bird | ë |
| /ɛ/ | bed, said, friend | e |
| /eɪ/ | day, eight | é |
| /ɪ/ | bit, women | i |
| /i/ | see, eat | í |
| /ɔ/ | thought, four | ö |
| /oʊ/ | go, though | ó |
| /ʊ/ | book, woman | u |
| /u/ | too, two, queue | ú |

The leftover diphthongs can stay as two letters, which is also honest: *ai* in
*night*, *au* in *now*, *oi* in *boy*. Hungarian is happy with digraphs for
consonants. We can be happy with them for the few gliding vowels. Consonants
are the easy part anyway: drop the silent letters, write *k* when it is *k*,
keep *th, sh, ch, ng*. No need to invent runes.

Once you do this, the famous jokes become ordinary words, and the ordinary words
become readable.

*Though the tough cough and hiccough plough him through*

becomes

*thó thë taf köf ënd hikap plau him thrú*

Same family of letters on the page, six different vowels, no crossword.

A few I like:

- *colonel* → *kërnel* (it was *kernel* all along)
- *queue* → *kjú* (four letters were unemployed)
- *women* / *woman* → *wimin* / *wumën*
- *one, two, eight, four* → *wan, tú, ét, för*
- *knight* → *nait* (and *night* is the same word, which it is)
- *choir* → *kwaiër*
- *yacht* → *yát*
- *Leicester* → *lestër*
- *beautiful* → *bjútifël*
- *knowledge* → *nálij*
- *I owe you* → *ai ó yú*

And the sentence they use to teach that English "long A" is one sound:

*The rain in Spain stays mainly in the plain*
→ *thë rén in spén stéz ménli in thë plén*

You can see it. That is the whole point.

Shaw's *ghoti* for *fish* dies instantly. Good. If your writing system lets
*gh-o-ti* spell *fish*, you do not have a writing system, you have folklore.

Would it be annoying to type, would dialects fight over *ö* versus *á*, would
etymologists complain. Yes. They already complain, and children already pay the
cost. Fifteen slots, thirteen sounds, Latin letters, no fancy dots. It is not a
lack of symbols. It is a lack of willingness to start from the sounds.

What do you think? Would you read English like this? *ï* and *ü* are still free.
Please comment below.









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
