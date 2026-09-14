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
stable: none, one dot, two dots, one accent, two accents.

- none: a e i o u
- one dot: ȧ ė ị ȯ ụ
- two dots: ä ë ï ö ü
- one accent: á é í ó ú
- two accents: a̋ e̋ i̋ ő ű

Five letters times five hats is twenty-five vocalic symbols. That used to look
greedy. Schoolbook English has "about thirteen vowels", and then three hats
already give you fifteen, so one dot and two accents look like Hungarian extras
we could drop. *ő* and *ű* exist because Hungarian *ö* and *ü* can be long.
Lowercase *i* already wears a tittle, so the overdot is ugly. Fifteen also
happens to be the boring Latin set that keyboards already have.

Then you count properly. [English phonology](https://en.wikipedia.org/wiki/English_phonology)
does not give you thirteen. In the system on that page there are 20–25 vowel
phonemes in Received Pronunciation, 14–16 in General American and 19–21 in
Australian English. The spread is mostly diphthongs, and whether *near*,
*square* and *cure* are vowels of their own or just *i, e, u* plus an *r*.

Three hats is a General American trick. Fifteen cells sit right on top of
14–16. Australian already spills over. RP walks through the ceiling. Four hats
would kiss the RP floor of twenty and miss the twenty-five. So yes: if you want
one letter per phoneme, and you want a page that still works in London, you need
all five hats. Twenty-five is not a luxury grid. It is the RP high count.

You can still cheat. Write *night, now, boy, here* as *nait, nau, noi, iė* and
you are back in a biphonemic analysis, two sounds, two letters. Wikipedia lists
that option too. It is honest. It is also not "one vocalic phoneme, one glyph",
which was the whole point of starting from phonetic transcription. For that
job, FACE, PRICE, CHOICE, GOAT, MOUTH, NEAR, SQUARE, CURE are eight extra
symbols, not eight spelling accidents.

A working RP-sized subset. Spare cells stay spare; Australian can have them for
*bad* versus *lad*.

| set | as in | write |
| --- | --- | --- |
| STRUT | but, blood | a |
| PALM | father, yacht | á |
| TRAP | cat | ä |
| PRICE | night, I | ȧ |
| MOUTH | now, plough | a̋ |
| DRESS | bed, said | e |
| FACE | day, eight | é |
| COMMA | the, about | ė |
| NURSE | bird, colonel | ë |
| SQUARE | hair, bear | e̋ |
| KIT | bit, women | i |
| FLEECE | see, eat | í |
| NEAR | here, beer | ï |
| LOT | cot, knowledge | o |
| GOAT | go, though | ó |
| THOUGHT | thought, four | ö |
| CHOICE | boy, noise | ő |
| FOOT | book, woman | u |
| GOOSE | too, two, queue | ú |
| CURE | sure, tour | ü |

Consonants are the easy part anyway: drop the silent letters, write *k* when it
is *k*, keep *th, sh, ch, ng*. No need to invent runes.

Once you do this, the famous jokes become ordinary words, and the ordinary words
become readable.

*Though the tough cough and hiccough plough him through*

becomes

*thó thė taf köf ėnd hikap pla̋ him thrú*

Same family of letters on the page, seven different vowels, no crossword.

A few I like:

- *colonel* → *kënėl* (it was *kernel* all along)
- *queue* → *kjú* (four letters were unemployed)
- *women* / *woman* → *wimin* / *wumėn*
- *one, two, eight, four* → *wan, tú, ét, för*
- *knight* → *nȧt* (and *night* is the same word, which it is)
- *choir* → *kwȧė*
- *yacht* → *yát*
- *Leicester* → *lestė*
- *beautiful* → *bjútifėl*
- *knowledge* → *nolij*
- *I owe you* → *ȧ ó yú*
- *here / hair / sure* → *hï / he̋ / shü*
- *cot / caught / cart* → *kot / köt / kát*

And the sentence they use to teach that English "long A" is one sound:

*The rain in Spain stays mainly in the plain*
→ *thė rén in spén stéz ménli in thė plén*

You can see it. That is the whole point.

Shaw's *ghoti* for *fish* dies instantly. Good. If your writing system lets
*gh-o-ti* spell *fish*, you do not have a writing system, you have folklore.

Would it be annoying to type, would dialects fight over *ö* versus *á*, would
etymologists complain. Yes. They already complain, and children already pay the
cost. Twenty-five slots, twenty to twenty-five RP sounds. General American can
leave cells empty. It is not a lack of symbols. It is a lack of willingness to
start from the sounds.

What do you think? Would you read English like this? *ị ȯ ụ i̋ ű* are still
free, which is exactly the slack between 20 and 25.
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
