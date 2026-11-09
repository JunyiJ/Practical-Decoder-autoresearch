# Selected model: qualitative generation demo

The adopted 16,351,859-parameter byte-input model captures theatrical formatting and vocabulary, but these four samples do not sustain coherent meaning. This is a qualitative spot check, not a benchmark or a new model comparison.

Selected representative checkpoint: `research_runs/byte-input-quality-v1-001/10/h200.pt`. GPT-2 tokens; temperature 0.8; top-k 40; CPU sampling generator seed 42 shared across prompts; 64 generated tokens each; MPS inference. All four first completions are included unchanged. All reached the token limit, so final fragments are truncation rather than evidence of a model failure. No training or weight changes.

Generation/load elapsed: 8.24 seconds. Guard: finished.

## Sample 1

Prompt:

```text
KING RICHARD:
My friends, the kingdom is in danger, and we must
```

Continuation:

```text

Where our great-wixters are we at us.
If they are not to me, for I have done
Myself to the crown and your grace.

QUEEN MARGARET:
I will not say.

KING RICHARD II:
Infer that we
```

The continuation fails to complete “we must,” invents “great-wixters,” and loses the initial danger scenario despite royal vocabulary.

## Sample 2

Prompt:

```text
JULIET:
If love be true, why must we part tonight?
ROMEO:

```

Continuation:

```text
That's I am too much to take my leave:
I'll make your majesty to thyself?

HENRY BOLINGBROKE:
Thou shalt to me, if you may be so.

DUCHESS OF YORK:
That I am sure for you.

```

It mentions taking leave, but the grammar is broken and the scene drifts to unrelated characters.

## Sample 3

Prompt:

```text
First Citizen:
We have no bread, and our children are hungry.
Second Citizen:

```

Continuation:

```text
What says your highness will make a tear?

Second Murderer:
'Tis a child too.

First Gentleman:
No more, I am a child.

ANGELO:
Well, go, I am too quick.

CAMILLO:
I must
```

It echoes “child,” but never addresses hunger or bread and rapidly changes speakers.

## Sample 4

Prompt:

```text
Question: Why does the sun rise in the morning?
Answer:
```

Continuation:

```text
 I pray thee,--my tongue is not my part;
The army of the business, and not a name,
As you are too long for a great thing and
Myself out the people's point and my own
That is a man, on a very man's eyes,
Which, by your
```

It does not answer the question. It continues theatrical prose, consistent with a Shakespeare completion model rather than an instruction-trained assistant.
