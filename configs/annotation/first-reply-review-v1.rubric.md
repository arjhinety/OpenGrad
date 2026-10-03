# First-reply review rubric (v1)

The rubric of Study 002's triage of the flag set (`study_002_prereg_v14`, 46 §5; `study_002_prereg_v15`, 47).
The annotation screen and every labelling batch show sections 1 to 3 only; this preamble is not shown, and the
labellers are not told how the items were chosen. The label definitions in section 2 are 46 §5's, word for word;
sections 1 and 3 say how to apply them. 46 §5 allows the procedure to be revised once, after the 100-record
trial, and never the floors.

## 1. What you see

Each item is the opening of one conversation between a user and an assistant:
- the **tools** the assistant was offered (possibly none);
- the **user's first message**;
- the assistant's **first reply** to it.

Judge the first reply only, as a reply to that first message. Nothing after it is shown, and nothing about where
the conversation came from.

## 2. The labels

- `NOT_A_DECLINE`: the reply does not decline.
- `DECLINE_JUSTIFIED`: it declines, and the request needs something the assistant lacks (current or private data,
  an action in the world, a tool not offered).
- `DECLINE_UNJUSTIFIED`: it declines, but a capable assistant could answer correctly from general knowledge or
  reasoning, without the tools.
- `UNKNOWN`: the item cannot be decided.

## 3. How to apply them

- **A decline** is a reply that does not do what the user asked and says, or plainly implies, that the assistant
  cannot or will not. Asking the user a question, or doing only part of what was asked, is not by itself a
  decline.
- **Justified or not** depends on the request, not on the wording of the reply. Ask: could a capable assistant
  with no tools at all answer this request correctly from general knowledge or reasoning? If yes, the decline is
  `DECLINE_UNJUSTIFIED`. If the request genuinely needs current or private data, an action in the world, or a
  tool the assistant was not offered, it is `DECLINE_JUSTIFIED`.
- **The offered tools** are shown so you can see what the assistant had. A decline does not become justified
  because the reply says the assistant's functions are limited; judge the request itself.
- **When none of the labels fits,** or two remain genuinely plausible, label the item `UNKNOWN` with the matching
  ambiguity reason. One case the definitions do not settle: the reply declines although an offered tool could
  have done what was asked. Label it `UNKNOWN` with `AMBIGUOUS_TWO_MODES`.
- **Each item on its own content.** There is no expected distribution and no quota.
