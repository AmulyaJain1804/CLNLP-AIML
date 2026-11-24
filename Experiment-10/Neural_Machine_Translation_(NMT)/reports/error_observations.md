1. **A large share of the "worst" sentences are not model errors.** 23 to 27 of every 64 lowest-scoring
   sentences are misaligned test pairs. A typical example is a long English sentence whose reference is
   only a Bible verse number ("১৪১ : ৫ ; হিতো."). These come from the organisers' training release, which we
   had to split ourselves (see `docs/task_notes.md`). NLLB often produces a *correct* translation of the
   source in these cases, and still gets a chrF++ of about 5. So absolute scores underestimate quality,
   by roughly the same amount for every model, and the model comparisons stay valid.
2. **Omission is the dominant genuine error of the small Transformers** (36/64 baseline, 37/64 BT,
   against 31/64 for NLLB). On long or rare-word inputs, the models trained from scratch stop early
   and output a short, generic sentence.
3. **Hallucination is mostly repetition loops** ("তহৎ , তহৎ , তহৎ …", "শ্বহীদ শ্বহীদ …"). They are
   triggered by lists of names (genealogies) and by abbreviations such as "Sh .". NLLB has slightly more
   of them (9) than the baselines (4 and 5). A repetition penalty or `no_repeat_ngram_size` at decoding
   time is the obvious fix.
4. **Named entities and numbers.** Back-translation *increased* named-entity and number errors (5 to 9).
   The synthetic corpus itself shows why: the generator duplicates and drops names ("Dan, Naphtali,
   Naphtali, Gad, Asher and Asher" in `outputs/synthetic/en-as/quality.json`), and the BT model learns
   from those noisy pairs.
5. **Script errors and untranslated words are essentially absent.** The joint SentencePiece models and
   NLLB's language tags keep every model in the correct target script, Bodo with the `hin_Deva` proxy
   tag included. Copied English words were not detected for any model.
