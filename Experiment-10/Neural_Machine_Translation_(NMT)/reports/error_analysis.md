# Error analysis

For every model we took the **8 lowest-scoring test sentences per direction** (sentence chrF++, references with at least 5 words), which gives up to 64 errors per model. Each one is labelled by the reproducible rules in `src/eval/error_analysis.py`. A sentence can carry several labels; the table counts every label, and the *primary* label is the first one that matched, in the order below.

| Category | Rule |
|---|---|
| Reference noise | the test pair itself is misaligned: reference < 0.4x or > 2.5x the source length (the model is not at fault) |
| Script error | under 80% of output letters in the target script |
| Hallucination / repetition | output > 1.6x reference length, or a 3-gram repeated 3+ times |
| Omission | output < 0.6x reference length |
| Untranslated words | 2+ source words copied into the output that are not in the reference |
| Named entity / number | a source number or capitalised name missing from the output |
| Mistranslation | none of the above (adequate length, wrong words or meaning) |

## Counts (all labels)

| Category | Transformer baseline | + Back-translation | NLLB-200 fine-tuned |
|---|---:|---:|---:|
| Reference noise (misaligned pair) | 27 | 23 | 24 |
| Script error | 0 | 1 | 0 |
| Hallucination / repetition | 4 | 5 | 9 |
| Omission (too short) | 36 | 37 | 31 |
| Untranslated words | 0 | 0 | 0 |
| Named entity / number | 5 | 9 | 6 |
| Mistranslation (lexical / meaning) | 20 | 18 | 24 |
| *Sentences analysed* | 64 | 64 | 64 |

## Primary category by direction

### Transformer baseline

| Direction | Reference noise | Script error | Hallucination / repetition | Omission | Untranslated words | Named entity / number | Mistranslation |
|---|---:|---:|---:|---:|---:|---:|---:|
| en-as | 3 | 0 | 1 | 3 | 0 | 0 | 1 |
| as-en | 6 | 0 | 2 | 0 | 0 | 0 | 0 |
| en-mni | 4 | 0 | 0 | 2 | 0 | 0 | 2 |
| mni-en | 2 | 0 | 0 | 1 | 0 | 2 | 3 |
| en-lus | 4 | 0 | 0 | 3 | 0 | 0 | 1 |
| lus-en | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| en-bodo | 0 | 0 | 0 | 2 | 0 | 0 | 6 |
| bodo-en | 0 | 0 | 0 | 0 | 0 | 1 | 7 |

### + Back-translation

| Direction | Reference noise | Script error | Hallucination / repetition | Omission | Untranslated words | Named entity / number | Mistranslation |
|---|---:|---:|---:|---:|---:|---:|---:|
| en-as | 1 | 1 | 3 | 2 | 0 | 0 | 1 |
| as-en | 7 | 0 | 0 | 1 | 0 | 0 | 0 |
| en-mni | 4 | 0 | 0 | 3 | 0 | 0 | 1 |
| mni-en | 2 | 0 | 0 | 1 | 0 | 2 | 3 |
| en-lus | 3 | 0 | 1 | 3 | 0 | 0 | 1 |
| lus-en | 6 | 0 | 0 | 1 | 0 | 0 | 1 |
| en-bodo | 0 | 0 | 0 | 3 | 0 | 0 | 5 |
| bodo-en | 0 | 0 | 0 | 2 | 0 | 0 | 6 |

### NLLB-200 fine-tuned

| Direction | Reference noise | Script error | Hallucination / repetition | Omission | Untranslated words | Named entity / number | Mistranslation |
|---|---:|---:|---:|---:|---:|---:|---:|
| en-as | 2 | 0 | 2 | 2 | 0 | 0 | 2 |
| as-en | 6 | 0 | 1 | 1 | 0 | 0 | 0 |
| en-mni | 4 | 0 | 1 | 2 | 0 | 0 | 1 |
| mni-en | 2 | 0 | 0 | 2 | 0 | 0 | 4 |
| en-lus | 4 | 0 | 2 | 1 | 0 | 0 | 1 |
| lus-en | 6 | 0 | 1 | 0 | 0 | 0 | 1 |
| en-bodo | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| bodo-en | 0 | 0 | 0 | 1 | 0 | 0 | 7 |

## Examples

Two examples per category and model (src / ref / hyp, sentence chrF++).

### Transformer baseline

**Reference noise (misaligned pair)**

- `en-as` #419 (chrF++ 4.3)
  - SRC: Then , minor faults or mildly irritating habits may become major problems .
  - REF: সেইহেতুকে পাঁচনি পৌলে খ্ৰীষ্টানসকলক এই বিপদৰ সম্পৰ্কে সতৰ্ক কৰি এইদৰে লিখিছিল : “ সেই সৰ্পে যেনেকৈ তাৰ ধুৰ্ত্ততাৰে হৱাক ভুলাইছিল , তেনেকৈ , খ্ৰীষ্টলৈ তোমালোকৰ য…
  - HYP: তাৰ পিছত মিল্ বা মৃদু সমস্যাৰ সম্মুখীন হ’ব পাৰে ।
- `en-as` #637 (chrF++ 4.5)
  - SRC: ( Romans 3 : 24 ) Paul was there talking about those who were chosen to be joint heirs with Christ in the heavenly Kingdom .
  - REF: ১৪১ : ৫ ; হিতো.
  - HYP: ( ৰোমীয়া ৩ : ২৪ ) গতিকে পাঁচনি পৌলে কোৱা লোকসকলক স্বৰ্গীয় ৰাজ্যৰ সৈতে কথা পাতিছিল ।

**Hallucination / repetition**

- `en-as` #126 (chrF++ 2.2)
  - SRC: AIDS Acquired Immune Deficiency Syndrome is an infectious disease caused by the HIV Human Immunodeficiency Virus .
  - REF: এইডছ অৰ্জিত ৰোগ প্ৰতিৰোধ অভাৱ লক্ষণ হৈছে এইচ.আই.ভি. হিউমান ইমিউনোডেফিচিয়েন্সি ভাইৰাছৰ ফলত হোৱা এক সংক্ৰামক ৰোগ ।
  - HYP: আই এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এ এচমিমিমিমিমিমিমিমিমিমিমিমিমিমিমিম…
- `as-en` #1394 (chrF++ 2.0)
  - SRC: শপত গ্ৰহণৰ সময়ত তাইৰ সংস্কৰণত তাই ক্ষমতাচ্যুত কৰিছিল যে তাই অভিযুক্তৰ দ্বাৰা বিবাহৰ প্ৰতিশ্ৰুতিৰে প্ৰলোভিত কৰিছিল আৰু এইকাৰণেই তাই তেওঁৰ সৈতে গৈছিল ।
  - REF: in her version on oath she had deposed that she had enticed by the accused with promise of marriage and this was the reason why she had accompanied him .
  - HYP: In her mid - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

**Omission (too short)**

- `en-as` #538 (chrF++ 2.2)
  - SRC: any threadlike structure or filament
  - REF: যিকোনো মিহি সূতাৰ দৰে বস্তু , অৰ্থাৎ মিহি তন্তু
  - HYP: গাঁঠি বা গাঁঠনি
- `en-as` #648 (chrF++ 3.5)
  - SRC: They assist the body of elders .
  - REF: মনোৰঞ্জন বাছনি কৰাৰ আগতে আমি নিজকে কি প্ৰশ্ন সোধা উচিত ?
  - HYP: বৃদ্ধলোকৰ শৰীৰ ।

**Named entity / number**

- `mni-en` #187 (chrF++ 6.8)
  - SRC: ফিৎ ৮x৮গী স্ক্বার বোর্দ অমদা মঙ্গে মরু নত্রগা লিকোন নত্রগা দাইসনা লংদুনা শান্নৈ ।
  - REF: Who amongst us would not have enjoyed playing Gilli-Danda in our childhood ?
  - HYP: fek skets in a funger or skkot or diarrheaus .
- `mni-en` #706 (chrF++ 8.1)
  - SRC: ২০০৯দা মীখল মতুংদা দি.এম.কে.
  - REF: No aim or vision for India’s growth .
  - HYP: We are about 2002 .K after the election .

**Mistranslation (lexical / meaning)**

- `en-as` #973 (chrF++ 4.6)
  - SRC: Jorhat today with a different look
  - REF: যোৰহাট আজি এক অন্য দৃষ্টিৰে
  - HYP: বেলেগ চাই থাকোঁতে যোহা
- `en-mni` #241 (chrF++ 5.2)
  - SRC: Provision for commencement of operations before COD .
  - REF: “চেঞ্জ ইন লো” গী অনৌবা দেফিনিসন হাপচিল্লগনি
  - HYP: থৌরম অদুদা ওপরেসন পাংথোক্ত্রিঙৈ মমাংদা পাংথোকখি ।

### + Back-translation

**Reference noise (misaligned pair)**

- `en-as` #850 (chrF++ 2.5)
  - SRC: “ Charm may be false , and prettiness may be vain , ” says Proverbs 31 : 30 , “ but the woman that fears Jehovah is the one that procures praise for herself . ”
  - REF: বাইবেলখন আমাৰ বাবে এখন ব্যৱহাৰিক কিতাপ
  - HYP: ( হিতোপদেশ ৩১ : ৩০ , ৩০ ; হিতোপদেশ ৩১ : ৩০ , ৩০ ; হিতোপদেশ ৩১ : ৩০ , ৩০ ; হিতোপদেশ ৩১ : ৩০ , ৩০ , ৩০.
- `as-en` #1407 (chrF++ 1.5)
  - SRC: ঈশ্বৰক প্ৰেম কৰা লোকসকলক প্ৰেম কৰক
  - REF: • How was King Jehoash affected by his associations , both early in his life and later ?
  - HYP: Love for God

**Script error**

- `en-as` #1665 (chrF++ 0.5)
  - SRC: The state produces 12.07 lakh tonnes of coconut and copper annually from 103 lakh hectares of land occupied by the fruit .
  - REF: ৰাজ্যখনত ফলমুল অধিকৃত ১০৩ লাখ হেক্টৰ মাটিৰ পৰা বছৰি ১২০৭ লাখ টন নাৰিকল আৰু তামোলৰ বাদে ফলমুল উৎপাদিত হয় ।
  - HYP: ১০.৫.৫.৫.৫.৫.৫.৫.৫.৫.৫.৫.৫.৫.৫.৫.৫.৫.৫.

**Hallucination / repetition**

- `en-as` #880 (chrF++ 1.1)
  - SRC: Curfew an app assured home delivery Punjab leaves little to chance in Covid19 battle .
  - REF: কাৰফিউ এটা এপ নিশ্চিত হোম ডেলিভাৰী পাঞ্জাৱে কোভিড১৯ যুদ্ধত অলপ সুযোগ এৰি দিছে ।
  - HYP: কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে
- `en-as` #1380 (chrF++ 1.8)
  - SRC: A laboratory for the study of live insects , their life histories , effects on plants , reaction to insecticides , etc .
  - REF: কীট পতংগৰ বংশবৃদ্ধি আৰু অন্য গছ গছনি অথবা পৰিৱেশৰ ওপৰত ইয়াৰ বাবে সুবিধা থকা ঘৰ ।
  - HYP: উৎপীড়ন , উৎপীড়ন , উৎপীড়ন , উৎপীড়ন , উৎপীড়ন , উৎপীড়ন , উৎপীড়ন বা উৎপীড়ন , উৎপীড়ন , উৎপী

**Omission (too short)**

- `en-as` #538 (chrF++ 1.9)
  - SRC: any threadlike structure or filament
  - REF: যিকোনো মিহি সূতাৰ দৰে বস্তু , অৰ্থাৎ মিহি তন্তু
  - HYP: গাঁৱ বা সদৃশ
- `en-as` #1491 (chrF++ 3.0)
  - SRC: A herbal menu is served at the farm in support of the Ayurveda Therapy options offered in Satya Health Farm , to cleanse the system of toxins .
  - REF: আমাৰ টেক্সিন ব্যৱস্থা পৰিষ্কাৰ কৰি ৰাখিবলৈ সত্য হেল্থ ফাৰ্মত আয়ুৰ্বেদিক চিকিৎসাত সহায়ক হোৱাকৈ লতা বা শাকজাতীয় খাদ্য সম্ভাৰ পৰিৱেশন কৰা হয় ।
  - HYP: আয়ুৰ আয়ুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰুৰু

**Named entity / number**

- `mni-en` #187 (chrF++ 7.9)
  - SRC: ফিৎ ৮x৮গী স্ক্বার বোর্দ অমদা মঙ্গে মরু নত্রগা লিকোন নত্রগা দাইসনা লংদুনা শান্নৈ ।
  - REF: Who amongst us would not have enjoyed playing Gilli-Danda in our childhood ?
  - HYP: foots or divines are used by a sky board or a sky board or diarrhea .
- `mni-en` #706 (chrF++ 9.2)
  - SRC: ২০০৯দা মীখল মতুংদা দি.এম.কে.
  - REF: No aim or vision for India’s growth .
  - HYP: The election will be defendant afternooned by 2003 .

**Mistranslation (lexical / meaning)**

- `en-as` #973 (chrF++ 1.8)
  - SRC: Jorhat today with a different look
  - REF: যোৰহাট আজি এক অন্য দৃষ্টিৰে
  - HYP: বেলেগ চাই থকা
- `en-mni` #241 (chrF++ 5.8)
  - SRC: Provision for commencement of operations before COD .
  - REF: “চেঞ্জ ইন লো” গী অনৌবা দেফিনিসন হাপচিল্লগনি
  - HYP: সি ও ওদিওগী মমাংদা ওপরেসনশিং পাংথোকখ্রে ।

### NLLB-200 fine-tuned

**Reference noise (misaligned pair)**

- `en-as` #637 (chrF++ 4.0)
  - SRC: ( Romans 3 : 24 ) Paul was there talking about those who were chosen to be joint heirs with Christ in the heavenly Kingdom .
  - REF: ১৪১ : ৫ ; হিতো.
  - HYP: ( ৰোমীয়া ৩ : ২৪ ) ইয়াত পাঁচনি পৌলে যিসকলক স্বৰ্গীয় ৰাজ্যত খ্ৰীষ্টৰ সহ-অধিপত্য হবলৈ মনোনীত কৰা হৈছিল , তেওঁলোকৰ বিষয়ে কৈছিল ।
- `en-as` #1171 (chrF++ 5.7)
  - SRC: ( Joshua 24 : 15 , 16 ) Because of Joshua’s fine attitude , he was among the few of his generation privileged to enter the Promised Land .
  - REF: ২২ : ৭-৯ ; ১ কৰি.
  - HYP: ( যিহোচূৱা ২৪ : ১৫ , ১৬ ) যিহোচূৱাৰ উত্তম মনোবৃত্তিয়ে তেওঁক প্ৰতিজ্ঞাত দেশত প্ৰৱেশ কৰাৰ সুযোগ প্ৰদান কৰিছিল ।

**Hallucination / repetition**

- `en-as` #1553 (chrF++ 2.8)
  - SRC: The son of Tahath , the son of Assir , the son of Ebiasaph , the son of Korah ,
  - REF: চফনিয়া আছিল তহতৰ পুত্র । তহৎ আছিল অচীৰৰ পুত্র । অচীৰ আছিল অবিয়াচফৰ পুত্র । অবিয়াচফ আছিল কোৰহৰ পুত্র ।
  - HYP: তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ , তহৎ ,
- `en-as` #439 (chrF++ 4.4)
  - SRC: • The projects were visited by a team under Chairman CWC during 2016 and by another team under Sh .
  - REF: উল্লেখ্য যে ৰাজস্থান ফিডাৰ কেনেল আৰু ছিৰহিন্দ ফিডাৰ কেনেলৰ বাবে অনুমোদিত ধনৰ পৰিমাণ ক্ৰমে ৬৭১.৪৭৮ কোটি টকা আৰু ১৩০৫.২৬৭ কোটি টকা হৈছিল ।
  - HYP: ২০১৬ চনত চিডব্লিউচিৰ অধ্যক্ষৰ নেতৃত্বত এটা দল আৰু শ্বহীদ শ্বহীদ শ্বহীদ শ্বহীদ শ্বহীদ শ্বহীদ শ্বহীদ শ্বহীদ শ্বহীদ শ্বহীদ শ্বহীদ শ্বহীদ শ্বহীদ শ্বহীদ শ্বহীদ শ্বহী…

**Omission (too short)**

- `en-as` #795 (chrF++ 3.0)
  - SRC: Having a clear perceptive mind
  - REF: এক মুকলি ধাৰণা বা বুদ্ধিশীল মনৰ অধিকাৰী
  - HYP: সুস্পষ্ট চেতনা থকা
- `en-as` #1570 (chrF++ 5.8)
  - SRC: Let this mind be in you , which was also in Christ Jesus
  - REF: বৰং নিজকে শূণ্য কৰিলে ।আৰু তেওঁ দাসৰ ৰূপ ধাৰণ কৰিলে ।তেওঁ মানুহৰ নিচিনা হৈ অৱতীৰ্ণ হ’ল ।আৰু তেওঁ আকাৰ প্ৰকাৰত মানুহৰ দৰে হ ল ।
  - HYP: খ্ৰীষ্ট যীচুত যি মনোবৃত্তি আছিল , সেয়ে আপোনালোকৰো হওক ।

**Mistranslation (lexical / meaning)**

- `en-as` #1747 (chrF++ 4.6)
  - SRC: See the box “ Should I Join in the Celebration ? ”
  - REF: উদাহৰণস্বৰূপে , সাধাৰণতে ইয়োবৰ এক উত্তম মনোভাব আছিল ।
  - HYP: মই এই উৎসৱত ভাগ লব লাগেনে ?
- `en-as` #1784 (chrF++ 5.0)
  - SRC: Characterized by rich or profuse growth
  - REF: বেছিকৈ ডাল পাত ওলাই বাঢ়া
  - HYP: ধনী বা প্ৰচুৰ পৰিমাণে বৃদ্ধি হোৱা

## Observations

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
