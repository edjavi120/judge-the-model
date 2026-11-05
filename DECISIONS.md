# Decisions

<!-- Fill in each section at the step the activity website gives, then commit and push.
     Write each answer after its label. Keep the headings and labels exactly as they are:
     `python -m analysis` reads them. Limit: 150 words for Before results.
     Fill in Recommendations and Design choice and AI use last, once your app is built. -->

## Before results
For each dataset: positive class; which error matters more for the job (missed positive or false alarm) and why.
- readmission: Positive class is a patient readmitted within 30 days. A missed positive matters more than a false alarm. Both errors are bad, but a missed patient who needs more treatment and doesn't receive it can get worse, and the health problem can grow bigger while going untreated.
- diabetes: Positive class is a doctor diagnosis of diabetes or HbA1c of 6.5% or higher. A missed positive matters more than a false alarm. If a patient's diabetes goes untreated it can worsen, and they could die several years later from complications; a patient who is treated has a much lower chance of dying.

## Recommendations
- readmission: <model and threshold, or "none is adequate for this job"> because ...
- diabetes: ...

## Design choice and AI use
A few sentences each. The design choice must be your own idea, not an AI tool's.
- design choice: <one thing you chose to show, or leave out, for your clinical lead> because ...
- AI suggestion I turned down: <what an AI tool suggested, and why you didn't use it, or "no AI tools used">
