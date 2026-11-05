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
- readmission: Logistic regression at threshold 0.10, with a physician reviewing every flagged patient. Missed readmissions are the worse mistake, so I want to catch more of them. At 0.10 it catches 44.3% of readmitted patients (159 of 359) and flags 26 of every 100 discharges. The cost is 893 false alarms and only 15.1% precision, so physician review is what makes those flags manageable. I picked logistic regression because its training and test AUROC are close (0.657 and 0.640), while random forest's are 0.997 and 0.652, which suggests it memorised its training patients. Limit: it still misses 200 of 359 readmissions (55.7%), and this is a historical 1999-2008 diabetes inpatient sample, so it needs checking on current patients first.
- diabetes: Logistic regression at threshold 0.20. It had the best test F1 (40.7% at 0.50) and the highest test AUROC (0.827), and its training and test results are close, so it isn't memorising. At 0.20 it catches 77.5% of adults with diabetes (45 missed of 200) and flags 37 of every 100 adults. I chose 0.20 over 0.11 to cut false alarms (27.6 vs 45.0 per 100 adults without diabetes), because extra testing can be expensive for people without health insurance. Limit: this is one 2017-2018 sample, so it should be checked on new patients before use.

## Design choice and AI use
A few sentences each. The design choice must be your own idea, not an AI tool's.
- design choice: I added a "Which mistake matters most for this job?" panel at the top of every section. It shows my reasoning from before I saw any results. A clinical lead could trust accuracy alone, which looks high (91% for readmission) even when the model misses most readmissions, or not know which error is worse for the job. The panel keeps the real question in front of them while they read the numbers: which mistake the clinic can't afford.
- AI suggestion I turned down: An AI suggested a graph of how sensitivity and false-positive rate change as the threshold moves. I didn't use it because a graph shows the trade-off but not which mistake matters most for the job, which was the key point for the clinical lead.
