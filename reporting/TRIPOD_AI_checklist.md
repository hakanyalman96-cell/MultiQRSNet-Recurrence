# TRIPOD+AI reporting checklist

*Paper:* Low unipolar voltage at the earliest activation site independently predicts recurrence after catheter ablation of premature ventricular complexes.

TRIPOD+AI (Collins GS, Moons KGM, et al. *BMJ* 2024;385:e078378). This study reports an exploratory prognostic-factor / incremental-value analysis rather than a deployable prediction model; items are addressed accordingly, and the exploratory, single-centre, validation-pending nature is stated throughout.

| # | TRIPOD+AI item | Where addressed |
|---|---|---|
| 1 | Title identifies study as developing/evaluating a prognostic marker | Title |
| 2 | Structured abstract | Abstract |
| 3a | Background and rationale | Introduction |
| 3b | Objective / research question | Introduction; Abstract |
| 4a | Source of data (retrospective single-centre cohort) | Methods 2.1 |
| 4b | Study dates / setting | Methods 2.1 |
| 5a | Eligibility criteria | Methods 2.1 |
| 5b | Treatments received (CARTO-guided RF ablation) | Methods 2.1, 2.3 |
| 6a | Outcome definition (<80% burden reduction @3-month Holter) | Methods 2.2 |
| 6b | Outcome/leakage handling (burden % excluded; train-only fitting) | Methods 2.2, 2.5 |
| 7a | Predictor definition/measurement (unipolar voltage at earliest site; LAT = Map − Reference) | Methods 2.3 |
| 7b | Predictors procedural, measured before outcome | Methods 2.3 |
| 8 | Sample size / events (159 → 157 analysed; 32 events) | Methods 2.1, 2.3; Results 3.1 |
| 9 | Missing data (invalid voltages discarded; median fill in CV) | Methods 2.3, 2.5 |
| 10a | Predictor handling (continuous; per-mV and Firth-penalised) | Methods 2.5 |
| 10b | Model type/analysis (logistic; multivariable; 5-fold patient-level CV) | Methods 2.5 |
| 10c | Multiplicity / selective-reporting safeguards (Bonferroni; full screen) | Methods 2.6; Online Resource 2 |
| 10d | Performance measures (AUROC, ΔAUROC, bootstrap CI, cut-off metrics) | Methods 2.5; Results 3.4 |
| 11 | Risk groups / cut-off (Youden) | Methods 2.5; Results 3.4 |
| 12 | Development vs validation (development only; no external validation) | Methods 2.6; Limitations |
| 13a | Participant flow / numbers analysed | Results 3.1 |
| 13b | Characteristics by outcome | Table 1 |
| 14a | Univariable and multivariable associations | Results 3.2–3.3; Table 2 |
| 14b | Performance with uncertainty | Results 3.4; Figure 1 |
| 15 | Model specification (covariates, estimates) | Table 2 |
| 16 | Performance/calibration | Results 3.4 |
| 17 | Limitations | Discussion 4.1 |
| 18 | Interpretation, generalisability, mechanism | Discussion 4 |
| 19 | Data / code availability | Declarations; this repository |
| 20 | Funding and conflicts | Declarations |
| — | AI transparency / human oversight | Descriptive statistical model with full code disclosure; Methods 2.6, Limitations |
