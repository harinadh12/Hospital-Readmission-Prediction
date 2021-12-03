# Hospital-Readmission-Prediction

This Classification problem is hosted as an in-class competition for course Intelligent Data Analytics during my masters in Data Science and Analytics at University of Oklahoma.

Team Name : **_Stronger With the Force!_**

A hospital readmission is an episode when a patient who had been discharged from a hospital is admitted again within a specified time interval. Readmission rates have increasingly been used as an outcome measure in health services research and as a quality benchmark for health systems. Hospital readmission rates were formally included in reimbursement decisions for the Centers for Medicare and Medicaid Services (CMS) as part of the Patient Protection and Affordable Care Act (ACA) of 2010, which penalizes health systems with higher than expected readmission rates through the Hospital Readmission Reduction Program.

The data in this problem is real-world data covering 10 years (1999–2008) of clinical care at 130 hospitals and integrated delivery networks throughout the United States. The specific patient cases used in this challenge also meet the following criteria:

 * It is generated from a hospital admission with length of stay between 1 and 14 days
 * Diabetes was entered to the system as one of the diagnoses
 * Laboratory tests were performed during the encounter
 * Medications were administered during the encounter

## The challenge: Readmissions

In this machine learning problem, you have the exacting task of trying to predict whether or not a patient will be readmitted to the hospital. The target here takes on binary values where 0 implies that the patient was not readmitted, and 1 implies that the patient was readmitted. You are predicting the latter.

Since the outcomes are relatively balanced, your model will be evaluated using accuracy.

In the data section, you can download hm7-train.csv to help you develop your model. Additionally, a hm7-test data file is available which does not include the  variable.

Approximately 25% of the hm7-test data is used to calculate a public log loss value -- you and all other competing teams can see this value. The remaining 75% of the test data is used to evaluate your private competition score -- no one other than the administrators can see this score until the competition is closed. The final quantitative ranking of your model performance will be based on this 70% holdout data set. 

## File descriptions
  * hm7-Train.csv - the training set
  * hm7-Test.csv - the test set

## Data fields
 * Patient ID: Unique identifier of a patient; Nominal
 * Race: Values: Caucasian, Asian, African American, Hispanic, and other; Nominal
 * Gender: Values: male, female, and unknown/invalid; Nominal
 * Age: Grouped in 10-year intervals: \[0, 10), \[10, 20), …, \[90, 100)\; Ordinal
 * Admission type: 8 distinct values, e.g., emergency, urgent, elective, and not available. See "codes" info; Nominal
 * Discharge disposition: 20+ distinct codes, e.g., discharged to home, expired, and not available. See "codes" info; Nominal
 * Admission source: 20+ distinct codes, e.g., physician referral, emergency room, and transfer. See "codes" documentation.; Nominal
 * Time in hospital: Integer number of days between admission and discharge; Numeric
 * Payer code: values are insurance, medicare, or self-pay; Nominal
 * Medical specialty: specialty of admitting physician, e.g., cardiology, internal medicine, family/general practice, and surgeon; Nominal
 * Number of lab procedures: Number of lab tests performed during the encounter; Numeric
 * Number of procedures: Number of procedures (other than lab tests) performed during the encounter; Numeric
 * Number of medications: Number of distinct generic names administered during the encounter; Numeric
 * Number of outpatient visits: Number of outpatient visits of the patient in the year preceding the encounter; Numeric
 * Number of emergency visits: Number of emergency visits of the patient in the year preceding the encounter; Numeric
 * Number of inpatient visits: Number of inpatient visits of the patient in the year preceding the encounter; Numeric
 * Diagnosis: The primary diagnosis (coded as first three digits of ICD9); 848 distinct values see http://icd9.chrisendres.com/index.php?action=contents for more information; Nominal
 * Number of diagnoses: Number of diagnoses entered to the system; Numeric
 * Glucose serum test result: Indicates the range of the result or if the test was not taken. Values: “>200,” “>300,” “normal,” and “none” if not measured; Nominal
 * A1c test result: Indicates the range of the result or if the test was not taken. Values: “>8” if the result was greater than 8%, “>7” if the result was greater than 7% but less than 8%, “normal” if the result was less than 7%, and “none” if not measured.; Nominal
 * Change of medications: Indicates if there was a change in diabetic medications (either dosage or generic name). Values: “change” and “no change”; Nominal
 * Diabetes medications: Indicates if there was any diabetic medication prescribed. Values: “yes” and “no”; Nominal
 * 24 features for medications: The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: “up” if the dosage was increased, “down” if the dosage was decreased, “steady” if the dosage did not change, and “no” if the drug was not prescribed; Nominal
 * Readmitted: Values: 1 if the patient was readmitted and 0 for no record of readmission.; Nominal

## Codes
### Admission Type
  * 1: Emergency
  * 2: Urgent
  * 3: Elective
  * 4: Newborn
  * 5: Not Available
  * 6: NULL
  * 7: Trauma Center
  * 8: Not Mapped
### Discharge Disposition

  * 1: Discharged to home
  * 2: Discharged/transferred to another short term hospital
  * 3: Discharged/transferred to SNF
  * 4: Discharged/transferred to ICF
  * 5: Discharged/transferred to another type of inpatient care institution
  * 6: Discharged/transferred to home with home health service
  * 7: Left AMA (against medical advice)
  * 8: Discharged/transferred to home under care of Home IV provider
  * 9: Admitted as an inpatient to this hospital
  * 10: Neonate discharged to another hospital for neonatal aftercare
  * 11: Expired
  * 12: Still patient or expected to return for outpatient services
  * 13: Hospice / home
  * 14: Hospice / medical facility
  * 15: Discharged/transferred within this institution to Medicare approved swing bed
  * 16: Discharged/transferred/referred another institution for outpatient services
  * 17: Discharged/transferred/referred to this institution for outpatient services
  * 18: NULL
  * 19: Expired at home. Medicaid only, hospice.
  * 20: Expired in a medical facility. Medicaid only, hospice.
  * 21: Expired, place unknown. Medicaid only, hospice.
  * 22: Discharged/transferred to another rehab fac including rehab units of a hospital .
  * 23: Discharged/transferred to a long term care hospital.
  * 24: Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.
  * 25: Not Mapped
  * 26: Unknown/Invalid
  * 30: Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere
  * 27: Discharged/transferred to a federal health care facility.
  * 28: Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital
  * 29: Discharged/transferred to a Critical Access Hospital (CAH)

### Admission Source
  * 1: Physician Referral
  * 2: Clinic Referral
  * 3: HMO Referral
  * 4: Transfer from a hospital
  * 5: Transfer from a Skilled Nursing Facility (SNF)
  * 6: Transfer from another health care facility
  * 7: Emergency Room
  * 8: Court/Law Enforcement
  * 9: Not Available
  * 10: Transfer from critial access hospital
  * 11: Normal Delivery
  * 12: Premature Delivery
  * 13: Sick Baby
  * 14: Extramural Birth
  * 15: Not Available
  * 17: NULL
  * 18: Transfer From Another Home Health Agency
  * 19: Readmission to Same Home Health Agency
  * 20: Not Mapped
  * 21: Unknown/Invalid
  * 22: Transfer from hospital inpt/same fac reslt in a sep claim
  * 23: Born inside this hospital
  * 24: Born outside this hospital
  * 25: Transfer from Ambulatory Surgery Center
  * 26: Transfer from Hospice

Code for this Problem can be found here [Hospital Readmissions](./hosp_readmission.R)

Kaggle Competition and LeaderBoard can be accessed here [Kaggle](https://www.kaggle.com/c/5103-ida-hm7-2021/leaderboard)
