# UIDAI-aadhaar-digital-identity-stress-analysis
This project transforms Aadhaar enrolment, demographic update, and biometric update datasets into actionable governance intelligence using exploratory data analysis, forecasting, and a Composite Digital Identity Stress Index.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##  Full Dataset & Outputs
Due to GitHub file size limits, the complete project archive (including large CSVs and output files) is available here:

🔗 Google Drive (Full Project):  
https://drive.google.com/drive/folders/1qTklIlyH_E5GWuRT3jdpzxPYHyTFeYuo?usp=drive_link

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##  Problem Statement
Unlock societal trends in Aadhaar enrolment and updates by identifying meaningful patterns, anomalies, and predictive indicators to support informed decision-making and system improvements.

---

##  Datasets Used
1. **Aadhaar Enrolment Dataset**
   - Date, State, District, Pincode
   - Age groups: 0–5, 5–17, 18+

2. **Aadhaar Demographic Update Dataset**
   - Updates to name, address, DOB, gender, mobile
   - Aggregated by time and geography

3. **Aadhaar Biometric Update Dataset**
   - Fingerprint, iris, face updates
   - Focus on lifecycle-driven updates

(Data provided by UIDAI for hackathon use)

---

##  Methodology
- Data cleaning & preprocessing
- Univariate, bivariate & trivariate EDA
- Outlier & anomaly detection
- Seasonality & trend analysis
- Short-term forecasting (linear trend models)
- Construction of a **Composite Digital Identity Stress Index**

---

##  Key Outputs
- Monthly trends & seasonality
- Outlier spike detection
- State-wise pressure indices
- Forecasted service demand
- Composite Stress Index (0–1 scale)

---

##  Composite Stress Index
Combines:
- Enrolment pressure
- Biometric update pressure
- Demographic update pressure  

Normalized using percentile ranks and aggregated with equal weights.

---

##  How to Run
```bash
python Enrollment/Enrollment\ codes/EDA.py
python Biometrics/Biometrics\ Codes/EDA.py
python Demographics/Demographics\ codes/EDA.py
python composite_stress_index.py



