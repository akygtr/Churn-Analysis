# What is Customer Churn?

Customer churn is defined as when customers or subscribers discontinue doing business with a firm or service.

Customers in the telecom industry can choose from a variety of service providers and actively switch from one to the next. The telecommunications business has an annual churn rate of 15-25 percent in this highly competitive market.

Individualized customer retention is tough because most firms have a large number of customers and can't afford to devote much time to each of them. The costs would be too great, outweighing the additional revenue. However, if a corporation could forecast which customers are likely to leave ahead of time, it could focus customer retention efforts only on these "high risk" clients. The ultimate goal is to expand its coverage area and retrieve more customers loyalty. The core to succeed in this market lies in the customer itself.

Customer churn is a critical metric because it is much less expensive to retain existing customers than it is to acquire new customers.

To detect early signs of potential churn, one must first develop a holistic view of the customers and their interactions across numerous channels.As a result, by addressing churn, these businesses may not only preserve their market position, but also grow and thrive. More customers they have in their network, the lower the cost of initiation and the larger the profit. As a result, the company's key focus for success is reducing client attrition and implementing effective retention strategy.

# Objectives
* Finding the % of Churn Customers and customers that keep in with the active services.
  
* Analysing the data in terms of various features responsible for customer Churn.
  
* Finding a most suited machine learning model for correct classification of Churn and non churn customers.

# Dataset:
[Telco Customer Churn](https://www.kaggle.com/code/bhartiprasad17/customer-churn-p)

The data set includes information about:
* Customers who left within the last month – the column is called Churn
* Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
* Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
* Demographic info about customers – gender, age range, and if they have partners and dependents

# Implementation:
**Libraries**: sklearn, Matplotlib, pandas, seaborn, and NumPy

**Few Glimpse of EDA**

**1. Churn Distribution:**
<img width="1298" height="170" alt="Screenshot 2025-09-10 131248" src="https://github.com/user-attachments/assets/679f63b8-53ac-4f18-b5ed-9cea7ba43939" />
26.99 % of customers switched to another firm.

***

**2. Churn distribution with respect to Gender:** Shows that male customers make up 64.15% of churn, indicating a gender-skewed attrition trend.

<img width="600" height="520" alt="Screenshot 2025-09-10 131911" src="https://github.com/user-attachments/assets/c1a091bc-43e0-4dab-badb-4c96720b6215" />

***

**3. Churn distribution w.r.t Customer Contract Distribution:** Month-to-Month contracts show the highest churn rate, signaling an urgent need to improve retention strategies for short-term customers.

<img width="600" height="520" alt="Contract Distribution" src="https://github.com/user-attachments/assets/39bfc6e8-9818-4e33-86d5-8e2e100a3615" />

***

**4. Churn distribution w.r.t Payment Methods:** Customers paying by mailed check show the highest churn rate, highlighting a need to promote more stable payment methods like credit cards

<img width="600" height="520" alt="Payment Menthod" src="https://github.com/user-attachments/assets/0584b405-0056-4b3b-88ce-f2b512d75802" />

***
 
**5. Churn distribution w.r.t Tenure Group:** Customers with less than 6 months of tenure show the highest churn volume, highlighting the need for stronger early engagement strategies.

<img width="600" height="520" alt="Tenure Group" src="https://github.com/user-attachments/assets/e047b544-0d41-404e-ab87-95a66386b119" />

***
 
**6. Churn distribution w.r.t Age Distribution:** Here churn rate shows a strong upward trend with increasing age. While the younger age groups have very low churn, the rate accelerates significantly for older customers

<img width="600" height="520" alt="Age Group" src="https://github.com/user-attachments/assets/ad09f122-0ca3-438c-9703-8ee68a10fb6c" />

***

**7. Churn distribution w.r.t Top 5 States:** This analysis suggests that geographic location is a significant factor in customer churn, with some states experiencing much higher rates than others.

<img width="600" height="520" alt="State" src="https://github.com/user-attachments/assets/87fb0c63-cfbb-4732-bce5-72033ac33a10" />

***

**8. Churn distribution w.r.t External Factors:**
The primary driver of customer churn is competition, followed by customer attitude and dissatisfaction, with price being the least common factor.

<img width="600" height="520" alt="Churn Factors" src="https://github.com/user-attachments/assets/afa34180-6ae3-4a49-92a6-8ce00ab42961" />

***

**9. Churn distribution w.r.t Internet Type:** Based on the chart, churn rate is highest among Fiber Optic customers at over 40%, suggesting a paradoxical trend where customers with the most advanced service are also the most likely to leave, while those with no internet have the lowest churn.

<img width="600" height="520" alt="Internet Type" src="https://github.com/user-attachments/assets/adb7dac6-3d04-4ef7-8395-db8715f7383f" />

***

**10. Churn distribution w.r.t Services:** Most customers use core services like Internet and Phone, but optional add-ons like Premium Support and Online Security are underutilized and highlighting upsell opportunities and churn risk among low-engagement users.

<img  width="600" height="520" alt="Services" src="https://github.com/user-attachments/assets/1549c98b-87c6-4e07-ad76-c9c4cc2a63d6" />

***

# CHURN ANALYSIS - PREDICTION

<img width="300" height="400" alt="Prediction profile" src="https://github.com/user-attachments/assets/19554a37-72cd-464b-9df1-979c469e0bea" />



Based on the **"Predicted Churner Profile" dashboard**, here’s a comprehensive analysis of churn prediction insights:

---

 **Overall Churn Distribution**
- **Female churners**: 246
- **Male churners**: 132  
 **Females are nearly twice as likely to churn** compared to males in this dataset.

---

**Female Churner Insights**

#### **Age Group**
- Highest churn: **>50 years** and **35–50 years**
- Lowest churn: **<20 years**  
 Older customers are more likely to churn.

#### **Tenure Group**
- Most churn: **≥24 months** and **12–18 months**
- Least churn: **<6 months**  
 Long-term customers are still churning, possibly due to dissatisfaction or lack of engagement.

#### **Marital Status**
- Slightly more churn among **unmarried** customers  
 Marital status may influence service needs or loyalty.

#### **Payment Method**
- Highest churn: **Credit Card**
- Moderate churn: **Bank Withdrawal**
- Lowest churn: **Mailed Check**  
 Digital payment users may churn more, possibly due to ease of switching or billing issues.

---

###  **Male Churner Insights**

#### **State-wise Distribution**
- Churn is spread across **Uttar Pradesh, Maharashtra, Tamil Nadu, Karnataka**, and others.
- No single state dominates, but **Uttar Pradesh and Maharashtra** show relatively higher churn.

#### **Contract Type**
- Overwhelming churn from **Monthly contracts**
- Very low churn from **One-year and Two-year contracts**  
 **Contract length is a strong retention factor** longer contracts reduce churn.

---

###  **Actionable Insights**

1. **Target long-tenure and older customers** with loyalty programs or personalized offers.
2. **Encourage longer contracts** to reduce churn, especially among monthly users.
3. **Investigate credit card billing experience** may be causing dissatisfaction.
4. **Segment by region and gender** for tailored retention strategies.
5. **Focus on digital engagement** for female customers, who show higher churn.

___

### **Potential Customers At Risk**

<img width="586" height="682" alt="image" src="https://github.com/user-attachments/assets/3dd19896-4052-4f9d-b18c-5b0061503106" />

***

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib







