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
```


```python
#file path
file_path = "Prediction Data.xlsx"

#Sheet Name
sheet_name = "vw_prodChurn"

# Read the data from the specified sheet into a pandas DataFrame
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Display the first few rows of the fetched data
print(data.head())
```

      Customer_ID  Gender  Age Married           State  Number_of_Referrals  \
    0   11098-MAD  Female   30     Yes  Madhya Pradesh                    0   
    1   11114-PUN    Male   51      No          Punjab                    5   
    2   11167-WES  Female   43     Yes     West Bengal                    3   
    3   11179-MAH    Male   35      No     Maharashtra                   10   
    4   11180-TAM    Male   75     Yes      Tamil Nadu                   12   
    
       Tenure_in_Months Value_Deal Phone_Service Multiple_Lines  ...  \
    0                31     Deal 1           Yes             No  ...   
    1                 9     Deal 5           Yes             No  ...   
    2                28     Deal 1           Yes            Yes  ...   
    3                12        NaN           Yes             No  ...   
    4                27     Deal 2           Yes             No  ...   
    
        Payment_Method Monthly_Charge Total_Charges Total_Refunds  \
    0  Bank Withdrawal      95.099998   6683.399902          0.00   
    1  Bank Withdrawal      49.150002    169.050003          0.00   
    2  Bank Withdrawal     116.050003   8297.500000         42.57   
    3      Credit Card      84.400002   5969.299805          0.00   
    4      Credit Card      72.599998   4084.350098          0.00   
    
      Total_Extra_Data_Charges Total_Long_Distance_Charges Total_Revenue  \
    0                        0                  631.719971   7315.120117   
    1                       10                  122.370003    301.420013   
    2                      110                 1872.979980  10237.910156   
    3                        0                  219.389999   6188.689941   
    4                      140                  332.079987   4556.430176   
    
      Customer_Status Churn_Category                   Churn_Reason  
    0          Stayed         Others                         Others  
    1         Churned     Competitor  Competitor had better devices  
    2          Stayed         Others                         Others  
    3          Stayed         Others                         Others  
    4          Stayed         Others                         Others  
    
    [5 rows x 32 columns]
    


```python
#Data_Processing

#Drop Columns that won't be used for prediction
data = data.drop(['Customer_ID', 'Churn_Category', 'Churn_Reason'], axis=1)
# List of columns to be label encoded

columns_to_encode = [

    'Gender', 'Married', 'State', 'Value_Deal', 'Phone_Service', 'Multiple_Lines',

    'Internet_Service', 'Internet_Type', 'Online_Security', 'Online_Backup',

    'Device_Protection_Plan', 'Premium_Support', 'Streaming_TV', 'Streaming_Movies',

    'Streaming_Music', 'Unlimited_Data', 'Contract', 'Paperless_Billing',

    'Payment_Method'

]

# Encode categorical variables except the target variable
label_encoders = {}
for column in columns_to_encode:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

#Manually encode the target variable 'Customer_Status'

data['Customer_Status'] = data['Customer_Status'].map({'Stayed':0, 'Churned':1})

# Split data into features and target

X = data.drop('Customer_Status', axis=1)

y = data['Customer_Status']

#Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

 

# Train the model

rf_model.fit(X_train, y_train)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(random_state=42)</pre></div> </div></div></div></div>



## Evaluate Model ##


```python

# Make predictions

y_pred = rf_model.predict(X_test)

# Evaluate the model

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")

print(classification_report(y_test, y_pred))
```

    Confusion Matrix:
    [[783  64]
     [126 229]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.86      0.92      0.89       847
               1       0.78      0.65      0.71       355
    
        accuracy                           0.84      1202
       macro avg       0.82      0.78      0.80      1202
    weighted avg       0.84      0.84      0.84      1202
    
    


```python
# Feature Selection using Feature Importance

importances = rf_model.feature_importances_

indices = np.argsort(importances)[::-1]
```


```python
# Plot the feature importances

plt.figure(figsize=(15, 6))
sns.barplot(x=importances[indices], y=X.columns[indices])

plt.title('Feature Importances')

plt.xlabel('Relative Importance')

plt.ylabel('Feature Names')

plt.show()


```


    
![png](output_7_0.png)
    


## Use Model for Prediction on New Data ##


```python
#Define Path 
file_path = "Prediction Data.xlsx"

#Define sheet name to read data from
sheet_name = 'vw_Join'
# Read the data from the specified sheet into a pandas DataFrame

new_data = pd.read_excel(file_path, sheet_name=sheet_name)

# Display the first few rows of the fetched data

print(new_data.head())

```

      Customer_ID  Gender  Age Married        State  Number_of_Referrals  \
    0   11751-TAM  Female   18      No   Tamil Nadu                    5   
    1   12056-WES    Male   27      No  West Bengal                    2   
    2   12136-RAJ  Female   25     Yes    Rajasthan                    2   
    3   12257-ASS  Female   39      No        Assam                    9   
    4   12340-DEL  Female   51     Yes        Delhi                    0   
    
       Tenure_in_Months Value_Deal Phone_Service Multiple_Lines  ...  \
    0                 7     Deal 5            No             No  ...   
    1                20        NaN           Yes             No  ...   
    2                35        NaN           Yes             No  ...   
    3                 1        NaN           Yes             No  ...   
    4                10        NaN           Yes             No  ...   
    
        Payment_Method Monthly_Charge Total_Charges Total_Refunds  \
    0     Mailed Check      24.299999     38.450001           0.0   
    1  Bank Withdrawal      90.400002    268.450012           0.0   
    2  Bank Withdrawal      19.900000     19.900000           0.0   
    3      Credit Card      19.549999     19.549999           0.0   
    4      Credit Card      62.799999     62.799999           0.0   
    
      Total_Extra_Data_Charges Total_Long_Distance_Charges Total_Revenue  \
    0                        0                    0.000000     38.450001   
    1                        0                   94.440002    362.890015   
    2                        0                   11.830000     31.730000   
    3                        0                   10.200000     29.750000   
    4                        0                   42.189999    104.989998   
    
      Customer_Status Churn_Category Churn_Reason  
    0          Joined         Others       Others  
    1          Joined         Others       Others  
    2          Joined         Others       Others  
    3          Joined         Others       Others  
    4          Joined         Others       Others  
    
    [5 rows x 32 columns]
    


```python
# Retain the original DataFrame to preserve unencoded columns

original_data = new_data.copy()

# Retain the Customer_ID column

customer_ids = new_data['Customer_ID']


```


```python
# Drop columns that won't be used for prediction in the encoded DataFrame

new_data = new_data.drop(['Customer_ID', 'Customer_Status', 'Churn_Category', 'Churn_Reason'], axis=1)
```


```python
# Encode categorical variables using the saved label encoders

for column in new_data.select_dtypes(include=['object']).columns:

    new_data[column] = label_encoders[column].transform(new_data[column])
```


```python
# Make predictions

new_predictions = rf_model.predict(new_data)
```


```python
# Add predictions to the original DataFrame

original_data['Customer_Status_Predicted'] = new_predictions
```


```python
# Filter the DataFrame to include only records predicted as "Churned"

original_data = original_data[original_data['Customer_Status_Predicted'] == 1]
```


```python
# Save the results

original_data.to_csv(r"C:\Users\akskumari\Desktop\Prediction Data.csv", index=False)
```


```python

```
