# 🛒 Market Basket Analysis — Online Retail Intelligence

An end-to-end retail analytics and recommendation engine project built using transactional e-commerce data from the UCI Online Retail dataset. This project applies Exploratory Data Analysis (EDA) and Market Basket Analysis (MBA) to uncover purchasing behavior, identify high-value cross-selling opportunities, and generate actionable merchandising recommendations using the **Apriori algorithm**.
---

## 📚 Dataset

Dataset Source:
[UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail?utm_source=chatgpt.com)

---
## 📊 Key Insights

| Insight                                        | Business Impact                                    |
| ---------------------------------------------- | -------------------------------------------------- |
| UK contributes ~85% of total revenue           | Primary market for personalization and retention   |
| Transactions peak between 10 AM – 2 PM         | Ideal window for campaigns and promotions          |
| November demand surges ~182% vs February       | Inventory planning should begin by September       |
| Herb marker products show lift values above 70 | Extremely strong bundle recommendation opportunity |
| Average UK basket size = 26.44 products        | Strong potential for cart-expansion strategies     |

---

## 🧠 Market Basket Analysis Summary

| Metric                   | Value |
| ------------------------ | ----- |
| Frequent 1-Itemsets      | 731   |
| Frequent 2-Itemsets      | 8,985 |
| Frequent 3-Itemsets      | 6,173 |
| Strong Association Rules | 565   |
| Min Support              | 0.5%  |
| Min Confidence           | 50%   |
| Min Lift                 | 5     |

---

## 📈 Analytics Covered

### Global Retail Analysis

* Revenue distribution by country
* Transaction distribution
* Basket size behavior
* Average order value comparison

### UK Customer Behavior Analysis

* Hourly transaction trends
* Monthly seasonality patterns
* Product demand analysis
* Basket size contribution

### Association Rule Mining

* 1 → 1 product recommendations
* Cross-sell recommendations
* Bundle recommendation strategy
* High-lift product combinations

---

## 🛠️ Tech Stack

| Category                 | Tools               |
| ------------------------ | ------------------- |
| Programming              | Python              |
| Data Processing          | pandas, numpy       |
| Visualization            | matplotlib, plotly  |
| MBA & Rule Mining        | mlxtend             |
| Dashboard / Presentation | HTML, CSS, Chart.js |
| Notebook Sync            | jupytext            |

---

## 🌐 Interactive Case Study

The project also includes a fully designed interactive HTML business case study featuring:

* Executive KPI dashboard
* Interactive visualizations
* Business recommendations
* Rule-mining outputs
* Strategic retail insights

---
## 📌 Business Recommendations

* Deploy “Frequently Bought Together” recommendation widgets
* Launch seasonal product bundles during Q4
* Prioritize UK retention campaigns
* Trigger dynamic cart-growth prompts
* Optimize inventory planning around seasonal spikes

---

## 👤 Author

Meghana Bommena
Retail Analytics • Customer Intelligence • Data Storytelling

---

*Built with Python, Apriori, and a dangerous amount of retail transaction logs.* 😄
