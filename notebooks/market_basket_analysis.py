# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## MARKET BASKET ANALYSIS

# %%
# Install library (run only once)
# #!pip install mlxtend
# #!pip install --upgrade plotly xarray
# #!pip install numpy==1.26.4
# #!pip install jupytext
# #!jupytext --set-formats ipynb,py mba.ipynb

# %%
# Import warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Import pandas for data manipulation and analysis
import pandas as pd

# Import numpy for numerical computations
import numpy as np

# Import matplotlib for visualizations
import matplotlib.pyplot as plt

# Import plotly for interactive visualizations
import plotly.express as px

# Import Apriori algorithm and association rules
from mlxtend.frequent_patterns import apriori, association_rules

# %%
df_main = pd.read_csv("E:/Market Basket Analysis/data/Online_Retail.csv", encoding='cp1252')
df = df_main

df.head()

# %%
df.info()

# %%
# Data Preprocessing

# Remove rows with missing product descriptions
df.dropna(subset=['Description'], inplace=True)

# Remove cancelled invoices
df = df[~df['InvoiceNo'].str.startswith('C')]

# Remove negative or zero quantities
df = df[df['Quantity'] > 0]

# Remove negative or zero unit prices
df = df[df['UnitPrice'] > 0]

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Standardize product descriptions
df['Description'] = df['Description'].str.strip().str.upper()

# %%
# List of non-product / operational entries
exclude_items = [
    'POSTAGE',
    'DOTCOM POSTAGE',
    'BANK CHARGES',
    'PACKING CHARGE',
    'MANUAL',
    'DOTCOMGIFTSHOP GIFT VOUCHER £10.00',
    'DOTCOMGIFTSHOP GIFT VOUCHER £20.00',
    'DOTCOMGIFTSHOP GIFT VOUCHER £50.00'
]

# Remove rows where Description is in exclude_items list
df = df[~df['Description'].isin(exclude_items)]

# %%
#Feature engineering

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['Hour'] = df['InvoiceDate'].dt.hour
df['Weekday'] = df['InvoiceDate'].dt.day_name()
df['Revenue'] = df['Quantity'] * df['UnitPrice']
df.head()

# %%

# %%
# # Export cleaned dataset to CSV & Excel

# df.to_csv(
#     r"E:\Market Basket Analysis\outputs\Online_Retail_Cleaned.csv",
#     index=False
# )

# df.to_excel(
#     r"E:\Market Basket Analysis\outputs\Online_Retail_Cleaned.xlsx",
#     index=False
# )

# print("CSV & Excel files are exported successfully!")

# %%
# Global KPIs
total_revenue = df['Revenue'].sum()
total_transactions = df['InvoiceNo'].nunique()  
total_items_sold = df['Quantity'].sum()
unique_customers = df['CustomerID'].nunique()
unique_products = df['Description'].nunique()
unique_countries = df['Country'].nunique()
print(f"Total Revenue: £{total_revenue:,.2f}")
print(f"Total Transactions: {total_transactions}")
print(f"Total Items Sold: {total_items_sold}")
print(f"Unique Customers: {unique_customers}")
print(f"Unique Products: {unique_products}")
print(f"Unique Countries: {unique_countries}")  


# %% [markdown]
#
# ## GLOBAL OVERVIEW ANALYSIS
#
#

# %%
# ==========================================
# 1. Top 10 Countries by Number of Transactions
# HORIZONTAL BAR CHART
# =========================================
top_transactions_country = (
    df.groupby('Country')['InvoiceNo']
      .nunique()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure(figsize=(12,6))

# Gradient Colors
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(top_transactions_country)))

bars = plt.bar(
    top_transactions_country.index,
    top_transactions_country.values,
    color=colors
)

# Title and Labels
plt.title('Top 10 Countries by Number of Transactions',
          fontsize=16,
          fontweight='bold')

plt.xlabel('Country', fontsize=12)
plt.ylabel('Number of Transactions', fontsize=12)

# Rotate country names
plt.xticks(rotation=45)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f'{int(height):,}',
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.tight_layout()
plt.show()


# ==========================================
# 2. TOP 10 REVENUE GENERATING COUNTRIES
# HORIZONTAL BAR CHART
# ==========================================

top_revenue_countries = (
    df.groupby('Country')['Revenue']
           .sum()
           .sort_values(ascending=False)
           .head(10)
)

plt.figure(figsize=(12,6))

# Create green gradient colors
colors = plt.cm.Greens(
    np.linspace(0.4, 0.9, 10)
)

# Plot chart
ax = top_revenue_countries.sort_values().plot(
    kind='barh',
    color=colors
)

plt.title('Top 10 Revenue Generating Countries')

plt.xlabel('Revenue (€)')
plt.ylabel('Country')

plt.ticklabel_format(style='plain', axis='x')

# Add labels
for i in ax.patches:

    value = i.get_width()

    ax.annotate(
        f'€{value:,.0f}',
        (
            value,
            i.get_y() + i.get_height()/2
        ),
        ha='left',
        va='center',
        fontsize=9,
        fontweight='bold'
    )

plt.show()

# ==========================================
# 3. BASKET SIZE DISTRIBUTION
# DONUT CHART
# ==========================================

# Basket Size = Number of unique products per transaction
basket_size = (
    df.groupby('InvoiceNo')['StockCode']
      .nunique()
)

# Basket Categories
basket_categories = pd.cut(
    basket_size,
    bins=[0, 2, 5, 10, 20, 100],
    labels=['1-2 Items', '3-5 Items', '6-10 Items',
            '11-20 Items', '20+ Items']
)

basket_distribution = basket_categories.value_counts().sort_index()

plt.figure(figsize=(10,8))

# Gradient Colors
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(basket_distribution)))

# Donut Chart
wedges, texts, autotexts = plt.pie(
    basket_distribution.values,
    labels=basket_distribution.index,      # Labels outside
    autopct='%1.1f%%',                     # Percentages inside
    startangle=140,
    colors=colors,
    pctdistance=0.75,                      # Move % inside donut
    labeldistance=1.1,                     # Move labels outside
    wedgeprops=dict(width=0.4),
    textprops=dict(fontsize=11)
)

# Style percentage text
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(10)
    autotext.set_weight('bold')

# Title
plt.title('Basket Size Distribution',
          fontsize=16,
          fontweight='bold')

plt.tight_layout()
plt.show()


# ==========================================
# 4. AVERAGE REVENUE PER TRANSACTION
# SCATTER PLOT
# ==========================================

revenue_per_transaction = (
    df.groupby('Country')
           .apply(
               lambda x:
               x['Revenue'].sum() /
               x['InvoiceNo'].nunique()
           )
           .sort_values(ascending=False)
           .head(10)
)

plt.figure(figsize=(10,5))

# Violet-Purple Gradient
colors = plt.cm.Purples(
    np.linspace(0.9, 0.4, 10)
)

plt.scatter(
    revenue_per_transaction.index,
    revenue_per_transaction.values,
    s=220,
    c=colors
)

plt.title('Top 10 Countries by Average Revenue per Transaction')

plt.xlabel('Country')
plt.ylabel('Average Revenue per Transaction (€)')

plt.xticks(rotation=45)

# Add labels
for x, y in zip(
    revenue_per_transaction.index,
    revenue_per_transaction.values
):

    plt.text(
        x,
        y + 40,
        f'€{y:,.0f}',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold'
    )

plt.grid(alpha=0.3)

plt.show()

# %% [markdown]
# ## Global Market Insights
#
# The United Kingdom overwhelmingly dominated the dataset in total revenue and transaction volume, indicating a significantly richer customer activity base compared to other countries.
# Although several countries exhibited larger average basket sizes and higher revenue per transaction, their overall transaction volumes were relatively small.
# Therefore, the United Kingdom was selected for deeper Market Basket Analysis to generate more reliable and scalable customer purchasing insights. 

# %% [markdown]
# ## United Kingdom Analysis

# %%
# Filter only United Kingdom data
df = df[df['Country'] == 'United Kingdom']

# Exporting UK data to Excel for further analysis
#df.to_excel("E:/Market Basket Analysis/outputs/uk_data.xlsx", index=False)

# %%
df.head()

# %%
df.info()

# %%
df.isnull().sum()

# %%
# UK KPIs
total_revenue = df['Revenue'].sum()
total_transactions = df['InvoiceNo'].nunique()  
average_order_value = total_revenue / total_transactions
total_items_sold = df['Quantity'].sum()
average_basket_size = df.groupby('InvoiceNo')['Description'].nunique().mean()

print(f"Total Revenue: £{total_revenue:,.2f}")
print(f"Total Transactions: {total_transactions}")
print(f"Average Order Value: £{average_order_value:,.2f}")
print(f"Total Items Sold: {total_items_sold}")
print(f"Average Basket Size: {average_basket_size:.2f}")
 


# %% [markdown]
# ## Hourly Transaction Trend - United Kingdom

# %%
hourly_transactions = (df.groupby('Hour')['InvoiceNo']
         .nunique()
)

plt.figure(figsize=(12,6))

plt.fill_between(
    hourly_transactions.index,
    hourly_transactions.values,
    alpha=0.7,
    color = 'crimson'
)

plt.plot(
    hourly_transactions.index,
    hourly_transactions.values,
    linewidth=2,
    color = 'teal'
)

plt.title('Hourly Transaction Trend - United Kingdom', fontsize=14)

plt.xlabel('Hour of Day')
plt.ylabel('Number of Transactions')

plt.xticks(range(0,24))

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()

# %% [markdown]
# ## Monthly Demand Trend - United Kingdom

# %%
# Create Year-Month column
df['YearMonth'] = (
    df['Year'].astype(str)
    + '-'
    + df['Month'].astype(str).str.zfill(2)
)

# Monthly demand trend
monthly_demand = (
    df.groupby('YearMonth')['Quantity']
      .sum()
)

plt.figure(figsize=(12,6))

plt.plot(
    monthly_demand.index,
    monthly_demand.values,
    marker='o',
    linewidth=3,
    color = 'navy'
)

plt.title('Monthly Demand Trend - United Kingdom')

plt.xlabel('Year-Month')
plt.ylabel('Quantity Sold')

plt.xticks(rotation=45)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()

# %% [markdown]
# ## Top 10 Selling Products by Quantity - United Kingdom

# %%
#Top 10 Selling Products by Quantity - United Kingdom
top_selling_products = (df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10))

plt.figure(figsize=(12,6))

colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))

ax = top_selling_products.sort_values().plot(
    kind='barh',
    color=colors
)

plt.title('Top 10 Selling Products by Quantity - United Kingdom', fontsize=14)

plt.xlabel('Quantity Sold')
plt.ylabel('Products')

# Quantity labels
for i in ax.patches:

    value = i.get_width()

    ax.annotate(
        f'{int(value):,}',
        (
            value,
            i.get_y() + i.get_height()/2
        ),
        ha='left',
        va='center',
        fontsize=10,
        fontweight='bold'
    )

plt.tight_layout()

plt.show()

# %% [markdown]
# ## Top Revenue Generating Products - United Kingdom

# %%
# Top Revenue Generating Products in UK

top_revenue_products = (df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10))

plt.figure(figsize=(12,6))

colors = plt.cm.Reds(np.linspace(0.4, 0.9, 10))

ax = top_revenue_products.sort_values().plot(
    kind='barh',
    color=colors
)

plt.title('Top 10 Revenue Generating Products - United Kingdom', fontsize=14)

plt.xlabel('Revenue (€)')
plt.ylabel('Products')

# Revenue labels
for i in ax.patches:

    value = i.get_width()

    ax.annotate(
        f'€{value:,.0f}',
        (
            value,
            i.get_y() + i.get_height()/2
        ),
        ha='left',
        va='center',
        fontsize=10,
        fontweight='bold'
    )

plt.tight_layout()

plt.show()

# %% [markdown]
# ## Basket Size Distribution - United Kingdom

# %%
# Calculate basket size per transaction
basket_size = (
    df.groupby('InvoiceNo')['Description']
      .nunique()
)

# Create custom business-friendly bins
bins = [1, 5, 10, 15, 20, 30, 50, 100, 200, 1000]

# Labels for display
labels = [
    '1-5', '6-10', '11-15', '16-20',
    '21-30', '31-50', '51-100',
    '101-200', '201+'
]

# Categorize basket sizes
basket_categories = pd.cut(
    basket_size,
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Count transactions in each category
basket_distribution = (
    basket_categories
    .value_counts()
    .sort_index()
)

# Plot
plt.figure(figsize=(12,6))

bars = plt.bar(
    basket_distribution.index.astype(str),
    basket_distribution.values,
    color='orange',
    edgecolor='black',
    alpha=0.9
)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 50,
        f'{int(height):,}',
        ha='center',
        fontsize=10
    )

# Titles and labels
plt.title(
    'Basket Size Contribution — United Kingdom',
    fontsize=16,
    fontweight='bold'
)

plt.xlabel('Unique Products per Transaction')
plt.ylabel('Number of Transactions')

plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## APRIORI ALGORITHM

# %%
# Create basket matrix
basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0))

# %%
# Convert quantities into 1 and 0
basket = (basket > 0).astype(int)

# Display basket matrix shape
print("Basket Matrix Shape:", basket.shape)

# Preview basket matrix
basket.head()

# %%
# STEP 2: REMOVE RARE PRODUCTS
# Keep products purchased in at least 200 transactions
basket_filtered = basket.loc[:,basket.sum(axis=0) >= 200]

# Display new shape
print("Filtered Basket Shape:", basket_filtered.shape)

# Preview filtered basket
basket_filtered.head()

# %%
# STEP 3: APPLY APRIORI ALGORITHM
# Generate frequent itemsets
frequent_itemsets = apriori(
    basket_filtered.astype(bool),
    min_support=0.005,
    use_colnames=True,
    low_memory=True,
    max_len=3
)

# Add itemset length
frequent_itemsets['Length'] = (
    frequent_itemsets['itemsets']
    .apply(len)
)

# Sort by support
frequent_itemsets = frequent_itemsets.sort_values(
    by='support',
    ascending=False
)

print("\nFrequent Itemsets Summary:")
print(frequent_itemsets['Length'].value_counts().sort_index())

# %% [markdown]
# ## Association Rules

# %%
rules = association_rules(
    frequent_itemsets,
    metric='lift',
    min_threshold=1
)

# Convert frozensets into readable strings
rules['antecedents'] = rules['antecedents'].apply(
    lambda x: ', '.join(list(x))
)

rules['consequents'] = rules['consequents'].apply(
    lambda x: ', '.join(list(x))
)

# Create rule length columns
rules['antecedent_len'] = rules['antecedents'].apply(
    lambda x: len(x.split(', '))
)

rules['consequent_len'] = rules['consequents'].apply(
    lambda x: len(x.split(', '))
)

# Filter strong business-relevant rules
strong_rules = rules[
    (rules['support'] >= 0.005) &
    (rules['confidence'] >= 0.5) &
    (rules['lift'] >= 5) &
    (rules['antecedent_len'] == 1) &
    (rules['consequent_len'] <= 2)
]

# Keep only useful columns
strong_rules = strong_rules[[
    'antecedents',
    'consequents',
    'support',
    'confidence',
    'lift'
]]

strong_rules = strong_rules.drop_duplicates()

# Sort by lift
strong_rules = strong_rules.sort_values(
    by='lift',
    ascending=False
)

# Round values
strong_rules = strong_rules.round(3)

strong_rules = strong_rules.rename(columns={
    'antecedents': 'Product Purchased',
    'consequents': 'Recommended Product'
})

# Reset index
strong_rules = strong_rules.reset_index(drop=True)
strong_rules

# %%
# # Print strong rules summary - Run only once to save to Excel when changed the above code.

# strong_rules.to_excel("E:/Market Basket Analysis/outputs/strong_association_rules.xlsx", index=False)
# print("Strong association rules saved to Excel successfully!")
