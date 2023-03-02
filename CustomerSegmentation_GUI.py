import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split  
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pickle
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

#--------------------------------- RFM Analysis -------------------------------------
# read RFM data
rfm_df =  pd.read_csv('RFM_data.csv')
# customer labeling
@st.cache_data
def rfm_label(df):
  if df.F == 1 or df.F==2 or df.F == 3:
    if (df.R == 1 or df.R == 2) and (df.M == 1 or df.M==2 or df.M == 3 or df.M == 4):
      return "Lost"
    elif df.R == 3 or df.R == 4 and (df.M == 1 or df.M==2 or df.M == 3 or df.M == 4):
      return "Regular"
    elif (df.R == 3 or df.R == 4) and df.M == 4:
      return "Potential"
  else:
    if (df.R == 1 or df.R == 2) and (df.M == 1 or df.M==2 or df.M == 3 or df.M == 4):
      return "Lost"
    elif df.R == 3 and (df.M == 1 or df.M == 2 or df.M == 3 or df.M == 4):
      return 'Treat'
    elif df.R == 4 and (df.M == 1 or df.M == 2 or df.M == 3):
      return "Loyal"
    elif df.R == 4 and df.M == 4:
      return "VIP"

rfm_df['RFM_label'] = rfm_df.apply(rfm_label, axis=1)
# RFM aggregration
rfm_agg = rfm_df.groupby('RFM_label').agg({
    'Recency' : 'mean',
    'Frequency' : 'mean',
    'Monetary' : ['mean', 'count']}).round(0)

rfm_agg.columns = rfm_agg.columns.droplevel()
rfm_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
rfm_agg['Percent'] = round(rfm_agg['Count']*100/rfm_agg.Count.sum(), 2)
rfm_agg = rfm_agg.reset_index()

# Clusters bubble plot
fig = px.scatter(rfm_agg, x="RecencyMean", y="FrequencyMean", size="MonetaryMean", color="RFM_label",
           hover_name="RFM_label", size_max=100)

# clusters by quantity
count = rfm_df.RFM_label.value_counts(normalize=True)*100
sum = rfm_df[['Monetary','RFM_label']].groupby('RFM_label').sum()
sum['percent'] = round(sum['Monetary']*100/rfm_df.Monetary.sum(),2)

plt.style.use('seaborn-whitegrid')
qua_re_fig = plt.figure(figsize = (10, 5))
plt.subplot(1,2,1)
ax_q = sns.barplot(data = count, 
            x = count.index.tolist(), y = count.values,
            orient = 'h',
            palette = 'Spectral')
ytick = count.sort_values().index.tolist()
ax_q.set_yticklabels(ytick, fontsize=13)
plt.setp(ax_q.get_xticklabels(), fontsize = 13)
ax_q.set_title("Customers' count by each cluster (%)", fontsize=17)
ax_q.set_ylabel('Labels', fontsize = 15)
ax_q.set_xlabel(None)
# clusters by revenues
plt.subplot(1,2,2)
ax_r = sns.barplot(y=sum.sort_values(by='percent').index, 
            x=sum.sort_values(by='percent').percent, 
            palette='Blues', orient='h')
ytick = sum.sort_values(by='percent').index.values.tolist()
ax_r.set_yticklabels(ytick, fontsize = 13)
ax_r.set_xlim(0, 60)
ax_r.set_ylabel(None)
ax_r.set_xlabel(None)
ax_r.set_title('Total revenue by customer clusters (%)', fontsize = 17)
plt.setp(ax_r.get_xticklabels(), fontsize=13)
plt.tight_layout()

#--------------------------------- KMeans Clustering -----------------------------------
df = rfm_df[['Recency', 'Frequency', 'Monetary']]
# distribution and boxplot
dis_box_fig = plt.figure(figsize=(10,8))
plt.subplot(3, 2, 1)
sns.distplot(df['Recency'])# Plot distribution of R
plt.subplot(3, 2, 3)
sns.distplot(df['Frequency'])# Plot distribution of F
plt.subplot(3, 2, 5)
sns.distplot(df['Monetary']) # Plot distribution of M
plt.subplot(3, 2, 2)
sns.boxplot(df.Recency)
plt.subplot(3, 2, 4)
sns.boxplot(df.Frequency)
plt.subplot(3, 2, 6)
sns.boxplot(df.Monetary)
plt.tight_layout()
# scale df
robust_scaler = preprocessing.RobustScaler()
scaled = robust_scaler.fit_transform(df)
scale_df = pd.DataFrame(scaled, columns=df.columns.values.tolist())

# Elbow method
wsse = []
K=[]
for k in range(2, 10):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(scale_df)
    wsse.append(kmeans.inertia_/scale_df.shape[0])
    K.append(k)
# plotting
elbow_fig = plt.figure(figsize=(10, 4))
plt.plot(K, wsse, 'bx-')
plt.xlabel('Number of centroids', fontsize = 15)
plt.ylabel('WSSE', fontsize = 15)
plt.xticks(K, fontsize=12)
plt.yticks(fontsize=12)
plt.title('Elbow Method for optimal k', fontsize = 18)

# Train model
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 5)
model.fit(scale_df)
# get centroids and labels
centroids = model.cluster_centers_
labels = model.labels_
df['K_label'] = pd.Series(labels)

# df aggregation
df_agg =df.groupby('K_label').agg({
    'Recency' : 'mean',
    'Frequency' : 'mean',
    'Monetary' : ['mean', 'count']}).round(0)
df_agg.columns = df_agg.columns.droplevel()
df_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
df_agg['Percent'] = round(df_agg['Count']*100/df_agg.Count.sum(), 2)
df_agg = df_agg.reset_index()

# bubble plot
fig_2 = px.scatter(df_agg, x="RecencyMean", y="FrequencyMean", size="MonetaryMean", color="K_label",
           hover_name="K_label", size_max=100)

# cluster by revenue
sum_2 = df[['Monetary','K_label']].groupby('K_label').sum()
sum_2['percent'] = round(sum_2['Monetary']*100/df.Monetary.sum(),2)
sum_2 = sum_2.sort_values(by='percent')
plt.figure(figsize = (10, 5))
ax_r_2 = sns.barplot(x = [str(x) for x in sum_2.index.tolist()], y=sum_2.percent, palette='flare')
ax_r_2.set_ylim(0, 60)
ax_r_2.set_title('Total revenue by Kmeans clusters (%)', fontsize = 15)
#------------------------ CLUSTERING WHOLE NEW FILE FROM USER --------------------------

# upload file
# upload_file = st.file_upload("Load your file", type = ['csv'])
# st.write('Your file should include fields: "Customer_id", "order_id", "Revenue"')
# if upload_file is not None:
#     data = pd.read_csv(upload_file, encoding = 'latin-1')

# Data preprocess

# -------------------------------- GUI Setting -----------------------------------------
# set page configuration
st.set_page_config(page_title='Customer Segmentation Project', layout='centered')

# create title
st.title('Customer Segmentation Project')
# create a navigation menu
# choice = option_menu(
#     options = ['Business Objective', 'RFM Analysis','Kmeans Clustering', 'New Prediction'],
#     menu_title = None,
#     icons = ['bullseye', 'bar-chart', 'robot', 'file-plus'],
#     menu_icon = [None],
#     orientation='horizontal')
menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == "Business Objective":
    st.write('## Business Objective')
    st.write('''Customer segmentation is the process of dividing a company's customers into smaller groups based on similar characteristics, such as demographics, behavior, needs, or preferences. Customer segmentation is important for several reasons:
    
    1. Better understanding of customers: 
    Customer segmentation allows a company to gain a deeper understanding of its customers, including their needs, preferences, and behaviors.
    
    2. Improved customer experience: 
    By tailoring products and services to specific customer segments, companies can provide a better customer experience.
    
    3. More effective marketing: 
    Customer segmentation allows companies to target their marketing efforts to specific customer segments. This can result in more effective marketing campaigns, and increased sales.
    
    4. Increased profitability: 
    By focusing on the most profitable customer segments, companies can increase their profitability.
    ''')
    st.image('customer-segmentation.jpg')
    st.write('''In this project, I perform segmenting customers based on 3 main factors:
             
    - Recency: The last time a customer made a purchase.
    - Frequency: The number of times a customer has made a purchase.
    - Monetary value: The total amount of money a customer has spent on purchases.
    ''')
    st.write("By using RFM analysis and Kmeans clustering algorithm on these 3 features, I expect to defferentiate customer groups' behaviors and values.")
elif choice == 'RFM Analysis':
    st.write("## RFM Analysis")
    st.write('### About The Data:')
    
    st.write('''The data used for analysis including 3 main features: "Recency", "Frequency", "Monetary Value"
             . The "R", "F", "M" features were engineered by calculating quantile for each feature.''')
    code = """ r_groups = pd.qcut(df_RFM['Recency'].rank(method='first'), q=4, labels=range(4, 0, -1))
f_groups = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=4, labels=range(1, 5, 1))
m_groups = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=4, labels=range(1, 5, 1))

df_rfm = df_RFM.assign(R = r_groups.values, F = f_groups.values,  M = m_groups.values)
    """
    st.code(code)
    st.dataframe(rfm_df.iloc[:, :-1].head(3))
    st.write('"RFM_label" was being assigned for each transaction by taking into consideration values of "R", "F", "M"')
    st.dataframe(rfm_df.tail(3))

    st.write('''I then performed aggregating RFM result for ploting and analyzing the difference between groups:
    ''')
    st.dataframe(rfm_agg)
    st.write('### RFM result:')
    st.plotly_chart(fig)
    st.write('Based on the result, The dataset was clustered into 5 different groups with following characteristics:')
    st.write('''
      - VIP : this group of customer bring the highest value to the company as they purchased very frequently with the mean of revenue for each transaction was $338, and still remain buying products over the past 2 months. Eventhough they only accounted for 14.7% of the total number of customers, over 50% revenue was from this group.
      - Loyal : this cluster represent for customers who purchased frequently and remained purchasing for the last few months. However, though they're considered to be loyalty, their revenue attribution was the lowest - about 5%.
      - Treat : the customers in this group had high mean of sale per transaction with the average frequency of buying is 5. Nevertheless, this group was saw to gradually leaving as it'd been approximately a year since their last purchase.
      - Regular : this type of customers represented for over 25% of the total number and their revenue attribution was less than 20%. They rarely made a purchase and had a high Recency mean.
      - Lost : customers from this group no longer bought products from the company as their frequency was low and it's over 1 and a half year since their last purchased.
    ''')
    
    st.pyplot(qua_re_fig)

elif choice == 'Kmeans Clustering':
    st.write('## Kmeans Clusering')
    st.write('### About The Data')
    st.dataframe(df.iloc[:, :-1].head())
    st.pyplot(dis_box_fig)
    st.write("As RFM features had lots of outliners, I perform Robust scaling to standardize data before the RFM dataframe was trained by Kmeans algorithm.")
    st.dataframe(scale_df.head())
    st.write('### Picking K centroids and Kmeans Modeling')
    st.pyplot(elbow_fig)

    st.plotly_chart(fig_2)
    
    st.pyplot(ax_r_2.figure)
else:
    st.write('## Making Predictions')
