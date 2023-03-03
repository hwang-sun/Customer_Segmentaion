from streamlit_option_menu import option_menu
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
import pickle
from joblib import load

#--------------------------------- RFM Analysis -------------------------------------
# read RFM data
@st.cache_data
def load_csv_df(df):
  df =  pd.read_csv(df)
  return df
rfm_df = load_csv_df(df = 'RFM_data.csv')    

# customer labeling
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
@st.cache_data
def rfm_labeling(rfm_df):
  rfm_df['RFM_label'] = rfm_df.apply(rfm_label, axis=1)
  return rfm_df

# RFM aggregration
@st.cache_data
def rfm_aggregation(df, label, agg_dict):
  rfm_agg = df.groupby(label).agg(agg_dict).round(0)
  rfm_agg.columns = rfm_agg.columns.droplevel()
  rfm_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
  rfm_agg['Percent'] = round(rfm_agg['Count']*100/rfm_agg.Count.sum(), 2)
  rfm_agg = rfm_agg.reset_index()
  return rfm_agg

# Clusters bubble plot
@st.cache_data
def bubble_plot(df_agg, label):
  fig = px.scatter(df_agg, x="RecencyMean", y="FrequencyMean", size="MonetaryMean", color=label,
                  hover_name=label, size_max=100)
  return fig

# clusters by quantity
@st.cache_data
def qua_rev_plot(df, label):
  count = df[label].value_counts(normalize=True)*100
  sum = df[['Monetary',label]].groupby(label).sum()
  sum['percent'] = round(sum['Monetary']*100/df.Monetary.sum(),2)

  plt.style.use('seaborn-whitegrid')
  qua_re_fig = plt.figure(figsize = (10, 5))
  plt.subplot(1,2,1)
  ax_q = sns.barplot(data = count, 
              x = count.index.tolist(), y = count.values,
              orient = 'h',
              palette = 'Spectral')
  ytick = [str(x) for x in count.index.tolist()]
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
  ytick = [str(x) for x in sum.sort_values(by='percent').index.values.tolist()]
  ax_r.set_yticklabels(ytick, fontsize = 13)
  ax_r.set_xlim(0, 60)
  ax_r.set_ylabel(None)
  ax_r.set_xlabel(None)
  ax_r.set_title('Total revenue by customer clusters (%)', fontsize = 17)
  plt.setp(ax_r.get_xticklabels(), fontsize=13)
  plt.tight_layout()
  return qua_re_fig
#--------------------------------- KMeans Clustering -----------------------------------
@st.cache_data
def extract_cols(df, col_lst):
  new_df = rfm_df[col_lst]
  return new_df

# distribution and boxplot
@st.cache_data
def dis_box_plot(df):
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
  return dis_box_fig

# scale df
@st.cache_data
def robust_scale(df):
  # log normalization
  log_features = df.copy()
  log_features['R_log'] = np.log1p(log_features['Recency'])
  log_features['F_log'] = np.log1p(log_features['Frequency'])
  log_features['M_log'] = np.log1p(log_features['Monetary'])
  col_names = ['R_log', 'F_log','M_log']
  features = log_features[col_names]
  # Robust scaling
  robust_scaler = preprocessing.RobustScaler()
  scaled = robust_scaler.fit_transform(features)
  scale_df = pd.DataFrame(scaled, columns=df.columns.values.tolist())
  return scale_df

# Picking best centroids with Elbow method
@st.cache_data
def k_best_plot(df):
  silhouette = []
  wsse = []
  K=[]
  for k in range(2, 10):
      kmeans = KMeans(n_clusters = k)
      kmeans.fit(df)
      wsse.append(kmeans.inertia_/df.shape[0])
      silhouette.append(silhouette_score(df, kmeans.labels_))
      K.append(k)
  
  # plotting
  k_best_fig = plt.figure(figsize=(10, 4))
  plt.subplot(2,1,1)
  plt.plot(K, wsse, 'bx-', alpha = 0.8)
  plt.ylabel('WSSE', fontsize = 12)
  plt.xticks(K, fontsize=10)
  plt.yticks(fontsize=10)
  plt.title('Elbow Method for optimal k', fontsize = 15)
  plt.subplot(2,1,2)
  plt.plot(K, silhouette, 'bx-', c='r', alpha=0.7)
  plt.xlabel('Number of centroids', fontsize = 12)
  plt.ylabel('Silhouette score', fontsize = 12)
  plt.xticks(K, fontsize=10)
  plt.yticks(fontsize=10)
  plt.title('Silhouette Method for optimal k', fontsize = 15)
  plt.tight_layout()
  return k_best_fig

# Train model
@st.cache_data
def kmeans_model(train_df, label_df):
  model = KMeans(n_clusters = 5)
  model.fit(train_df)
  # get centroids and labels
  centroids = model.cluster_centers_
  labels = model.labels_
  label_df['K_label'] = pd.Series(labels)
  return centroids, label_df

@st.cache_data
def df_aggregation(df, label, agg_dict):
  df_agg = df.groupby(label).agg(agg_dict).round(0)
  df_agg.columns = df_agg.columns.droplevel()
  df_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
  df_agg['Percent'] = round(df_agg['Count']*100/df_agg.Count.sum(), 2)
  df_agg = df_agg.reset_index()
  return df_agg
#------------------------ CLUSTERING WHOLE NEW FILE FROM USER --------------------------




# upload file
# upload_file = st.file_upload("Load your file", type = ['csv'])
# st.write('Your file should include fields: "Customer_id", "order_id", "Revenue"')
# if upload_file is not None:
#     data = pd.read_csv(upload_file, encoding = 'latin-1')

# Data preprocess


# load model
@st.cache(allow_output_mutation=True)
def load_model(model_name):  
  clf = load(model_name)
  return clf

# -------------------------------- GUI Setting -----------------------------------------
# set page configuration
# st.set_page_config(page_title='Customer Segmentation Project', layout='centered')

# create title
st.title('Customer Segmentation Project')
#create a navigation menu
choice = option_menu(
    options = ['Business Objective', 'RFM Analysis','Kmeans Clustering', 'New Prediction'],
    menu_title = None,
    icons = ['bullseye', 'bar-chart', 'robot', 'file-plus'],
    menu_icon = [None])

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
    code = """ 
r_groups = pd.qcut(df_RFM['Recency'].rank(method='first'), q=4, labels=range(4, 0, -1))
f_groups = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=4, labels=range(1, 5, 1))
m_groups = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=4, labels=range(1, 5, 1))

df_rfm = df_RFM.assign(R = r_groups.values, F = f_groups.values,  M = m_groups.values)
    """
    st.code(code)
    st.dataframe(rfm_df.head(3))
    st.write('"RFM_label" was being assigned for each transaction by taking into consideration values of "R", "F", "M"')
    rfm_df = rfm_labeling(rfm_df)
    st.dataframe(rfm_df.head(3))

    st.write('''I then performed aggregating RFM result for ploting and analyzing the difference between groups:
    ''')
    rfm_agg = rfm_aggregation(df = rfm_df, label = 'RFM_label', agg_dict = {
      'Recency' : 'mean',
      'Frequency' : 'mean',
      'Monetary' : ['mean', 'count']})
    st.dataframe(rfm_agg)
    
    st.write('### RFM result:')
    fig = bubble_plot(df_agg = rfm_agg, label = 'RFM_label')
    st.plotly_chart(fig)
    st.write('Based on the result, The dataset was clustered into 5 different groups with following characteristics:')
    st.write('''
      - VIP : this group of customer bring the highest value to the company as they purchased very frequently with the mean of revenue for each transaction was $338, and still remain buying products over the past 2 months. Eventhough they only accounted for 14.7% of the total number of customers, over 50% revenue was from this group.
      - Loyal : this cluster represent for customers who purchased frequently and remained purchasing for the last few months. However, though they're considered to be loyalty, their revenue attribution was the lowest - about 5%.
      - Treat : the customers in this group had high mean of sale per transaction with the average frequency of buying is 5. Nevertheless, this group was saw to gradually leaving as it'd been approximately a year since their last purchase.
      - Regular : this type of customers represented for over 25% of the total number and their revenue attribution was less than 20%. They rarely made a purchase and had a high Recency mean.
      - Lost : customers from this group no longer bought products from the company as their frequency was low and it's over 1 and a half year since their last purchased.
    ''')
    
    qua_re_fig = qua_rev_plot(df = rfm_df, label = 'RFM_label')
    st.pyplot(qua_re_fig)

elif choice == 'Kmeans Clustering':
    st.write('## Kmeans Clusering')
    st.write('### About The Data')
    
    df = extract_cols(df = rfm_df, col_lst = ['Recency', 'Frequency', 'Monetary'])
    st.dataframe(df.head())
    
    dis_box_fig = dis_box_plot(df = df)
    st.pyplot(dis_box_fig)
    st.write("As RFM features had lots of outliners, I perform Robust scaling to standardize data before the RFM dataframe was trained by Kmeans algorithm.")
    
    scale_df = robust_scale(df = df)
    st.dataframe(scale_df.head())

    st.write('### Picking K centroids and Kmeans Modeling')
    k_best_fig = k_best_plot(df=scale_df)
    st.pyplot(k_best_fig)

    centroids, k_df = kmeans_model(train_df = scale_df, label_df = df)
    df_agg = df_aggregation(df = k_df, label = 'K_label', agg_dict = {
      'Recency' : 'mean',
      'Frequency' : 'mean',
      'Monetary' : ['mean', 'count']})
    fig_2 = bubble_plot(df_agg = df_agg, label = 'K_label')
    st.plotly_chart(fig_2)
    
    qua_re_fig = qua_rev_plot(df = k_df, label = 'K_label')
    st.pyplot(qua_re_fig.figure)
else:
    st.write('## How to predict?')
    model_select = load_csv_df('Clf_model/Clf_select.csv')
    st.dataframe(model_select)

    # load model
    clf = load_model('Clf_model/DC_clf.joblib')
    
    # Model evaluation
    st.write('## Model Evaluation')
    score_option = st.radio(
      'What score you want to see?',
      ['Accuracy', 'Weighted Scores', 'Classification report', 'Confusion matrix']
    )
    if score_option == 'Accuracy':
      score_df = load_csv_df('Clf_model/score_df.csv')
      st.dataframe(score_df)
      st.write('Model perform well and not being underfiting or overfiting')
    elif score_option == 'Weighted Scores':
      score_df2 = load_csv_df('Clf_model/score_df2.csv')
      st.dataframe(score_df2)
    elif score_option == 'Classification report':
      report_df = load_csv_df('Clf_model/classification_report.csv')
      st.dataframe(report_df)
    else:
      st.image('Clf_model/confusion_matrix.png')
    
    # Making predictions
    st.write('## Making Predictions')
    
    pred_option = st.selectbox(
      'How would you like to make prediction?',
      ['Upload your own data', 'Input values']
    )
    if pred_option == 'Upload your own data':
      upload_file = st.file_uploader("Choose a csv file", type = ['txt', 'csv'])
      if upload_file is not None:
        new_df_1 = pd.read_csv(upload_file)
        st.dataframe(new_df_1.head(5))
        line_1 = new_df_1[0]
        flag = 1
    elif pred_option == 'Input values':
      recency = st.slider('Days since your last purchase:', 0, 500, 0)
      frequency = st.slider('Range of total times you have made purchases:', 0, 200, (1, 20))
      monetary = st.slider('Range of total money you have spent ($):', 4, 14000, (4, 100))
      new_df_2 = pd.DataFrame({
        'Recency' : recency,
        'Frequency' :  sum(frequency)/len(frequency),
        'Monetary' : sum(monetary)/len(monetary)}, 
        index = [0])
      st.dataframe(new_df_2)
      line_2 = np.array(new_df_2)
      flag = 2
    
    if flag == 2:
      st.write('Prediction:')
      if len(line_2) > 0:
        x_scale = robust_scale(new_df_2)
        y_pred = clf.predict(x_scale)
        st.code("You belong to" + str(y_pred)[0] "group of customer") 
     elif flag == 1:
      st.write('Prediction:')
      if len(line_1) > 0:
        x_scale = robust_scale(new_df_1)
        y_pred = clf.predict(x_scale)
        new_df_1['label'] = y_pred
        st.dataframe(new_df_1.head())
    
    
