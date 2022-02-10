#!/usr/bin/env python
# coding: utf-8

# # 高级数据分析师最趁手的利器 （Atoti）
# 
# ## 前言
# 我是一个懒人。
# 
# 虽然我会plotly, matplotlib 等等很多可视化库，但是毕竟还要写代码，即便是复制粘贴过来还要改改代码呢。
# 
# 我也尝试过BI，比如Microstrategy，Power BI，Grafana，KNIME等，KNIME是我最喜欢的，因为它包含数据分析和可视化，甚至是机器学习。
# 我公众号第一篇文章就是介绍的它。
# 
# 但是数据源的复杂性，往往又得让我不得不依赖于python，尤其是做数据处理，分析，以及建模。我最常用的就是Jupyter家族，notebook或者lab。
# 
# 今天重磅安利一款我喜欢的jupyter BI 插件，让我使用python处理数据时，又能轻松创建可视化的需求，甚至说发布临时dashboard给客户看。
# 
# 它就是Atoti。
# 
# 我关注Atoti 已经有一年之久了，一直没有安利，是因为它的版本不稳定，现在终于稍微稳定了，我觉得是时候介绍一下，未来可期。

# ## Atoti 特征总结
# - 优点：
#     - 定位轻量版的BI，可能是我孤陋寡闻，这是我见过最轻量的BI
#     - BI意味着，你托托拽拽就可以生成图表，或者是dashboard，饼图，散点，等常见图表应有尽有
#     - 集成在JupyterLab 中，几乎让我无缝将数据挖掘和BI结合
#     - 当然，针对BI的聚合功能也是包含了。count，mean，sum等统计特征自动帮你聚合好
#     - 最吸引我的，就是what-if功能。 很明显，设计者深知这个需求，这个功能让我们模拟不同情况下，KPI的指标如何变化
#     - 我最常用的数据源是csv，Atoti还支持其他，excel，spark，database，aws，azure等
#     - 竟然还考虑了数据安全，这点对我来说优点鸡肋，它可以创建允许访问dashboard的用户
#     
# - 缺点：
#     - 可视化目前没有地图插件，但是我估计会有，因为技术上可行。
#     - 初始使用习惯不同，学习曲线优点抖。不过几个小时还是可以搞定的
#     
# 

# 官方提供了很多范例，范围比较广：
# 
# 行政、航空公司、数字营销、能源、金融、食品加工、医疗保健、保险、 制药、零售、社交、社交媒体、体育、电信
# 
# https://github.com/atoti/notebooks
# 
# 
# 这里我高度参考了这些范例。如果看到这里，你感兴趣，那就体验一下吧。

# ## 开始实战
# 
# 首先是像平时一样，读取数据，做数据预处理等。
# 
# 唯一不同的是，需要安装并导入atoti.
# 
# 这里我们演示Atoti的两种使用场景： 
# - EDA
# - 建立dashboard（讲故事）

# In[1]:


import atoti as tt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# In[2]:


train_file = '../data/insurance.csv'
test_file = '../data/insurance_test.csv'
df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
df.head()
df['Sale']= df['Sale'].astype(str)


# # EDA
# 
# 简单介绍一下这个数据集，这个数据集来自kaggle，https://www.kaggle.com/ranja7/vehicle-insurance-customer-data
# 
# 数据包含了保险客户的社会经济数据以及被保险车辆的详细信息，这对于了解客户购买行为至关重要。我们采用这个数据集预测客户是否最后购买该保险。

# In[3]:


session = tt.create_session()
eda = session.read_pandas(
    df, table_name="eda", keys=["cust_id"]
)
eda_cube = session.create_cube(eda, name="eda")


# 比如我们可以快速查看不同教育程度的人在不同的销售渠道的到访情况。
# 我们只需要运行code
# ```session.visualize``` 就创建了一个图表，在这个图表了可以自由的探索了。
# - 选择一个图表类型，比如饼图
# - 选择对应的参数，比如value，类别，还可以创建多个子图。
# - Done！

# ![vis_2.png](../img/use_vis.gif)

# In[4]:


session.visualize("Sales by Sales Channel")


# 当然，BI可以任你发挥，创建各种图表

# In[5]:


session.visualize("bar")


# In[6]:


session.visualize("treemap")


# ## 创建模型进行模拟
# 
# 首先是常规操作，选择你喜欢的模型，预测各种数值。

# In[7]:


df['Sale']= df['Sale'].astype(int)



train_df, val_df = train_test_split(df, test_size=0.1, random_state=36)

train_df = train_df.dropna()
val_df = val_df.dropna()
test_df = test_df.dropna()

# Training data
X_train = train_df.iloc[:, 2:]
Y_train = train_df.iloc[:, 1]

# Validation data
X_val = val_df.iloc[:, 2:]
Y_val = val_df.iloc[:, 1]

# Test data
X_test = test_df.iloc[:, 2:]
Y_test = test_df.iloc[:, 1]
# split data into the features (X) and labels (y)
features_to_encode = X_train.columns[X_train.dtypes==object].tolist()  
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
col_trans = make_column_transformer(
                        (OneHotEncoder(),features_to_encode),
                        remainder = "passthrough"
                        )




# In[8]:


rf = RandomForestClassifier()
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(col_trans, rf)
pipe.fit(X_train, Y_train)
test_result = pipe.predict(X_test)


# ## 创建预测副本

# In[9]:


prediction_df = test_df.copy()
prediction_df.drop("Sale", axis=1, inplace=True)
prediction_df["Sales_prediction"] = test_result.tolist()

rf_predictions = pipe.predict_proba(X_test)
probability = []
for i in range(len(rf_predictions)):
    probability.append(rf_predictions[i][1])
    
prediction_df["sales_prediction_probability"] = probability


# In[10]:


predictions = session.read_pandas(
    prediction_df, table_name="scenarios", keys=["cust_id"]
)
predict_cube = session.create_cube(predictions, name="scenarios")

predictions.head()


# ## 模拟降价调整
# 我们模拟一下杀熟拉新的手段，也就是对于购买概率比较高的客户提高价格，对于购买意愿比较低的人降低价格。
# 
# 然后对于调整后的数据，进行重新预测，之后将结果保存在```scenarios```里面。

# In[11]:


# updating the pricing based on above pareto simulation

prediction_df_pareto = prediction_df.copy()

prediction_df_pareto.loc[
    (prediction_df_pareto.sales_prediction_probability > 0.8), "Price"
] = prediction_df_pareto["Price"].apply(lambda x: x * 1.2)

prediction_df_pareto.loc[
    (prediction_df_pareto.sales_prediction_probability < 0.2), "Price"
] = prediction_df_pareto["Price"].apply(lambda x: x * 0.8)

#
X_test_pareto = X_test.copy()
X_test_pareto["Price"] = prediction_df_pareto["Price"]

test_result_pareto = pipe.predict(X_test_pareto)

prediction_df_pareto["Sales_prediction"] = test_result_pareto.tolist()


# In[12]:


predictions.scenarios["Pareto price change"].load_pandas(prediction_df_pareto)


# 或者，你可以对特定的用户进行价格调整，比如只买基本保险的人，提高价格。或者非个人用户降低价格等。

# In[13]:


# updating the pricingi
prediction_df_personal_auto = prediction_df.copy()

prediction_df_personal_auto.loc[
    (prediction_df_personal_auto.Policy_Type != "Personal Auto"), "Price"
] = prediction_df_personal_auto["Price"].apply(lambda x: x * 1.25)

prediction_df_personal_auto.loc[
    (prediction_df_personal_auto.Coverage_Type == "Basic"), "Price"
] = prediction_df_personal_auto["Price"].apply(lambda x: x * 1.15)


# In[14]:


X_test_personal_auto = X_test.copy()
X_test_personal_auto["Price"] = prediction_df_personal_auto["Price"]

test_result_personal_auto = pipe.predict(X_test_personal_auto)
prediction_df_personal_auto["Sales_prediction"] = test_result_personal_auto.tolist()


# In[15]:


predictions.scenarios["Policy Change"].load_pandas(prediction_df_personal_auto)


# ## 对比模拟结果
# 同样的可视化，只不过，这里我们有多个scenarios副本信息，这些scenarios信息可以轻松加入图表。
# 在这之前，我们可以再做一些统计，比如利润的计算，我们可以采用pandas 单独算好每个副本。也可以使用atoti对所有副本同时计算。

# In[16]:


# this is sum of quotes from the predicted sales
predict_cube.measures["revenue_realised"] = tt.agg.sum(
    predict_cube.measures["sales_prediction_probability.SUM"] * predict_cube.measures["Price.SUM"],
    scope=tt.scope.origin(predict_cube.levels["cust_id"]),
)


# In[17]:


session.visualize()


# ## 给别人秀秀
# 
# 最后一步，分析的结果可以给别人展示了。
# 
# 首先是建立一个dashboard link，然后，可以添加上面我们分析的图表到dashboard了。
# 

# In[18]:


session.link(path="#/sales")


# ![use_dash.gif](../img/use_dash.gif)

# ![dash1.png](../img/dash1.png)

# In[ ]:




