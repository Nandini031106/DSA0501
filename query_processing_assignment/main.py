import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix, lag_plot

BASE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(BASE,'data'); RESULTS=os.path.join(BASE,'results'); os.makedirs(DATA,exist_ok=True); os.makedirs(RESULTS,exist_ok=True)
rng=np.random.default_rng(42)

def create_data():
    n=500; ids=np.arange(1001,1101)
    customers=pd.DataFrame({'Customer_ID':ids,'Customer_Name':[f'Customer_{i}' for i in ids],'City':rng.choice(['Chennai','Bengaluru','Hyderabad','Coimbatore','Madurai'],100)})
    sales=pd.DataFrame({'Order_ID':range(1,n+1),'Customer_ID':rng.choice(ids,n),'Date':pd.date_range('2024-01-01',periods=n),'Product':rng.choice(['Laptop','Phone','Tablet','Monitor','Keyboard'],n),'Category':rng.choice(['Electronics','Accessories'],n),'Quantity':rng.integers(1,8,n),'Unit_Price':rng.uniform(500,80000,n).round(2),'Discount_pct':rng.uniform(0,25,n).round(2),'Rating':rng.uniform(2.5,5,n).round(1)})
    sales['Sales_Amount']=(sales.Quantity*sales.Unit_Price*(1-sales.Discount_pct/100)).round(2)
    sales.loc[[15,72,188],'Rating']=np.nan; sales.loc[[31,144],'Discount_pct']=np.nan
    customers.to_csv(os.path.join(DATA,'customers.csv'),index=False); sales.to_csv(os.path.join(DATA,'sales.csv'),index=False); return sales,customers

def main():
    sales,customers=create_data(); print('=== DATA IMPORTING AND EXPLORATION ==='); print('Shape:',sales.shape); print(sales.head()); print('\nMissing BEFORE:\n',sales.isna().sum())
    sales.Rating=sales.Rating.fillna(sales.Rating.median()); sales.Discount_pct=sales.Discount_pct.fillna(sales.Discount_pct.median()); sales.Date=pd.to_datetime(sales.Date); sales['Month']=sales.Date.dt.month; sales['Month_Name']=sales.Date.dt.month_name(); sales.to_csv(os.path.join(DATA,'cleaned_sales.csv'),index=False)
    merged=sales.merge(customers,on='Customer_ID',how='left'); merged.to_csv(os.path.join(DATA,'merged_sales_customers.csv'),index=False)
    city=merged.groupby('City').Sales_Amount.agg(['sum','mean','count']).sort_values('sum',ascending=False); product=merged.groupby('Product').Sales_Amount.agg(['sum','mean','count']).sort_values('sum',ascending=False); city.to_csv(os.path.join(DATA,'city_summary.csv')); product.to_csv(os.path.join(DATA,'product_summary.csv'))
    numeric=merged[['Quantity','Unit_Price','Discount_pct','Rating','Sales_Amount']]; corr=numeric.corr(); corr.to_csv(os.path.join(RESULTS,'correlation_matrix.csv'))
    q1,q3=merged.Sales_Amount.quantile([.25,.75]); iqr=q3-q1; out=merged[(merged.Sales_Amount<q1-1.5*iqr)|(merged.Sales_Amount>q3+1.5*iqr)]; out.to_csv(os.path.join(RESULTS,'sales_outliers.csv'),index=False); print('\nMissing AFTER:\n',merged.isna().sum())
    daily=merged.groupby('Date').Sales_Amount.sum(); plt.figure(figsize=(9,5)); plt.plot(daily.index,daily.values); plt.title('Daily Sales Trend'); plt.xlabel('Date'); plt.ylabel('Sales Amount'); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(os.path.join(RESULTS,'line_chart.png')); plt.close()
    plt.figure(figsize=(8,5)); product['sum'].sort_values().plot(kind='barh'); plt.title('Product-wise Total Sales'); plt.xlabel('Sales Amount'); plt.tight_layout(); plt.savefig(os.path.join(RESULTS,'bar_chart.png')); plt.close()
    plt.figure(figsize=(8,5)); plt.hist(merged.Sales_Amount,bins=25); plt.title('Distribution of Sales Amount'); plt.xlabel('Sales Amount'); plt.ylabel('Frequency'); plt.tight_layout(); plt.savefig(os.path.join(RESULTS,'histogram.png')); plt.close()
    plt.figure(figsize=(8,5)); plt.scatter(merged.Unit_Price,merged.Sales_Amount,alpha=.5); plt.title('Unit Price vs Sales Amount'); plt.xlabel('Unit Price'); plt.ylabel('Sales Amount'); plt.tight_layout(); plt.savefig(os.path.join(RESULTS,'scatter_plot.png')); plt.close()
    plt.figure(figsize=(8,5)); plt.boxplot(merged.Sales_Amount); plt.title('Box Plot of Sales Amount'); plt.ylabel('Sales Amount'); plt.tight_layout(); plt.savefig(os.path.join(RESULTS,'box_plot.png')); plt.close()
    scatter_matrix(numeric,figsize=(12,10),diagonal='hist'); plt.suptitle('Scatter Matrix'); plt.tight_layout(); plt.savefig(os.path.join(RESULTS,'scatter_matrix.png')); plt.close('all')
    plt.figure(figsize=(7,5)); lag_plot(merged.sort_values('Date').Sales_Amount.reset_index(drop=True)); plt.title('Lag Plot of Sales Amount'); plt.tight_layout(); plt.savefig(os.path.join(RESULTS,'lag_plot.png')); plt.close()
    s=merged.sort_values('Date').Sales_Amount.reset_index(drop=True); lags=range(1,31); ac=[s.autocorr(lag=i) for i in lags]; plt.figure(figsize=(8,5)); plt.stem(lags,ac); plt.title('Autocorrelation Plot'); plt.xlabel('Lag'); plt.ylabel('Autocorrelation'); plt.tight_layout(); plt.savefig(os.path.join(RESULTS,'autocorrelation.png')); plt.close()
    means=[rng.choice(merged.Sales_Amount.to_numpy(),len(merged),replace=True).mean() for _ in range(1000)]; lo,hi=np.percentile(means,[2.5,97.5]); plt.figure(figsize=(8,5)); plt.hist(means,bins=30); plt.axvline(lo,linestyle='--',label='2.5%'); plt.axvline(hi,linestyle='--',label='97.5%'); plt.title('Bootstrap Distribution of Mean Sales'); plt.xlabel('Bootstrap Mean'); plt.ylabel('Frequency'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(RESULTS,'bootstrap_plot.png')); plt.close(); print(f'Bootstrap 95% CI: {lo:.2f} to {hi:.2f}')
    pd.DataFrame({'Technique':['Line Chart','Bar Chart','Histogram','Scatter Plot','Box Plot','Scatter Matrix','Lag Plot','Autocorrelation Plot','Bootstrap Plot'],'Purpose':['Time trends','Category comparison','Distribution','Variable relationship','Outliers','Multiple relationships','Sequential dependence','Time-series dependence','Uncertainty']}).to_csv(os.path.join(RESULTS,'visualization_evaluation.csv'),index=False)
    print('\n=== COMPLETE ==='); print('Dataset files:',DATA); print('Results:',RESULTS)
if __name__=='__main__': main()
