
# coding: utf-8

# In[ ]:


# Hypothesis: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the 
# ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession
# bottom.


# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy import stats


# In[2]:


def get_list_of_university_towns():
    
    # import data
    df = pd.read_table('university_towns.txt',header=None)

    dataStorage = []
    
# Cleaning the dataSet from inconsistent formating and unnecessary elements included in each record. The state
# records has the word '[edit]' included, i.e. Texas[edit]. The region records has unnecessary parentheses included, i.e.
# Paloma (...). Both state and region information are stored within the same unlabeled column.
     
    for element in df[0]:
    
        # Locate state. Split word on the first bracket. Store state name within variable. 
        if '[edit]' in element:
            x = element.split('[')[0]
        
        # Locate region. Split word on the first parenthese. Store both state and region record within list. 
        if ' (' in element:
            y = element.split(' (')[0]
            dataStorage.append((x,y))
            
        # Locate region that do not include either parentheses or brackets.Store both state and region record within list. 
        if ' (' not in element and '[edit]' not in element:
            dataStorage.append((x,element))
    
    # Convert list into a dataframe with labeled columns.
    tempDF = pd.DataFrame(dataStorage,columns=('State','RegionName'))

    return tempDF


# In[3]:


def get_recession_start():

    # Import Data.  
    df = pd.read_excel('gdplev.xls',skiprows=7)
        
    # Rename colomns that will be used.
    df = df.rename(index=str,columns={"Unnamed: 4":"Quarterly","Unnamed: 6":"GDP 2009_quarterly"})

    # Filter DataSet to include data from the first quarter year 2000  to latest record entry.
    df = df[df['Quarterly']>='2000q1']
    
    # Create new dataframe with specific columns. 
    df = df[['Quarterly','GDP 2009_quarterly']]

    # reset the index and drop the previous index. 
    df.reset_index(inplace=True)
    df.drop('index',axis=1,inplace=True)

    # Create new column within dataframe.
    df['GDP Change'] = 0

    # Stored the difference in GDP between quarters.
    for element in range(0,65):
        
        df['GDP Change'][element+1] = df.loc[element+1]['GDP 2009_quarterly'] - df.loc[element]['GDP 2009_quarterly']

        
    # Created a new column where the values will be either 'increase' or 'decline', which will be based on the GDP Change. 
    # This column will help determine where the recession began. A recession is determined to be when there's a decline in 
    # the GDP for two consecutive quarters.
    df['Change'] = 'NaN'
    x=0

    for element in df['GDP Change']:
        
        if element >= 0:
            df['Change'].loc[x] = 'increase'
            x+=1
            
        else:
            df['Change'].loc[x] = 'decline'
            x+=1


    # Filtering through the newly added column 'Change' to identify the quarter where the GDP was in a decline for two
    # consecutive quarters.
    
    s1='decline'
    s2='increase'
    x=0
    dataStorage=[]
    for element in df['Change']:

        if element==s1 and df['Change'].loc[x+1]==s1:

            recession_start = df.loc[x]['Quarterly']
            before_recession = df.loc[x-1]['Quarterly']
            break
            
        else:
            x+=1    
            
            
    # Identifying the quarter where the recession ended, which is when the GDP is in a decline for two consecutive 
    # quarters followed by an increase in two consecutive quarters.
    x=0
    for element in df['Change']:

        if element==s1 and df['Change'].loc[x+1]==s1 and df['Change'].loc[x+2]==s2 and df['Change'].loc[x+3]==s2:

            recession_end = df.loc[x+3]['Quarterly']
            break

        else:
            x+=1
            
    # Identifying the quarter where the recession bottom occurs, which is when the GDP is in a decline for two consecutive 
    # quarters followed by an increase in the following quarter.
    x=0
    for element in df['Change']:

        if element==s1 and df['Change'].loc[x+1]==s1 and df['Change'].loc[x+2]==s2:

            recession_bottom = df.loc[x+1]['Quarterly']
            break

        else:
            x+=1

    
    return recession_start,before_recession,recession_end,recession_bottom


# In[4]:


def convert_housing_data_to_quarters():
    
    # import dataSet
    df = pd.read_csv('City_Zhvi_AllHomes.csv')
    
    # keys that will be used to change the state's appreviation to the state's full name.
    states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 
              'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 
              'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia',
              'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii',
              'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona',
              'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas',
              'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri',
              'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas',
              'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California',
              'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island',
              'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia',
              'ND': 'North Dakota', 'VA': 'Virginia'}

    # Mapping the keys to the 'State' column.
    df['State'] = df['State'].map(states)

    # Setting the state and region name to be the index of the dataframe. 
    df.set_index((['State','RegionName']),inplace=True)

    x=df.columns

    # Selecting the names of the columns from year 2000 to 2016. 
    sel_cols = x[(x>='2000-01') & (x<='2016-08')]

    # Used the previous column selection to create a new dataframe.
    df = df[sel_cols]

    # Columns converted from string to date_time. This is done in order to use the 'resample()' function.
    df = df[df.columns].rename(columns=pd.to_datetime)

    # resampled the data in quarterly intervals to determine the mean.
    mdf = df[df.columns].resample('Q',axis=1).mean()

    # Converted the columns from a date_time back to a string, in it's original formate of '%Y-%M' and stored it.
    sColumns = mdf.columns.strftime('%Y-%m')

    # stored the columns as a date_time
    tsColumns =  mdf.columns

    counter=0

    
    # Renamed the date_time column back to it's series name for the dataframe.
    for element in range(0,len(sColumns)):

        mdf = mdf.rename(index=str,columns={tsColumns[counter]:sColumns[counter]},)
        counter+=1

    # Stored the columns to the new dataframe.
    x1 = mdf.columns 

    # Split each column name by a dash '-' in order to store the year and month separately in a list.
    # i.e. 2015-01 -> ['2015','01']
    for element in range(0,len(x1)):

        x = x1[element]

        y = x1[element].split('-')

        # Converting the month, second element in the list, to it's respective quarter then renaming the dataframe column
        # to it's updated format. i.e. 2015-01 -> ['2015','01'] -> 2015q1
        if y[1]=='01' or  y[1]=='02' or  y[1]=='03':

            y[1] = 'q1'
            y = y[0]+y[1]

            mdf = mdf.rename(index=str,columns={x:y},)

        elif y[1]=='04' or  y[1]=='05' or  y[1]=='06':

            y[1] = 'q2'
            y = y[0]+y[1]

            mdf = mdf.rename(index=str,columns={x:y},)

        elif y[1]=='07' or  y[1]=='08' or  y[1]=='09':

            y[1] = 'q3'
            y = y[0]+y[1]

            mdf = mdf.rename(index=str,columns={x:y},)

        elif y[1]=='10' or  y[1]=='11' or  y[1]=='12':

            y[1] = 'q4'
            y = y[0]+y[1]

            mdf = mdf.rename(index=str,columns={x:y},)
            
    return mdf


# In[5]:



def run_ttest():

    # Imported data from previously created functions that cleaned and formatted the dataset.

    # rs -> recession start, br -> before recession, re -> recession end, rbb -> recession bottom
    rs,br,re,rb = get_recession_start() 
    ut = get_list_of_university_towns()
    house_df = convert_housing_data_to_quarters()

    # Filtering the housing dataset to include the quarter before the recession began and the recession bottom.
    house_df = house_df[[br,rb]]

    # Created a price ratio column to compare the pricing of homes before the recession occured to the end of the recession.
    house_df['PriceRatio'] = house_df[br].div(house_df[rb])


    # Convert the records of university town, excluding the index, to be stored as a list. i.e. [(state,region)...]
    subset_list = ut.to_records(index=False).tolist()

    # Created new dataframe to include records in house_df that exist in university towns.
    university_towns = house_df.loc[subset_list] 

    # Created new dataframe to includes records of house_df but excluding those that exist in university towns. For example, 
    # (Alaska, Fairbanks) exist in university_towns therefore it will not be included in non_university_towns.
    non_university_towns = house_df.loc[~house_df.index.isin(subset_list)] 

    # Stored the statistic value and p value from the T-Test.  
    stat,pvalue = stats.ttest_ind(university_towns['PriceRatio'],non_university_towns['PriceRatio'],nan_policy='omit')

    # pvalue condition statement. If less then 0.01 then we can reject the 'null hypothesis'. If above 0.01 then we failed  
    # to reject the 'null hypothesis'.
    different = pvalue < 0.01


    # Compared the mean value of price ratio of university_towns vs non_university_towns in order to determine which group 
    # was less affected in terms of housing price.
    if university_towns['PriceRatio'].mean() < non_university_towns['PriceRatio'].mean():

        better = 'university town'

    else:

        better = 'non-university town'


    return (different, pvalue, better)


# In[ ]:


'''
Full Disclosure: The project was completed as part of a milestone project for a Coursera Data Scientist course.  
'''

