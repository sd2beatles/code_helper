# 1. Time
## 1.1 Time Conversion

```python
#data time is now converted into the format as 'year-month-date'
pd.to_datetime(data['column'])
```

For more specific function to use, we need to implement the followings
![image](https://user-images.githubusercontent.com/53164959/83241135-5a1a4b00-a1d5-11ea-8ff0-850a5894d4ba.png)

![image](https://user-images.githubusercontent.com/53164959/83241168-6a322a80-a1d5-11ea-9086-c9009d756015.png)

![image](https://user-images.githubusercontent.com/53164959/83241196-74ecbf80-a1d5-11ea-8099-48f2b4ca4274.png)

## 1.2 Time Type

``` python
data['time_column']=data['time_column'].astype('datatime64[ns]')

```


## 2. Json File Read

```
def json_read(path,columns):
    df = pd.read_csv(path, 
                     converters={column: json.loads for column in columns}, # loading the json columns properly
                     dtype={'fullVisitorId': 'str'})# Number of rows that will be imported randomly
    
    for column in columns: #loop to finally transform the columns in data frame
        #It will normalize and set the json to a table
        column_as_df = json_normalize(df[column]) 
        # here will be set the name using the category and subcategory of json columns
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns] 
        # after extracting the values, let drop the original columns
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
    return df 

```
