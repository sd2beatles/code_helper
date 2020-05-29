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
