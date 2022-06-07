initial_balance = 100000
strike_price = 100
premium = 2
no_of_shares = 1000
for i in range(100,200):
    if(i < premium +strike_price):
        profit = -premium*no_of_shares
    else:
        profit = no_of_shares*(i - (premium +strike_price))
    per_inc = (i- strike_price)
    per_inc = per_inc*100/strike_price
    print(per_inc,profit)