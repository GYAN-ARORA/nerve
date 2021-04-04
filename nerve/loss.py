def mse(error):
    return (error**2).mean()

def rmse(error):
    return mse(error)**0.5

