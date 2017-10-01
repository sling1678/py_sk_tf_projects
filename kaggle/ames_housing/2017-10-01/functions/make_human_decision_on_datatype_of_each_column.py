def make_human_decision_on_datatype_of_each_column(df):


#Data columns (total 79 columns): dtypes: float64(3), int64(33), object(43)


  df.MSSubClass = df.MSSubClass.astype(str)
# 1460 non-null int64
# MSZoning         1460 non-null object
# LotFrontage      1201 non-null float64
# LotArea          1460 non-null int64
# Street           1460 non-null object
###### Alley            91 non-null object
# LotShape         1460 non-null object
# LandContour      1460 non-null object
# Utilities        1460 non-null object
# LotConfig        1460 non-null object
# LandSlope        1460 non-null object
# Neighborhood     1460 non-null object
# Condition1       1460 non-null object
# Condition2       1460 non-null object
# BldgType         1460 non-null object
# HouseStyle       1460 non-null object
# OverallQual      1460 non-null int64
  df.OverallCond = df.OverallCond.astype(str)  #1460 non-null int64
# YearBuilt        1460 non-null int64
# YearRemodAdd     1460 non-null int64
# RoofStyle        1460 non-null object
# RoofMatl         1460 non-null object
# Exterior1st      1460 non-null object
# Exterior2nd      1460 non-null object
# MasVnrType       1452 non-null object
# MasVnrArea       1452 non-null float64
# ExterQual        1460 non-null object
# ExterCond        1460 non-null object
# Foundation       1460 non-null object
# BsmtQual         1423 non-null object
# BsmtCond         1423 non-null object
# BsmtExposure     1422 non-null object
# BsmtFinType1     1423 non-null object
# BsmtFinSF1       1460 non-null int64
# BsmtFinType2     1422 non-null object
# BsmtFinSF2       1460 non-null int64
# BsmtUnfSF        1460 non-null int64
# TotalBsmtSF      1460 non-null int64
# Heating          1460 non-null object
# HeatingQC        1460 non-null object
# CentralAir       1460 non-null object
# Electrical       1459 non-null object
# 1stFlrSF         1460 non-null int64
# 2ndFlrSF         1460 non-null int64
# LowQualFinSF     1460 non-null int64
# GrLivArea        1460 non-null int64
  df.BsmtFullBath = df.BsmtFullBath.astype(str) #  1460 non-null int64
  df.BsmtHalfBath = df.BsmtHalfBath.astype(str) #  1460 non-null int64
  df.FullBath = df.FullBath.astype(str)    #   1460 non-null int64
  df.HalfBath = df.HalfBath.astype(str)    #   1460 non-null int64
# BedroomAbvGr     1460 non-null int64
# KitchenAbvGr      1460 non-null int64
# KitchenQual      1460 non-null object
# TotRmsAbvGrd     1460 non-null int64
# Functional       1460 non-null object
# Fireplaces       1460 non-null int64
# FireplaceQu      770 non-null object
# GarageType       1379 non-null object
# GarageYrBlt      1379 non-null float64
# GarageFinish     1379 non-null object
# GarageCars       1460 non-null int64
# GarageArea       1460 non-null int64
# GarageQual       1379 non-null object
# GarageCond       1379 non-null object
# PavedDrive       1460 non-null object
# WoodDeckSF       1460 non-null int64
# OpenPorchSF      1460 non-null int64
# EnclosedPorch    1460 non-null int64
# 3SsnPorch        1460 non-null int64
# ScreenPorch      1460 non-null int64
# PoolArea         1460 non-null int64
####### PoolQC           7 non-null object
####### Fence            281 non-null object
####### MiscFeature      54 non-null object
# MiscVal          1460 non-null int64
# MoSold           1460 non-null int64
# YrSold           1460 non-null int64
# SaleType         1460 non-null object
# SaleCondition    1460 non-null object
  return df