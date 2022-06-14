import random as rd
import pandas as pd
import numpy as np
from seaborn import boxplot
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

class Preprocessing_Data:
    color_list = ['azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']

# =====================================[redandant columns]==========================================
    def detected_redundant(df, threshold=80, drop_columns=False):
        df.insert(0,'idididid', df.index + 1)
        list_duplicated = []
        total = len(df)

        if total > 0:
            for col in df:
                d = df.groupby([col])['idididid'].count().max()
                present = (d/total)*100
                if present >= threshold:
                    list_duplicated.append(col)

        print('redundant features: ', list_duplicated)

        if drop_columns==True:
            df.drop(columns=list_duplicated, inplace=True, axis=1)
            print('redundant features are dropped')
        df.drop(['idididid'], axis=1, inplace=True)
        return df
    
# =====================================[symbol encoding]==========================================
    def detected_symbols(df, symbols =['!','?','#','$','%','^','&','*','~']):
        exists_symbol_col = df[df.isin(symbols)].count().reset_index()\
            .rename(columns={0:'count_symbols'}).where(lambda x: x['count_symbols'] > 0).dropna()
        return exists_symbol_col
    
# =====================================[check outlier]==========================================
    def detected_outliers(df, threshold=1.5, clampping=False):
        outlier_dict = dict()
        if len(df) > 0:
            for col in df:
                Q1, Q3 = np.quantile(df[col], [0.25, 0.75])
                IQR = Q3 - Q1

                lower_Q = Q1 - threshold * IQR
                upper_Q = Q3 + threshold * IQR
                # get outliers
                outlier_dict[col] = df[(df[col] < lower_Q) | (df[col] > upper_Q)].index.tolist()
                        
                # clamping outliers
                if clampping == True and outlier_dict[col] != []:
                    df[col] = df[col].clip(lower=lower_Q, upper=upper_Q)
                    print('dropped outliers in column: ', col)
                if outlier_dict[col] != []:
                    print('- outliers in column: ', col)
        return df

# =====================================[boxplot]==========================================
    def boxplot(df, figsize=(None,None)):
        n = len(df.columns)
        plt.figure(figsize=figsize)
        for i in range(n):
            plt.subplot(round(n/2), 3, i+1)
            boxplot(data=df, y=df.columns[i], color=rd.choice(Preprocessing_Data().color_list));
        

    def histplot(df, bins=30):
        m,n=df.shape
        df.hist(figsize=(n,n), bins=bins)
# =====================================[determine precentage of null value ]==========================================
    def detected_null_value(df, presentage_of_null=0, drop_columns=False, drop_rows=False):
        total = len(df)
        # count null value
        _null = df.isna().sum().reset_index().rename(columns={0:'Counter'})
        # sorting descending by null value
        _null = _null[_null['Counter'] > 0].sort_values(by=['Counter'], ascending=False)
        # calculate percentage of null value
        _null = _null[(_null['Counter']/total)* 100 > presentage_of_null]
        # add presentage of null value
        _null['presentage of null value'] = round(_null['Counter']/total* 100, 2)

        if drop_columns==True:
            df.drop(columns=_null['index'].tolist(), inplace=True)
            print('drop columns: ', _null['index'].tolist())
        
        if drop_rows==True:
            df.dropna(subset=_null['index'].tolist(), axis=0, inplace=True)
            print('drop rows in : ', _null['index'].tolist())
        
        return _null


# =====================================[separated data]==========================================
    def separate_data(df):
        numeric = df.select_dtypes(include=np.number)
        categorical = df.select_dtypes(exclude=np.number)
        return numeric, categorical

#====================================[determine corr]==========================================
    def detected_highly_corr(df, target_name, threshold=70):
        # exclude categorical data, and target column
        num_cols_without_target = df.drop(columns=[target_name]).select_dtypes(include=np.number)
        # correlation matrix
        corr_matrix = num_cols_without_target.corr()
        # number of features
        n = len(corr_matrix)

        for row_index in range(n):
            for col_index in range(row_index+1, n):
                if abs(corr_matrix.values[row_index, col_index]) > threshold*100:
                    row_corr_with_target = abs(np.corrcoef(df[target_name], df[num_cols_without_target.columns[row_index]])[0, 1])
                    col_corr_with_target = abs(np.corrcoef(df[target_name], df[num_cols_without_target.columns[col_index]])[0, 1])
                    
                    if row_corr_with_target > col_corr_with_target:
                        print('we will drop the column ', num_cols_without_target.columns[col_index])
                        df = df.drop(columns = [num_cols_without_target.columns[col_index]])
                    else:
                        print('we will drop the column ', num_cols_without_target.columns[row_index])
                        df = df.drop(columns = [num_cols_without_target.columns[row_index]])
        return df

# =====================================[determine skewness]==========================================
    def skewness_calc(num_df, threshold=80, transform=False):
        mean = np.mean(num_df, axis=0)
        std = np.std(num_df, axis=0)
        n = len(num_df)
        col_list =[]

        skewness = np.sum((num_df - mean)**3, axis=0) / ((n-1) * std**3)

        for i in range(len(skewness)):
            if abs(skewness[i]*100) >= threshold:
                col_list.append(num_df.columns[i])
        print('-Skewness Columns: ', col_list)
                
                
        if transform==True:
            for col_name in col_list:
                num_df[col_name] = np.log(1 + num_df[col_name].abs())
                print('|+| Transformed: ', col_name)
                    
        return num_df

# =====================================[determine Shapiro-Wilk]==========================================
    def shapiro_test(df, alpha=0.001, n_sample=500):
        normal_dist = []
        non_normal_dist = []
        sample = df.sample(replace=False, n=n_sample)

        for col in df.columns:
            stat, p_value = shapiro(sample[col])
            if p_value > alpha:
                normal_dist.append(col)
            else:
                non_normal_dist.append(col)

        print(f'+ Column \'{normal_dist}\' is normally distributed (std)')
        print(f'- Column \'{non_normal_dist}\' is \'not\' normally distributed (minmax)')

# ====================================[Imbalance data] ========================================
    # def handling_binary_imbalance(df, columns=None, unique_values=None, frac_list=None, random_state=2020):
    #     df_out = pd.DataFrame()
    #     columns.value_counts().plot(kind='bar', cmap='Paired_r')
        
    #     for col in columns:
    #         for i, frac in zip(unique_values, frac_list):
    #             samples = df[df[col] == i].sample(frac=frac, random_state=random_state, replace=False)
    #             # print(samples)
    #             df_out = df_out.append(samples)
        
    #     columns.value_counts().plot(kind='bar', cmap='Paired_r')
    #     return df_out.sample(frac=1, random_state=201).reset_index(drop=True)

    def handling_binary_imbalance(df, columns_name=None, unique_values=None, frac_list=None, random_state=2020):
        df_out = pd.DataFrame()
        df[columns_name].value_counts().plot(kind='bar', cmap='Paired_r')
            
        for col in columns_name:
            for i, frac in zip(unique_values, frac_list):
                samples = df[df[col] == i].sample(frac=frac, random_state=random_state, replace=False)
                    # print(samples)
                df_out = df_out.append(samples)
            
        X = df_out.sample(frac=1, random_state=201).reset_index(drop=True)
        X[columns_name].value_counts().plot(kind='bar', cmap='Dark2')
        plt.title('plot Resampling ~ '+str(X[columns_name].value_counts().to_dict()))
        plt.legend(['plot before', 'plot after'])
        
        return X

# =====================================[Encoding Categorical]==========================================
    def encoding_ordinal_cat(ordinal_cats, df_train=None, df_test=None):
        
        if df_test is not None:
            ordinal_test_df = pd.DataFrame()    
        if df_train is not None:
            ordinal_train_df = pd.DataFrame()
        
        for col in ordinal_cats:
            label_enc_model = LabelEncoder()

            if df_train is not None:
                ordinal_train_df[col] = label_enc_model.fit_transform(df_train[col])
            if df_test is not None:
                ordinal_test_df[col] = label_enc_model.transform(df_test[col])
        
        if df_test is None and df_train is not None:
            return ordinal_train_df
        elif df_test is not None and df_train is None:
            return ordinal_test_df
        else:
            return ordinal_train_df, ordinal_test_df


    def encoding_nominal_cat(nominal_cats, df_train=None, df_test=None):
        one_hot_enc_model = OneHotEncoder(sparse = False)
        if df_train is not None:
            nominal_train_data = one_hot_enc_model.fit_transform(df_train[nominal_cats])
        if df_test is not None:
            nominal_test_data = one_hot_enc_model.transform(df_test[nominal_cats])

        col_names=[]
        
        for list_ in one_hot_enc_model.categories_:
            for element in list_:
                col_names.append(element)
        
        if df_train is not None:
            nominal_train_df = pd.DataFrame(nominal_train_data, columns=col_names)
        
        if df_test is not None:
            nominal_test_df = pd.DataFrame(nominal_test_data, columns=col_names)
        
        
        if df_test is None and df_train is not None:
            return nominal_train_df
        elif df_test is not None and df_train is None:
            return nominal_test_df
        else:
            return nominal_train_df, nominal_test_df

# =====================================[Scaling]==========================================
    def scaling(X_train, X_test, scaling_type='standard'):
        if scaling_type == 'standard':
            scaler = StandardScaler()
            return scaler.fit_transform(X_train), scaler.transform(X_test)
        elif scaling_type == 'minmax':
            scaler = MinMaxScaler()
            return scaler.fit_transform(X_train), scaler.transform(X_test)
        elif scaling_type == 'robust':
            scaler = RobustScaler()
            return scaler.fit_transform(X_train), scaler.transform(X_test)
        elif scaling_type == 'normal':
            scaler = Normalizer()
            return scaler.fit_transform(X_train), scaler.transform(X_test)


# =====================================[MultiColinearity]==========================================
    # def multi_colinearity(df, threshold=20, drop_cols=False):
    #     vif_data = pd.DataFrame()
    #     col_drop_list = []

    #     vif_data["feature"] = df.columns
    #     vif_data["VIF"] = [vif(df.values, i) for i in range(len(df.columns))]
            
    #     out_data = vif_data[vif_data['VIF'] > threshold]

    #     for col in out_data['feature']:
    #         col_drop_list.append(col)


    #     if drop_cols == True:
    #         df = np.delete(df.values ,col_drop_list, axis=1)
    #         print(f'We will drop the columns \'{col_drop_list}\'')
    #         return df
    #     else:
    #         return vif_data.sort_values(by="VIF", ascending=False)
    def multi_colinearity(df, threshold=20, drop_cols=False):
        vif_data = pd.DataFrame()
        col_drop_list = []

        vif_data["feature"] = df.columns
        vif_data["VIF"] = [vif(df.values, i) for i in range(len(df.columns))]
                
        out_data = vif_data[vif_data['VIF'] > threshold]

        for col in out_data['feature']:
            col_drop_list.append(col)


        if drop_cols == True:
            df = df.drop(col_drop_list, axis=1)
            print(f'We will drop the columns \'{col_drop_list}\'')
            return df
        else:
            return vif_data.sort_values(by="VIF", ascending=False)