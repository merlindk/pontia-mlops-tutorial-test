from sklearn.model_selection import train_test_split

def build_features(df):
    df = df.dropna()
    X = df.drop('income', axis=1)
    y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    return train_test_split(X, y, test_size=0.2, random_state=42)
