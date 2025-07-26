import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """CSV 파일을 불러와서 DataFrame으로 반환"""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """기본 전처리: 결측치 제거 + 라벨 인코딩"""
    df = df.dropna()
    
    if 'label' in df.columns:
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])
        
    return df

if __name__ == "__main__":
    # 테스트용 코드
    test_df = load_data("../data/sample.csv")
    clean_df = preprocess_data(test_df)
    print(clean_df.head())
