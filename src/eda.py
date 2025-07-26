import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_basic_info(df):
    print("📊 데이터 정보:")
    print(df.info())
    print("\n📈 통계 요약:")
    print(df.describe())

def show_class_distribution(df):
    if 'label' in df.columns:
        print("\n🎯 클래스 분포:")
        print(df['label'].value_counts())

        sns.countplot(x='label', data=df)
        plt.title('Label Distribution')
        plt.show()

def plot_sensor_data(df, n_rows=500):
    cols = [col for col in df.columns if col not in ['label', 'time']]
    if len(cols) == 0:
        print("📉 시각화할 센서 데이터가 없습니다.")
        return

    df_plot = df.head(n_rows)
    plt.figure(figsize=(12, 6))
    for col in cols:
        plt.plot(df_plot[col], label=col)
    plt.legend()
    plt.title(f'센서 데이터 (상위 {n_rows}개 샘플)')
    plt.xlabel('샘플 인덱스')
    plt.ylabel('센서값')
    plt.grid()
    plt.show()
