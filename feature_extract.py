from openfe import OpenFE, transform, tree_to_formula
import pandas as pd
import matplotlib.pyplot as plt

def label_dataset():
    df = pd.read_pickle("./data/train/month/2022/2022_01_1m.pkl")
    df['position'] = None
    
    for interval, group in df.resample('12H'):
        if not group.empty:
            max_idx = group['close'].idxmax()
            min_idx = group['close'].idxmin()
            
            df.at[max_idx, 'position'] = 'Short'
            df.at[min_idx, 'position'] = 'Long'
    
    return df


def plot_data(df):
    # Plotting the price data
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Close', color='gray', alpha=0.7)

    # Highlighting Long positions with green dots
    long_positions = df[df['position'] == 'Long']
    plt.scatter(long_positions.index, long_positions['close'], color='green', label='Long', marker='^', s=100)

    # Highlighting Short positions with red dots
    short_positions = df[df['position'] == 'Short']
    plt.scatter(short_positions.index, short_positions['close'], color='red', label='Short', marker='v', s=100)

    # Adding titles and labels
    plt.title('Price Data with Long and Short Positions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Display the plot
    plt.show()


def find_features(df):
    ofe = OpenFE()
    features = ofe.fit(data=df, label=df['position'])

    for feature in ofe.new_features_list[:10]:
        print(tree_to_formula(feature))


if __name__ == "__main__":
    df = label_dataset()
    plot_data(df)
    # find_features(df)
