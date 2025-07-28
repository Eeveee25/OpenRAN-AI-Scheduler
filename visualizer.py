import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_results():
    """
    Loads simulation results and creates plots to compare AI and Baseline performance.
    """
    df = pd.read_csv('simulation_results.csv')

    sns.set_theme(style="whitegrid")

    # Plot 1: Throughput over Time
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['ai_throughput'].rolling(window=50).mean(), label='AI-Driven (Smoothed)', color='blue')
    plt.plot(df.index, df['pf_throughput'].rolling(window=50).mean(), label='Proportional Fairness (Smoothed)', color='orange', linestyle='--')
    plt.title('System Throughput Comparison (50-step Moving Average)')
    plt.xlabel('Time Step')
    plt.ylabel('Total Throughput (Mbps)')
    plt.legend()
    plt.savefig('throughput_comparison.png')
    plt.show()

    # Plot 2: Fairness over Time
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['ai_fairness'].rolling(window=50).mean(), label='AI-Driven (Smoothed)', color='blue')
    plt.plot(df.index, df['pf_fairness'].rolling(window=50).mean(), label='Proportional Fairness (Smoothed)', color='orange', linestyle='--')
    plt.title("Jain's Fairness Index Comparison (50-step Moving Average)")
    plt.xlabel('Time Step')
    plt.ylabel('Fairness Index (1 = Perfect Fairness)')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig('fairness_comparison.png')
    plt.show()

    # Plot 3: Summary Statistics Bar Chart
    summary = {
        'Method': ['AI-Driven', 'Proportional Fairness'],
        'Average Throughput (Mbps)': [df['ai_throughput'].mean(), df['pf_throughput'].mean()],
        'Average Fairness Index': [df['ai_fairness'].mean(), df['pf_fairness'].mean()]
    }
    df_summary = pd.DataFrame(summary)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=df_summary, x='Method', y='Average Throughput (Mbps)', ax=ax[0], palette='viridis')
    ax[0].set_title('Average System Throughput')

    sns.barplot(data=df_summary, x='Method', y='Average Fairness Index', ax=ax[1], palette='plasma')
    ax[1].set_title('Average Fairness Index')
    ax[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('summary_kpis.png')
    plt.show()

if __name__ == '__main__':
    visualize_results()