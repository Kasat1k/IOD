import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from statistics import mode

data_file = "data.txt"
raw_data = np.loadtxt(data_file, delimiter=',')

def get_stat(data):
    mean = np.mean(data)
    harmonic_mean = np.mean(1 / data)
    geometric_mean = np.exp(np.mean(np.log(np.abs(data))))
    dispersion = np.std(data)
    gini_diff = np.mean(np.abs(np.subtract.outer(data, data)))
    median = np.median(data)
    res_mode = mode(data)
    skewness = np.mean((data - mean) ** 3) / (dispersion ** 3)
    excess = np.mean((data - mean) ** 4) / (dispersion ** 4)

    standardized_data = (data - np.mean(data)) / np.std(data)
    stat, p = normaltest(standardized_data)

    return {
        "середнє арифметичне": mean,
        "середнє гармонійне": harmonic_mean,
        "середнє геометричне": geometric_mean,
        "дисперсія": dispersion,
        "коефіцієнт Джині": gini_diff,
        "медіана": median,
        "мода": res_mode,
        "коефіцієнт асиметрії": skewness,
        "коефіцієнт ексцесу": excess,
        "нормальний закон розподілу": (stat, p),
        "standardized_data": standardized_data
    }

def display_stat(stats, channel_index):
    print(f"Канал {channel_index + 1} Statistics:")
    for key, value in stats.items():
        if key in ['середнє арифметичне', 'середнє гармонійне', 'середнє геометричне', 'дисперсія', 'коефіцієнт Джині', 'медіана', 'мода', 'коефіцієнт асиметрії', ' коефіцієнт ексцесу']:
            print(f"{key.capitalize()}: {value:.2f}")
        elif key == 'нормальний закон розподілу':
            print(f"Normal Test Statistic: {value[0]:.2f}, p-value: {value[1]:.3f}")

    # Plotting histogram
    plt.figure(figsize=(10, 4))
    plt.hist(stats['standardized_data'], bins=30, alpha=0.7, color='blue')
    plt.title(f"Канал  {channel_index + 1} Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    # Plotting cardiogram
    plt.figure(figsize=(10, 4))
    plt.plot(stats['standardized_data'], color='red')
    plt.title(f"Channel {channel_index + 1} Cardiogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()

    print()

if __name__ == "__main__":
    raw_data = raw_data.T  # Transpose to iterate over channels
    for i, channel_data in enumerate(raw_data):
        stats = get_stat(channel_data)
        display_stat(stats, i)
