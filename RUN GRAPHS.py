import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv(r'C:\Users\DELL\Desktop\AsthmaDetectionProject\asthma_disease_data.csv')

# Define numerical features
numerical_features = ['Age', 'BMI', 'LungFunctionFVC', 'LungFunctionFEV1', 'PollenExposure',
                      'SleepQuality', 'PollutionExposure', 'DietQuality', 'DustExposure', 'PhysicalActivity']

# Function to plot and save box plots
def save_box_plots(features, filename):
    plt.figure(figsize=(20, 5))  # Adjust height to 5 for horizontal layout
    for i, feature in enumerate(features):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(data[feature])
        plt.title(f'Box Plot of {feature}')
        plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Split the numerical features into chunks of 3
for i in range(0, len(numerical_features), 3):
    chunk = numerical_features[i:i+3]
    save_box_plots(chunk, f'C:\\Users\\DELL\\Desktop\\AsthmaDetectionProject\\boxplot_set_{i//3 + 1}.png')

# Define selected features for pair plots
selected_features = ['Age', 'BMI', 'LungFunctionFVC', 'LungFunctionFEV1', 'PollenExposure',
                     'SleepQuality', 'PollutionExposure', 'DietQuality', 'DustExposure', 'PhysicalActivity', 'Diagnosis']

# Function to plot pair plots and save images
def save_pair_plots(features, filename):
    # Ensure 'Diagnosis' is present in the data and correctly formatted
    pairplot_data = data[features].copy()
    if 'Diagnosis' in pairplot_data.columns:
        pairplot_data['Diagnosis'] = pairplot_data['Diagnosis'].astype(str)
    sns.pairplot(pairplot_data, hue='Diagnosis')
    plt.savefig(filename)
    plt.close()

# Check that all selected features are in the DataFrame
missing_features = [feature for feature in selected_features if feature not in data.columns]
if missing_features:
    raise ValueError(f"Missing features in data: {missing_features}")

# Split the selected features into chunks of 3 for pair plots
for i in range(0, len(selected_features) - 1, 3):  # Exclude 'Diagnosis' from chunking
    chunk = selected_features[i:i+3] + ['Diagnosis']  # Add 'Diagnosis' to each chunk
    save_pair_plots(chunk, f'C:\\Users\\DELL\\Desktop\\AsthmaDetectionProject\\pairplot_set_{i//3 + 1}.png')
