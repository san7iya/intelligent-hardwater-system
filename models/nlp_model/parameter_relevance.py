import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt

df = pd.read_csv("data/synthetic_complaints_dataset.csv")

def clean_text(complaint_text):
    complaint_text = str(complaint_text).lower()
    complaint_text = re.sub(r'[^a-z\s]', ' ', complaint_text)
    return complaint_text

df['cleaned_text'] = df['complaint_text'].apply(clean_text)

parameters = {
    'Hardness': [
        'hard', 'hardness', 'mineral', 'scale', 'scaling', 'calcium', 'magnesium',
        'dry hair', 'white stains', 'soap', 'chalky', 'spots on dishes', 'residue',
        'rough water', 'lime', 'build up', 'deposit'
    ],
    'Solids': [
        'tds', 'total dissolved solids', 'salty', 'salinity', 'minerals', 'taste salty',
        'brackish', 'saline', 'metallic taste', 'saline taste', 'salty taste'
    ],
    'Chloramines': [
        'chlorine', 'chloramine', 'smell', 'bleachy', 'swimming pool smell', 
        'chemical smell', 'disinfectant', 'odor', 'bleach', 'chlorinated', 
        'strong smell', 'chemical odor'
    ],
    'Sulfate': [
        'sulfate', 'sulphate', 'bitter taste', 'laxative effect', 'taste bitter', 
        'rotten egg smell', 'sulfur', 'sulphur', 'egg smell'
    ],
    'Conductivity': [
        'conductivity', 'metallic taste', 'electric', 'shock', 'current', 'static',
        'tingling water', 'metal taste'
    ],
    'Organic_carbon': [
        'organic', 'carbon', 'contamination', 'decay', 'rotting', 'mold', 'fungus', 
        'musty smell', 'algae', 'biofilm', 'dirty tank'
    ],
    'Trihalomethanes': [
        'trihalomethane', 'chemical', 'disinfectant', 'toxic', 'byproduct', 
        'chemical residue', 'strong odor', 'eye irritation'
    ],
    'Turbidity': [
        'turbid', 'murky', 'cloudy', 'clear', 'muddy', 'dirty', 'unclear', 
        'foggy', 'sediment', 'particles', 'floating stuff', 'visible dirt', 
        'brown water', 'yellow water'
    ]
}

param_counts = Counter({param: 0 for param in parameters.keys()})

for complaint_text in df['cleaned_text']:
    for param, keywords in parameters.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', complaint_text):
                param_counts[param] += 1

count_df = pd.DataFrame(param_counts.items(), columns=['Parameter', 'Mentions'])
count_df = count_df.sort_values(by='Mentions', ascending=False)

print("\nParameter Relevance Based on NLP Mentions")
print(count_df)

output_path = "parameter_relevance.csv"
count_df.to_csv(output_path, index=False)
print(f"\nRanked parameter relevance saved to '{output_path}'")

plt.figure(figsize=(8, 5))
plt.barh(count_df['Parameter'], count_df['Mentions'], color='skyblue')
plt.xlabel('Number of Mentions')
plt.ylabel('Parameter')
plt.title('Relevance of Water Quality Parameters (based on NLP dataset)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

