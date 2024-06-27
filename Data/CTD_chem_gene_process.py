import pandas as pd

# Load the CSV file
csv_file = 'Data/CTD_chem_gene_interaction.csv'  # Replace with your actual file path
df = pd.read_csv(csv_file)

# DataFrame 1: InteractionActions count for each Gene
# Split InteractionActions and explode
df1 = df.copy()
df1['InteractionActions'] = df1['InteractionActions'].str.split('|')
df1 = df1.explode('InteractionActions')

# Create a pivot table
df1 = df1.pivot_table(index='GeneSymbol', columns='InteractionActions', aggfunc='size', fill_value=0)

# DataFrame 2: Chemical appearance for each Gene
# Create a pivot table
#df2 = df.pivot_table(index='GeneSymbol', columns='ChemicalID', aggfunc='size', fill_value=0)


def filter_columns_by_sum(df, percentage_to_eliminate):
    column_sums = df.sum()
    sorted_columns = column_sums.sort_values(ascending=False)
    num_columns_to_eliminate = int(len(sorted_columns) * (percentage_to_eliminate / 100))
    columns_to_keep = sorted_columns.index[:-num_columns_to_eliminate]
    filtered_df = df[columns_to_keep]
    return filtered_df

# Set the percentage of columns to eliminate
percentage_to_eliminate1 = 70  # Replace with your desired percentage
percentage_to_eliminate2 = 50

# Filter columns in both DataFrames
df1_filtered = filter_columns_by_sum(df1, percentage_to_eliminate1)
#df2_filtered = filter_columns_by_sum(df2, percentage_to_eliminate2)

# Print summary of both DataFrames
print("Summary of DataFrame 1 after filtering:")
print(df1_filtered.describe())
#print("\nSummary of DataFrame 2 after filtering:")
#print(df2_filtered.describe())



# Save the DataFrames to CSV files (optional)
df1_filtered.to_csv('Data/CTD_chem_feature.csv')
#df2_filtered.to_csv('Data/CTD_Chem_Interaction/CTD_chem_gene_chemicals.csv')

# Display the DataFrames
print("DataFrame 1: InteractionActions count for each Gene")
print(df1.head())

#print("\nDataFrame 2: Chemical appearance for each Gene")
#print(df2.head())