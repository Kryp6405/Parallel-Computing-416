import hatchet as ht
import sys

N = int(sys.argv[1])

gf8 = ht.GraphFrame.from_caliper("lulesh-8cores.json")
gf64 = ht.GraphFrame.from_caliper("lulesh-64cores.json")

gf8.drop_index_levels()
gf64.drop_index_levels()

gf_diff = gf64 - gf8
df_diff = gf_diff.dataframe
df_diff = df_diff[df_diff['time'] > 0]
sorted_df_diff = df_diff.sort_values(by='time', ascending=False)

top_n_diff = sorted_df_diff.head(N)
for idx, row in top_n_diff.iterrows():
    func_name = row['name']
    time_diff = row['time']
    print(f"{func_name} {time_diff}")
