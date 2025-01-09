import hatchet as ht
import sys

N = int(sys.argv[1])

gf = ht.GraphFrame.from_caliper("lulesh-1core.json")
df = gf.dataframe

sorted_df = df.sort_values(by='time', ascending=False)
top_n = sorted_df.head(N)

for idx, row in top_n.iterrows():
    func_name = row['name']
    time_exc = row['time']
    print(f"{func_name} {time_exc}")