import hatchet as ht 
import sys

X = int(sys.argv[1])

gf = ht.GraphFrame.from_caliper("lulesh-64cores.json")
gf = gf.load_imbalance(verbose=True)
df = gf.dataframe.sort_values(by='time.imbalance', ascending=False)

target_row = df.iloc[X - 1] 
processes = target_row['time.ranks']

print(processes)