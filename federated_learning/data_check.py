import pandas as pd
import numpy as np

benign = pd.read_csv("../csv_data/BenignTraffic.pcap_Flow.csv", low_memory=False)
attack = pd.read_csv("../csv_data/DDoS_TCP_SYN_Flood_attack.csv", low_memory=False)

print("Benign columns:", len(benign.columns))
print(list(benign.columns))

print("\nAttack columns:", len(attack.columns))
print(list(attack.columns))

benign_num = benign.select_dtypes(include=[np.number])
attack_num = attack.select_dtypes(include=[np.number])

print("\nBenign numeric:", len(benign_num.columns))
print(list(benign_num.columns))

print("\nAttack numeric:", len(attack_num.columns))
print(list(attack_num.columns))

shared = set(benign_num.columns).intersection(set(attack_num.columns))
print("\nShared numeric:", len(shared))
print(list(shared))