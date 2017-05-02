import numpy as np
import argparse
from collections import Counter

parser = argparse.ArgumentParser()

parser.add_argument("--cluster_num", action ="append", help="the number of clusters")
parser.add_argument("--style_layers",help="the number of style layers",type = int)


args = parser.parse_args()
print('Enter python')


cluster_num = args.cluster_num
style_layers = args.style_layers

print(cluster_num)
k = cluster_num[0]
lt = k.split(',')
print(lt)
for i in range(len(lt)):
    print(int(lt[i]))

print(style_layers)
