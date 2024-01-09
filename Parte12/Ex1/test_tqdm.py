#!/usr/bin/env python3


from time import sleep
from tqdm import tqdm


names = [' miguel', ' ze', ' bruno', ' carolina']
names2 = {
    "Miguel": ['gay'],
    "Mestre":['megagay']    
}
for name in tqdm(names2.items()):
    print(name)
    sleep(1)
