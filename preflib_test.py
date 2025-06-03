import os
import numpy as np
from preflibtools.instances import OrdinalInstance
from preflibtools.properties import borda_scores, has_condorcet
from voting_rules import *

instance = OrdinalInstance()
instance.parse_file("voting_data/type_soc/breakfast_prefs_overall.soc")

new_entries = [((12,), (11,), (4,), (6,), (5,), (13,), (3,), (7,), (14,), (9,), (8,), (2,), (1,), (15,), (10,)), ((14,), (6,), (3,), (15,), (11,), (1,), (12,), (5,), (8,), (9,), (4,), (13,), (7,), (10,), (2,))]
#instance.append_order_list(new_entries)
flat = instance.flatten_strict()

#print(borda_scores(instance))
#print(borda(instance))
print(IRV(instance))
