from .sampler import data_sampler
from .collator import data_collator

ospnext_samplers = {}
ospnext_samplers.update(data_sampler)

ospnext_collators = {}
ospnext_collators.update(data_collator)