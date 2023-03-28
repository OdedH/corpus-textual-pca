from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")

###
# First try, using parrot T5 paraphraser: https://huggingface.co/prithivida/parrot_paraphraser_on_T5
# There also more paraphraser at huggingface spaces at the link above
# Paraphraser makes sense since we want encoder-decoder that retain the meaning of the text
###

