from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")

###
# First try, using parrot T5 paraphraser: https://huggingface.co/prithivida/parrot_paraphraser_on_T5
# There also more paraphraser at huggingface spaces at the link above
# Paraphraser makes sense since we want encoder-decoder that retain the meaning of the text
###

def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)

parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

phrases = ["Can you recommed some upscale restaurants in Newyork?",
           "What are the famous places we should not miss in Russia?"
]
parrot.rephrase("testing")
for phrase in phrases:
  print("-"*100)
  print("Input_phrase: ", phrase)
  print("-"*100)
  para_phrases = parrot.augment(input_phrase=phrase)
  for para_phrase in para_phrases:
   print(para_phrase)