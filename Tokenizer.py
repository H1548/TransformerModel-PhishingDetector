from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

trainer = BpeTrainer(special_tokens=["[UNK]", "[BOS]", "[SEP]", "[PAD]", "[MASK]","[EOS]"], vocab_size=35000, show_progress = True )
tokenizer.pre_tokenizer = Whitespace()



file= ["bigger_sampled_OWT.txt"]


tokenizer.train(file,trainer)


tokenizer.save("tokenizer.json")
