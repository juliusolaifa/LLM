from tokenizers import Tokenizer
from tokenizers.models import BPE  #'BPE', 'Model', 'Unigram', 'WordLevel', 'WordPiece'
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from datasets import load_dataset

special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

tokenizer = Tokenizer(BPE(unk_token=special_tokens[0]))
trainer = BpeTrainer(special_tokens=special_tokens)
tokenizer.pre_tokenizer = Whitespace()

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

tokenizer.train_from_iterator(dataset["train"]["text"], trainer)
tokenizer.save("tokenizer.json")

tokenizer = Tokenizer.from_file("tokenizer.json")
input = tokenizer.encode("Hello, world!")
