from transformers import BartTokenizerFast

input = "d(20sin^8(4n^7))/dn=4480n^6*cos(4n^7)*sin^7(4n^7)".replace("d(", "").replace(")/d", "")

tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
tokenizer.add_tokens(" *")
a = tokenizer.tokenize(input)
print(a)
input_id = tokenizer(text=input)['input_ids']
print(input_id)
for i in input_id:
    print(tokenizer.convert_ids_to_tokens(i))
