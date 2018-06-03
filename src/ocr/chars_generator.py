from utils import char_mapper

# print(char_mapper.vocab_letter_to_class)

alphabet_lower = list(char_mapper.alphabet_lower_mapper_letter_to_class.keys())
alphabet_lower = sorted(alphabet_lower)

alphabet_upper = list(char_mapper.alphabet_upper_mapper_letter_to_class.keys())
alphabet_upper = sorted(alphabet_upper)

numbers = list(char_mapper.number_letter_to_class.keys())
numbers = sorted(numbers)

specials = list(char_mapper.special_letter_to_class.keys())
specials = sorted(specials)

all_chars = [*alphabet_lower, *alphabet_upper, *numbers, *specials]

print("".join(all_chars))