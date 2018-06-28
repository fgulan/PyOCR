# -*- coding: utf-8 -*-

alphabet_upper_mapper_letter_to_class = {
    'A': 'upper_a',
    'B': 'upper_b',
    'D': 'upper_d',
    'Đ': 'upper_d1',
    'E': 'upper_e',
    'F': 'upper_f',
    'G': 'upper_g',
    'H': 'upper_h',
    'I': 'upper_i',
    'J': 'upper_j',
    'K': 'upper_k',
    'L': 'upper_l',
    'M': 'upper_m',
    'N': 'upper_n',
    'P': 'upper_p',
    'R': 'upper_r',
    'T': 'upper_t',
    'U': 'upper_u',
    'Q': 'upper_q',
}

alphabet_lower_mapper_letter_to_class = {
    'a': 'lower_a',
    'b': 'lower_b',
    'd': 'lower_d',
    'đ': 'lower_d1',
    'e': 'lower_e',
    'f': 'lower_f',
    'g': 'lower_g',
    'h': 'lower_h',
    'i': 'lower_i',
    'j': 'lower_j',
    'k': 'lower_k',
    'l': 'lower_l',
    'm': 'lower_m',
    'n': 'lower_n',
    'p': 'lower_p',
    'r': 'lower_r',
    't': 'lower_t',
    'u': 'lower_u',
    'q': 'lower_q',
}

unique_letter_to_class = {
    'W': 'unique_w',
    'V': 'unique_v',
    'Z': 'unique_z',
    'Ž': 'unique_z1',
    'X': 'unique_x',
    'Y': 'unique_y',
    'S': 'unique_s',
    'Š': 'unique_s1',
    'C': 'unique_c',
    'Č': 'unique_c1',
    'Ć': 'unique_c2',
    'O': 'unique_o',

    'w': 'unique_w',
    'x': 'unique_x',
    'v': 'unique_v',
    'z': 'unique_z',
    'ž': 'unique_z1',
    's': 'unique_s',
    'š': 'unique_s1',
    'o': 'unique_o',
    'c': 'unique_c',
    'č': 'unique_c1',
    'ć': 'unique_c2',
    'y': 'unique_y',
}

unique_small_letter_to_class = {
    'w': 'unique_w',
    'x': 'unique_x',
    'v': 'unique_v',
    'z': 'unique_z',
    'ž': 'unique_z1',
    's': 'unique_s',
    'š': 'unique_s1',
    'o': 'unique_o',
    'c': 'unique_c',
    'č': 'unique_c1',
    'ć': 'unique_c2',
    'y': 'unique_y',
}

number_letter_to_class = {
    '0': 'number_0',
    '1': 'number_1',
    '2': 'number_2',
    '3': 'number_3',
    '4': 'number_4',
    '5': 'number_5',
    '6': 'number_6',
    '7': 'number_7',
    '8': 'number_8',
    '9': 'number_9',
}

special_letter_to_class = {
    '.': 'special_dot',
    ',': 'special_comma',
    '?': 'special_question',
    '!': 'special_exclamation',
    '-': 'special_minus',
    '(': 'special_left_bracket',
    ')': 'special_right_bracket'
}

classifier_out_to_class = {
    0: 'lower_a', 1: 'lower_b', 2: 'lower_d', 
    3: 'lower_d1', 4: 'lower_e', 5: 'lower_f', 
    6: 'lower_g', 7: 'lower_h', 8: 'lower_i', 
    9: 'lower_j', 10: 'lower_k', 11: 'lower_l', 
    12: 'lower_m', 13: 'lower_n', 14: 'lower_p', 
    15: 'lower_q', 16: 'lower_r', 17: 'lower_t', 
    18: 'lower_u', 19: 'number_0', 20: 'number_1', 
    21: 'number_2', 22: 'number_3', 23: 'number_4', 
    24: 'number_5', 25: 'number_6', 26: 'number_7', 
    27: 'number_8', 28: 'number_9', 29: 'special_comma',
    30: 'special_dot', 31: 'special_exclamation', 
    32: 'special_left_bracket', 33: 'special_minus', 
    34: 'special_question', 35: 'special_right_bracket', 
    36: 'unique_c', 37: 'unique_c1', 38: 'unique_c2', 
    39: 'unique_o', 40: 'unique_s', 41: 'unique_s1', 
    42: 'unique_v', 43: 'unique_w', 44: 'unique_x', 
    45: 'unique_y', 46: 'unique_z', 47: 'unique_z1', 
    48: 'upper_a', 49: 'upper_b', 50: 'upper_d', 
    51: 'upper_d1', 52: 'upper_e', 53: 'upper_f', 
    54: 'upper_g', 55: 'upper_h', 56: 'upper_i', 
    57: 'upper_j', 58: 'upper_k', 59: 'upper_l', 
    60: 'upper_m', 61: 'upper_n', 62: 'upper_p', 
    63: 'upper_q', 64: 'upper_r', 65: 'upper_t',
    66: 'upper_u'}

vocab_letter_to_class = {
    **alphabet_lower_mapper_letter_to_class,
    **alphabet_upper_mapper_letter_to_class,
    **number_letter_to_class,
    **special_letter_to_class,
    **unique_small_letter_to_class
}

class_to_vocab_letter = {v: k for k, v in vocab_letter_to_class.items()}

def classifier_out_to_vocab_letter(index):
    class_letter = classifier_out_to_class[index]
    return class_to_vocab_letter[class_letter]