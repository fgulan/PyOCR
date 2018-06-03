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

vocab_letter_to_class = {
    **alphabet_lower_mapper_letter_to_class,
    **alphabet_upper_mapper_letter_to_class,
    **number_letter_to_class,
    **special_letter_to_class,
    **unique_letter_to_class
}