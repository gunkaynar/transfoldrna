import RNA
from Bio.Seq import Seq


#rna_str = "AUGAAAGCGCUGAUGCUGUUGACACUGUCAGUCUUACUCUGCUGGGUAUCUGCCGAUAUUCGUUGUCAUAGCUGUUAUAAGGUGCCUGUGCUCGGUUGUGUGGACCGACAGAGUUGCAGACUGGAGCCCGGACAGCAAUGUUUGACCACCCACGCUUACCUGGGCAAAAUGUGGGUAUUUAGUAACCUGAGGUGUGGCACCCCCGAAGAACCAUGUCAAGAGGCAUUCAACCAGACCAACCGCAAGCUAGGUCUGACAUACAACACAACCUGCUGUAAUAAAGAUAAUUGCAAUUCCGCA"
#RNAfold_compound = RNA.fold_compound(rna_str)
#print(RNAfold_compound)
#mfe = RNA.fold_compound.mfe(RNAfold_compound)[1]
#import pdb; pdb.set_trace()



#RNA.compare_structures(RNAfold_compound, RNAfold_compound)

#print(RNA.inverse_fold('AUGUUCAAUCCACACGCUCUGGACGUCCCAGCUGUCAUCUUUGACAAUGGCAGCGGCCUGUGCAAAGCGGGCCUGUCAGGCGAAAUCGGCCCUCGGCAUGUGAUUAGCUCUGUUCUGGGACACUGUAAAUUCAAUGUGCCCCUAGCCCGCCUGAAUCAGAAGUAUUUUGUCGGCCAGGAAGCCCUGUACAAGUACGAGGCCCUGCACCUACAUUACCCCAUUGAGAGGGGGCUGGUUACGGGCUGGGACGAUAUGGAGAAGCUAUGGAAGCAUCUUUUUGAACGAGAGCUGGGCGUAAAG','..........(((((((((((((((((((((((((((((...)))..)))))(((((((((.....))((((((((((((((.....((((((((((((((((...(((((((((((((((((((.........)))))))...)))((((((..............(((()))).((..((((((........)))((((((((...........(((.......))))))))))(.((((((((((((((.((((((((((.(((((((.(((((((...((.(.((((((((.....'))


f = open('test_examples.txt', 'r')
lines = f.readlines()
f.close()

#Input RNA: 
#True Structure: 
#Predicted Structure: 

dic = {}
for line in lines:
    if line.startswith('Input RNA'):
        rna_str = line.split(': ')[1].strip()
        dic[rna_str] = []
    elif line.startswith('True Structure'):
        true_str = line.split(': ')[1].strip()
        dic[rna_str].append(true_str)
    elif line.startswith('Predicted Structure'):
        pred_str = line.split(': ')[1].strip()
        dic[rna_str].append(pred_str)
#print(dic)
def balance_dot_bracket(struct):

    chars = list(struct)
    stack = []  # indices of unmatched '(' so far

    for i, ch in enumerate(chars):
        if ch == '(':
            stack.append(i)                 # remember where the '(' is
        elif ch == ')':
            if stack:
                stack.pop()                 # match with most recent '('
            else:
                chars[i] = '.'              # unmatched ')': convert to dot

    # any '(' left in stack are unmatched → convert them to dots
    for idx in stack:
        chars[idx] = '.'

    return ''.join(chars)


def fix_dot_bracket_preserve_parens(s):

    chars  = list(s)
    depth  = 0          # current stack depth
    n      = len(chars)

    # Pass 1: left‑to‑right — ensure we never go negative
    for i in range(n):
        if chars[i] == '(':
            depth += 1
        elif chars[i] == ')':
            depth -= 1
            if depth < 0:              # too many closers so far
                chars[i] = '('         # flip to opener
                depth   = 1            # depth was –1, becomes +1

    # Pass 2: right‑to‑left — fix any excess openers
    # depth now equals the number of unmatched '('.
    for i in range(n - 1, -1, -1):
        if depth == 0:
            break
        if chars[i] == '(':            # flip some '(' to ')'
            chars[i] = ')'
            depth  -= 2                # removes two from surplus

    return ''.join(chars)
#fixed = fix_dot_bracket_preserve_parens('..(((...((((((((..((((((((((((((((((((((....((((((((((((.....))((((((((((((((((((((....((((((((((((..(((((((((((((((((...(......((((((((((((((((((((..(..))(...)))..))..((((...((.......((((((..............((.......((((((((())((((((((((.((....((((((((((((((.(((.((((...(((....(((((......).))))(((((((((')

#print(RNA.inverse_fold('AUGCCUAAGCGAGCCCACUGGGGCGCCCUGUCUGUGGUGCUUAUCCUCCUGUGGGGACAUCCCAGGGUAGCCCUCGCCUGUCCUCAUCCUUGCGCCUGUUAUGUGCCCUCCGAGGUGCAUUGUACAUUCCGGUCCUUGGCUAGUGUCCCCGCCGGGAUCGCGAAGCACGUCGAAAGAAUCAAUUUAGGAUUUAAUUCUAUUCAAGCACUCUCAGAGACGAGCUUUGCCGGCCUGACGAAACUGGAGCUGUUGAUGAUUCACGGCAAUGAGAUCCCCUCCAUACCAGAUGGGGCUCUCCGG',fixed))

for rna_str, structures in dic.items():
    true_str = structures[0]
    pred_str = structures[1]
    #balance the pred_str dot-bracket parentheses
    #pred_str = balance_dot_bracket(pred_str)
    pred_str = fix_dot_bracket_preserve_parens(pred_str)


    #print(rna_str)
    #print(true_str)
    #print(pred_str)
    try:
        predicted = RNA.inverse_fold(rna_str, pred_str)
        print(predicted)
    except Exception as e:
        print(f"Error processing {rna_str}: {e}")
        continue


