from string import ascii_lowercase
import itertools

def num2str(n):
    res = ''
    while(n>0):
        res = res + chr(97+(n%26))
        n = n // 26
    if res == '':
        return 'a'
    return res[::-1]

def str2num(s):
    res = 0
    for c in s:
        res = res * 26 + ord(c)-97
    return res

def get_alphabetic_list(n):
    ret = []
    for i in range(n):
        ret.append(num2str(i))
    return ret

def get_last_alphabetic(st):
    num = str2num(st)
    return num2str(num-1)


if __name__ == '__main__':
    print(num2str(13))
    print(get_last_alphabetic('bac'))
