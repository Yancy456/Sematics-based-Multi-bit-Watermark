def _radix_convert( message: str, input_r: int, output_r: int,message_len:int) -> str:
    '''convert input_r radix input string into output_r radix output string'''
    # Convert binary string to an integer
    try:
        num = int(message, input_r)
    except ValueError:
        print('Embedded message must be binary string')
        return

    digits = []
    while num:
        digits.append(int(num % output_r))
        num //= output_r
    return ''.join(str(x) for x in digits[::-1])

print(_radix_convert('101',2,4,3))