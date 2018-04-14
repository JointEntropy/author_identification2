
ALLOWED_CHARS = {chr(chr_idx) for chr_idx in range(ord('а'), ord('я')+1)}
ALLOWED_CHARS |= set('ё,.—?!: \t\n') #«»


def filter_chars(ser, allowed=ALLOWED_CHARS):
    ser = ser.str.strip()
    ser = ser.str.lower()
    return ser.apply(lambda item: ''.join(filter(lambda ch: ch in allowed, item)))


if __name__ == '__main__':
    #filter_chars(data['text'], allowed=ALLOWED_CHARS)
    pass


