"""
File:        hamming.py
Author:        Dominic Carrano (carrano.dominic@gmail.com)
Created:    December 18, 2017

Implementation of Hamming single error correction, double error detection
(SECDED) codes for arbitrary length bitstrings, represented as Python bitarrays.
All parity bits use even parity.

For more, see: https://en.wikipedia.org/wiki/Hamming_code
"""

from bitarray import bitarray
from math import floor, ceil, log2

DEBUG = False
BITS_PER_BYTE =8 

# CORE API

def encode(data: bitarray):
    """
    Given a bitstring 'data', returns a new bitstring containing the original bits
    and Hamming (even) parity bits to allow for SECDED.
    """
    # cache due to constant reuse
    data_length     = len(data) 
    num_parity_bits = _num_parity_bits_needed(data_length)
    encoded_length  = data_length + num_parity_bits + 1 # need +1 for the overall parity bit
    if(DEBUG):
        print(f"\tfunc({encode.__name__}: {data_length=}, {num_parity_bits=}")

    # the Hamming SECDED encoded bitstring
    encoded = bitarray(encoded_length) 
    
    # set parity bits
    for parity_bit_index in _powers_of_two(num_parity_bits):
        encoded[parity_bit_index] = _calculate_parity(data, parity_bit_index)
    
    # set data bits
    data_index = 0
    for encoded_index in range(3, len(encoded)): # start at 3 instead of 1 to skip first two parity bits
        if not _is_power_of_two(encoded_index):
            encoded[encoded_index] = data[data_index]
            data_index += 1

    # compute and set overall parity for the entire encoded data (not including the overall parity bit itself)
    encoded[0] = _calculate_parity(encoded[1:], 0)

    # all done!
    return encoded


def decode(encoded: bitarray):
    """
    Given a bitstring 'encoded' with Hamming SECDED parity bits, returns the original data bitstring,
    correcting single errors and reporting if two errors are found.
    """
    encoded_length  = len(encoded)
    num_parity_bits = floor(log2(encoded_length - 1)) + 1 # -1 to remove overall parity bit from length, +1 since first parity bit at len 1 (2 ** 0)
    if(DEBUG):
        print(f'\tfunc({decode.__name__}):{num_parity_bits=}')
    index_of_error  = 0 # index of bit in error, relative to DATA + PARITY bitstring

    # the original data bits, which may be corrupted
    decoded = _extract_data(encoded)
    if(DEBUG):
        print(f'\tfunc({decode.__name__}): {decoded=}')

    # check overall parity bit
    overall_expected = _calculate_parity(encoded[1:], 0)
    overall_actual   = encoded[0]
    overall_correct  = overall_expected == overall_actual
    if(DEBUG):
        print(f'\tfunc({decode.__name__}): {overall_expected=}, {overall_actual=}')

    # check individual parities - each parity bit's index (besides overall parity) is a power of two
    for parity_bit_index in _powers_of_two(num_parity_bits):
        expected = _calculate_parity(decoded, parity_bit_index)
        actual   = encoded[parity_bit_index]
        if not expected == actual:
            index_of_error += parity_bit_index
    if(DEBUG):
        print(f"\tfunc({decode.__name__}): {index_of_error=}")

    # report results
    index_out_of_bounds = index_of_error > (encoded_length - 1) # -1 to omit overall bit from bounds calculation
    if index_of_error and overall_correct:          # two errors found
        raise ValueError("Two errors detected.")
    elif not index_out_of_bounds and index_of_error and not overall_correct:    # one error found - flip the bit in error and we're good
        encoded[index_of_error] = not encoded[index_of_error]
    elif not index_of_error and overall_correct:    # zero errors found
        pass
    elif not index_of_error and not overall_correct: # one error, first bit
        encoded[0] = not encoded[0]
    elif index_out_of_bounds:
        raise IndexError(f"Multiple errors detected. {index_of_error=}")
    decoded = _extract_data(encoded)                 # extract new, corrected data and return it
    return decoded

# HELPER FUNCTIONS - The functions' names begin with an underscore to denote these being module private

def _num_parity_bits_needed(length: int):
    """
    Given the length of a DATA bitstring, returns the number of parity bits needed for Hamming SEC codes.
    An additional parity bit beyond this number of parity bits is needed to achieve SECDED codes.
    """
    n = _next_power_of_two(length)
    lower_bin = floor(log2(n))
    upper_bin = lower_bin + 1
    data_bit_boundary = n - lower_bin - 1                    
    return lower_bin if length <= data_bit_boundary else upper_bin

def _calculate_parity(data: bitarray, parity: int):
    """
    Calculates the specified Hamming parity bit (1, 2, 4, 8, etc.) for the given data.
    Assumes even parity to allow for easier computation of parity using XOR.

    If 0 is passed in to parity, then the overall parity is computed - that is, parity over
    the entire sequence.
    """
    retval = 0 # 0 is the XOR identity

    if parity == 0: # special case - compute the overall parity
        for bit in data:
            retval ^= bit
    else:
        for data_index in _data_bits_covered(parity, len(data)):
            retval ^= data[data_index]
    return retval

def _data_bits_covered(parity: int, lim: int):
    """
    Yields the indices of all data bits covered by a specified parity bit in a bitstring
    of length lim. The indices are relative to DATA BITSTRING ITSELF, NOT including
    parity bits.
    """
    if not _is_power_of_two(parity):
        raise ValueError("All hamming parity bits are indexed by powers of two.")

    # use 1-based indexing for simpler computational logic
    data_index  = 1        # bit we're currently at in the DATA bitstring
    total_index = 3     # bit we're currently at in the OVERALL bitstring

    while data_index <= lim:
        curr_bit_is_data = not _is_power_of_two(total_index)
        if curr_bit_is_data and (total_index % (parity << 1)) >= parity:    
            yield data_index - 1 # adjust output to be zero indexed
        data_index += curr_bit_is_data
        total_index += 1
    return None

def _extract_data(encoded: bitarray):
    """
    Assuming encoded is a Hamming SECDED encoded bitstring, returns the substring that is the data bits.
    """
    data = bitarray()
    for i in range(3, len(encoded)):
        if not _is_power_of_two(i):
            data.append(encoded[i])
    return data

def _next_power_of_two(n: int):
    """
    Given an integer n, returns the next power of two after n.

    >>> _next_power_of_two(768)
    1024
    >>> _next_power_of_two(4)
    8
    """
    if (not (type(n) == int)) or (n <= 0):
        raise ValueError("Argument must be a positive integer.")
    elif _is_power_of_two(n):
        return n << 1
    return 2 ** ceil(log2(n))

def _is_power_of_two(n: int):
    """
    Returns if the given non-negative integer n is a power of two.
    Credit: https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
    """
    return (not (n == 0)) and ((n & (n - 1)) == 0)

def _powers_of_two(n: int):
    """
    Yields the first n powers of two.

    >>> [x for x in _powers_of_two(5)]
    [1, 2, 4, 8, 16]
    """
    power, i = 1, 0
    while i < n:
        yield power
        power <<= 1
        i += 1
    return None

# Further utility functions - not used anywhere in hamming.py, but would be useful for dealing with
# Hamming codes of bytearrays (useful for sending strings over an unreliable network, for instance)

def bytes_to_bits(byte_stream: bytearray, nzfill=BITS_PER_BYTE):
    """
    Converts the given bytearray to a bitarray by converting  each successive byte into its 
    appropriate binary data bits and appending them to the bitarray.
    """
    if(DEBUG):
        print(f"\tfunc({bytes_to_bits.__name__}{byte_stream=}")
    out = bitarray()
    for byte in byte_stream:
        data = bin(byte)[2:].zfill(nzfill)
        for bit in data:
            out.append(0 if bit == '0' else 1) # note that all bits go to 1 if we just append bit, since it's a non-null string
    if(DEBUG):
        print(f"\tfunc({bytes_to_bits.__name__}): {out=}")
    return out

def bits_to_bytes(bits: bitarray):
    """
    Converts the given bitarray bits to a bytearray. 

    Assumes the bits of the last byte or fraction of a byte are to be interpreted as the least 
    significant bits of the last byte of data, e.g. 0b100 would map to the byte 0b00000100.
    """
    out = bytearray()
    for i in range(0, len(bits) // BITS_PER_BYTE * BITS_PER_BYTE, BITS_PER_BYTE):
        byte, k = 0, 0
        for bit in bits[i:i + BITS_PER_BYTE][::-1]:
            byte += bit * (1 << k)
            k += 1
        out.append(byte)

    # tail case - necessary if bitstring length isn't a multiple of 8
    if len(bits) % BITS_PER_BYTE:
        byte, k = 0, 0
        for bit in bits[int(len(bits) // BITS_PER_BYTE * BITS_PER_BYTE):][::-1]:
            byte += bit * (1 << k)
            k += 1
        out.append(byte)

    return out

def _get_codewords(num_data_bits):
    codewords = []
    for i in range(num_data_bits**2):
        data = bytes_to_bits(i.to_bytes(), num_data_bits)
        data_w_parity = encode(data.copy())
        if(DEBUG):
            print(f"\tfunc({_get_codewords.__name__}): {data=} : {data_w_parity=}")
        codewords.append(data_w_parity)
    return codewords

"""
There should be maximum 3 states for all points in the graph
given a graph that consists of 2**N points where N is the 
length, in bits, of a codeword. The 3 states are:
1. codeword with 0 error - exactly 2**M points, where M is the number of data bits
2. codeword with 1 error (correctable) - exactly N*(2**M) points
3. codeword with 2 errors (uncorrectable) - exactly (2**N - 2**M - N*(2**M)) points
Any message with 3 or more errors will be interpreted as being in one of
the three states above.
"""
def test_hamming_decode(num_data_bits=4):
    codewords = _get_codewords(num_data_bits)
    num_zero_errors = 0
    num_one_errors = 0
    num_bits = len(codewords[0])
    for i in range(2**num_bits):
        message = bytes_to_bits(i.to_bytes(), num_bits)
        try:
            decoded = decode(message.copy())
            encoded_bytes = bits_to_bytes(encode(decoded))
            message_bytes = bits_to_bytes(message)
            if(DEBUG):
                print(f"\tfunc({test_hamming_decode.__name__}): decoded: {encoded_bytes}, message: {message_bytes}", end='')
            if(encoded_bytes != message_bytes):
                num_one_errors += 1
                if(DEBUG):
                    print(', one error')
            else:
                num_zero_errors += 1
                if(DEBUG):
                    print(', zero error')
            if(DEBUG):
                print(f"\tfunc({test_hamming_decode.__name__}): {message}: {decoded}")
        except ValueError:
            if(DEBUG):
                print(f"\tfunc({test_hamming_decode.__name__}): {message}: ValueError")
        except IndexError as e:
            if(DEBUG):
                print(f"\tfunc({test_hamming_decode.__name__}): {message}: IndexError {e}")
    assert num_zero_errors == num_data_bits**2
    assert num_one_errors == (num_bits * (2**num_data_bits))
    if(DEBUG):
        print(f"\tfunc{test_hamming_decode.__name__}): {num_zero_errors=}, {num_one_errors=}")

def calculate_hamming_distance(num_data_bits=4):
    codewords = _get_codewords(num_data_bits)
    min_distance = len(codewords[0])
    for i in range(len(codewords)):
        test_codeword = codewords[i]
        for j in range(len(codewords)):
            if(i == j):
                continue
            distance = sum([int(x) for x in (test_codeword ^ codewords[j])])
            if(distance < min_distance):
                min_distance = distance
    if(DEBUG):
        print(f"\tfunc({calculate_hamming_distance.__name__}): {num_data_bits=}, {min_distance=}")
    return min_distance

if __name__ == "__main__":
    padding = 25
    data = bitarray('101010')
    print(f'Data:      {str(data):>{padding}}')
    data_with_parity = encode(data)
    print(f'Encoded:   {str(data_with_parity):>{padding}}')
    data_with_parity[3] = not data_with_parity
    print(f'Corrupted: {str(data_with_parity):>{padding}}')
    print(f'Decoded:   {str(decode(data_with_parity)):>{padding}}')
    print(f"Hamming Distance is {calculate_hamming_distance()}")
    test_hamming_decode()
