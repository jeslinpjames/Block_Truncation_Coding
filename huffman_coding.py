import heapq
from collections import defaultdict
import os

class Node:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    frequency = defaultdict(int)
    for byte in data:
        frequency[byte] += 1

    priority_queue = [Node(value, freq) for value, freq in frequency.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(priority_queue, merged)

    return priority_queue[0]

def generate_huffman_codes(root, current_code, codes):
    if root is None:
        return

    if root.value is not None:
        codes[root.value] = current_code
        return

    generate_huffman_codes(root.left, current_code + '0', codes)
    generate_huffman_codes(root.right, current_code + '1', codes)

def huffman_encode(input_path, output_path):
    with open(input_path, 'rb') as file:
        data = file.read()

    root = build_huffman_tree(data)
    codes = {}
    generate_huffman_codes(root, '', codes)

    encoded_data = ''.join(codes[byte] for byte in data)

    # Write the encoded data and codes to the output file
    with open(output_path, 'wb') as file:
        file.write(int(encoded_data, 2).to_bytes((len(encoded_data) + 7) // 8, byteorder='big'))

    return root  # Return the root for decoding

def huffman_decode(input_path, output_path, root):
    with open(input_path, 'rb') as file:
        encoded_data = bin(int.from_bytes(file.read(), byteorder='big'))[2:]

    decoded_data = decode_huffman(encoded_data, root)

    # Write the decoded data to the output file
    with open(output_path, 'wb') as file:
        file.write(decoded_data)

def decode_huffman(encoded_data, root):
    current_node = root
    decoded_data = bytearray()

    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        elif bit == '1':
            current_node = current_node.right

        if current_node.value is not None:
            decoded_data.append(current_node.value)
            current_node = root

    return bytes(decoded_data)

# Example usage:
mean_path = "D:/git/Block_Truncation_Coding/compressed/mean.txt"
variance_path = "D:/git/Block_Truncation_Coding/compressed/variance.txt"
blocks_path = "D:/git/Block_Truncation_Coding/compressed/blocks.txt"

# Encoding
root_mean = huffman_encode(mean_path, "D:/git/Block_Truncation_Coding/compressed/mean_encoded.txt")
root_variance = huffman_encode(variance_path, "D:/git/Block_Truncation_Coding/compressed/variance_encoded.txt")
root_blocks = huffman_encode(blocks_path, "D:/git/Block_Truncation_Coding/compressed/blocks_encoded.txt")

# Decoding
huffman_decode("D:/git/Block_Truncation_Coding/compressed/mean_encoded.txt", "D:/git/Block_Truncation_Coding/compressed/mean_decoded.txt", root_mean)
huffman_decode("D:/git/Block_Truncation_Coding/compressed/variance_encoded.txt", "D:/git/Block_Truncation_Coding/compressed/variance_decoded.txt", root_variance)
huffman_decode("D:/git/Block_Truncation_Coding/compressed/blocks_encoded.txt", "D:/git/Block_Truncation_Coding/compressed/blocks_decoded.txt", root_blocks)
