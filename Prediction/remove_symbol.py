import string


def remove_symbols(text):
    # Use the string module's `punctuation` property to get all ASCII symbols
    symbols = string.punctuation
    # Iterate through the text and replace any symbol with an empty string
    for symbol in symbols:
        text = text.replace(symbol, '')

    text = text.replace('\n', ' ')
    return text


# Read the text file
with open('dataset/Tunnel_text.txt', 'r') as file:
    text = file.read()

# Remove the symbols from the text
text_without_symbols = remove_symbols(text)

# Write the resulting text to a new text file
with open('Tunnel_text_no_symbols.txt', 'w') as file:
    file.write(text_without_symbols)
