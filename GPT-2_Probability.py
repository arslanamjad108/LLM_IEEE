import matplotlib.pyplot as plt
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define a function to generate tokens and probabilities from a given input phrase
def generate_tokens_and_probs(input_phrase):
    # Tokenize the input phrase
    input_ids = tokenizer.encode(input_phrase, return_tensors='pt')
    # Generate output probabilities using the model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs[0][:, -1, :]
        probs = torch.softmax(logits, dim=-1)[0]
    # Convert token IDs and probabilities to lists
    token_ids = input_ids.tolist()[0]
    token_probs = probs.tolist()
    # Remove the first token ID (which is the <BOS> token)
    token_ids.pop(0)
    token_probs.pop(0)
    # Return the token IDs and probabilities
    return token_probs

# Define an example input phrase
input_phrase = "Hello, how are you?"

# Generate the token probabilities for the example input phrase
token_probs = generate_tokens_and_probs(input_phrase)

# Plot a graph of token probabilities vs. index
plt.plot(range(len(token_probs)), token_probs)
plt.xlabel('Index')
plt.ylabel('Probability')
plt.title('Token Probabilities for Input Phrase: "{}"'.format(input_phrase))
plt.show()

