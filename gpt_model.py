# -*- coding: utf-8 -*-
"""GPT_Model

Original file is located at
    https://colab.research.google.com/drive/1Dv1rA3DkrR9PZYws3OXPNJaIPdpSpqr3

##**Importing libraries**
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

"""##**Choosing a pre-trained language model**"""

# Load pre-trained tokenizer and model
model_name = "ericzzz/falcon-rw-1b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map='auto',torch_dtype=torch.bfloat16)

"""##**Creating an empty chat history**"""

chat_history = []
val=True

"""##**Generating a loop for continuous user feed**"""

while val:
    # Prompt user for input
    user_input = input('You: ')

    if user_input.lower()=='exit':
      print('Bot: Thanks for talking with me')
      val=False

    else:
      # Add user input to chat history
      chat_history.append({"role": "user", "content": user_input})

      # Tokenize the updated chat history
      input_ids = tokenizer.apply_chat_template(
          chat_history, tokenize=True, add_generation_prompt=True, return_tensors="pt"
      ).to(model.device)

      # Generate the attention mask for the input sequence
      attention_mask = torch.ones_like(input_ids)

      # Generate model response
      output_tokens = model.generate(
          input_ids,
          attention_mask=attention_mask,  # Set the attention mask
          pad_token_id=tokenizer.eos_token_id,
          do_sample=True,
          temperature=0.7,
          repetition_penalty=1.05,
          max_new_tokens=200,
      )

      # Decode the generated output into text
      output_text = tokenizer.decode(
          output_tokens[0][len(input_ids[0]):], skip_special_tokens=True
      )

      # Add the assistant's response to the chat history
      chat_history.append({"role": "assistant", "content": output_text})

      # Print the assistant's response
      print('Bot:', output_text)