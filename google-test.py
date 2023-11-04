import google.generativeai as palm

palm.configure(api_key="")
# Create a new conversation
response = palm.chat(messages="how are you")

# Last contains the model's response:
print(response.last)
