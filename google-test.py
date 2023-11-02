import google.generativeai as palm

palm.configure(api_key='AIzaSyBrgRH5OWpYx5lP9c_hAf38LxbaaIq4c28')
# Create a new conversation
response = palm.chat(messages='how are you')

# Last contains the model's response:
print(response.last)