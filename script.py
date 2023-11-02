import pprint
import google.generativeai as palm
from load_creds import load_creds

creds = load_creds()

palm.configure(credentials=creds)

print()
print('Available base models:', [m.name for m in palm.list_tuned_models()])
print('My tuned models:', [m.name for m in palm.list_tuned_models()])