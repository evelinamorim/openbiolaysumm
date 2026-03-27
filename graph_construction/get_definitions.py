import requests, os 

# https://documentation.uts.nlm.nih.gov/rest/home.html

# base-url: https://uts-ws.nlm.nih.gov/rest
# Concept definitions - /content/{version}/CUI/{CUI}/definitions
# Semnantic type definitions - /semantic-network/{version}/TUI/{id}

base_url = "https://uts-ws.nlm.nih.gov/rest"
# Inserted real UMLS API key provided by your teacher
api_key = "a0a84d62-1695-4717-9b21-1ad7893187b9"
version = "current"
query = {'apiKey': api_key}

with open("../UMLS_Files/umls_concepts_used.txt", "r") as in_file:
  all_concepts = in_file.readlines()
  all_concepts = [c.strip() for c in all_concepts]

with open("../UMLS_Files/umls_semtypes.txt", "r") as in_file:
  all_semtypes = in_file.readlines()
  all_semtypes = [s.strip().split("|")[1] for s in all_semtypes]


# Get semantic type definitions
semtype_definitions = {}
with open("../UMLS_Files/semtype_definitions.txt", "w") as out_file:
  for tui in all_semtypes:
    path = f"/semantic-network/{version}/TUI/{tui}/"
    output = requests.get(base_url+path, params=query)
    output.encoding = 'utf-8'
    outputJson = output.json()
    
    if 'status' in outputJson and outputJson['status'] == 404:
      continue

    defin = outputJson['result']['definition']
    out_file.write(f"{tui}|{defin}\n")


# Get concept definitions
with open("../UMLS_Files/umls_concept_definitions.txt", "w") as out_file:
  for i, cui in enumerate(all_concepts):
    
    path = f"/content/{version}/CUI/{cui}/definitions"
    output = requests.get(base_url+path, params=query)
    output.encoding = 'utf-8'
    outputJson = output.json()
    
    if 'status' in outputJson and outputJson['status'] == 404:
      continue

    defin = outputJson['result'][0]['value']
    out_file.write(f"{cui}|{defin}\n")

