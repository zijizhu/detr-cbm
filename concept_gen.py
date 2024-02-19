import json

if __name__ == '__main__':
    with open('data/concepts_cleaned.json', 'r') as fp:
        concepts_dict = json.load(fp=fp)
    concepts = []
    for v in concepts_dict.values():
        concepts += v['gpt']
    concepts = list(set(concepts))
    concepts = [c.lower() for c in concepts] + ['unknown']
    with open('data/concepts_cleaned_list.json', 'w') as fp:
        json.dump(concepts, fp=fp)
