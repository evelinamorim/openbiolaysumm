import json
import logging
from os import path
from datetime import datetime
from enum import Enum

logging.basicConfig(
  filename="./logs/create_discourse_graphs.log.{}".format(datetime.timestamp(datetime.now())), 
  level=logging.INFO,
  format = '%(asctime)s | %(levelname)s | %(message)s'
)

class Discourse_Relations(Enum):
  CONTAINS = "contains"
  HAS_TITLE = "has_title"
  HAS_KEYWORD = "has_keyword"
  WAS_PUBLISHED_IN = "was_published_in" # ref. year 
  
ISA_RELATION = 'T186'



def load_datafile(fp):
    with open(fp, "r", encoding="utf-8") as in_file:
        data = json.load(in_file)
        sections = [x["sections"] for x in data]
        section_names = [x['headings'] for x in data]
        abstracts = [x['abstract'] for x in data]
        titles = [x["title"] for x in data]
        keywords = [x["keywords"] for x in data]
        ids = [x["id"] for x in data]
        years = [x["year"] for x in data]
    return ids, sections, section_names, abstracts, titles, keywords, years


def get_concepts_for_article(article_id, concepts_dict):
    # Returns the list of concept dicts for this article
    return concepts_dict.get(article_id, [])


def get_discourse_graph(document_dict, concepts_dict):
    nodes = set()
    edges = set()
    nodes.add(document_dict['id']) # document node

    # Title nodes / relations
    if document_dict['title'] != "":
        nodes.add(document_dict['title']) # title node
        edges.add((document_dict['id'], Discourse_Relations.HAS_TITLE.value, document_dict['title']))

    # Year nodes / relations
    if document_dict['year'] != "":
        nodes.add(document_dict['year'])
        edges.add((document_dict['id'], Discourse_Relations.WAS_PUBLISHED_IN.value, document_dict['year']))

    # Abstract nodes / relations
    if document_dict['abstract'] != "":
        abstract_node = document_dict['id'] + "_Abs"
        nodes.add(abstract_node)
        edges.add((document_dict['id'], Discourse_Relations.CONTAINS.value, abstract_node))

        # Use pre-extracted concepts for this article
        for c in get_concepts_for_article(document_dict['id'], concepts_dict):
            nodes.add(c['cui'])
            edges.add((abstract_node, Discourse_Relations.CONTAINS.value, c['cui']))
            for stype in c.get('semtypes', []):
                nodes.add(stype)
                edges.add((c['cui'], ISA_RELATION, stype))

    # Keyword nodes / relations
    for kw in document_dict['keywords']:
        nodes.add(kw)
        edges.add((document_dict['id'], Discourse_Relations.HAS_KEYWORD.value, kw))

    # Section nodes / relations
    for i, section in enumerate(document_dict['sections']):
        section_heading = document_dict['section_names'][i]
        sec_node = document_dict['id'] + "_Sec" + str(i)
        nodes.add(sec_node)
        edges.add((document_dict['id'], Discourse_Relations.CONTAINS.value, sec_node))
        nodes.add(section_heading)
        edges.add((sec_node, Discourse_Relations.HAS_TITLE.value, section_heading))
        # (Optional) You can also add section-level concepts if you want

    return { 'edges': list(edges), "nodes": list(nodes) }


# Loop over splits
for split in ["train", "val", "test"]:
    # Load concepts for this split
    with open(f"../DSplit/elife_umls_concepts_{split}.json", "r", encoding="utf-8") as f:
        concepts_data = json.load(f)
    # Build a dict: article_id -> list of concepts
    concepts_dict = {art['id']: art['matches'] for art in concepts_data}

    # Load article data (now from main folder)
    fp = f"../{split}.json"
    ids, sections, section_names, abstracts, titles, keywords, years = load_datafile(fp)
    out_path = fp.replace(".json", "_disc_graphs.jsonl")
    is_existing = path.exists(out_path)
    o_type = "r+" if is_existing else "w"

    with open(out_path, o_type, encoding="utf-8") as out_file:
        i = len(out_file.readlines()) if is_existing else 0
        for ind in range(i, len(ids)):
            logging.info(f'idx={ind}, id={ids[ind]}')
            data_dict = {
                "id": ids[ind],
                "sections": sections[ind],
                "section_names": section_names[ind],
                "abstract": abstracts[ind],
                "title": titles[ind],
                "keywords": keywords[ind],
                "year": years[ind],
            }
            out_dict = get_discourse_graph(data_dict, concepts_dict)
            out_dict['id'] = ids[ind]
            out_file.write(json.dumps(out_dict))
            out_file.write("\n")
