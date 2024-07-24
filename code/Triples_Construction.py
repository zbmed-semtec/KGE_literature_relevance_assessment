import pandas as pd
import json
import argparse
import os


def reduce_nonrelevant_triples_function(triples_df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    id_counter = {}
    for index, row in triples_df.iterrows():
        if row[0] in id_counter:
            id_counter[row[0]] += 1
        else:
            id_counter[row[0]] = 1
        if row[2] in id_counter:
            id_counter[row[2]] += 1
        else:
            id_counter[row[2]] = 1
    heads = []
    relations = []
    tails = []
    for index, row in triples_df.iterrows():
        if id_counter[row[0]] >= min_count and id_counter[row[2]] >= min_count:
            heads.append(row[0])
            relations.append(row[1])
            tails.append(row[2])
    df = pd.DataFrame(
        {'heads': heads, 'relations': relations, 'tails': tails})
    return df


def create_chunked_triples(knowledge_base_json: str, citation_json: str, triples_chunk_output_directory: str, relevance_matrix_tsv: str):
    chunks = []
    kb = {}
    triples = set()
    citations = {}
    relevance_matrix = pd.read_csv(relevance_matrix_tsv, sep='\t')
    relevance_matrix = relevance_matrix.astype(str)
    last_pmid = ""
    keywords_count = 0
    authors = 0
    if not os.path.exists(triples_chunk_output_directory):
        os.makedirs(triples_chunk_output_directory)
    chunk = []
    iteration = 0
    seedPMIDList = list(relevance_matrix["PMID1"])
    secondPMIDList = list(relevance_matrix["PMID1"])
    counter = 0
    for index in range(len(seedPMIDList)):
        if (last_pmid == ""):
            last_pmid = seedPMIDList[index]
            chunk.append(seedPMIDList[index])
        elif (last_pmid != seedPMIDList[index]):
            print(f"{last_pmid} | {seedPMIDList[index]}")
            counter += 1
            last_pmid = seedPMIDList[index]
            chunks.append(chunk)
            chunk = []
        chunk.append(secondPMIDList[index])
    if (len(chunk) > 0):
        chunks.append(chunk)

    with open(knowledge_base_json, 'r') as f:
        kb = json.load(f)
    with open(citation_json, 'r') as f:
        citations = json.load(f)
    for current_chunk in chunks:
        triples = set()
        for pmid in current_chunk:
            if pmid in kb:
                current_kb = kb[pmid]
                id = current_kb["id"].split("https://openalex.org/")[1]
                for author in current_kb["authorships"]:
                    authors += 1
                    triples.add(tuple([id, "authoredBy", author["author"]
                                ["id"].split("https://openalex.org/")[1]]))
                if pmid in citations:
                    for citation_id in citations[pmid]:
                        triples.add((citation_id, "references", id))
                    if "topics" in current_kb:
                        for topic in current_kb["topics"]:
                            triples.add(
                                (id, "has", topic["id"].split("https://openalex.org/")[1]))
                            triples.add(
                                (id, "has", topic["subfield"]["id"].split("https://openalex.org/")[1]))
                            triples.add((id, "has",
                                        topic["field"]["id"].split("https://openalex.org/")[1]))
                            triples.add(tuple(
                                [id, "has", topic["domain"]["id"].split("https://openalex.org/")[1]]))
                if ("keywords" in current_kb and len(current_kb["keywords"]) > 0):
                    for keyword in current_kb["keywords"]:
                        if "id" in keyword:
                            keywords_count += 1
                            triples.add(
                                (id, "has", keyword["id"].split("https://openalex.org/")[1]))
                if ("concepts" in current_kb and len(current_kb["concepts"]) > 0):
                    for concept in current_kb["concepts"]:
                        triples.add(
                            (id, "has", concept["id"].split("https://openalex.org/")[1]))
                if ("mesh" in current_kb and len(current_kb["mesh"]) > 0):
                    for mesh in current_kb["mesh"]:
                        triples.add((id, "has", mesh["descriptor_ui"]))
                if ("referenced_works" in current_kb and len(current_kb["referenced_works"]) > 0):
                    for referenced_work in current_kb["referenced_works"]:
                        triples.add(
                            (id, "references", referenced_work.split("https://openalex.org/")[1]))
                if ("related_works" in current_kb and len(current_kb["related_works"]) > 0):
                    for related_work in current_kb["related_works"]:
                        triples.add(
                            (id, "relatedTo", related_work.split("https://openalex.org/")[1]))
        heads = ['head']
        relations = ['relation']
        tails = ['tail']
        for entry in triples:
            heads.append(entry[0])
            relations.append(entry[1])
            tails.append(entry[2])
        df = pd.DataFrame(
            {'heads': heads, 'relations': relations, 'tails': tails})

        # Save RELISH chunk
        df = reduce_nonrelevant_triples_function(df, 2)
        df.to_csv(f"{triples_chunk_output_directory}/Triples{iteration}.tsv",
                  sep='\t', index=False, header=False)
        iteration += 1
        # print(f"{iteration}/{len(chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ik', '--inputKG', type=str,
                        help="Path for input OpenAlex knowledge base JSON.")
    parser.add_argument('-ic', '--inputCitation', type=str,
                        help="Path for input cited-by JSON.")
    parser.add_argument('-ir', '--inputRelevanceMatrix', type=str,
                        help="Input of the three column RELISH PMID pair relevance matrix TSV.")
    parser.add_argument('-o', '--output', type=str,
                        help="Directory path for chunked triples.")
    args = parser.parse_args()

    create_chunked_triples(args.inputKG, args.inputCitation,
                           args.output, args.inputRelevanceMatrix)
