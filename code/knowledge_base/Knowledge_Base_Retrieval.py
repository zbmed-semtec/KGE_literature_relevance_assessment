import requests
import pandas as pd
import json
import argparse


def get_unique_pmids(relish_pmids_tsv: str, pmids_output: str):
    df = pd.read_csv(relish_pmids_tsv, sep='\t')
    columns = list(df.columns)
    firstColumn = columns[0]
    secondColumn = columns[1]
    pmids = df[f"{firstColumn}"].values.tolist()
    pmids.extend(df[f"{secondColumn}"].values.tolist())
    pmids.append(columns[0])
    pmids.append(columns[1])
    unique_pmids = list(set(pmids))
    with open(pmids_output, 'w') as f:
        json.dump(unique_pmids, f)


def retrieve_knowledge_base(pmids_json: str, kb_output: str, breakpoint_iteration: int = 0):
    pmids = {}
    with open(pmids_json, 'r') as f:
        pmids = json.load(f)
    pmids_chunks = [pmids[i:i + 100] for i in range(0, len(pmids), 100)]
    iteration = 0
    full_RELISH_JSON_LIST = {}
    for chunk in pmids_chunks:
        if breakpoint_iteration <= iteration:
            api_string = "https://api.openalex.org/works?filter=pmid:"
            for pmid in chunk:
                api_string += f"{pmid}|"
            api_string = api_string[:-1] + "&per-page=100"
            response = requests.get(api_string)
            if (response.status_code != 200):
                print(f"Status error: {response.status_code}")
                last_iteration = {"iteration": iteration}
                json.dump(last_iteration, "data/last_iteration.json")
                break
            response_json = response.json()
            for article in response_json["results"]:
                full_RELISH_JSON_LIST[article["ids"]["pmid"].split(
                    "https://pubmed.ncbi.nlm.nih.gov/")[1]] = article
            with open(f"data/chunks/OPENALEX{iteration}.json", 'w') as out:
                json.dump(response_json, out)
        else:
            with open(f"data/chunks/OPENALEX{iteration}.json", 'w') as old_file:
                old_json = json.load(old_file)
                for article in old_json["results"]:
                    full_RELISH_JSON_LIST[article["ids"]["pmid"].split(
                        "https://pubmed.ncbi.nlm.nih.gov/")[1]] = article
        iteration += 1
        if (iteration % 10 == 0):
            print(f"{iteration}/{len(pmids_chunks)}")
    with open(f"{kb_output}.json", 'w') as full_file:
        json.dump(full_RELISH_JSON_LIST, full_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help="Path for RELISH 3 column TSV file.")
    parser.add_argument('-o', '--output', type=str,
                        help="Path for OpenAlex knowledge base JSON.")
    args = parser.parse_args()

    unique_pmid_json = "data/unique_pmids.json"
    get_unique_pmids(args.input, unique_pmid_json)
    retrieve_knowledge_base(unique_pmid_json, args.output)
