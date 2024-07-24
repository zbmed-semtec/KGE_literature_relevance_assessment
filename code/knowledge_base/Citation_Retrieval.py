import requests
import json
import requests
import sys
import argparse
import warnings
import os


def retrieve_cited_by(input_kb_json: str, cited_by_json: str, breakpoint_iteration: int = 0):
    kb_dict = {}
    with open(input_kb_json, 'r') as f:
        kb_dict = json.load(f)
    iteration = 1
    full_citation_dict = {}
    chunk_list = []
    chunk_iteration = 4
    if not os.path.exists("data/citation_chunks"):
        os.makedirs("data/citation_chunks")
    for key in kb_dict:
        if breakpoint_iteration <= iteration:
            current_page = 1
            final_page = False
            content = []
            while final_page != True:
                api_string = kb_dict[key]["cited_by_api_url"] + \
                    f"?per-page=100&page={current_page}"
                response = requests.get(api_string)
                if (response.status_code != 200):
                    warnings.warn(f"Status error: {response.status_code}")
                    # Ignore this chunk and continue to next one
                    continue
                response_json = response.json()
                for entry in response_json["results"]:
                    content.append(entry["id"].split(
                        "https://openalex.org/")[1])
                if (response_json["meta"]["count"] > (100 * current_page)):
                    current_page += 1
                else:
                    final_page = True
            json_content = {}
            full_citation_dict[key] = content
            json_content[key] = content
            chunk_list.append(json_content)
            if iteration % 100 == 0:
                with open(f"data/citation_chunks/OPENALEX{int(iteration/100)}.json", 'w') as out:
                    json.dump(chunk_list, out)
                chunk_list = []
        else:
            try:
                with open(f"data/citation_chunks/OPENALEX{chunk_iteration}.json", 'r') as old_file:
                    old_json = json.load(old_file)
                    for element in old_json:
                        for key in element:
                            full_citation_dict[key] = element[key]
            except:
                warnings.warn(
                    f"Could not find JSON: data/citation_chunks/OPENALEX{chunk_iteration}.json")
            chunk_iteration += 1
        iteration += 1
        if (iteration % 100 == 0):
            print(f"{iteration}/{len(kb_dict)}")
    with open(f"{cited_by_json}", 'w') as full_file:
        json.dump(full_citation_dict, full_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help="Path for OpenAlex knowledge base JSON.")
    parser.add_argument('-o', '--output', type=str,
                        help="Path for cited-by JSON.")
    args = parser.parse_args()

    retrieve_cited_by(args.input, args.output)
