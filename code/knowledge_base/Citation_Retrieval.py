import requests
import json
import requests
import sys
import argparse
import warnings


def retrieve_cited_by(input_kb_json: str, cited_by_json: str, breakpoint_iteration: int = 0):
    kg_list = {}
    with open(input_kb_json, 'r') as f:
        kg_list = json.load(f)
    iteration = 1
    full_citation_list = {}
    chunk_list = []
    chunk_iteration = 4
    for key in kg_list:
        if breakpoint_iteration <= iteration:
            current_page = 1
            final_page = False
            content = []
            while final_page != True:
                api_string = kg_list[key]["cited_by_api_url"] + \
                    f"?per-page=100&page={current_page}&mailto=Geist@zbmed.de"
                response = requests.get(api_string)
                if (response.status_code != 200):
                    warnings.warn(f"Status error: {response.status_code}")
                    last_iteration = {"iteration": iteration / 100}
                    if iteration > 100:
                        with open(f"data/last_iteration.json", 'w') as iteration_json:
                            json.dump(last_iteration, iteration_json)
                    sys.exit()
                response_json = response.json()
                for entry in response_json["results"]:
                    content.append(entry["id"].split(
                        "https://openalex.org/")[1])
                if (response_json["meta"]["count"] > (100 * current_page)):
                    current_page += 1
                else:
                    final_page = True
            json_content = {}
            full_citation_list[key] = content
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
                            full_citation_list[key] = element[key]
            except:
                warnings.warn("Could not open JSON")
            chunk_iteration += 1
        iteration += 1
        if (iteration % 100 == 0):
            print(f"{iteration}/{len(kg_list)}")
    with open(f"{cited_by_json}", 'r') as full_file:
        json.dump(full_citation_list, full_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help="Path for OpenAlex knowledge base JSON.")
    parser.add_argument('-o', '--output', type=str,
                        help="Path for cited-by JSON.")
    args = parser.parse_args()

    retrieve_cited_by(args.input, args.output)
