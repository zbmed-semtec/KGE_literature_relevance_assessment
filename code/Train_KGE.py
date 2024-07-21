import pandas as pd
import json
import argparse
from scipy import spatial
import json
import pandas as pd
import torch
import time
import sys
from pykeen.triples import TriplesFactory
from pykeen.triples.utils import get_entities


def generate_knowledge_graph_embeddings(triples_input_directory: str, embeddings_output_directory: str, triples_factory_output_directory: str, kge_model: str, training_approach: str, processing_time_json_output: str):
    from pykeen.models import TransE, TransH, TransR, TransD, RotatE, ConvE
    from pykeen.training import SLCWATrainingLoop, LCWATrainingLoop
    from pykeen.losses import CrossEntropyLoss
    start = time.time()
    cuda = torch.device('cuda')

    # Create models dictionary
    models_dict = {"TransE": TransE(triples_factory=triples_factory, embedding_dim=200, loss=CrossEntropyLoss(), random_seed=42),
                   "TransR": TransH(triples_factory=triples_factory, embedding_dim=200, loss=CrossEntropyLoss(), random_seed=42),
                   "TransH": TransR(triples_factory=triples_factory, embedding_dim=200, loss=CrossEntropyLoss(), random_seed=42),
                   "TransD": TransD(triples_factory=triples_factory, embedding_dim=200, loss=CrossEntropyLoss(), random_seed=42),
                   "RotatE": RotatE(triples_factory=triples_factory, embedding_dim=200, loss=CrossEntropyLoss(), random_seed=42),
                   "ConvE": ConvE(triples_factory=triples_factory, embedding_dim=200, loss=CrossEntropyLoss(), random_seed=42)}
    if kge_model not in models_dict:
        sys.exit(f"Model {kge_model} not supported by KGE pipeline.")

    for i in range(3194):
        df = pd.read_csv(
            f"{triples_input_directory}/Triples{i}.tsv", sep='\t', header=None)
        df.columns = ['head', 'relation', 'tail']
        df = df.astype(str)

        triples_factory = TriplesFactory.from_labeled_triples(
            triples=df[['head', 'relation', 'tail']].values,
            create_inverse_triples=True
        )
        model = models_dict[kge_model]
        model = model.to('cuda')

        training_loop = None
        if training_approach == "sLCWA" or training_approach == "SLCWA" or training_approach == "slcwa":
            training_loop = SLCWATrainingLoop(
                model=model,
                triples_factory=triples_factory,
            )
        elif training_approach == "LCWA" or training_approach == "LCWA" or training_approach == "lcwa":
            training_loop = LCWATrainingLoop(
                model=model,
                triples_factory=triples_factory,
            )
        else:
            sys.exit(f"Training approach {training_approach} is invalid")

        num_epochs = 10
        batch_size = 256
        if kge_model == "ConvE":
            batch_size = 32  # Reduce batch size for ConvE

        training_loop.train(
            triples_factory=triples_factory,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_tqdm_batch=False,
        )
        triples_factory.to_path_binary(
            f"{triples_factory_output_directory}/triples_factory{i}.pkl")
        entity_embeddings = model.entity_representations[0](
            indices=None).detach().cpu().numpy()
        print(f"Completed {i} / 3194")
        torch.save(entity_embeddings,
                   f"{embeddings_output_directory}/kge_chunk{i}.pt")
    processing_time = {}
    processing_time["Processing time in s"] = (time.time() - start)
    with open(f"{processing_time_json_output}", 'w') as process_time_json:
        json.dump(processing_time, process_time_json)


def evaluate_embedding_chunks(embeddings_directory: str, input_knowledge_base: str, triples_factory_directory: str, relevance_matrix: str, output_cosine_similarity_matrix: str):
    with open(input_knowledge_base, 'r') as knowledge_base:
        relish_knowledge_base = json.load(knowledge_base)
    id_map = {}
    for entry in relish_knowledge_base:
        id_map[entry] = relish_knowledge_base[entry]["id"].split(
            "https://openalex.org/")[1]
    embeddings_dict = {}
    for i in range(3194):  # Expected to be 3194 chunks on RELISH dataset
        entity_embeddings = torch.load(
            f"{embeddings_directory}/kge_chunk{i}.pt")
        triples_factory = TriplesFactory.from_path_binary(
            f"{triples_factory_directory}/triples_factory{i}.pkl")
        entities_list = list(get_entities(triples_factory.mapped_triples))
        index = 0
        for entity in entity_embeddings:
            entity_name = triples_factory.entity_id_to_label[entities_list[index]]
            embeddings_dict[entity_name] = entity
            index += 1
        print(f"Processed {i} / 3193")

    relevance_matrix = pd.read_csv(relevance_matrix, sep='\t', header=None)
    relevance_matrix = relevance_matrix.astype(str)
    first_pmids = ['PMID1']
    second_pmids = ['PMID2']
    relevancy = ['relevance']
    cosine_similarity = ['Cosine Similarity']
    for index, row in relevance_matrix.iterrows():
        try:
            cosine_similarity.append(
                1 - spatial.distance.cosine(embeddings_dict[id_map[row[0]]], embeddings_dict[id_map[row[1]]]))
            relevancy.append(row[2])
            second_pmids.append(row[1])
            first_pmids.append(row[0])
        except:
            # print(f"Could not find pmid {row[0]} or pmid {row[1]}")
            continue

    output_df = pd.DataFrame(
        {'PMID1': first_pmids, 'PMID2': second_pmids, 'relevance': relevancy, 'Cosine Similarity': cosine_similarity})
    output_df.to_csv(output_cosine_similarity_matrix,
                     sep='\t', index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ik', '--inputKG', type=str,
                        help="Path for input OpenAlex knowledge base JSON.")
    parser.add_argument('-it', '--inputTriples', type=str,
                        help="Directory path for input triples.")
    parser.add_argument('-ir', '--inputRelevanceMatrix', type=str,
                        help="Input of RELISH PMID relevance matrix TSV")
    parser.add_argument('-m', '--kgeModel', type=str,
                        help="Name of knowledge graph embedding model.")
    parser.add_argument('-t', '--trainingApproach', type=str,
                        help="Name of training approach, it can either be sLCWA or LCWA.")
    parser.add_argument('-oe', '--outputEmbeddings', type=str,
                        help="Directory path for embeddings output.")
    parser.add_argument('-ot', '--outputTriples', type=str,
                        help="Directory path for TriplesFactory entities output.")
    parser.add_argument('-oe', '--outputEmbeddings', type=str,
                        help="Directory path for embeddings output.")
    parser.add_argument('-ot', '--outputProcessingTime', type=str,
                        help="Output of processing time JSON.")
    parser.add_argument('-oc', '--outputCosineSimilarity', type=str,
                        help="Output of cosine similarity matrix TSV.")
    args = parser.parse_args()

    generate_knowledge_graph_embeddings(args.inputTriples, args.outputEmbeddings,
                                        args.outputTriples, args.kgeModel, args.trainingApproach, args.outputProcessingTime)
    evaluate_embedding_chunks(args.outputEmbeddings, args.inputKG, args.outputTriples,
                              args.inputRelevanceMatrix, args.outputCosineSimilarity)
