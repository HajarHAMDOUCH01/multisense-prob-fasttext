import numpy as np
import os
import scipy.spatial.distance # Needed for cosine_similarity if not already imported
import argparse # Import argparse to handle command-line arguments
from multift import MultiFastText # Import your MultiFastText class

# Your existing cosine_similarity function (keep this)
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# Your existing find_most_similar function (keep this)
def find_most_similar(query_vec, all_vectors_dict, top_n=10, exclude_word=None):
    similarities = []
    for word, vec in all_vectors_dict.items():
        if exclude_word and word == exclude_word:
            continue
        # Check for exact vector equality, avoiding self-similarity if not explicitly excluded
        if np.array_equal(query_vec, vec) and exclude_word is None:
            continue

        # --- NEW FILTERING LOGIC HERE ---
        # Basic filter: exclude very short words (e.g., 1 or 2 characters)
        # Also exclude words that are not purely alphabetic (e.g., "!”" or ".”")
        if len(word) < 3 or not word.isalpha():
            continue
        # --- END NEW FILTERING LOGIC ---

        sim = cosine_similarity(query_vec, vec)
        similarities.append((word, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# --- Main Analysis Block ---
if __name__ == "__main__":
    # Setup argument parser to accept model_basename from command line
    parser = argparse.ArgumentParser(description='Analyze MultiFastText word prototypes.')
    parser.add_argument('--model_basename',
                        default='model', # Default to 'model' if not specified
                        help='The basename of your FastText model files (e.g., "model" for model.words, model.in, model.out).')
    parser.add_argument('--word',
                        default='car', # Default word to analyze
                        help='The word to analyze for its prototypes and nearest neighbors.')

    args = parser.parse_args()

    model_basename = args.model_basename
    word_to_analyze = args.word

    print("--- Loading Model Components using MultiFastText ---")
    try:
        # Load the model using the MultiFastText class
        # Ensure 'multi=True' matches your training flag if it's a multi-sense model
        ft = MultiFastText(model_basename, multi=True)

        # Create the all_vectors_dict needed by find_most_similar
        all_proto1_vectors_dict = {ft.id2word[i]: ft.emb[i] for i in range(len(ft.id2word))}
        all_proto2_vectors_dict = {ft.id2word[i]: ft.emb2[i] for i in range(len(ft.id2word))}

        # Use the word_to_id from the loaded ft object
        word_id = ft.word2id.get(word_to_analyze)
        if word_id is None:
            print(f"Error: '{word_to_analyze}' not found in the vocabulary.")
            exit()

        proto1_vec = ft.emb[word_id]    # Prototype 1 vector for the word
        proto2_vec = ft.emb2[word_id]   # Prototype 2 vector for the word

    except Exception as e:
        print(f"Error loading model with MultiFastText: {e}")
        print("Ensure 'multift.py' is in the same directory and model files exist at the specified basename.")
        exit()

    print(f"\n--- Analyzing '{word_to_analyze}' Vectors ---")

    # --- Step 1: Calculate Cosine Similarity Between the Two Prototypes of the Word ---
    print("\n--- Cosine Similarity Between Prototypes ---")
    similarity_proto1_proto2 = cosine_similarity(proto1_vec, proto2_vec)
    print(f"Cosine similarity between '{word_to_analyze}' (prototype 1) and '{word_to_analyze}' (prototype 2): {similarity_proto1_proto2:.4f}")

    if similarity_proto1_proto2 < 0.5: # Example threshold for distinctness
        print(f"   --> The prototypes for '{word_to_analyze}' appear distinct (similarity {similarity_proto1_proto2:.4f} is low).")
    else:
        print(f"   --> The prototypes for '{word_to_analyze}' are quite similar (similarity {similarity_proto1_proto2:.4f} is high).")
        print("     This might mean the model did not strongly differentiate them for this corpus, or their raw form is not meant to be highly distinct.")

    # --- Step 2: Find Nearest Neighbors for Each Prototype (Initial Pass) ---
    print(f"\n--- Initial Nearest Neighbors for '{word_to_analyze}' Prototype 1 ---")
    neighbors_prototype1_initial = find_most_similar(proto1_vec, all_proto1_vectors_dict, top_n=10, exclude_word=word_to_analyze)
    print(f"Most similar words to '{word_to_analyze}' (prototype 1):")
    for word, sim in neighbors_prototype1_initial:
        print(f"   - {word}: {sim:.4f}")

    print(f"\n--- Initial Nearest Neighbors for '{word_to_analyze}' Prototype 2 ---")
    neighbors_prototype2_initial = find_most_similar(proto2_vec, all_proto2_vectors_dict, top_n=10, exclude_word=word_to_analyze)
    print(f"Most similar words to '{word_to_analyze}' (prototype 2):")
    for word, sim in neighbors_prototype2_initial:
        print(f"   - {word}: {sim:.4f}")

    # --- Step 3: Cross-Prototype Similarity Analysis ---
    print(f"\n--- Cross-Prototype Similarity Analysis for '{word_to_analyze}' ---")

    print(f"\n--- How '{word_to_analyze}' Prototype 2 relates to Prototype 1's neighbors ---")
    print(f"Similarity of '{word_to_analyze}' (prototype 2) to words most similar to '{word_to_analyze}' (prototype 1):")
    for word, sim_proto1 in neighbors_prototype1_initial:
        word_id_neighbor = ft.word2id.get(word)
        if word_id_neighbor is not None:
            # Get the Prototype 2 vector for the neighbor word
            neighbor_proto2_vec = ft.emb2[word_id_neighbor]
            # Calculate similarity of original word's Proto2 to the neighbor's Proto2
            sim_cross = cosine_similarity(proto2_vec, neighbor_proto2_vec)
            print(f"   - {word} (Proto1 Sim: {sim_proto1:.4f}) vs {word_to_analyze} (Proto2): {sim_cross:.4f}")
        else:
            print(f"   - {word} (Proto1 Sim: {sim_proto1:.4f}) -- Word not found for cross-analysis in Proto2 set.")


    print(f"\n--- How '{word_to_analyze}' Prototype 1 relates to Prototype 2's neighbors ---")
    print(f"Similarity of '{word_to_analyze}' (prototype 1) to words most similar to '{word_to_analyze}' (prototype 2):")
    for word, sim_proto2 in neighbors_prototype2_initial:
        word_id_neighbor = ft.word2id.get(word)
        if word_id_neighbor is not None:
            # Get the Prototype 1 vector for the neighbor word
            neighbor_proto1_vec = ft.emb[word_id_neighbor]
            # Calculate similarity of original word's Proto1 to the neighbor's Proto1
            sim_cross = cosine_similarity(proto1_vec, neighbor_proto1_vec)
            print(f"   - {word} (Proto2 Sim: {sim_proto2:.4f}) vs {word_to_analyze} (Proto1): {sim_cross:.4f}")
        else:
            print(f"   - {word} (Proto2 Sim: {sim_proto2:.4f}) -- Word not found for cross-analysis in Proto1 set.")


    print("\n--- Analysis Complete ---")
    print("Carefully review the nearest neighbors for each prototype.")
    print("Look for distinct semantic clusters that align with the intended senses.")
    print("For cross-prototype analysis, observe if the similarity drops significantly for 'unrelated' senses.")