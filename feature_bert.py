from sentence_transformers import SentenceTransformer, util
import numpy as np

def get_embeddings(texts):
    """
    Generate semantic embeddings for a list of texts using a pre-trained Sentence Transformer model.
    """
    # Initialize the multilingual BERT model
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    # Encode the texts to obtain embeddings
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def rank_resumes_semantic(job_description, resumes):
    """
    Rank resumes based on semantic similarity to the job description.
    """
    # Combine the job description and resumes into a single list
    all_texts = [job_description] + resumes
    # Get embeddings for all texts
    embeddings = get_embeddings(all_texts)

    # Separate the embeddings for the job description and resumes
    job_embedding = embeddings[0]
    resume_embeddings = embeddings[1:]

    # Calculate cosine similarities between the job description and each resume
    similarities = util.cos_sim(job_embedding, resume_embeddings)[0]

    # Create a list of dictionaries containing index and similarity score
    rankings = sorted(
        [{'Index': idx, 'Similarity': sim.item()} for idx, sim in enumerate(similarities)],
        key=lambda x: x['Similarity'], reverse=True
    )

    return rankings
