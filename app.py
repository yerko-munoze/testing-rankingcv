import streamlit as st
import pandas as pd
from transformation import extract_text_from_pdf
from feature_bert import rank_resumes_semantic

# Set up the Streamlit user interface
st.title("CV Match with Semantic Search")

# Input for the job description
job_description = st.text_area("Enter a job description", height=100)

# Input for essential keywords (optional)
essential_keywords_input = st.text_input("Enter essential keywords separated by commas (optional)", "")
# Process the input into a list of keywords
essential_keywords = [word.strip() for word in essential_keywords_input.split(",")] if essential_keywords_input else []

# Input for uploading multiple PDF CVs
uploaded_pdfs = st.file_uploader("Upload CVs (PDF Format)", type="pdf", accept_multiple_files=True)

# Parameter for selecting the number of CVs to display
top_n = st.number_input("Number of CVs to select", min_value=1, max_value=50, value=5)

# Check if both job description and CVs have been provided
if job_description and uploaded_pdfs:
    resumes = []
    resume_filenames = []

    # Process each uploaded PDF to extract text and filenames
    for pdf in uploaded_pdfs:
        # Extract text from the PDF file
        resume_text = extract_text_from_pdf(pdf)
        resumes.append(resume_text)
        resume_filenames.append(pdf.name)

    # Execute the ranking of CVs using semantic search
    rankings = rank_resumes_semantic(job_description, resumes)

    # Select the top N CVs based on the ranking
    top_rankings = rankings[:top_n]

    # Create a DataFrame to display the results
    df_ranking = pd.DataFrame(top_rankings)
    # Map the index to the corresponding filename
    df_ranking['File'] = df_ranking['Index'].apply(lambda x: resume_filenames[x])
    # Format the similarity score as a percentage
    df_ranking['Similarity Score'] = df_ranking['Similarity'].apply(lambda x: f"{x*100:.2f}%")
    # Add a ranking order column
    df_ranking['Rank'] = df_ranking.index + 1

    # Display the table of results
    st.write("CV Matching Results:")
    st.dataframe(df_ranking[['Rank', 'File', 'Similarity Score']], use_container_width=True)

else:
    # Show a warning if inputs are missing
    st.warning("Please enter a job description and upload at least one CV in PDF format")
